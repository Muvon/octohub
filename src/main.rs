mod api;
mod auth;
mod config;
mod proxy;
mod storage;

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::Context;
use bytes::Bytes;
use clap::Parser;
use http_body_util::Full;
use hyper::service::service_fn;
use hyper::{Method, Request, Response};
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto;
use tokio::net::TcpListener;

use config::Config;
use proxy::engine::ProxyEngine;
use storage::Storage;

type BoxBody = Full<Bytes>;

#[derive(Parser, Debug)]
#[command(name = "octohub")]
#[command(about = "High-performance LLM proxy server", long_about = None)]
struct Args {
    /// Path to configuration file
    #[arg(short = 'c', long)]
    config: Option<String>,
    /// Bind to HTTP server on host:port (e.g., "0.0.0.0:8080") - overrides config
    #[arg(long)]
    bind: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    let mut config = Config::load(args.config)?;

    // Override bind address if specified
    if let Some(bind) = args.bind {
        let parts: Vec<&str> = bind.splitn(2, ':').collect();
        if parts.len() != 2 {
            anyhow::bail!("Invalid bind format '{}': expected HOST:PORT", bind);
        }
        config.server.host = parts[0].to_string();
        config.server.port = parts[1].parse().context("Invalid port in bind argument")?;
    }

    if config.server.api_key.is_empty() {
        tracing::warn!("Master API key is empty (server.api_key). Consider setting a strong key.");
    }

    let config = Arc::new(config);

    // Initialize storage from DSN
    let storage: Arc<dyn Storage> = storage::from_url(&config.server.db_url)?;
    tracing::info!("Database initialized: {}", config.server.db_url);

    // Initialize proxy engine
    let engine = Arc::new(ProxyEngine::new(storage.clone(), config.clone()));

    let addr: SocketAddr = format!("{}:{}", config.server.host, config.server.port).parse()?;
    let listener = TcpListener::bind(addr).await?;
    tracing::info!("OctoHub server listening on {}", addr);
    tracing::info!("Authentication enabled (master key configured)");

    loop {
        let (stream, remote_addr) = listener.accept().await?;
        let io = TokioIo::new(stream);
        let engine = engine.clone();
        let storage = storage.clone();
        let master_key = config.server.api_key.clone();

        tokio::task::spawn(async move {
            let service = service_fn(move |req: Request<hyper::body::Incoming>| {
                let engine = engine.clone();
                let storage = storage.clone();
                let master_key = master_key.clone();
                async move {
                    Ok::<_, hyper::Error>(
                        route(req, engine, storage, &master_key, remote_addr).await,
                    )
                }
            });

            // auto::Builder negotiates HTTP/1 or HTTP/2 from what the client speaks
            // (HTTP/2 prior-knowledge for plaintext, ALPN for TLS — TLS terminated
            // upstream). HTTP/1 keep-alive stays disabled so each request closes
            // its connection — prevents clients from reusing a stale pooled
            // connection silently dropped by NAT/firewall during long LLM gaps.
            let mut builder = auto::Builder::new(TokioExecutor::new());
            builder.http1().keep_alive(false);
            if let Err(err) = builder.serve_connection(io, service).await {
                tracing::error!("Connection error from {}: {:?}", remote_addr, err);
            }
        });
    }
}

async fn route(
    req: Request<hyper::body::Incoming>,
    engine: Arc<ProxyEngine>,
    storage: Arc<dyn Storage>,
    master_key: &str,
    remote_addr: SocketAddr,
) -> Response<BoxBody> {
    let method = req.method().clone();
    let path = req.uri().path().to_string();

    tracing::info!("{} {} from {}", method, path, remote_addr);

    // Admin endpoints: /v1/admin/* (master key auth)
    if path.starts_with("/v1/admin/") {
        return route_admin(req, method, &path, storage, master_key).await;
    }

    // Client endpoints (api_keys table auth)
    match (method, path.as_str()) {
        (Method::POST, "/v1/completions") => {
            api::handler::handle_create_completion(req, engine, storage).await
        }
        (Method::POST, "/v1/embeddings") => {
            api::handler::handle_create_embedding(req, engine, storage).await
        }
        (Method::GET, "/health") => api::handler::handle_health(),
        _ => not_found(),
    }
}

async fn route_admin(
    req: Request<hyper::body::Incoming>,
    method: Method,
    path: &str,
    storage: Arc<dyn Storage>,
    master_key: &str,
) -> Response<BoxBody> {
    // Parse /v1/admin/keys/:id and /v1/admin/keys/:id/revoke
    let segments: Vec<&str> = path
        .trim_start_matches("/v1/admin/")
        .split('/')
        .filter(|s| !s.is_empty())
        .collect();

    match (method, segments.as_slice()) {
        // POST /v1/admin/keys
        (Method::POST, ["keys"]) => api::admin::handle_create_key(req, storage, master_key).await,
        // GET /v1/admin/keys
        (Method::GET, ["keys"]) => api::admin::handle_list_keys(req, storage, master_key).await,
        // GET /v1/admin/keys/:id
        (Method::GET, ["keys", id]) => {
            let Ok(key_id) = id.parse::<i64>() else {
                return error_response(hyper::StatusCode::BAD_REQUEST, "Invalid key ID");
            };
            api::admin::handle_get_key(req, storage, master_key, key_id).await
        }
        // POST /v1/admin/keys/:id/revoke
        (Method::POST, ["keys", id, "revoke"]) => {
            let Ok(key_id) = id.parse::<i64>() else {
                return error_response(hyper::StatusCode::BAD_REQUEST, "Invalid key ID");
            };
            api::admin::handle_revoke_key(req, storage, master_key, key_id).await
        }
        // GET /v1/admin/usage
        (Method::GET, ["usage"]) => api::admin::handle_usage(req, storage, master_key).await,
        // GET /v1/admin/completions
        (Method::GET, ["completions"]) => {
            api::admin::handle_list_completions(req, storage, master_key).await
        }
        // GET /v1/admin/embeddings
        (Method::GET, ["embeddings"]) => {
            api::admin::handle_list_embeddings(req, storage, master_key).await
        }
        _ => not_found(),
    }
}

fn not_found() -> Response<BoxBody> {
    Response::builder()
        .status(404)
        .header("Content-Type", "application/json")
        .body(Full::new(Bytes::from(
            r#"{"error":{"message":"Not found","type":"not_found"}}"#,
        )))
        .unwrap()
}

fn error_response(status: hyper::StatusCode, message: &str) -> Response<BoxBody> {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": "invalid_request_error"
        }
    });
    let body_bytes = serde_json::to_vec(&body).unwrap_or_default();
    Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(Full::new(Bytes::from(body_bytes)))
        .unwrap()
}
