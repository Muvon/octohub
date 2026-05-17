mod api;
mod auth;
mod config;
mod http_util;
mod logging;
mod metrics;
mod proxy;
mod storage;

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Context;
use bytes::Bytes;
use clap::Parser;
use http_body_util::Full;
use hyper::service::service_fn;
use hyper::{HeaderMap, Method, Request, Response};
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto;
use tokio::net::TcpListener;
use tracing::Instrument;
use ulid::Ulid;

use config::Config;
use proxy::engine::ProxyEngine;
use proxy::limiter::ProviderLimiter;
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
    let args = Args::parse();
    let mut config = Config::load(args.config)?;

    // Initialize logging (must happen after config load so we have LogFormat)
    logging::init(&config.logging)?;

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
        tracing::warn!("Master API key is empty (server.api_key). Admin endpoints disabled.");
    }

    let config = Arc::new(config);

    // Initialize storage from DSN
    let storage: Arc<dyn Storage> = storage::from_url(&config.server.db_url)?;

    // Per-provider concurrency gate. Unconfigured providers run unthrottled.
    let limiter = Arc::new(ProviderLimiter::from_config(&config));

    // Initialize metrics
    if let Some(handle) = metrics::init(&config.metrics)? {
        let bind = config.metrics.bind.clone();
        let limiter_clone = limiter.clone();
        tokio::spawn(metrics::serve(handle, bind));
        tokio::spawn(metrics::provider_gauge_loop(limiter_clone));
    }

    // Initialize proxy engine
    let engine = Arc::new(ProxyEngine::new(storage.clone(), config.clone(), limiter));

    let addr: SocketAddr = format!("{}:{}", config.server.host, config.server.port).parse()?;
    let listener = TcpListener::bind(addr).await?;

    // Structured startup banner
    let db_kind = if config.server.db_url.starts_with("mysql://") {
        "mysql"
    } else if config.server.db_url.starts_with("postgres://")
        || config.server.db_url.starts_with("postgresql://")
    {
        "postgres"
    } else {
        "sqlite"
    };
    let provider_summary = build_provider_summary(&config);
    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        bind = %addr,
        db = db_kind,
        admin_auth = !config.server.api_key.is_empty(),
        providers = %provider_summary,
        models = config.models.len(),
        embed_models = config.embedding_models.len(),
        metrics = config.metrics.enabled,
        "octohub starting"
    );
    if config.metrics.enabled {
        tracing::info!(bind = %config.metrics.bind, "metrics endpoint listening");
    }

    loop {
        let (stream, remote_addr) = listener.accept().await?;
        let io = TokioIo::new(stream);
        let engine = engine.clone();
        let storage = storage.clone();
        let config = config.clone();

        tokio::task::spawn(async move {
            let service = service_fn(move |req: Request<hyper::body::Incoming>| {
                let engine = engine.clone();
                let storage = storage.clone();
                let config = config.clone();
                async move {
                    Ok::<_, hyper::Error>(route(req, engine, storage, &config, remote_addr).await)
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
                tracing::error!(remote = %remote_addr, error = %err, "connection error");
            }
        });
    }
}

/// Build a concise provider summary for the startup banner, e.g. "ollama:4,minimax:8"
fn build_provider_summary(config: &Config) -> String {
    if config.providers.is_empty() {
        return "none".to_string();
    }
    let mut parts: Vec<String> = config
        .providers
        .iter()
        .map(|(name, cfg)| match cfg.concurrency {
            Some(c) => format!("{}:{}", name, c),
            None => format!("{}:unlimited", name),
        })
        .collect();
    parts.sort();
    parts.join(",")
}

fn classify_route(path: &str) -> &'static str {
    if path == "/v1/completions" {
        "/v1/completions"
    } else if path == "/v1/embeddings" {
        "/v1/embeddings"
    } else if path.starts_with("/v1/admin/keys") {
        "/v1/admin/keys"
    } else if path == "/v1/admin/usage" {
        "/v1/admin/usage"
    } else if path == "/v1/admin/completions" {
        "/v1/admin/completions"
    } else if path == "/v1/admin/embeddings" {
        "/v1/admin/embeddings"
    } else if path == "/health" {
        "/health"
    } else {
        "other"
    }
}

fn extract_request_id(headers: &HeaderMap) -> String {
    if let Some(val) = headers.get("X-Request-Id").and_then(|v| v.to_str().ok()) {
        // Validate: 1-64 chars, alphanumeric + .-_
        if !val.is_empty()
            && val.len() <= 64
            && val
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_')
        {
            return val.to_string();
        }
    }
    Ulid::new().to_string()
}

fn attach_request_id(mut response: Response<BoxBody>, request_id: &str) -> Response<BoxBody> {
    response.headers_mut().insert(
        "X-Request-Id",
        request_id
            .parse()
            .unwrap_or_else(|_| "unknown".parse().unwrap()),
    );
    response
}

async fn route(
    req: Request<hyper::body::Incoming>,
    engine: Arc<ProxyEngine>,
    storage: Arc<dyn Storage>,
    config: &Arc<Config>,
    remote_addr: SocketAddr,
) -> Response<BoxBody> {
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let route_label = classify_route(&path);
    let request_id = extract_request_id(req.headers());
    let effective_remote = http_util::effective_remote(
        req.headers(),
        remote_addr,
        config.server.trust_forwarded_for,
    );

    let span = tracing::info_span!(
        "request",
        req_id = %request_id,
        method = %method,
        route = %route_label,
        path = %path,
        remote = %effective_remote,
        status = tracing::field::Empty,
        dur_ms = tracing::field::Empty,
        api_key_id = tracing::field::Empty,
        model = tracing::field::Empty,
        provider = tracing::field::Empty,
        queued_ms = tracing::field::Empty,
        tok_in = tracing::field::Empty,
        tok_out = tracing::field::Empty,
    );

    let start = Instant::now();
    let _flight = metrics::in_flight_guard(route_label);
    let master_key = config.server.api_key.clone();
    let method_str = method.as_str().to_owned();

    let response = async {
        // Admin endpoints: /v1/admin/* (master key auth)
        if path.starts_with("/v1/admin/") {
            return route_admin(req, method, &path, storage, &master_key).await;
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
    .instrument(span.clone())
    .await;

    let elapsed = start.elapsed();
    span.record("status", response.status().as_u16());
    span.record("dur_ms", elapsed.as_millis() as u64);

    metrics::record_request(
        route_label,
        &method_str,
        response.status().as_u16(),
        elapsed,
    );

    tracing::info!(parent: &span, "request completed");

    attach_request_id(response, &request_id)
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
