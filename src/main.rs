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
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Method, Request, Response};
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;

use config::Config;
use proxy::engine::ProxyEngine;
use storage::sqlite::SqliteStorage;

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

    let config = Arc::new(config);

    // Initialize storage
    let storage = Arc::new(SqliteStorage::new(&config.server.db_path)?);
    tracing::info!("Database initialized at: {}", config.server.db_path);

    // Initialize proxy engine
    let engine = Arc::new(ProxyEngine::new(storage, config.clone()));

    let addr: SocketAddr = format!("{}:{}", config.server.host, config.server.port).parse()?;
    let listener = TcpListener::bind(addr).await?;
    tracing::info!("OctoHub server listening on {}", addr);

    if config.server.api_key.is_some() {
        tracing::info!("API key authentication enabled");
    } else {
        tracing::warn!("No API key configured - authentication disabled");
    }

    loop {
        let (stream, remote_addr) = listener.accept().await?;
        let io = TokioIo::new(stream);
        let engine = engine.clone();
        let api_key = config.server.api_key.clone();

        tokio::task::spawn(async move {
            let service = service_fn(move |req: Request<hyper::body::Incoming>| {
                let engine = engine.clone();
                let api_key = api_key.clone();
                async move { Ok::<_, hyper::Error>(route(req, engine, api_key, remote_addr).await) }
            });

            if let Err(err) = http1::Builder::new().serve_connection(io, service).await {
                tracing::error!("Connection error from {}: {:?}", remote_addr, err);
            }
        });
    }
}

async fn route(
    req: Request<hyper::body::Incoming>,
    engine: Arc<ProxyEngine>,
    api_key: Option<String>,
    remote_addr: SocketAddr,
) -> Response<BoxBody> {
    let method = req.method().clone();
    let path = req.uri().path().to_string();

    tracing::info!("{} {} from {}", method, path, remote_addr);

    match (method, path.as_str()) {
        (Method::POST, "/v1/responses") => {
            api::handler::handle_create_response(req, engine, api_key).await
        }
        (Method::GET, "/health") => api::handler::handle_health(),
        _ => Response::builder()
            .status(404)
            .header("Content-Type", "application/json")
            .body(Full::new(Bytes::from(
                r#"{"error":{"message":"Not found","type":"not_found"}}"#,
            )))
            .unwrap(),
    }
}
