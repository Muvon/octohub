mod api;
mod auth;
mod config;
mod health;
mod http_util;
mod logging;
mod metrics;
mod proxy;
mod storage;

use std::net::SocketAddr;
use std::sync::{Arc, RwLock};
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

/// A config value (or something derived from it) that SIGHUP reload can swap
/// atomically. Reads clone the inner `Arc` under a brief read lock; reload takes
/// the write lock to replace it.
pub type Live<T> = Arc<RwLock<Arc<T>>>;

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

/// Load config from `path` and apply the CLI `--bind` override. Shared by
/// startup and every SIGHUP reload so both build the config identically.
fn load_config(path: Option<String>, bind: Option<&str>) -> anyhow::Result<Config> {
    let mut config = Config::load(path)?;
    if let Some(bind) = bind {
        let parts: Vec<&str> = bind.splitn(2, ':').collect();
        if parts.len() != 2 {
            anyhow::bail!("Invalid bind format '{}': expected HOST:PORT", bind);
        }
        config.server.host = parts[0].to_string();
        config.server.port = parts[1].parse().context("Invalid port in bind argument")?;
    }
    Ok(config)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let config = load_config(args.config.clone(), args.bind.as_deref())?;

    // OpenRouter attribution: octolib defaults X-Title to "octolib" unless
    // these are set, and upstream calls run in this process (not the client's).
    if std::env::var("OPENROUTER_APP_TITLE").is_err() {
        std::env::set_var("OPENROUTER_APP_TITLE", "Octohub");
    }
    if std::env::var("OPENROUTER_HTTP_REFERER").is_err() {
        std::env::set_var(
            "OPENROUTER_HTTP_REFERER",
            "https://octomind.run/products/octohub",
        );
    }
    // Same idea for the User-Agent octolib sends upstream: name this proxy
    // rather than the generic library default.
    octolib::set_user_agent(concat!("Octohub/", env!("CARGO_PKG_VERSION")));

    // Media adapters read their base URL from the process environment, so a
    // configured override has to land there before the first request.
    config.export_media_provider_endpoints();

    // Initialize logging (must happen after config load so we have LogFormat)
    logging::init(&config.logging)?;

    if config.server.api_key.is_empty() {
        tracing::warn!("Master API key is empty (server.api_key). Admin endpoints disabled.");
    }

    // Immutable startup snapshot for the parts that can't be hot-reloaded
    // (bind address, DB, logging/metrics init).
    let cfg = Arc::new(config);

    // Initialize storage from DSN
    let storage: Arc<dyn Storage> = storage::from_url(&cfg.server.db_url)?;

    // Live handles swapped on SIGHUP; requests read a fresh snapshot each time.
    let config_live: Live<Config> = Arc::new(RwLock::new(cfg.clone()));
    let limiter_live: Live<ProviderLimiter> =
        Arc::new(RwLock::new(Arc::new(ProviderLimiter::from_config(&cfg))));

    // Initialize metrics
    if let Some(handle) = metrics::init(&cfg.metrics)? {
        let bind = cfg.metrics.bind.clone();
        tokio::spawn(metrics::serve(handle.clone(), bind));
        tokio::spawn(metrics::provider_gauge_loop(limiter_live.clone(), handle));
    }

    // Reload config + limiter on SIGHUP without dropping the listener. Bind
    // address, DB, logging and metrics stay fixed at their startup values.
    #[cfg(unix)]
    {
        let config_live = config_live.clone();
        let limiter_live = limiter_live.clone();
        let config_path = args.config.clone();
        let bind = args.bind.clone();
        tokio::spawn(async move {
            use tokio::signal::unix::{signal, SignalKind};
            let mut hup = match signal(SignalKind::hangup()) {
                Ok(sig) => sig,
                Err(e) => {
                    tracing::error!(error = %e, "failed to install SIGHUP handler; reload disabled");
                    return;
                }
            };
            while hup.recv().await.is_some() {
                match load_config(config_path.clone(), bind.as_deref()) {
                    Ok(new) => {
                        let new = Arc::new(new);
                        *limiter_live.write().unwrap() =
                            Arc::new(ProviderLimiter::from_config(&new));
                        *config_live.write().unwrap() = new;
                        tracing::info!("SIGHUP: reloaded models, providers and server timeouts");
                    }
                    Err(e) => {
                        tracing::error!(error = %e, "SIGHUP: reload failed, keeping current config");
                    }
                }
            }
        });
    }

    // Initialize proxy engine
    let engine = Arc::new(ProxyEngine::new(storage.clone(), config_live, limiter_live));

    let addr: SocketAddr = format!("{}:{}", cfg.server.host, cfg.server.port).parse()?;
    let listener = TcpListener::bind(addr).await?;

    // Structured startup banner
    let db_kind = if cfg.server.db_url.starts_with("mysql://") {
        "mysql"
    } else if cfg.server.db_url.starts_with("postgres://")
        || cfg.server.db_url.starts_with("postgresql://")
    {
        "postgres"
    } else {
        "sqlite"
    };
    let provider_summary = build_provider_summary(&cfg);
    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        bind = %addr,
        db = db_kind,
        admin_auth = !cfg.server.api_key.is_empty(),
        providers = %provider_summary,
        models = cfg.models.len(),
        embed_models = cfg.embedding_models.len(),
        media_models = cfg.media_models.len(),
        metrics = cfg.metrics.enabled,
        provider_queue_timeout_secs = cfg.server.provider_queue_timeout_secs,
        upstream_timeout_secs = cfg.server.upstream_timeout_secs,
        "octohub starting"
    );
    if cfg.metrics.enabled {
        tracing::info!(bind = %cfg.metrics.bind, "metrics endpoint listening");
    }

    loop {
        // A failed accept (EMFILE, ECONNABORTED, ...) must not kill the server —
        // log, back off briefly, and keep accepting.
        let (stream, remote_addr) = match listener.accept().await {
            Ok(conn) => conn,
            Err(e) => {
                tracing::warn!(error = %e, "accept failed");
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                continue;
            }
        };
        let io = TokioIo::new(stream);
        let engine = engine.clone();
        let storage = storage.clone();

        tokio::task::spawn(async move {
            let service = service_fn(move |req: Request<hyper::body::Incoming>| {
                let engine = engine.clone();
                let storage = storage.clone();
                async move { Ok::<_, hyper::Error>(route(req, engine, storage, remote_addr).await) }
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
    } else if path == "/v1/chat/completions" {
        "/v1/chat/completions"
    } else if path == "/v1/embeddings" {
        "/v1/embeddings"
    } else if path == "/v1/images/generations" {
        "/v1/images/generations"
    } else if path == "/v1/videos" {
        "/v1/videos"
    } else if path == "/v1/audio/speech" {
        "/v1/audio/speech"
    } else if path == "/v1/audio/transcriptions" {
        "/v1/audio/transcriptions"
    } else if path == "/v1/media/models" {
        "/v1/media/models"
    } else if path.starts_with("/v1/media/") {
        // Collapsed so a per-record label cannot explode metric cardinality.
        "/v1/media/{id}"
    } else if path.starts_with("/v1/admin/keys") {
        "/v1/admin/keys"
    } else if path == "/v1/admin/usage" {
        "/v1/admin/usage"
    } else if path == "/v1/admin/completions" {
        "/v1/admin/completions"
    } else if path == "/v1/admin/embeddings" {
        "/v1/admin/embeddings"
    } else if path == "/v1/admin/media" {
        "/v1/admin/media"
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
    remote_addr: SocketAddr,
) -> Response<BoxBody> {
    // Snapshot the live config for this request (picks up SIGHUP reloads).
    let config = engine.config();
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
        chain_ms = tracing::field::Empty,
        upstream_ms = tracing::field::Empty,
        store_ms = tracing::field::Empty,
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
            (Method::POST, "/v1/chat/completions") => {
                api::handler::handle_chat_completion(req, engine, storage).await
            }
            (Method::POST, "/v1/embeddings") => {
                api::handler::handle_create_embedding(req, engine, storage).await
            }
            (Method::POST, "/v1/images/generations") => {
                api::handler::handle_image_generation(req, engine, storage).await
            }
            (Method::POST, "/v1/videos") => {
                api::handler::handle_video_generation(req, engine, storage).await
            }
            (Method::POST, "/v1/audio/speech") => {
                api::handler::handle_speech(req, engine, storage).await
            }
            (Method::POST, "/v1/audio/transcriptions") => {
                api::handler::handle_transcription(req, engine, storage).await
            }
            (Method::GET, "/v1/media/models") => {
                api::handler::handle_media_models(req, engine, storage).await
            }
            (Method::GET, "/health") => api::handler::handle_health(),
            (method, path) => route_media_record(req, method, path, engine, storage).await,
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

/// `/v1/media/{id}` and `/v1/media/{id}/cancel`. Split out because these are
/// the only client routes with a path parameter.
async fn route_media_record(
    req: Request<hyper::body::Incoming>,
    method: Method,
    path: &str,
    engine: Arc<ProxyEngine>,
    storage: Arc<dyn Storage>,
) -> Response<BoxBody> {
    let Some(rest) = path.strip_prefix("/v1/media/") else {
        return not_found();
    };
    let segments: Vec<&str> = rest.split('/').filter(|s| !s.is_empty()).collect();
    match (method, segments.as_slice()) {
        (Method::GET, [id]) => api::handler::handle_media_get(req, engine, storage, id).await,
        (Method::POST, [id, "cancel"]) => {
            api::handler::handle_media_cancel(req, engine, storage, id).await
        }
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
        // POST /v1/admin/keys/:id/models — replace the allowed-models list
        (Method::POST, ["keys", id, "models"]) => {
            let Ok(key_id) = id.parse::<i64>() else {
                return error_response(hyper::StatusCode::BAD_REQUEST, "Invalid key ID");
            };
            api::admin::handle_update_key_models(req, storage, master_key, key_id).await
        }
        // POST /v1/admin/keys/:id/owner — replace the owner grouping + shared
        // concurrency budget (in place, key value untouched)
        (Method::POST, ["keys", id, "owner"]) => {
            let Ok(key_id) = id.parse::<i64>() else {
                return error_response(hyper::StatusCode::BAD_REQUEST, "Invalid key ID");
            };
            api::admin::handle_update_key_owner(req, storage, master_key, key_id).await
        }
        // GET/PUT /v1/admin/owners/:owner/auto — the owner's purpose→alias
        // override map for the virtual `auto` model
        (Method::GET, ["owners", owner, "auto"]) => {
            api::admin::handle_get_owner_auto(req, storage, master_key, owner).await
        }
        (Method::PUT, ["owners", owner, "auto"]) => {
            api::admin::handle_set_owner_auto(req, storage, master_key, owner).await
        }
        // GET /v1/admin/status — per-model health observed from real traffic
        (Method::GET, ["status"]) => api::admin::handle_status(req, master_key).await,
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
        // GET /v1/admin/media
        (Method::GET, ["media"]) => api::admin::handle_list_media(req, storage, master_key).await,
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
