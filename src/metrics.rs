use std::time::Duration;

use anyhow::Result;
use bytes::Bytes;
use http_body_util::Full;
use hyper::service::service_fn;
use hyper::{Request, Response};
use hyper_util::rt::TokioIo;
use metrics::{counter, describe_counter, describe_gauge, describe_histogram, gauge, histogram};
use metrics_exporter_prometheus::PrometheusBuilder;
use tokio::net::TcpListener;

use crate::config::MetricsConfig;
use crate::proxy::limiter::ProviderLimiter;

const REQUEST_DURATION_BUCKETS: &[f64] = &[
    0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0,
];
const QUEUE_WAIT_BUCKETS: &[f64] = &[0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0];

pub fn init(cfg: &MetricsConfig) -> Result<Option<metrics_exporter_prometheus::PrometheusHandle>> {
    if !cfg.enabled {
        return Ok(None);
    }

    let recorder = PrometheusBuilder::new();
    let recorder = recorder
        .set_buckets_for_metric(
            metrics_exporter_prometheus::Matcher::Full(
                "octohub_request_duration_seconds".to_owned(),
            ),
            REQUEST_DURATION_BUCKETS,
        )?
        .set_buckets_for_metric(
            metrics_exporter_prometheus::Matcher::Full(
                "octohub_completion_duration_seconds".to_owned(),
            ),
            REQUEST_DURATION_BUCKETS,
        )?
        .set_buckets_for_metric(
            metrics_exporter_prometheus::Matcher::Full(
                "octohub_embedding_duration_seconds".to_owned(),
            ),
            REQUEST_DURATION_BUCKETS,
        )?
        .set_buckets_for_metric(
            metrics_exporter_prometheus::Matcher::Full(
                "octohub_provider_queue_wait_seconds".to_owned(),
            ),
            QUEUE_WAIT_BUCKETS,
        )?;

    let handle = recorder.install_recorder()?;

    // Describe all metrics for HELP text in Prometheus exposition
    describe_counter!(
        "octohub_requests_total",
        "Total number of HTTP requests processed"
    );
    describe_histogram!(
        "octohub_request_duration_seconds",
        "HTTP request duration in seconds"
    );
    describe_gauge!(
        "octohub_requests_in_flight",
        "Number of requests currently being processed"
    );
    describe_counter!(
        "octohub_completions_total",
        "Total number of completion requests"
    );
    describe_histogram!(
        "octohub_completion_duration_seconds",
        "Completion upstream call duration in seconds"
    );
    describe_counter!(
        "octohub_completion_tokens_total",
        "Total tokens processed by completions"
    );
    describe_counter!(
        "octohub_embeddings_total",
        "Total number of embedding requests"
    );
    describe_histogram!(
        "octohub_embedding_duration_seconds",
        "Embedding upstream call duration in seconds"
    );
    describe_counter!(
        "octohub_embedding_tokens_total",
        "Total tokens processed by embeddings"
    );
    describe_histogram!(
        "octohub_provider_queue_wait_seconds",
        "Time spent waiting for a provider concurrency permit"
    );
    describe_gauge!(
        "octohub_provider_permits_available",
        "Available concurrency permits per provider"
    );
    describe_gauge!(
        "octohub_provider_in_flight",
        "In-flight requests per provider"
    );

    // Build info gauge — constant, identifies the running version.
    gauge!("octohub_build_info", "version" => env!("CARGO_PKG_VERSION")).set(1.0);

    Ok(Some(handle))
}

pub async fn serve(handle: metrics_exporter_prometheus::PrometheusHandle, bind: String) {
    let listener = match TcpListener::bind(&bind).await {
        Ok(l) => l,
        Err(e) => {
            tracing::warn!(error = %e, "metrics endpoint bind failed");
            return;
        }
    };

    loop {
        let (stream, _) = match listener.accept().await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(error = %e, "metrics accept failed");
                continue;
            }
        };
        let io = TokioIo::new(stream);
        let handle = handle.clone();

        tokio::task::spawn(async move {
            let service = service_fn(move |_req: Request<hyper::body::Incoming>| {
                let handle = handle.clone();
                async move {
                    if _req.uri().path() == "/metrics" {
                        let body = handle.render();
                        Ok::<_, hyper::Error>(
                            Response::builder()
                                .header("Content-Type", "text/plain; version=0.0.4")
                                .body(Full::new(Bytes::from(body)))
                                .unwrap(),
                        )
                    } else {
                        Ok(Response::builder()
                            .status(404)
                            .body(Full::new(Bytes::from("not found")))
                            .unwrap())
                    }
                }
            });

            if let Err(e) =
                hyper_util::server::conn::auto::Builder::new(hyper_util::rt::TokioExecutor::new())
                    .serve_connection(io, service)
                    .await
            {
                tracing::warn!(error = %e, "metrics connection error");
            }
        });
    }
}

pub async fn provider_gauge_loop(
    limiter: crate::Live<ProviderLimiter>,
    handle: metrics_exporter_prometheus::PrometheusHandle,
) {
    let mut interval = tokio::time::interval(Duration::from_secs(5));
    loop {
        interval.tick().await;
        // Drain accumulated histogram samples into bucketed distributions.
        // Without this, samples only drain on /metrics scrape — an unscraped
        // exporter grows memory unboundedly with every recorded request.
        handle.run_upkeep();
        // Read the current limiter each tick so gauges track post-reload state.
        let snapshot = limiter.read().unwrap().snapshot();
        for (name, available, max) in snapshot {
            gauge!("octohub_provider_permits_available", "provider" => name.clone())
                .set(available as f64);
            gauge!("octohub_provider_in_flight", "provider" => name).set((max - available) as f64);
        }
    }
}

// ── Recording helpers ────────────────────────────────────────────────

pub fn record_request(route: &str, method: &str, status: u16, duration: Duration) {
    let route = route.to_owned();
    let method = method.to_owned();
    let status_str = status.to_string();
    counter!("octohub_requests_total", "route" => route.clone(), "method" => method.clone(), "status" => status_str).increment(1);
    histogram!("octohub_request_duration_seconds", "route" => route, "method" => method)
        .record(duration.as_secs_f64());
}

pub fn in_flight_guard(route: &'static str) -> InFlightGuard {
    gauge!("octohub_requests_in_flight", "route" => route).increment(1.0);
    InFlightGuard { route }
}

pub struct InFlightGuard {
    route: &'static str,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        gauge!("octohub_requests_in_flight", "route" => self.route).decrement(1.0);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn record_completion(
    model: &str,
    provider: &str,
    status: &str,
    duration: Duration,
    tok_in: u64,
    tok_out: u64,
    api_key_id: Option<i64>,
    per_key: bool,
) {
    let model = model.to_owned();
    let provider = provider.to_owned();
    let status = status.to_owned();
    let mut labels: Vec<(&str, String)> = vec![
        ("model", model.clone()),
        ("provider", provider.clone()),
        ("status", status),
    ];
    if per_key {
        if let Some(id) = api_key_id {
            labels.push(("api_key_id", id.to_string()));
        }
    }

    counter!("octohub_completions_total", &labels).increment(1);
    histogram!("octohub_completion_duration_seconds", "model" => model.clone(), "provider" => provider.clone())
        .record(duration.as_secs_f64());

    let in_labels: Vec<(&str, String)> = vec![
        ("model", model.clone()),
        ("provider", provider.clone()),
        ("direction", "in".to_owned()),
    ];
    let out_labels: Vec<(&str, String)> = vec![
        ("model", model),
        ("provider", provider),
        ("direction", "out".to_owned()),
    ];
    counter!("octohub_completion_tokens_total", &in_labels).increment(tok_in);
    counter!("octohub_completion_tokens_total", &out_labels).increment(tok_out);
}

pub fn record_embedding(
    model: &str,
    provider: &str,
    status: &str,
    duration: Duration,
    tok_in: u64,
    api_key_id: Option<i64>,
    per_key: bool,
) {
    let model = model.to_owned();
    let provider = provider.to_owned();
    let status = status.to_owned();
    let mut labels: Vec<(&str, String)> = vec![
        ("model", model.clone()),
        ("provider", provider.clone()),
        ("status", status),
    ];
    if per_key {
        if let Some(id) = api_key_id {
            labels.push(("api_key_id", id.to_string()));
        }
    }

    counter!("octohub_embeddings_total", &labels).increment(1);
    histogram!("octohub_embedding_duration_seconds", "model" => model.clone(), "provider" => provider.clone())
        .record(duration.as_secs_f64());

    let tok_labels: Vec<(&str, String)> = vec![
        ("model", model),
        ("provider", provider),
        ("direction", "in".to_owned()),
    ];
    counter!("octohub_embedding_tokens_total", &tok_labels).increment(tok_in);
}

pub fn record_queue_wait(provider: &str, duration: Duration) {
    let provider = provider.to_owned();
    histogram!("octohub_provider_queue_wait_seconds", "provider" => provider)
        .record(duration.as_secs_f64());
}
