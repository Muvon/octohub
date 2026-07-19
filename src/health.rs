//! Per-model health derived from REAL traffic.
//!
//! Octohub is the chokepoint every model call passes through, so it already sees
//! the only signal that matters: did the last calls to this model succeed, and
//! how fast. Recording that here means health costs nothing — no synthetic probe
//! traffic, no probe credential, and no spend on models nobody asked for.
//!
//! Deliberately in-memory: health is a statement about *now*, so a restart
//! correctly resets every model to "no observations yet" rather than serving a
//! stale verdict from before the process (or the upstream) changed.

use std::collections::HashMap;
use std::sync::RwLock;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// A successful call slower than this means the model is up but struggling.
const DEGRADED_MS: u128 = 8_000;
/// Consecutive failures before a model is called down. One blip is not an
/// outage — a single upstream hiccup must not paint a red light on a pricing page.
const DOWN_AFTER_FAILURES: u32 = 2;

#[derive(Debug, Default, Clone)]
struct ModelStat {
    provider: String,
    ok_total: u64,
    error_total: u64,
    consecutive_failures: u32,
    last_latency_ms: u128,
    last_ok_at: u64,
    last_error_at: u64,
    last_error: String,
}

static REGISTRY: RwLock<Option<HashMap<String, ModelStat>>> = RwLock::new(None);

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Record the outcome of one real request. Called from the metrics funnel that
/// every completion and embedding path already goes through, so no call site can
/// silently forget to report.
pub fn record(model: &str, provider: &str, ok: bool, duration: Duration, error: &str) {
    if model.is_empty() {
        return;
    }
    let Ok(mut guard) = REGISTRY.write() else {
        return; // a poisoned lock must never take the proxy down
    };
    let map = guard.get_or_insert_with(HashMap::new);
    let stat = map.entry(model.to_owned()).or_default();

    if !provider.is_empty() && provider != "unknown" {
        stat.provider = provider.to_owned();
    }
    if ok {
        stat.ok_total += 1;
        stat.consecutive_failures = 0;
        stat.last_latency_ms = duration.as_millis();
        stat.last_ok_at = now_secs();
    } else {
        stat.error_total += 1;
        stat.consecutive_failures += 1;
        stat.last_error_at = now_secs();
        stat.last_error = error.chars().take(200).collect();
    }
}

/// up | degraded | down for one model's accumulated stats.
fn classify(stat: &ModelStat) -> &'static str {
    if stat.consecutive_failures >= DOWN_AFTER_FAILURES {
        return "down";
    }
    if stat.consecutive_failures > 0 {
        return "degraded";
    }
    if stat.last_latency_ms > DEGRADED_MS {
        return "degraded";
    }
    "up"
}

/// Everything observed so far, for `GET /v1/admin/status`. Models absent from
/// this list have had no traffic — the caller reports them as unknown rather
/// than inventing a verdict.
pub fn snapshot() -> serde_json::Value {
    let guard = match REGISTRY.read() {
        Ok(g) => g,
        Err(_) => return serde_json::json!({ "models": [] }),
    };
    let Some(map) = guard.as_ref() else {
        return serde_json::json!({ "models": [] });
    };

    let mut models: Vec<serde_json::Value> = map
        .iter()
        .map(|(model, stat)| {
            serde_json::json!({
                "model": model,
                "provider": stat.provider,
                "status": classify(stat),
                "consecutive_failures": stat.consecutive_failures,
                "last_latency_ms": stat.last_latency_ms as u64,
                "last_ok_at": stat.last_ok_at,
                "last_error_at": stat.last_error_at,
                "last_error": stat.last_error,
                "ok_total": stat.ok_total,
                "error_total": stat.error_total,
            })
        })
        .collect();
    models.sort_by(|a, b| a["model"].as_str().cmp(&b["model"].as_str()));

    serde_json::json!({ "observed_at": now_secs(), "models": models })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stat(consecutive_failures: u32, last_latency_ms: u128) -> ModelStat {
        ModelStat {
            consecutive_failures,
            last_latency_ms,
            ..Default::default()
        }
    }

    #[test]
    fn healthy_fast_call_is_up() {
        assert_eq!(classify(&stat(0, 120)), "up");
    }

    #[test]
    fn healthy_but_slow_call_is_degraded() {
        assert_eq!(classify(&stat(0, DEGRADED_MS)), "up");
        assert_eq!(classify(&stat(0, DEGRADED_MS + 1)), "degraded");
    }

    #[test]
    fn one_failure_is_degraded_not_down() {
        // A single blip must never report an outage.
        assert_eq!(classify(&stat(1, 100)), "degraded");
    }

    #[test]
    fn repeated_failures_are_down() {
        assert_eq!(classify(&stat(DOWN_AFTER_FAILURES, 100)), "down");
        assert_eq!(classify(&stat(9, 100)), "down");
    }

    #[test]
    fn success_clears_the_failure_streak() {
        record(
            "m-recovers",
            "together",
            false,
            Duration::from_millis(10),
            "boom",
        );
        record(
            "m-recovers",
            "together",
            false,
            Duration::from_millis(10),
            "boom",
        );
        let snap = snapshot();
        let down = snap["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["model"] == "m-recovers")
            .unwrap()
            .clone();
        assert_eq!(down["status"], "down");

        record(
            "m-recovers",
            "together",
            true,
            Duration::from_millis(50),
            "",
        );
        let snap = snapshot();
        let up = snap["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["model"] == "m-recovers")
            .unwrap()
            .clone();
        assert_eq!(up["status"], "up");
        assert_eq!(up["consecutive_failures"], 0);
    }
}
