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
    // `auto` is a routing instruction, never a servable model. The success path
    // reports the RESOLVED model while the error path only knows the string the
    // client sent, so an auto-routed failure lands under "auto" while its successes
    // land under the real model — an entry that can only ever accumulate failures,
    // and is therefore guaranteed to reach "down" and stay there. A request that
    // never resolved to a model is evidence about nothing.
    if model.is_empty() || model == crate::config::AUTO_MODEL {
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
///
/// FAILURES ONLY. Duration is recorded for display but must never classify: a
/// completion's total latency is time-to-first-token plus one increment per
/// generated token, so it scales with output length, context size and the model's
/// own speed. A reasoning model writing a long answer breaches any fixed bound
/// while working perfectly — which is exactly what a threshold here did, painting
/// healthy models "degraded" with zero failures. The length-independent metric is
/// TTFT; until it is measured separately, latency is not a verdict.
fn classify(stat: &ModelStat) -> &'static str {
    if stat.consecutive_failures >= DOWN_AFTER_FAILURES {
        return "down";
    }
    if stat.consecutive_failures > 0 {
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
    fn slow_is_not_unhealthy_at_any_duration() {
        // A long answer is not an outage. Total latency scales with output length,
        // so no fixed bound can separate "struggling" from "writing a lot".
        assert_eq!(classify(&stat(0, 8_000)), "up");
        assert_eq!(classify(&stat(0, 600_000)), "up");
    }

    #[test]
    fn auto_is_never_recorded_as_a_model() {
        // Failures on an auto-routed call would otherwise accrue to a phantom model
        // that never receives the matching successes.
        record(
            crate::config::AUTO_MODEL,
            "openrouter",
            false,
            Duration::from_millis(10),
            "boom",
        );
        let snap = snapshot();
        let models = snap["models"].as_array().cloned().unwrap_or_default();
        assert!(
            !models
                .iter()
                .any(|m| m["model"] == crate::config::AUTO_MODEL),
            "'auto' leaked into the health registry"
        );
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
