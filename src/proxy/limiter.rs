use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::config::{Config, ProviderConfig};

/// Per-provider concurrency gate.
///
/// Holds one `(configured_max, Arc<Semaphore>)` per configured provider.
/// Providers that don't appear in `[providers]` are unlimited. Callers
/// acquire a permit before dispatching to the provider; the permit is
/// dropped when the request finishes (or fails), which wakes the next
/// queued waiter.
///
/// When a provider is at its limit, `acquire().await` parks the caller until
/// a permit is available. From the client's perspective this surfaces as a
/// hanging HTTP request — intentional throttling, not an error response.
pub struct ProviderLimiter {
    semaphores: HashMap<String, (u32, Arc<Semaphore>)>,
}

impl ProviderLimiter {
    /// Build a limiter from `[providers]` config. Only providers with a
    /// positive `concurrency` value get a semaphore; the rest are unlimited.
    pub fn from_config(config: &Config) -> Self {
        let mut semaphores = HashMap::new();
        for (name, provider_cfg) in &config.providers {
            if let Some(limit) = provider_cfg.concurrency {
                if limit > 0 {
                    let key = name.to_ascii_lowercase();
                    semaphores.insert(key, (limit, Arc::new(Semaphore::new(limit as usize))));
                }
            }
        }
        for (name, (limit, _)) in &semaphores {
            tracing::info!(provider = %name, concurrency = limit, "provider concurrency configured");
        }
        Self { semaphores }
    }

    /// Wait for a permit on `provider`. Returns `None` immediately if the
    /// provider is unlimited; the returned `Some(permit)` must be held for
    /// the entire duration of the upstream call.
    ///
    /// Provider names are matched case-insensitively. `Semaphore::close` is
    /// never called by us, so `acquire_owned` cannot fail — we surface that
    /// as `None` rather than a hard error.
    pub async fn acquire(&self, provider: &str) -> Option<OwnedSemaphorePermit> {
        let key = provider.to_ascii_lowercase();
        let (_, sem) = self.semaphores.get(&key)?;
        sem.clone().acquire_owned().await.ok()
    }

    /// Snapshot of all configured providers: (name, available_permits, configured_max).
    pub fn snapshot(&self) -> Vec<(String, usize, usize)> {
        self.semaphores
            .iter()
            .map(|(name, (max, sem))| (name.clone(), sem.available_permits(), *max as usize))
            .collect()
    }
}

/// Per-OWNER concurrency gate — the multi-tenant counterpart of
/// [`ProviderLimiter`].
///
/// Keys that share an `owner` label (see `storage::ApiKey::owner`) share one
/// in-flight budget across completions AND embeddings. The budget rides on
/// the authenticated key row (`owner_concurrency`), so there is no config and
/// nothing to reload: semaphores are created lazily per owner and resized
/// when the stored budget changes (a resize swaps in a fresh semaphore — the
/// permits already in flight drain on the old one, so a brief over-admission
/// window during a downsize is accepted by design).
///
/// Unlike the provider limiter (which parks callers silently — capacity
/// protection is OUR problem), a saturated owner budget waits at most
/// `OWNER_QUEUE_WAIT` and then fails the request: the caller is the
/// bottleneck and must hear about it (the handler maps it to HTTP 429).
pub struct OwnerLimiter {
    slots: std::sync::Mutex<HashMap<String, (u32, Arc<Semaphore>)>>,
}

/// How long a request may queue for an owner slot before 429ing. Long enough
/// to smooth an agent's parallel tool-call burst, short enough that a truly
/// saturated tenant gets told instead of silently serialized.
pub const OWNER_QUEUE_WAIT: std::time::Duration = std::time::Duration::from_secs(30);

impl OwnerLimiter {
    pub fn new() -> Self {
        Self {
            slots: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Wait up to `wait` for a slot in `owner`'s budget of `capacity`.
    /// `Ok(permit)` must be held for the whole upstream call; `Err(())` means
    /// the budget stayed saturated for the full wait.
    pub async fn acquire(
        &self,
        owner: &str,
        capacity: u32,
        wait: std::time::Duration,
    ) -> Result<OwnedSemaphorePermit, ()> {
        let sem = {
            let mut slots = self.slots.lock().unwrap();
            match slots.get(owner) {
                Some((cap, sem)) if *cap == capacity => sem.clone(),
                _ => {
                    let sem = Arc::new(Semaphore::new(capacity as usize));
                    slots.insert(owner.to_string(), (capacity, sem.clone()));
                    sem
                }
            }
        };
        match tokio::time::timeout(wait, sem.acquire_owned()).await {
            Ok(Ok(permit)) => Ok(permit),
            _ => Err(()),
        }
    }
}

impl Default for OwnerLimiter {
    fn default() -> Self {
        Self::new()
    }
}

/// Fixed 60s window backing the `*_per_minute` limits.
const MINUTE_WINDOW: Duration = Duration::from_secs(60);
const SECS_PER_DAY: u64 = 86_400;

/// Per-provider request/token accounting over two fixed windows — 60s and
/// UTC day — backing the `requests_per_minute` / `tokens_per_minute` /
/// `requests_per_day` / `tokens_per_day` provider limits (absent or 0 =
/// unlimited, same contract as `concurrency`).
///
/// Held as a plain `Arc` (like [`OwnerLimiter`]) so counters survive SIGHUP
/// config reloads: only the counters live here, the limits are read from the
/// live config snapshot at admission time. Counters are in-memory only — a
/// process restart forgets the day's usage.
///
/// ponytail: fixed windows can admit ~2x a limit across a window boundary
/// while providers meter rolling windows — configure limits below the real
/// quota; sliding windows are the upgrade if that headroom ever hurts.
pub struct ProviderRateTracker {
    states: std::sync::Mutex<HashMap<String, RateState>>,
}

struct RateState {
    minute_start: Instant,
    minute_requests: u64,
    minute_tokens: u64,
    /// UTC day index (unix seconds / 86400).
    day: u64,
    day_requests: u64,
    day_tokens: u64,
}

impl RateState {
    fn new(now: Instant, day: u64) -> Self {
        Self {
            minute_start: now,
            minute_requests: 0,
            minute_tokens: 0,
            day,
            day_requests: 0,
            day_tokens: 0,
        }
    }
}

/// Effective limit: positive value, or unlimited.
fn limit(value: Option<u64>) -> Option<u64> {
    value.filter(|&n| n > 0)
}

/// Current UTC day index and time remaining until the next UTC midnight.
fn utc_day() -> (u64, Duration) {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    (
        secs / SECS_PER_DAY,
        Duration::from_secs(SECS_PER_DAY - secs % SECS_PER_DAY),
    )
}

impl ProviderRateTracker {
    pub fn new() -> Self {
        Self {
            states: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Admit one request against `provider`'s configured windows, counting
    /// it on success. `Err(retry_after)` means at least one window is
    /// exhausted; the duration is when the LAST violated window resets —
    /// retrying earlier is guaranteed to fail again.
    ///
    /// Tokens are never pre-reserved: in-flight requests contribute via
    /// [`record_tokens`](Self::record_tokens) only once their response
    /// arrives, so a parallel burst can overshoot a token window.
    /// ponytail: pre-reserving max_tokens at admission is the upgrade path.
    pub fn try_admit(&self, provider: &str, cfg: &ProviderConfig) -> Result<(), Duration> {
        let (day, day_remaining) = utc_day();
        self.admit_at(provider, cfg, Instant::now(), day, day_remaining)
    }

    fn admit_at(
        &self,
        provider: &str,
        cfg: &ProviderConfig,
        now: Instant,
        day: u64,
        day_remaining: Duration,
    ) -> Result<(), Duration> {
        if !cfg.has_rate_limits() {
            return Ok(()); // No state tracked — unlimited providers cost nothing.
        }
        let mut states = self.states.lock().unwrap();
        let state = states
            .entry(provider.to_ascii_lowercase())
            .or_insert_with(|| RateState::new(now, day));

        // Roll expired windows before checking.
        if now.duration_since(state.minute_start) >= MINUTE_WINDOW {
            state.minute_start = now;
            state.minute_requests = 0;
            state.minute_tokens = 0;
        }
        if state.day != day {
            state.day = day;
            state.day_requests = 0;
            state.day_tokens = 0;
        }

        let minute_reset = MINUTE_WINDOW - now.duration_since(state.minute_start);
        let mut retry: Option<Duration> = None;
        let mut violated = |window_reset: Duration| {
            retry = Some(retry.map_or(window_reset, |r| r.max(window_reset)));
        };
        if limit(cfg.requests_per_minute).is_some_and(|l| state.minute_requests >= l) {
            violated(minute_reset);
        }
        if limit(cfg.tokens_per_minute).is_some_and(|l| state.minute_tokens >= l) {
            violated(minute_reset);
        }
        if limit(cfg.requests_per_day).is_some_and(|l| state.day_requests >= l) {
            violated(day_remaining);
        }
        if limit(cfg.tokens_per_day).is_some_and(|l| state.day_tokens >= l) {
            violated(day_remaining);
        }
        if let Some(retry) = retry {
            return Err(retry);
        }

        state.minute_requests += 1;
        state.day_requests += 1;
        Ok(())
    }

    /// Add a finished request's provider-reported token usage to the current
    /// windows. No-op for providers without configured limits (no state).
    pub fn record_tokens(&self, provider: &str, tokens: u64) {
        if tokens == 0 {
            return;
        }
        let mut states = self.states.lock().unwrap();
        if let Some(state) = states.get_mut(&provider.to_ascii_lowercase()) {
            state.minute_tokens += tokens;
            state.day_tokens += tokens;
        }
    }
}

impl Default for ProviderRateTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Consecutive provider-side failures before a provider is put on cooldown.
/// octolib already retries transient errors internally, so three surfaced
/// failures in a row mean the provider is genuinely struggling — one blip
/// must not sideline it.
const COOLDOWN_FAILURE_THRESHOLD: u32 = 3;

/// Per-provider failure tracking behind `server.provider_error_cooldown_secs`.
///
/// After [`COOLDOWN_FAILURE_THRESHOLD`] consecutive provider-side failures
/// the provider "cools" for the configured duration. Cooling providers are
/// DEPRIORITIZED at resolution — sorted behind healthy candidates — never
/// hard-blocked: a model whose every candidate is cooling still dispatches
/// rather than failing, and the first request after the cooldown lapses is
/// the recovery probe. Any success resets the streak. Plain in-memory state
/// held as `Arc` (like [`ProviderRateTracker`]), so it survives SIGHUP.
pub struct ProviderHealth {
    states: std::sync::Mutex<HashMap<String, HealthState>>,
}

#[derive(Default)]
struct HealthState {
    consecutive_failures: u32,
    cooling_until: Option<Instant>,
}

impl ProviderHealth {
    pub fn new() -> Self {
        Self {
            states: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Count a provider-side failure; trips the cooldown at the threshold.
    /// No-op when `cooldown` is zero (feature disabled) — nothing is tracked.
    pub fn record_failure(&self, provider: &str, cooldown: Duration) {
        self.record_failure_at(provider, cooldown, Instant::now())
    }

    fn record_failure_at(&self, provider: &str, cooldown: Duration, now: Instant) {
        if cooldown.is_zero() {
            return;
        }
        let mut states = self.states.lock().unwrap();
        let state = states.entry(provider.to_ascii_lowercase()).or_default();
        state.consecutive_failures += 1;
        if state.consecutive_failures >= COOLDOWN_FAILURE_THRESHOLD {
            state.cooling_until = Some(now + cooldown);
            tracing::warn!(
                provider = %provider,
                failures = state.consecutive_failures,
                cooldown_s = cooldown.as_secs(),
                "provider cooling down after consecutive failures"
            );
        }
    }

    /// Any success ends the failure streak and lifts an active cooldown.
    pub fn record_success(&self, provider: &str) {
        let mut states = self.states.lock().unwrap();
        if let Some(state) = states.get_mut(&provider.to_ascii_lowercase()) {
            state.consecutive_failures = 0;
            state.cooling_until = None;
        }
    }

    /// Whether `provider` is currently inside a cooldown window.
    pub fn is_cooling(&self, provider: &str) -> bool {
        self.is_cooling_at(provider, Instant::now())
    }

    fn is_cooling_at(&self, provider: &str, now: Instant) -> bool {
        let states = self.states.lock().unwrap();
        states
            .get(&provider.to_ascii_lowercase())
            .and_then(|s| s.cooling_until)
            .is_some_and(|until| now < until)
    }
}

impl Default for ProviderHealth {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, ProviderConfig};
    use std::collections::HashMap;
    use std::time::Duration;
    use tokio::time::timeout;

    fn limiter_with(provider: &str, concurrency: u32) -> ProviderLimiter {
        let mut providers = HashMap::new();
        providers.insert(
            provider.to_string(),
            ProviderConfig {
                concurrency: Some(concurrency),
                ..Default::default()
            },
        );
        let config = Config {
            server: Default::default(),
            models: HashMap::new(),
            embedding_models: HashMap::new(),
            providers,
            logging: Default::default(),
            metrics: Default::default(),
        };
        ProviderLimiter::from_config(&config)
    }

    #[tokio::test]
    async fn unconfigured_provider_is_unlimited() {
        let limiter = limiter_with("ollama", 1);
        let permit = limiter.acquire("openai").await;
        assert!(
            permit.is_none(),
            "openai isn't configured, acquire returns None"
        );
    }

    #[tokio::test]
    async fn acquire_is_case_insensitive() {
        let limiter = limiter_with("ollama", 1);
        let permit = limiter.acquire("OLLAMA").await;
        assert!(permit.is_some());
    }

    #[tokio::test]
    async fn excess_requests_queue_until_permit_released() {
        // concurrency=1 — first acquire takes it, second must wait. Verify
        // the waiter actually parks (would otherwise complete immediately)
        // and unblocks the moment the first permit drops.
        let limiter = std::sync::Arc::new(limiter_with("ollama", 1));
        let held = limiter.acquire("ollama").await;
        assert!(held.is_some());

        let limiter2 = limiter.clone();
        let waiter = tokio::spawn(async move { limiter2.acquire("ollama").await });

        // Confirm the second acquire is blocked.
        assert!(timeout(Duration::from_millis(50), async {
            // We can't peek into the JoinHandle without consuming it, so
            // race a sleep against a clone of the limiter's future.
            tokio::time::sleep(Duration::from_millis(30)).await;
        })
        .await
        .is_ok());
        assert!(!waiter.is_finished(), "second acquire must still be queued");

        drop(held);
        let result = timeout(Duration::from_millis(100), waiter).await;
        let joined = result.expect("waiter should unblock after permit drop");
        assert!(joined.unwrap().is_some());
    }

    #[tokio::test]
    async fn owner_budget_is_shared_and_times_out() {
        // capacity=1: the first acquire holds the only slot; a second acquire
        // (any key of the same owner) must time out with Err — that's the 429.
        let limiter = OwnerLimiter::new();
        let held = limiter
            .acquire("acct-1", 1, Duration::from_millis(50))
            .await
            .expect("first slot");
        let second = limiter
            .acquire("acct-1", 1, Duration::from_millis(50))
            .await;
        assert!(second.is_err(), "saturated owner budget must reject");

        // A DIFFERENT owner is unaffected.
        let other = limiter
            .acquire("acct-2", 1, Duration::from_millis(50))
            .await;
        assert!(other.is_ok(), "owners must not share budgets");

        // Releasing frees the slot for the same owner.
        drop(held);
        let third = limiter
            .acquire("acct-1", 1, Duration::from_millis(50))
            .await;
        assert!(third.is_ok(), "released slot must be reusable");
    }

    fn rate_cfg(
        rpm: Option<u64>,
        tpm: Option<u64>,
        rpd: Option<u64>,
        tpd: Option<u64>,
    ) -> ProviderConfig {
        ProviderConfig {
            requests_per_minute: rpm,
            tokens_per_minute: tpm,
            requests_per_day: rpd,
            tokens_per_day: tpd,
            ..Default::default()
        }
    }

    const DAY_LEFT: Duration = Duration::from_secs(1000);

    #[test]
    fn rpm_exhausts_and_resets_next_window() {
        let tracker = ProviderRateTracker::new();
        let cfg = rate_cfg(Some(2), None, None, None);
        let t0 = Instant::now();

        assert!(tracker.admit_at("moonshot", &cfg, t0, 0, DAY_LEFT).is_ok());
        assert!(tracker.admit_at("moonshot", &cfg, t0, 0, DAY_LEFT).is_ok());
        let retry = tracker
            .admit_at("moonshot", &cfg, t0, 0, DAY_LEFT)
            .expect_err("third request must exceed rpm=2");
        assert!(retry <= MINUTE_WINDOW && retry > Duration::ZERO);

        // Next fixed window: counters reset, requests admitted again.
        let t1 = t0 + MINUTE_WINDOW + Duration::from_secs(1);
        assert!(tracker.admit_at("moonshot", &cfg, t1, 0, DAY_LEFT).is_ok());
    }

    #[test]
    fn recorded_tokens_block_and_expire_with_the_window() {
        let tracker = ProviderRateTracker::new();
        let cfg = rate_cfg(None, Some(100), None, None);
        let t0 = Instant::now();

        assert!(tracker.admit_at("openai", &cfg, t0, 0, DAY_LEFT).is_ok());
        tracker.record_tokens("openai", 150);
        assert!(
            tracker.admit_at("openai", &cfg, t0, 0, DAY_LEFT).is_err(),
            "tpm budget already spent"
        );

        let t1 = t0 + MINUTE_WINDOW;
        assert!(
            tracker.admit_at("openai", &cfg, t1, 0, DAY_LEFT).is_ok(),
            "minute tokens reset with the window"
        );
    }

    #[test]
    fn day_limits_block_until_next_utc_day() {
        let tracker = ProviderRateTracker::new();
        let cfg = rate_cfg(None, None, Some(1), Some(500));
        let t0 = Instant::now();

        assert!(tracker.admit_at("minimax", &cfg, t0, 10, DAY_LEFT).is_ok());
        let retry = tracker
            .admit_at("minimax", &cfg, t0, 10, DAY_LEFT)
            .expect_err("rpd=1 exhausted");
        assert_eq!(retry, DAY_LEFT, "retry points at the UTC day rollover");

        // Same instant, next UTC day: counters reset.
        assert!(tracker.admit_at("minimax", &cfg, t0, 11, DAY_LEFT).is_ok());
        tracker.record_tokens("minimax", 600);
        assert!(
            tracker.admit_at("minimax", &cfg, t0, 11, DAY_LEFT).is_err(),
            "tpd=500 exceeded by recorded usage"
        );
    }

    #[test]
    fn retry_after_is_the_latest_violated_window() {
        // Both rpm and rpd exhausted: retrying at the minute reset would
        // still hit the day cap, so retry_after must be the day reset.
        let tracker = ProviderRateTracker::new();
        let cfg = rate_cfg(Some(1), None, Some(1), None);
        let t0 = Instant::now();

        assert!(tracker.admit_at("openai", &cfg, t0, 0, DAY_LEFT).is_ok());
        let retry = tracker
            .admit_at("openai", &cfg, t0, 0, DAY_LEFT)
            .expect_err("both windows exhausted");
        assert_eq!(retry, DAY_LEFT);
    }

    #[test]
    fn zero_and_unset_limits_admit_everything() {
        let tracker = ProviderRateTracker::new();
        let cfg = rate_cfg(Some(0), Some(0), None, None);
        let t0 = Instant::now();
        for _ in 0..100 {
            assert!(tracker.admit_at("ollama", &cfg, t0, 0, DAY_LEFT).is_ok());
        }
        // No state tracked → recording tokens is a harmless no-op.
        tracker.record_tokens("ollama", 1_000_000);
        assert!(tracker.admit_at("ollama", &cfg, t0, 0, DAY_LEFT).is_ok());
    }

    #[test]
    fn rate_state_is_shared_case_insensitively() {
        let tracker = ProviderRateTracker::new();
        let cfg = rate_cfg(Some(1), None, None, None);
        let t0 = Instant::now();

        assert!(tracker.admit_at("OpenAI", &cfg, t0, 0, DAY_LEFT).is_ok());
        assert!(
            tracker.admit_at("openai", &cfg, t0, 0, DAY_LEFT).is_err(),
            "differently-cased names must share one budget"
        );
    }

    #[test]
    fn cooldown_trips_at_threshold_and_expires() {
        let health = ProviderHealth::new();
        let cooldown = Duration::from_secs(120);
        let t0 = Instant::now();

        health.record_failure_at("zai", cooldown, t0);
        health.record_failure_at("zai", cooldown, t0);
        assert!(
            !health.is_cooling_at("zai", t0),
            "below threshold — still healthy"
        );

        health.record_failure_at("zai", cooldown, t0);
        assert!(health.is_cooling_at("zai", t0), "third failure trips it");
        assert!(
            !health.is_cooling_at("zai", t0 + cooldown),
            "cooldown lapses on its own"
        );
    }

    #[test]
    fn success_resets_streak_and_lifts_cooldown() {
        let health = ProviderHealth::new();
        let cooldown = Duration::from_secs(120);
        let t0 = Instant::now();

        for _ in 0..3 {
            health.record_failure_at("openai", cooldown, t0);
        }
        assert!(health.is_cooling_at("openai", t0));

        health.record_success("openai");
        assert!(
            !health.is_cooling_at("openai", t0),
            "success lifts cooldown"
        );

        // Streak restarts from zero: two failures don't re-trip it.
        health.record_failure_at("openai", cooldown, t0);
        health.record_failure_at("openai", cooldown, t0);
        assert!(!health.is_cooling_at("openai", t0));
    }

    #[test]
    fn zero_cooldown_disables_health_tracking() {
        let health = ProviderHealth::new();
        let t0 = Instant::now();
        for _ in 0..10 {
            health.record_failure_at("minimax", Duration::ZERO, t0);
        }
        assert!(!health.is_cooling_at("minimax", t0));
    }

    #[test]
    fn cooldown_is_case_insensitive() {
        let health = ProviderHealth::new();
        let cooldown = Duration::from_secs(60);
        let t0 = Instant::now();
        for _ in 0..3 {
            health.record_failure_at("OpenAI", cooldown, t0);
        }
        assert!(health.is_cooling_at("openai", t0));
    }

    #[tokio::test]
    async fn owner_budget_resize_takes_effect() {
        // A plan upgrade mid-flight: capacity change swaps the semaphore, so
        // the next acquire sees the new budget immediately.
        let limiter = OwnerLimiter::new();
        let _held = limiter
            .acquire("acct-1", 1, Duration::from_millis(50))
            .await
            .expect("first slot");
        // Budget grows 1 → 2: a fresh semaphore admits even though the old
        // one is fully held.
        let upgraded = limiter
            .acquire("acct-1", 2, Duration::from_millis(50))
            .await;
        assert!(
            upgraded.is_ok(),
            "resized budget must apply to new acquires"
        );
    }
}
