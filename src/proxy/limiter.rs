use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::config::Config;

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
