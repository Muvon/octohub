use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::config::Config;

/// Per-provider concurrency gate.
///
/// Holds one `Arc<Semaphore>` per configured provider. Providers that don't
/// appear in `[providers]` are unlimited. Callers acquire a permit before
/// dispatching to the provider; the permit is dropped when the request
/// finishes (or fails), which wakes the next queued waiter.
///
/// When a provider is at its limit, `acquire().await` parks the caller until
/// a permit is available. From the client's perspective this surfaces as a
/// hanging HTTP request — intentional throttling, not an error response.
pub struct ProviderLimiter {
    semaphores: HashMap<String, Arc<Semaphore>>,
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
                    semaphores.insert(key, Arc::new(Semaphore::new(limit as usize)));
                }
            }
        }
        if !semaphores.is_empty() {
            let summary: Vec<String> = semaphores
                .iter()
                .map(|(k, s)| format!("{}={}", k, s.available_permits()))
                .collect();
            tracing::info!("Provider concurrency limits: {}", summary.join(", "));
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
        let sem = self.semaphores.get(&key)?.clone();
        sem.acquire_owned().await.ok()
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
}
