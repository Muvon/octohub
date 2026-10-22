use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use crate::storage::{ApiKey, Storage};

/// Result of authenticating a client API key (from api_keys table)
pub enum ClientAuth {
	/// Authenticated with this API key
	Ok(ApiKey),
	/// No Authorization header provided
	Missing,
	/// Key not found or revoked
	Invalid,
}

/// Extract bearer token from Authorization header
fn extract_bearer(header: Option<&str>) -> Option<&str> {
	header.and_then(|h| h.strip_prefix("Bearer "))
}

/// Authenticate a client request against the api_keys table.
/// Returns the matching active ApiKey on success.
pub fn authenticate_client(auth_header: Option<&str>, storage: &Arc<dyn Storage>) -> ClientAuth {
	let Some(token) = extract_bearer(auth_header) else {
		return ClientAuth::Missing;
	};

	match storage.get_api_key_by_key(token) {
		Ok(Some(key)) if key.status == "active" => ClientAuth::Ok(key),
		_ => ClientAuth::Invalid,
	}
}

/// Authenticate an admin request against the master key from config.
/// Returns true if the token matches the configured master key.
pub fn authenticate_admin(auth_header: Option<&str>, master_key: &str) -> bool {
	let Some(token) = extract_bearer(auth_header) else {
		return false;
	};
	token == master_key
}

const CREDENTIAL_NONCE_BYTES: usize = 12;
const CREDENTIAL_TAG_BYTES: usize = 16;
// Local writes invalidate at once; the TTL bounds staleness if another octohub instance writes the same DB.
const CREDENTIAL_CACHE_TTL: Duration = Duration::from_secs(60);

pub const BYOK_PROVIDERS: &[(&str, &str)] = &[
	("openai", "OpenAI"),
	("anthropic", "Anthropic"),
	("google-studio", "Google AI Studio"),
	("openrouter", "OpenRouter"),
	("ollama", "Ollama Cloud"),
	("together", "Together AI"),
	("deepseek", "DeepSeek"),
	("zai", "Z.ai"),
	("moonshot", "Moonshot AI"),
	("alibaba", "Alibaba Cloud"),
	("cerebras", "Cerebras"),
];

pub fn byok_provider_label(provider: &str) -> Option<&'static str> {
	BYOK_PROVIDERS
		.iter()
		.find_map(|(id, label)| (*id == provider).then_some(*label))
}

pub struct Credentials {
	storage: Arc<dyn Storage>,
	cipher: Option<aes_gcm::Aes256Gcm>,
	cache: RwLock<CredentialCache>,
	ttl: Duration,
}

/// Decrypted credentials per owner, empty results included. Bounded by the owner count.
#[derive(Default)]
struct CredentialCache {
	entries: HashMap<String, (Instant, Arc<Vec<OwnerCredential>>)>,
	// Bumped by every invalidation so a load that read the database before a write cannot cache it.
	generation: u64,
}

// Neither this type nor the cipher is Debug/Serialize: plaintext exists only for dispatch.
pub struct OwnerCredential {
	pub record: crate::storage::credentials::Credential,
	pub key: String,
}

impl Credentials {
	pub fn from_env(storage: Arc<dyn Storage>) -> anyhow::Result<Self> {
		let key = match std::env::var("OCTOHUB_CREDENTIALS_KEY") {
			Ok(key) => Some(key),
			Err(std::env::VarError::NotPresent) => None,
			Err(_) => anyhow::bail!("OCTOHUB_CREDENTIALS_KEY must be valid base64 of 32 bytes"),
		};
		Self::new(storage, key.as_deref())
	}

	pub fn new(storage: Arc<dyn Storage>, key: Option<&str>) -> anyhow::Result<Self> {
		use aes_gcm::KeyInit;
		use base64::Engine;
		let cipher = match key {
			Some(key) => {
				let decoded = base64::engine::general_purpose::STANDARD
					.decode(key)
					.map_err(|_| anyhow::anyhow!("OCTOHUB_CREDENTIALS_KEY must be valid base64 of 32 bytes"))?;
				Some(
					aes_gcm::Aes256Gcm::new_from_slice(&decoded)
						.map_err(|_| anyhow::anyhow!("OCTOHUB_CREDENTIALS_KEY must decode to exactly 32 bytes"))?,
				)
			}
			None => {
				anyhow::ensure!(
					!storage.has_credentials()?,
					"OCTOHUB_CREDENTIALS_KEY is required while owner credentials exist"
				);
				None
			}
		};
		Ok(Self {
			storage,
			cipher,
			cache: RwLock::default(),
			ttl: CREDENTIAL_CACHE_TTL,
		})
	}

	pub fn enabled(&self) -> bool {
		self.cipher.is_some()
	}

	pub fn put(
		&self,
		owner: &str,
		provider: &str,
		key: &str,
	) -> anyhow::Result<crate::storage::credentials::CredentialChange> {
		use aes_gcm::aead::{Aead, Generate, Payload};
		use base64::Engine;
		let cipher = self
			.cipher
			.as_ref()
			.ok_or_else(|| anyhow::anyhow!("Credentials disabled"))?;
		let nonce = aes_gcm::Nonce::generate();
		let aad = serde_json::to_vec(&(owner, provider))?;
		let encrypted = cipher
			.encrypt(
				&nonce,
				Payload {
					msg: key.as_bytes(),
					aad: &aad,
				},
			)
			.map_err(|_| anyhow::anyhow!("Failed to encrypt owner credential"))?;
		let mut bytes = nonce.to_vec();
		bytes.extend(encrypted);
		Ok(crate::storage::credentials::CredentialChange::Put {
			provider: provider.to_string(),
			secret: base64::engine::general_purpose::STANDARD.encode(bytes),
			key_hint: if key.chars().count() > 4 {
				crate::storage::make_key_hint(key)
			} else {
				// A hint must not reveal a short credential in full.
				"...".to_owned()
			},
		})
	}

	/// Apply an owner credential change. Every write goes through here so routing never serves
	/// a replaced, deleted or reordered key from the cache.
	pub fn update(
		&self,
		owner: &str,
		change: crate::storage::credentials::CredentialChange,
	) -> anyhow::Result<Vec<crate::storage::credentials::Credential>> {
		// Telemetry (last_used_at/last_error) does not change routing; keep the entry warm.
		let routing = !matches!(change, crate::storage::credentials::CredentialChange::Record { .. });
		let result = self.storage.update_credentials(owner, change);
		if routing {
			// Invalidate even on error: a failed write may still have committed.
			let mut cache = self.cache.write().unwrap();
			cache.entries.remove(owner);
			cache.generation += 1;
		}
		result
	}

	/// Decrypted credentials for routing. Cached `record.last_used_at/last_error` may be stale.
	pub fn load(&self, owner: &str) -> anyhow::Result<Arc<Vec<OwnerCredential>>> {
		let generation = {
			let cache = self.cache.read().unwrap();
			if let Some((loaded, credentials)) = cache.entries.get(owner) {
				if loaded.elapsed() < self.ttl {
					return Ok(credentials.clone());
				}
			}
			cache.generation
		};
		// Decryption failures are not cached: the next request retries.
		let credentials = Arc::new(self.decrypt_all(owner)?);
		let mut cache = self.cache.write().unwrap();
		if cache.generation == generation {
			cache
				.entries
				.insert(owner.to_owned(), (Instant::now(), credentials.clone()));
		}
		Ok(credentials)
	}

	fn decrypt_all(&self, owner: &str) -> anyhow::Result<Vec<OwnerCredential>> {
		use aes_gcm::aead::{Aead, Payload};
		use base64::Engine;
		let rows = self.storage.owner_credentials(owner)?;
		rows.into_iter()
			.map(|record| {
				let cipher = self.cipher.as_ref().ok_or_else(|| {
					anyhow::anyhow!("OCTOHUB_CREDENTIALS_KEY is required while owner credentials exist")
				})?;
				let decrypt = || -> anyhow::Result<String> {
					let bytes = base64::engine::general_purpose::STANDARD.decode(&record.secret)?;
					anyhow::ensure!(
						bytes.len() >= CREDENTIAL_NONCE_BYTES + CREDENTIAL_TAG_BYTES,
						"Invalid encrypted credential length"
					);
					let (nonce, ciphertext) = bytes.split_at(CREDENTIAL_NONCE_BYTES);
					let aad = serde_json::to_vec(&(owner, &record.provider))?;
					let plaintext = cipher
						.decrypt(
							aes_gcm::Nonce::try_from(nonce)
								.map_err(|_| anyhow::anyhow!("Invalid credential nonce"))?
								.as_ref(),
							Payload {
								msg: ciphertext,
								aad: &aad,
							},
						)
						.map_err(|_| anyhow::anyhow!("Credential authentication failed"))?;
					Ok(String::from_utf8(plaintext)?)
				};
				let key = decrypt().map_err(|_| {
					anyhow::anyhow!("Cannot decrypt owner credential for provider '{}'", record.provider)
				})?;
				Ok(OwnerCredential { record, key })
			})
			.collect()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_extract_bearer() {
		assert_eq!(extract_bearer(Some("Bearer abc123")), Some("abc123"));
		assert_eq!(extract_bearer(Some("abc123")), None);
		assert_eq!(extract_bearer(None), None);
	}

	#[test]
	fn test_admin_auth_no_master_key() {
		// Empty master key — only matches empty bearer token
		assert!(!authenticate_admin(Some("Bearer anything"), ""));
	}

	#[test]
	fn test_admin_auth_empty_key_match() {
		assert!(authenticate_admin(Some("Bearer "), ""));
	}

	#[test]
	fn test_admin_auth_valid() {
		assert!(authenticate_admin(Some("Bearer master-secret"), "master-secret"));
	}

	#[test]
	fn test_admin_auth_invalid() {
		assert!(!authenticate_admin(Some("Bearer wrong"), "master-secret"));
	}

	#[test]
	fn test_admin_auth_missing_header() {
		assert!(!authenticate_admin(None, "master-secret"));
	}

	fn cached_credentials(ttl: Duration) -> (Arc<dyn Storage>, Credentials) {
		use base64::Engine;
		let storage: Arc<dyn Storage> = Arc::new(crate::storage::sqlite::SqliteStorage::new(":memory:").unwrap());
		let key = base64::engine::general_purpose::STANDARD.encode([7; 32]);
		let mut credentials = Credentials::new(storage.clone(), Some(&key)).unwrap();
		credentials.ttl = ttl;
		(storage, credentials)
	}

	fn providers(loaded: &[OwnerCredential]) -> Vec<&str> {
		loaded.iter().map(|c| c.record.provider.as_str()).collect()
	}

	// Writing to storage directly bypasses the cache, so a stale load proves no storage call.
	#[test]
	fn test_credential_cache_hits_until_routing_write() {
		use crate::storage::credentials::CredentialChange;
		let (storage, credentials) = cached_credentials(CREDENTIAL_CACHE_TTL);
		credentials
			.update("a", credentials.put("a", "zai", "first-key").unwrap())
			.unwrap();
		let first = credentials.load("a").unwrap();
		assert_eq!(first[0].key, "first-key");
		storage
			.update_credentials("a", credentials.put("a", "ollama", "bypass-key").unwrap())
			.unwrap();
		assert!(Arc::ptr_eq(&first, &credentials.load("a").unwrap()));
		credentials
			.update(
				"a",
				CredentialChange::Record {
					provider: "zai".into(),
					secret: first[0].record.secret.clone(),
					error: None,
				},
			)
			.unwrap();
		assert!(storage.owner_credentials("a").unwrap()[0].last_used_at.is_some());
		assert!(
			Arc::ptr_eq(&first, &credentials.load("a").unwrap()),
			"telemetry must not invalidate"
		);
		credentials
			.update("a", CredentialChange::Order(vec!["ollama".into(), "zai".into()]))
			.unwrap();
		let loaded = credentials.load("a").unwrap();
		assert_eq!(providers(&loaded), ["ollama", "zai"]);
		assert_eq!(loaded[0].key, "bypass-key");
		credentials
			.update("a", credentials.put("a", "zai", "second-key").unwrap())
			.unwrap();
		assert_eq!(credentials.load("a").unwrap()[1].key, "second-key");
		credentials
			.update("a", CredentialChange::Delete("ollama".into()))
			.unwrap();
		assert_eq!(providers(&credentials.load("a").unwrap()), ["zai"]);
		credentials.update("a", CredentialChange::Clear).unwrap();
		assert!(credentials.load("a").unwrap().is_empty());
	}

	#[test]
	fn test_credential_cache_remembers_owners_without_credentials() {
		let (storage, credentials) = cached_credentials(CREDENTIAL_CACHE_TTL);
		assert!(credentials.load("a").unwrap().is_empty());
		storage
			.update_credentials("a", credentials.put("a", "zai", "bypass-key").unwrap())
			.unwrap();
		assert!(credentials.load("a").unwrap().is_empty());
		credentials
			.update("a", credentials.put("a", "zai", "admin-key").unwrap())
			.unwrap();
		assert_eq!(credentials.load("a").unwrap()[0].key, "admin-key");
	}

	#[test]
	fn test_credential_cache_expires_and_skips_failures() {
		use crate::storage::credentials::CredentialChange;
		let (storage, credentials) = cached_credentials(Duration::ZERO);
		assert!(credentials.load("a").unwrap().is_empty());
		storage
			.update_credentials("a", credentials.put("a", "zai", "bypass-key").unwrap())
			.unwrap();
		assert_eq!(credentials.load("a").unwrap()[0].key, "bypass-key");

		let (storage, credentials) = cached_credentials(CREDENTIAL_CACHE_TTL);
		storage
			.update_credentials(
				"a",
				CredentialChange::Put {
					provider: "zai".into(),
					secret: "corrupt".into(),
					key_hint: "...hint".into(),
				},
			)
			.unwrap();
		assert!(credentials.load("a").is_err());
		storage
			.update_credentials("a", credentials.put("a", "zai", "repaired-key").unwrap())
			.unwrap();
		assert_eq!(credentials.load("a").unwrap()[0].key, "repaired-key");
	}
}
