use std::sync::Arc;

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
pub fn authenticate_admin(auth_header: Option<&str>, master_key: Option<&str>) -> bool {
    let Some(expected) = master_key else {
        return false; // Admin endpoints require a master key
    };
    let Some(token) = extract_bearer(auth_header) else {
        return false;
    };
    token == expected
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
        // Admin endpoints disabled when no master key configured
        assert!(!authenticate_admin(Some("Bearer anything"), None));
    }

    #[test]
    fn test_admin_auth_valid() {
        assert!(authenticate_admin(
            Some("Bearer master-secret"),
            Some("master-secret")
        ));
    }

    #[test]
    fn test_admin_auth_invalid() {
        assert!(!authenticate_admin(
            Some("Bearer wrong"),
            Some("master-secret")
        ));
    }

    #[test]
    fn test_admin_auth_missing_header() {
        assert!(!authenticate_admin(None, Some("master-secret")));
    }
}
