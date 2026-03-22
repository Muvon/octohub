/// Validate bearer token from Authorization header
/// Returns true if auth is disabled (no API key configured) or token matches
pub fn check_auth(auth_header: Option<&str>, api_key: Option<&str>) -> bool {
    let Some(expected_key) = api_key else {
        return true; // No key configured = auth disabled
    };

    let Some(header) = auth_header else {
        return false; // Key required but no header
    };

    // Accept "Bearer <token>" format
    header
        .strip_prefix("Bearer ")
        .map(|token| token == expected_key)
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_key_configured() {
        assert!(check_auth(None, None));
        assert!(check_auth(Some("Bearer anything"), None));
    }

    #[test]
    fn test_valid_bearer() {
        assert!(check_auth(Some("Bearer secret123"), Some("secret123")));
    }

    #[test]
    fn test_invalid_bearer() {
        assert!(!check_auth(Some("Bearer wrong"), Some("secret123")));
    }

    #[test]
    fn test_missing_header() {
        assert!(!check_auth(None, Some("secret123")));
    }

    #[test]
    fn test_no_bearer_prefix() {
        assert!(!check_auth(Some("secret123"), Some("secret123")));
    }
}
