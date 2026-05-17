use std::net::{IpAddr, SocketAddr};

use hyper::HeaderMap;

/// Determine the effective remote address for a request.
///
/// When `trust` is false, always returns the peer address.
/// When `trust` is true, tries `Forwarded` (RFC 7239) then `X-Forwarded-For`.
pub fn effective_remote(headers: &HeaderMap, peer: SocketAddr, trust: bool) -> String {
    if !trust {
        return peer.to_string();
    }

    // Try Forwarded header (RFC 7239)
    if let Some(fwd) = headers.get("forwarded").and_then(|v| v.to_str().ok()) {
        if let Some(ip) = parse_forwarded_for(fwd) {
            return ip;
        }
    }

    // Fall back to X-Forwarded-For: first entry
    if let Some(xff) = headers.get("x-forwarded-for").and_then(|v| v.to_str().ok()) {
        if let Some(first) = xff.split(',').next() {
            let trimmed = first.trim();
            if trimmed.parse::<IpAddr>().is_ok() {
                return trimmed.to_string();
            }
        }
    }

    peer.to_string()
}

/// Parse the first `for=` parameter from a `Forwarded` header value.
/// Handles: `for=192.0.2.43`, `for="[2001:db8::1]:8080"`, quoted forms.
fn parse_forwarded_for(value: &str) -> Option<String> {
    for part in value.split(';') {
        for pair in part.split(',') {
            let trimmed = pair.trim();
            let Some(for_val) = trimmed.strip_prefix("for=") else {
                continue;
            };

            // Strip surrounding quotes
            let for_val = for_val
                .strip_prefix('"')
                .and_then(|v| v.strip_suffix('"'))
                .unwrap_or(for_val);

            // Strip port suffix for bracketed IPv6: [::1]:8080 → [::1]
            let candidate = if for_val.starts_with('[') {
                // IPv6 bracketed form
                if let Some(close) = for_val.find(']') {
                    &for_val[..=close]
                } else {
                    for_val
                }
            } else {
                // IPv4 — strip :port
                for_val.split(':').next().unwrap_or(for_val)
            };

            // Validate: for bracketed IPv6 we need to strip brackets for IpAddr parse
            let ip_str = if candidate.starts_with('[') && candidate.ends_with(']') {
                &candidate[1..candidate.len() - 1]
            } else {
                candidate
            };

            if ip_str.parse::<IpAddr>().is_ok() {
                return Some(candidate.to_string());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyper::header::HeaderValue;

    fn headers() -> HeaderMap {
        HeaderMap::new()
    }

    fn peer() -> SocketAddr {
        "10.0.0.1:12345".parse().unwrap()
    }

    #[test]
    fn no_headers_returns_peer() {
        assert_eq!(
            effective_remote(&headers(), peer(), false),
            "10.0.0.1:12345"
        );
    }

    #[test]
    fn trust_false_ignores_headers() {
        let mut h = headers();
        h.insert("x-forwarded-for", HeaderValue::from_static("1.2.3.4"));
        assert_eq!(effective_remote(&h, peer(), false), "10.0.0.1:12345");
    }

    #[test]
    fn forwarded_ipv4() {
        let mut h = headers();
        h.insert("forwarded", HeaderValue::from_static("for=192.0.2.43"));
        assert_eq!(effective_remote(&h, peer(), true), "192.0.2.43");
    }

    #[test]
    fn forwarded_ipv6_with_port() {
        let mut h = headers();
        h.insert(
            "forwarded",
            HeaderValue::from_static("for=\"[2001:db8::1]:8080\""),
        );
        assert_eq!(effective_remote(&h, peer(), true), "[2001:db8::1]");
    }

    #[test]
    fn xff_multi_hop_uses_first() {
        let mut h = headers();
        h.insert(
            "x-forwarded-for",
            HeaderValue::from_static("203.0.113.5, 10.0.0.1"),
        );
        assert_eq!(effective_remote(&h, peer(), true), "203.0.113.5");
    }

    #[test]
    fn malformed_xff_falls_back() {
        let mut h = headers();
        h.insert("x-forwarded-for", HeaderValue::from_static("not-an-ip"));
        assert_eq!(effective_remote(&h, peer(), true), "10.0.0.1:12345");
    }

    #[test]
    fn no_forward_headers_returns_peer_with_trust() {
        assert_eq!(effective_remote(&headers(), peer(), true), "10.0.0.1:12345");
    }
}
