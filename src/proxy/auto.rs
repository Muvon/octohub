//! Resolution of the virtual `auto` model into a real `[models]` alias.
//!
//! `auto` is ONE indirection: it resolves to another model *alias*, never to a
//! `provider:model` pair — the alias then goes through the ordinary candidate
//! routing. That keeps this module ignorant of providers, and routing ignorant
//! of purposes.
//!
//! The chain is caller-first, then deployment floor:
//!
//! 1. owner map `[purpose]` — the key owner's stored override
//! 2. owner map `[default]`
//! 3. config `[auto].[purpose]` — the deployment's floor
//! 4. config `[auto].[default]`
//!
//! Within each map, purposes fall back hierarchically by dash segment
//! (`supervisor-gate` → `supervisor` → `default`), so one `supervisor` entry
//! covers every supervisor mechanic until a specific one is redefined.
//!
//! An owner who set only `default` means "use this for everything" — their
//! choice beats the config's purpose-specific entry, which is why the owner
//! chain runs to its own default before the config chain starts. Entries whose
//! target is not a known `[models]` alias are skipped (an owner map may
//! outlive a config edit); the config floor is load-validated, so the chain
//! terminates in practice whenever `[auto]` is configured.

use std::collections::HashMap;

use crate::config::AUTO_DEFAULT_KEY;

/// Purposes are hierarchical by dash segments: `supervisor-gate` falls back to
/// `supervisor` before a map's `default`. One `supervisor` entry covers every
/// supervisor mechanic; a specific `supervisor-gate` entry overrides just that
/// one. Returns the match candidates for `purpose` within ONE map, most
/// specific first.
fn lookup<'m>(map: &'m HashMap<String, String>, purpose: Option<&str>) -> Vec<&'m String> {
    let mut out = Vec::new();
    if let Some(p) = purpose {
        let mut key = p;
        loop {
            if let Some(target) = map.get(key) {
                out.push(target);
            }
            match key.rfind('-') {
                Some(i) => key = &key[..i],
                None => break,
            }
        }
    }
    if let Some(target) = map.get(AUTO_DEFAULT_KEY) {
        out.push(target);
    }
    out
}

/// Pick the model alias for an `auto` request. `purpose` is the raw
/// `X-Model-Purpose` header value (unknown/missing purposes just fall to the
/// defaults — a typo must degrade, not fail). `known` filters each candidate
/// against the live `[models]` table. `None` = nothing usable anywhere, which
/// the caller reports as an unresolvable model.
pub fn resolve(
    purpose: Option<&str>,
    owner_map: Option<&HashMap<String, String>>,
    config_map: &HashMap<String, String>,
    known: impl Fn(&str) -> bool,
) -> Option<String> {
    let owner_chain = owner_map.map(|m| lookup(m, purpose)).unwrap_or_default();
    let config_chain = lookup(config_map, purpose);
    for target in owner_chain.into_iter().chain(config_chain) {
        if known(target) {
            return Some(target.clone());
        }
        tracing::warn!(target = %target, "auto map entry points at unknown model — skipping");
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn owner_purpose_wins_over_everything() {
        let owner = map(&[("compression", "cheap"), ("default", "mid")]);
        let config = map(&[("compression", "other"), ("default", "strong")]);
        let got = resolve(Some("compression"), Some(&owner), &config, |_| true);
        assert_eq!(got.as_deref(), Some("cheap"));
    }

    #[test]
    fn owner_default_beats_config_purpose() {
        // An owner who set only `default` said "use this for everything".
        let owner = map(&[("default", "mine")]);
        let config = map(&[("compression", "cheap"), ("default", "strong")]);
        let got = resolve(Some("compression"), Some(&owner), &config, |_| true);
        assert_eq!(got.as_deref(), Some("mine"));
    }

    #[test]
    fn config_floor_serves_unknown_owner_and_missing_purpose() {
        let config = map(&[("compression", "cheap"), ("default", "strong")]);
        // No owner map at all → config purpose.
        let got = resolve(Some("compression"), None, &config, |_| true);
        assert_eq!(got.as_deref(), Some("cheap"));
        // No purpose header → config default.
        let got = resolve(None, None, &config, |_| true);
        assert_eq!(got.as_deref(), Some("strong"));
        // Unknown purpose degrades to default, never fails.
        let got = resolve(Some("no-such-purpose"), None, &config, |_| true);
        assert_eq!(got.as_deref(), Some("strong"));
    }

    #[test]
    fn stale_owner_entries_fall_through_to_the_floor() {
        // The owner map references a model that was removed from [models]
        // after the map was stored — skip it rather than 4xx the request.
        let owner = map(&[("compression", "removed"), ("default", "also-removed")]);
        let config = map(&[("default", "strong")]);
        let got = resolve(Some("compression"), Some(&owner), &config, |m| {
            m == "strong"
        });
        assert_eq!(got.as_deref(), Some("strong"));
    }

    #[test]
    fn purpose_hierarchy_falls_back_by_dash_segment() {
        // One `supervisor` row covers all mechanics...
        let owner = map(&[("supervisor", "family"), ("default", "mine")]);
        let got = resolve(
            Some("supervisor-gate"),
            Some(&owner),
            &HashMap::new(),
            |_| true,
        );
        assert_eq!(got.as_deref(), Some("family"));

        // ...until a specific mechanic is redefined.
        let owner = map(&[("supervisor-gate", "special"), ("supervisor", "family")]);
        let got = resolve(
            Some("supervisor-gate"),
            Some(&owner),
            &HashMap::new(),
            |_| true,
        );
        assert_eq!(got.as_deref(), Some("special"));
        let got = resolve(
            Some("supervisor-condense"),
            Some(&owner),
            &HashMap::new(),
            |_| true,
        );
        assert_eq!(got.as_deref(), Some("family"));

        // The owner's family entry still beats the config's exact entry —
        // owner-first, then specificity within each map.
        let owner = map(&[("supervisor", "family")]);
        let config = map(&[("supervisor-gate", "cfg-exact"), ("default", "floor")]);
        let got = resolve(Some("supervisor-gate"), Some(&owner), &config, |_| true);
        assert_eq!(got.as_deref(), Some("family"));
    }

    #[test]
    fn nothing_usable_is_none() {
        let config = map(&[("default", "gone")]);
        assert_eq!(resolve(None, None, &config, |_| false), None);
        assert_eq!(resolve(Some("x"), None, &HashMap::new(), |_| true), None);
    }
}
