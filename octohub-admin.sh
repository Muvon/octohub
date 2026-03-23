#!/usr/bin/env bash
# OctoHub Admin CLI
# Manage API keys and query usage/logs via the admin API.
#
# Usage: ./octohub-admin.sh <command> [args]
#
# Environment:
#   OCTOHUB_URL       Base URL of the server (default: http://127.0.0.1:8080)
#   OCTOHUB_MASTER_KEY  Master API key (server.api_key from octohub.toml)

set -euo pipefail

BASE_URL="${OCTOHUB_URL:-http://127.0.0.1:8080}"
MASTER_KEY="${OCTOHUB_MASTER_KEY:-}"

# --- helpers ---

die() { echo "ERROR: $*" >&2; exit 1; }

require_master_key() {
  [[ -v OCTOHUB_MASTER_KEY ]] || die "OCTOHUB_MASTER_KEY is not set"
}

admin_curl() {
  require_master_key
  curl -sf \
    -H "Authorization: Bearer $MASTER_KEY" \
    -H "Content-Type: application/json" \
    "$@"
}

pretty() {
  # pretty-print JSON if jq is available, otherwise raw
  if command -v jq &>/dev/null; then
    jq .
  else
    cat
  fi
}

usage() {
  cat <<EOF
OctoHub Admin CLI

Usage: $(basename "$0") <command> [args]

Environment variables:
  OCTOHUB_URL         Server base URL  (default: http://127.0.0.1:8080)
  OCTOHUB_MASTER_KEY  Master API key   (required for all commands)

Commands:
  keys list                        List all API keys
  keys get <id>                    Get a single API key by ID
  keys create <name>               Create a new API key
  keys revoke <id>                 Revoke an API key

  usage [--key <id,...>] [--bucket hour|day|week|month] [--since <ts>] [--until <ts>]
                                   Aggregated usage stats

  completions [--key <id,...>] [--since <ts>] [--until <ts>] [--limit <n>] [--offset <n>]
                                   List raw completion records

  embeddings  [--key <id,...>] [--since <ts>] [--until <ts>] [--limit <n>] [--offset <n>]
                                   List raw embedding records

  help                             Show this help

Examples:
  OCTOHUB_MASTER_KEY=secret ./octohub-admin.sh keys list
  OCTOHUB_MASTER_KEY=secret ./octohub-admin.sh keys create my-app
  OCTOHUB_MASTER_KEY=secret ./octohub-admin.sh keys revoke 3
  OCTOHUB_MASTER_KEY=secret ./octohub-admin.sh usage --bucket day --key 1,2
  OCTOHUB_MASTER_KEY=secret ./octohub-admin.sh completions --limit 20
EOF
}

# --- commands ---

cmd_keys() {
  local sub="${1:-}"; shift || true
  case "$sub" in
    list)
      admin_curl "$BASE_URL/v1/admin/keys" | pretty
      ;;
    get)
      local id="${1:-}"; [[ -n "$id" ]] || die "Usage: keys get <id>"
      admin_curl "$BASE_URL/v1/admin/keys/$id" | pretty
      ;;
    create)
      local name="${1:-}"; [[ -n "$name" ]] || die "Usage: keys create <name>"
      admin_curl -X POST "$BASE_URL/v1/admin/keys" \
        -d "{\"name\": \"$name\"}" | pretty
      ;;
    revoke)
      local id="${1:-}"; [[ -n "$id" ]] || die "Usage: keys revoke <id>"
      admin_curl -X POST "$BASE_URL/v1/admin/keys/$id/revoke" | pretty
      ;;
    *)
      die "Unknown keys subcommand: '$sub'. Use: list, get, create, revoke"
      ;;
  esac
}

# Build query string from --key, --bucket, --since, --until, --limit, --offset flags
build_query() {
  local params=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --key)    params+=("key_id=$2");    shift 2 ;;
      --bucket) params+=("bucket=$2");   shift 2 ;;
      --since)  params+=("since=$2");    shift 2 ;;
      --until)  params+=("until=$2");    shift 2 ;;
      --limit)  params+=("limit=$2");    shift 2 ;;
      --offset) params+=("offset=$2");   shift 2 ;;
      *) die "Unknown flag: $1" ;;
    esac
  done
  local qs=""
  for p in "${params[@]}"; do
    qs="${qs:+$qs&}$p"
  done
  echo "$qs"
}

cmd_usage() {
  local qs; qs=$(build_query "$@")
  local url="$BASE_URL/v1/admin/usage"
  [[ -n "$qs" ]] && url="$url?$qs"
  admin_curl "$url" | pretty
}

cmd_completions() {
  local qs; qs=$(build_query "$@")
  local url="$BASE_URL/v1/admin/completions"
  [[ -n "$qs" ]] && url="$url?$qs"
  admin_curl "$url" | pretty
}

cmd_embeddings() {
  local qs; qs=$(build_query "$@")
  local url="$BASE_URL/v1/admin/embeddings"
  [[ -n "$qs" ]] && url="$url?$qs"
  admin_curl "$url" | pretty
}

# --- dispatch ---

COMMAND="${1:-help}"; shift || true

case "$COMMAND" in
  keys)         cmd_keys "$@" ;;
  usage)        cmd_usage "$@" ;;
  completions)  cmd_completions "$@" ;;
  embeddings)   cmd_embeddings "$@" ;;
  help|--help|-h) usage ;;
  *) die "Unknown command: '$COMMAND'. Run '$(basename "$0") help' for usage." ;;
esac
