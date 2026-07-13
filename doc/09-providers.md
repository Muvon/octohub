# 09 — Providers

OctoHub delegates the actual LLM call to **[octolib](https://crates.io/crates/octolib)**
(referenced in `Cargo.toml` as the `octolib` dependency, default
features disabled). octolib ships a `ProviderFactory` that knows how to
parse a `provider:model` string, construct the right client, and
dispatch the request.

This doc lists the providers the upstream library exposes today. **It
does not enumerate them exhaustively** — the source of truth is
octolib's `ProviderFactory::parse_model`. If you build octolib from a
newer revision, you may see additional providers not listed here.

Source: octolib `ProviderFactory::parse_model` (consumed via
`src/proxy/engine.rs`; called for every model resolution).

## The `provider:model` syntax

Every model that OctoHub dispatches is identified by a string of the
form `provider:model`. Examples:

```
openai:gpt-5
anthropic:claude-sonnet-4-5
ollama:llama3.3:70b-instruct-q5_K_M
minimax:minimax-m2.7
voyage:voyage-4
deepseek:deepseek-reasoner
```

The provider prefix is the **lowercase** short name that octolib
recognizes. It must contain a literal colon. If the `model` field on an
incoming request has no colon, OctoHub looks it up in the `[models]`
(or `[embedding_models]`) table in `octohub.toml` and picks a random
entry — see [03 — Configuration](./03-configuration.md#models-and-embedding_models).

## Naming in the config and metrics

The provider string is used verbatim as the `provider` label in
metrics, the `provider` field in stored completions, and the
`provider` value in logs. The case you write it in **is** the case that
shows up — `ollama:llama3.3` produces a metric label `provider="ollama"`.

The `[providers.<name>]` config keys (used to set per-provider
concurrency) are matched **case-insensitively** at runtime
(`src/proxy/limiter.rs:50`). So `[providers.ollama]` matches both
`ollama` and `Ollama` from the metric side.

## Common providers

The following are the providers most likely to be in active use. For
each, the practical notes you need:

### `openai`

OpenAI's hosted models: GPT-5 family, GPT-4o, o1 / o3 reasoning models,
embeddings (`text-embedding-3-*`), etc.

- API key typically from `OPENAI_API_KEY` env var, read by octolib's
  OpenAI provider implementation.
- **Reasoning models** (o1, o3) accept `reasoning_effort` (low/medium/
  high/xhigh/max) — the value is proxied through unchanged by OctoHub
  (`src/api/types.rs:34`–`38`).

### `anthropic`

Anthropic's Claude models: claude-sonnet-4-5, claude-opus-4-1, etc.

- API key from `ANTHROPIC_API_KEY` env var, read by octolib.
- Anthropic requires a `max_tokens` parameter; if the request omits
  `max_output_tokens`, octolib's Anthropic provider supplies a default.
- Reasoning effort is **not** an Anthropic concept — sending
  `reasoning_effort` is silently ignored by Anthropic.

### `google` / `gemini`

Google AI Studio / Gemini API.

### `ollama`

Self-hosted models running on an Ollama server.

- The `model` portion can include colons (e.g.
  `ollama:llama3.3:70b-instruct-q5_K_M`) — these are **part of the
  model identifier**, not a `provider:model` separator. OctoHub splits
  on the **first** colon only, when separating the provider from the
  model.
- **Concurrency limit is critical** here. Local GPU is the bottleneck;
  without `[providers.ollama].concurrency`, you'll happily issue 100
  parallel requests and saturate the box. Set
  `concurrency = <gpu_parallel>` (often 1–8) and the surplus queues
  in-process — see [03 — Configuration](./03-configuration.md#providersname).

### `voyage`

Voyage AI embedding models (`voyage-3.5`, `voyage-4`, …).

- Use in `[embedding_models]`, not `[models]`. The two tables are
  separate and aliases don't cross.

### `deepseek`

DeepSeek's hosted models, including `deepseek-reasoner`.

- Reasoning models: DeepSeek requires that prior chain-of-thought be
  replayed on every turn. Use `previous_completion_id` — see
  [05 — Client API](./05-api-client.md#multi-turn-chains-previous_completion_id).
  Without it, the model loses its thinking between turns.

### `minimax`

Provider used in the repo's example config (`octohub.toml`). This is a
placeholder/example provider in the upstream demo; replace with your own
provider name in production.

### `modal` / `vllm` / self-hosted

Modal-hosted vLLM instances, or any custom HTTP endpoint. octolib has
adapters for these. Configuration (model name, base URL, auth) is
provided through octolib's provider-specific config layer, not through
OctoHub's `octohub.toml`.

For Modal deployments, a common pattern is one vLLM instance per model,
all fronted by a single OctoHub.

### `openrouter` / `groq` / `together` / `fireworks`

Hosted aggregators or alternative inference providers, exposed as
distinct provider names by octolib. Each has its own auth env var and
rate limit characteristics.

## Per-provider tuning

The `[providers.<name>]` table in `octohub.toml` exposes `concurrency`
plus four rate windows. All keys optional; unset or `0` = unlimited.
Source: `ProviderConfig` in `src/config.rs`.

```toml
[providers.ollama]
concurrency = 4     # max 4 in-flight requests to ollama

[providers.openai]
concurrency = 30           # cap below your rate limit
requests_per_minute = 500  # provider RPM
tokens_per_minute = 30000  # provider TPM (actual input+output tokens)
# requests_per_day = 10000 # provider RPD
# tokens_per_day = 2500000 # provider TPD
```

What it does:

- The proxy builds a `tokio::sync::Semaphore` with `concurrency` permits
  for each provider that has the section (`src/proxy/limiter.rs:32`).
- Before each upstream call, the proxy **awaits** a permit
  (`src/proxy/limiter.rs:49`).
- When the limit is hit, the HTTP request **hangs** (the client
  connection blocks) until a permit frees up. This is intentional —
  the alternative is 429s, and clients with no retry policy then fail.
- The `octohub_provider_queue_wait_seconds` histogram records how long
  each request waited. The `octohub_provider_permits_available` and
  `octohub_provider_in_flight` gauges (5s poll) show saturation.

What it does **not** do:

- It does not rate-limit per API key. A single client can still
  consume all `concurrency` permits.
- It is per-process, not cluster-wide. If you run multiple OctoHub
  instances, you get `concurrency × instances` total — see
  [08 — Deployment](./08-deployment.md#multiple-instances).
- It does not retry on upstream failure by itself. A provider returning
  5xx releases its permit and the client gets the 5xx — unless
  `[server].failover_on_error` is enabled and the model alias has
  another candidate (see below).

### Rate windows (`requests_per_minute`, `tokens_per_minute`, …)

The four window knobs track provider RPM/TPM/RPD/TPD quotas so the
proxy stops sending before the provider starts 429ing
(`ProviderRateTracker` in `src/proxy/limiter.rs`):

- Requests are counted at dispatch; tokens from the provider-reported
  usage after each response (completions and embeddings drain the same
  windows).
- A provider with an exhausted window is **skipped** during model
  resolution when the alias lists other candidates. When every
  candidate is exhausted, the client gets `429` with a `Retry-After`
  header pointing at the soonest window reset — unlike `concurrency`,
  the request is not queued (a day window may be hours away).
- Windows are **fixed** (60s / UTC day), while providers meter rolling
  windows — set values below your real quota for headroom. Counters
  are in-memory: they survive SIGHUP reloads but reset on restart.
- The `octohub_provider_rate_limited_total{provider}` counter tracks
  every skip/rejection.

Grounded baselines for Moonshot/Kimi, Z.ai, MiniMax, OpenAI, and
Anthropic — with links to each provider's rate-limit docs — ship
commented-out in [`octohub.toml`](../../octohub.toml). Two provider
quirks worth knowing: Moonshot meters TPM as
`input + max_completion_tokens` (pre-charged, so leave extra headroom),
and Anthropic splits input/output limits (ITPM/OTPM) — set
`tokens_per_minute` to your ITPM.

### Failover and cooldown (`[server]` knobs, both off by default)

- `failover_on_error = true` — when an upstream call fails with a
  **provider-side** error (upstream timeout, connect failure, 429, 5xx),
  the request is re-sent to the next admitted candidate instead of
  surfacing the error. Embedded 4xx (context overflow, bad request)
  never fails over — every provider would reject it the same way. Each
  failover re-sends the full request and is counted in
  `octohub_provider_failovers_total{provider}` (labeled with the
  provider that FAILED).
- `provider_error_cooldown_secs = N` — after **3 consecutive**
  provider-side failures, the provider is deprioritized for `N`
  seconds: sorted behind healthy candidates at resolution, used only
  when nothing healthy can admit. It is never hard-blocked, so a
  single-provider model keeps dispatching, and the first request after
  the cooldown lapses is the recovery probe. Any success resets the
  streak. State is in-memory (survives SIGHUP, reset on restart).

The two are independent: cooldown works without failover (it shapes
*future* routing) and failover works without cooldown (it saves the
*current* request).

## Picking providers for a model alias

The `[models]` table is just a `HashMap<String, Vec<String>>`
(`src/config.rs:66`). Each value is a list of `provider:model` strings.
At request time, OctoHub starts from a random entry and takes the first
whose provider rate windows admit the request (see
[rate windows](#rate-windows-requests_per_minute-tokens_per_minute-)).
One exception to the random start: a request continuing a chain
(`previous_completion_id`) prefers the provider that served the
previous turn, so provider-side prompt caches survive multi-turn
sessions — it only moves to another candidate when that provider's
rate windows are exhausted. Providers on an error cooldown (see
[failover and cooldown](#failover-and-cooldown-server-knobs-both-off-by-default))
are sorted behind healthy candidates. There's no other stickiness and
no round-robin state.

```toml
[models]
# One model, one provider — simplest case
"minimax-m2.7" = ["minimax:minimax-m2.7"]

# Cheap model: load-balanced across two providers
"fast" = ["openai:gpt-5-nano", "groq:llama-3.1-8b-instant"]

# Tiered "best": same model, fallback providers
"best" = ["anthropic:claude-opus-4-1", "openai:gpt-5"]
```

By default, if the chosen provider errors, the client gets the error —
OctoHub does not retry the next entry. Enable
`[server].failover_on_error` to re-route provider-side failures to the
remaining candidates, and `provider_error_cooldown_secs` to
deprioritize repeatedly-failing providers — see
[failover and cooldown](#failover-and-cooldown-server-knobs-both-off-by-default).

## Adding a new provider

OctoHub itself is provider-agnostic. To add a new provider, you add it
to **octolib** (the `ProviderFactory::parse_model` function), bump the
octolib version in `Cargo.toml`, and rebuild. OctoHub picks it up
automatically.

Configuration that the new provider needs (API keys, base URLs, etc.)
comes from the environment the octolib provider implementation reads —
not from `octohub.toml`. OctoHub's config is intentionally minimal: it
only owns things that are global to the proxy (auth, logging, metrics,
model aliasing, per-provider concurrency).

## Quick reference

| Provider | Typical use | Watch out for |
|---|---|---|
| `openai`         | Hosted GPT, embeddings, reasoning | Rate limits — set `[providers.openai].concurrency` |
| `anthropic`      | Hosted Claude                  | `max_tokens` required; reasoning models don't accept `reasoning_effort` |
| `ollama`         | Self-hosted open weights       | **Always** set `[providers.ollama].concurrency` |
| `voyage`         | Hosted embeddings              | Embeddings only — register under `[embedding_models]` |
| `deepseek`       | Reasoning / CoT                | **Use `previous_completion_id`** to preserve thinking across turns |
| `minimax`        | Example in default config      | Placeholder; replace for production |
| `modal`/`vllm`   | Self-hosted on Modal/your HW   | Configure base URL / auth in octolib env, not `octohub.toml` |
| `openrouter`     | Multi-model aggregator         | Check octolib for the exact provider string |
| `groq`           | Fast Llama inference           | Watch the rate limits |
