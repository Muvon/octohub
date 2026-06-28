# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-06-28

### 📋 Release Summary

This release introduces a high-performance LLM proxy with OpenAI-compatible chat completions, support for structured outputs, reasoning, and multi-modal inputs (7843acae, 905a4750, 031502d8, ba17848f). Key enhancements include multi-tenant API key management, comprehensive observability metrics, and expanded provider support for vLLM, GLM-5.1, and Kimi-K2.6 (7aa979bd, 98fd378d, f3e982bd, eb873f19). System stability and reliability are further improved through refined error handling, optimized cache control, and the addition of MySQL and PostgreSQL storage backends (ff3999d7, 751be9b3, daaf4fbc).


### ✨ New Features & Enhancements

- **proxy**: improve error handling and cache control `ff3999d7`
- **api**: implement OpenAI compatible chat completions `7843acae`
- **proxy**: implement configurable provider and upstream timeouts `a0fb4d30`
- **api**: add structured output and finish reason to responses `905a4750`
- **api**: add sampling and structured output support `67e59c25`
- **proxy**: forward cache control markers and ttl `238e5fbd`
- **proxy**: improve telemetry and metrics tracking `e90e0cc0`
- **obs**: implement comprehensive observability and metrics `98fd378d`
- **proxy**: add model whitelisting and provider concurrency limits `6932fe90`
- **server**: enable http2 support and disable keep-alive `15fcdf97`
- **api**: support replaying tool calls and reasoning `9f3e5039`
- **api**: add support for model reasoning output `031502d8`
- **api**: support structured content and cache control `c4a9229e`
- **admin**: improve connection config and errors `8f460aed`
- **modal**: add GLM-5.1 and Kimi-K2.6 inference servers `f3e982bd`
- **modal**: add multiple vLLM model deployments `eb873f19`
- **storage**: add MySQL and PostgreSQL backends `751be9b3`
- **auth**: add multi-tenant API key management `7aa979bd`
- **api**: add admin endpoints for key management and usage tracking `66f3b251`
- **storage**: persist embeddings for observability `48fe90d3`
- **storage**: add session tracking for response chains `31c819b7`

### 🔧 Improvements & Optimizations

- **workflow**: migrate to shared ci-workflow templates `f2970fb1`
- **workflow**: add brief check step `96165933`
- **api**: reformat assertions in types tests `e384e0d5`
- **workflow**: upgrade rust toolchain to 1.95.0 `fc93b68f`
- **auth**: require master key for all admin endpoints `fc5540fa`
- **storage**: rename response to completion `e0ad43cf`
- **config**: replace env config with TOML file and model resolution `bf68043d`

### 🐛 Bug Fixes & Stability

- **api**: improve error reporting and cache control `33a07461`
- **proxy**: handle unknown previous completion IDs `6bc72afc`
- **engine**: unify assistant messages with text and tool_calls `55c87fff`
- **api**: classify engine errors into proper HTTP status codes `daaf4fbc`
- **api**: simplify embedding response structure `cdb220d3`

### 📚 Documentation & Examples

- add comprehensive project documentation `e43fa373`
- **instructions**: add authentication architecture and admin CLI script `8c4801fc`
- **api**: rename response to completion and simplify embedding format `bd42c83f`

### 🔄 Other Changes

9 maintenance, dependency, and tooling updates not listed individually.
