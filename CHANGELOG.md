# Changelog

## [0.7.9] - 2026-08-29

### 📋 Release Summary

Updated the setup and API documentation to make getting started with octohub and using its interfaces clearer (8b62f233). Refreshed project dependencies for ongoing maintenance and compatibility (b4d8ac3b, 0414b914).


### 📚 Documentation & Examples

- **readme**: rework setup and API guide `8b62f233`

### 🔄 Other Changes

2 maintenance, dependency, and tooling updates not listed individually.

## [0.7.8] - 2026-08-27

### 📋 Release Summary

This release includes routine dependency updates to ensure project stability and security (f4867cba).


### 🔄 Other Changes

1 maintenance, dependency, and tooling update not listed individually.

## [0.7.7] - 2026-08-26

### 📋 Release Summary

This release introduces enhanced modality handling and improved content parsing (24def18e).


### ✨ New Features & Enhancements

- **api**: enhance modality handling and content parsing `24def18e`

## [0.7.6] - 2026-08-22

### 📋 Release Summary

This release includes a dependency update for octolib to version 0.34.2 (0ff01330).


### 🔄 Other Changes

1 maintenance, dependency, and tooling update not listed individually.

## [0.7.5] - 2026-08-22

### 📋 Release Summary

This release includes a dependency update to octolib v0.34.1 to ensure improved system stability and performance (ad6eee16).


### 🔄 Other Changes

1 maintenance, dependency, and tooling update not listed individually.

## [0.7.4] - 2026-08-21

### 📋 Release Summary

This release includes dependency updates to improve system stability and performance (9d3385da, fa6adb8b).


### 🔄 Other Changes

2 maintenance, dependency, and tooling updates not listed individually.

## [0.7.3] - 2026-08-17

### 📋 Release Summary

This release includes general system updates and dependency optimizations to improve overall stability and performance (a8e271b6, 29ab3deb, c734bc12).


### 🔄 Other Changes

3 maintenance, dependency, and tooling updates not listed individually.

## [0.7.2] - 2026-08-10

### 📋 Release Summary

This release includes updated Rust dependencies to ensure improved system stability and performance (fcf09267).


### 🔄 Other Changes

1 maintenance, dependency, and tooling update not listed individually.

## [0.7.1] - 2026-08-08

### 📋 Release Summary

This release includes a bug fix to improve the reliability of health checks by refining how model thresholds are handled (440b8fec).


### 🐛 Bug Fixes & Stability

- **health**: ignore auto-model and latency thresholds `440b8fec`

## [0.7.0] - 2026-08-08

### 📋 Release Summary

This release introduces support for emulated SSE streaming in chat completions to enhance real-time response delivery (f255d172). Additionally, internal dependencies have been updated to ensure improved system stability and performance (1081d523).


### ✨ New Features & Enhancements

- **api**: implement emulated SSE streaming for chat completions `f255d172`

### 🔄 Other Changes

1 maintenance, dependency, and tooling update not listed individually.

## [0.6.5] - 2026-08-04

### 📋 Release Summary

This release includes routine dependency updates to ensure optimal performance and security (ce8ee7e3).


### 🔄 Other Changes

1 maintenance, dependency, and tooling update not listed individually.

## [0.6.4] - 2026-08-01

### 📋 Release Summary

This release introduces modality compatibility checks to ensure more reliable interactions between clients and LLMs (0cb6a931). Additionally, various core dependencies have been updated to improve overall system stability and performance (d2aa1b73, 7a697129, 31c5ffc6).


### ✨ New Features & Enhancements

- **proxy**: add modality compatibility checks `0cb6a931`

### 🔄 Other Changes

3 maintenance, dependency, and tooling updates not listed individually.

## [0.6.3] - 2026-07-31

### 📋 Release Summary

This release includes an update to the core library dependencies to ensure improved stability and performance (9fec4f24).


### 🔄 Other Changes

1 maintenance, dependency, and tooling update not listed individually.

## [0.6.2] - 2026-07-30

### 📋 Release Summary

This release introduces support for the Google Studio provider, expanding the range of available LLM integrations (c0df0e67).


### ✨ New Features & Enhancements

- **octolib**: add Google Studio provider support `c0df0e67`

## [0.6.1] - 2026-07-26

### 📋 Release Summary

This release includes routine dependency updates to ensure optimal performance and security (d97304fe).


### 🔄 Other Changes

1 maintenance, dependency, and tooling update not listed individually.

## [0.6.0] - 2026-07-21

### 📋 Release Summary

This release introduces purpose-based auto model resolution to optimize how the proxy selects the most appropriate model for a given task (3d56248c, 107812b1). Additionally, the update includes expanded API documentation for the admin status endpoint and general system stability improvements (57391558, c08dbe63).


### ✨ New Features & Enhancements

- **proxy**: add hierarchical purpose resolution for auto models `3d56248c`
- **routing**: implement purpose-based auto model resolution `107812b1`

### 🔧 Improvements & Optimizations

- **proxy**: format resolve calls in tests `dbeacf9a`

### 📚 Documentation & Examples

- **api**: document admin status endpoint `57391558`

### 🔄 Other Changes

1 maintenance, dependency, and tooling update not listed individually.

## [0.5.0] - 2026-07-19

### 📋 Release Summary

This release introduces a new model health monitoring endpoint to improve administrative oversight of system status (a5db7e11). General code refinements were also implemented to ensure better maintainability and stability (e7dcfc38).


### ✨ New Features & Enhancements

- **admin**: add model health monitoring endpoint `a5db7e11`

### 🔧 Improvements & Optimizations

- **health**: reformat test function calls `e7dcfc38`

## [0.4.0] - 2026-07-13

### 📋 Release Summary

This release introduces advanced provider management, including automated failover, health tracking, and intelligent rate limiting to ensure higher service availability (34ad5616, d5cae7ef). Additionally, session stability is improved through new provider stickiness for multi-turn conversations (c826a39d).


### ✨ New Features & Enhancements

- **proxy**: implement provider failover and health tracking `34ad5616`
- **proxy**: add provider stickiness for multi-turn sessions `c826a39d`
- **proxy**: implement provider rate limiting and rotation `d5cae7ef`

### 🔄 Other Changes

1 maintenance, dependency, and tooling update not listed individually.

## [0.3.0] - 2026-07-12

### 📋 Release Summary

This release introduces improved concurrency and scoping for shared owners to enhance request management (dc4ba3e7). Additionally, internal dependencies have been updated to ensure optimal performance and stability (69b692be).


### ✨ New Features & Enhancements

- **proxy**: implement shared owner concurrency and scoping `dc4ba3e7`

### 🔄 Other Changes

1 maintenance, dependency, and tooling update not listed individually.

## [0.2.0] - 2026-07-11

### 📋 Release Summary

This release introduces administrative controls for API key model permissions, enforced chat completions, and the ability to reload configurations without restarting the server (3d1846d5, 676098a7, 3723b06d). System stability and accuracy have been improved through updated dependencies and fixes for embedding token counts and server connection handling (8b7cb32d, ed19d639, 25609b28, 19be2aa2).


### ✨ New Features & Enhancements

- **admin**: add endpoint to update API key allowed models `3d1846d5`
- **proxy**: implement enforced chat completions `676098a7`
- **core**: implement SIGHUP configuration reloading `3723b06d`

### 🐛 Bug Fixes & Stability

- **proxy**: use actual token counts for embeddings `8b7cb32d`
- **server**: prevent crash on accept errors and leak `ed19d639`

### 🔄 Other Changes

2 maintenance, dependency, and tooling updates not listed individually.

## [0.1.1] - 2026-06-28

### 📋 Release Summary

This release expands platform support by adding a target for Intel-based Macs and introducing Homebrew notifications (5d13294b). General system stability and performance have been improved through updated dependencies (43e15126).


### 🔧 Improvements & Optimizations

- **release**: add intel mac target and brew notification `5d13294b`

### 🔄 Other Changes

1 maintenance, dependency, and tooling update not listed individually.

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
