//! Media proxying: submit, resume, cancel and describe.
//!
//! The shape here differs from `engine::process` in one way that matters. A
//! chat completion is a single call that either returns or fails; a media job
//! commits money upstream at submit and may outlive the request that started
//! it. So the row and its `JobHandle` are persisted the moment the provider
//! accepts the work, *before* any waiting — a restart, a timeout or a hung
//! client can never orphan a paid job.
//!
//! Pricing is not this module's business. octolib's adapters resolve a rate
//! from `media::reference_pricing` and report it on `MediaUsage`; OctoHub reads
//! `best_available_cost()` and records it, exactly as it does for completions.

use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};
use octolib::media::{
	FailureCategory, GenerationFailure, ImageGenerationProvider, ImageGenerationRequest, ImageGenerationResult,
	JobHandle, MediaError, MediaProviderFactory, MediaTask, MediaUsage, Operation, OperationStatus, ProviderWarning,
	RequestOptions, SafetyReport, SpeechSynthesisProvider, SpeechSynthesisRequest, SpeechSynthesisResult,
	TranscriptionProvider, TranscriptionRequest, TranscriptionResult, VideoGenerationProvider, VideoGenerationRequest,
	VideoGenerationResult,
};
use uuid::Uuid;

use crate::api::media_types::*;
use crate::config::{Config, LaneStrategy};
use crate::proxy::engine::{
	deprioritize_saturated, ensure_model_allowed, pick_admitted, ProxyEngine, ProxyTimeoutError,
};
use crate::storage::{ApiKey, StoredMedia};

/// How often the abandoned-job sweep runs.
const MEDIA_SWEEP_INTERVAL: Duration = Duration::from_secs(60);
/// Rows handled per sweep pass, so one pass stays bounded however big the backlog.
const MEDIA_SWEEP_BATCH: u32 = 50;
/// Younger jobs still belong to a client that is waiting or polling.
const MEDIA_SWEEP_MIN_AGE_SECS: u64 = 5 * 60;
/// A job the provider has not finished by now never will; stop polling it.
const MEDIA_ABANDONED_AFTER_SECS: u64 = 24 * 60 * 60;

/// One media request in any of the four task shapes.
pub enum MediaRequest {
	Image(Box<ImageRequest>),
	Video(Box<VideoRequest>),
	Speech(Box<SpeechRequest>),
	Transcription(Box<TranscriptionRequestBody>),
}

impl MediaRequest {
	fn common(&self) -> &MediaCommon {
		match self {
			Self::Image(r) => &r.common,
			Self::Video(r) => &r.common,
			Self::Speech(r) => &r.common,
			Self::Transcription(r) => &r.common,
		}
	}

	/// The task as declared by the endpoint plus the request's own `mode`.
	fn task(&self) -> Result<MediaTask> {
		Ok(match self {
			Self::Image(r) => match r.mode.as_deref().unwrap_or("generate") {
				"generate" => MediaTask::TextToImage,
				"edit" => MediaTask::ImageEdit,
				"inpaint" => MediaTask::Inpainting,
				"variation" => MediaTask::ImageVariation,
				other => return Err(anyhow!("Invalid request: unknown image mode '{other}'")),
			},
			Self::Video(r) => match r.mode.as_deref().unwrap_or("text_to_video") {
				"text_to_video" => MediaTask::TextToVideo,
				"image_to_video" => MediaTask::ImageToVideo,
				"reference_to_video" => MediaTask::ReferenceToVideo,
				"extend" => MediaTask::VideoExtend,
				"edit" => MediaTask::VideoEdit,
				other => return Err(anyhow!("Invalid request: unknown video mode '{other}'")),
			},
			Self::Speech(_) => MediaTask::TextToSpeech,
			Self::Transcription(_) => MediaTask::SpeechToText,
		})
	}

	/// The model as the client sent it — the alias, not the resolved model,
	/// so metrics stay stable when routing moves between mirrors.
	pub fn model_label(&self) -> &str {
		&self.common().model
	}

	/// Coarse task label for metrics. Deliberately the endpoint's task family
	/// rather than the exact `MediaTask`, to keep label cardinality bounded.
	pub fn task_label(&self) -> &'static str {
		match self {
			Self::Image(_) => "image",
			Self::Video(_) => "video",
			Self::Speech(_) => "speech",
			Self::Transcription(_) => "transcription",
		}
	}

	/// The request as persisted: identical to what the client sent, minus any
	/// inline binary payload.
	fn redacted(&self) -> serde_json::Value {
		let common = self.common();
		let mut value = serde_json::json!({
			"model": common.model,
			"wait": common.wait,
			"provider_options": common.provider_options,
		});
		let map = value.as_object_mut().expect("object literal");
		match self {
			Self::Image(r) => {
				map.insert("task".into(), "image".into());
				map.insert("prompt".into(), r.prompt.clone().into());
				map.insert(
					"source_images".into(),
					r.source_images.iter().map(|s| s.redacted()).collect(),
				);
				if let Some(mask) = &r.mask {
					map.insert("mask".into(), mask.redacted());
				}
				insert_opt(map, "count", r.count);
				insert_opt(map, "seed", r.seed);
				insert_opt(map, "size", r.size.clone());
			}
			Self::Video(r) => {
				map.insert("task".into(), "video".into());
				map.insert("prompt".into(), r.prompt.clone().into());
				if let Some(frame) = &r.first_frame {
					map.insert("first_frame".into(), frame.redacted());
				}
				if let Some(frame) = &r.last_frame {
					map.insert("last_frame".into(), frame.redacted());
				}
				map.insert(
					"reference_images".into(),
					r.reference_images.iter().map(|s| s.redacted()).collect(),
				);
				if let Some(video) = &r.source_video {
					map.insert("source_video".into(), video.redacted());
				}
				insert_opt(map, "count", r.count);
				insert_opt(map, "seed", r.seed);
				insert_opt(map, "duration_secs", r.duration_secs);
				insert_opt(map, "size", r.size.clone());
			}
			Self::Speech(r) => {
				map.insert("task".into(), "speech".into());
				// Prompt text is the billable quantity and is worth keeping;
				// it is not binary and does not bloat the row.
				map.insert("input".into(), r.input.clone().into());
				map.insert("voice".into(), r.voice.clone().into());
				insert_opt(map, "language", r.language.clone());
			}
			Self::Transcription(r) => {
				map.insert("task".into(), "transcription".into());
				map.insert("audio".into(), r.audio.redacted());
				insert_opt(map, "language", r.language.clone());
			}
		}
		value
	}
}

fn insert_opt<T: Into<serde_json::Value>>(
	map: &mut serde_json::Map<String, serde_json::Value>,
	key: &str,
	value: Option<T>,
) {
	if let Some(value) = value {
		map.insert(key.to_string(), value.into());
	}
}

/// A completed `process_media` / `poll_media` call.
pub struct MediaOutcome {
	pub response: MediaResponse,
	/// True when the operation is still running: the handler answers 202 and
	/// the client resumes with `GET /v1/media/{id}`.
	pub accepted: bool,
	/// True when this call took the job to its outcome upstream. A poll that
	/// only replayed a stored outcome is false, so the outcome is recorded once.
	pub finished_now: bool,
	/// The model as the client asked for it, the key every metric uses.
	pub alias: String,
	pub upstream_duration: Duration,
}

/// Every task's result flattened into one shape, so persistence and the wire
/// envelope have a single code path.
struct Normalized {
	status: OperationStatus,
	progress: Option<f32>,
	handle: Option<JobHandle>,
	artifacts: Vec<WireArtifact>,
	result: Option<serde_json::Value>,
	usage: Option<MediaUsage>,
	warnings: Vec<ProviderWarning>,
	safety: SafetyReport,
	error: Option<serde_json::Value>,
}

fn now_unix() -> u64 {
	SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.unwrap_or_default()
		.as_secs()
}

fn is_terminal(status: OperationStatus) -> bool {
	status.is_terminal()
}

fn failure_json(error: &octolib::media::GenerationFailure) -> serde_json::Value {
	serde_json::to_value(error).unwrap_or(serde_json::Value::Null)
}

/// The provider-independent half of every operation: lifecycle, resumability
/// and failure. The task-specific normalizers fill in the rest.
fn skeleton<T>(operation: &Operation<T>) -> Normalized {
	Normalized {
		status: operation.status,
		progress: operation.progress,
		handle: operation.handle.clone(),
		artifacts: Vec::new(),
		result: None,
		usage: None,
		warnings: Vec::new(),
		safety: SafetyReport::default(),
		error: operation.error.as_ref().map(failure_json),
	}
}

fn normalize_image(operation: &Operation<ImageGenerationResult>) -> Normalized {
	let mut normalized = skeleton(operation);
	if let Some(result) = operation.result.as_ref() {
		normalized.artifacts = result.artifacts.iter().map(WireArtifact::from).collect();
		normalized.usage = result.usage.clone();
		normalized.warnings = result.warnings.clone();
		normalized.safety = result.safety.clone();
	}
	normalized
}

fn normalize_video(operation: &Operation<VideoGenerationResult>) -> Normalized {
	let mut normalized = skeleton(operation);
	if let Some(result) = operation.result.as_ref() {
		normalized.artifacts = result.artifacts.iter().map(WireArtifact::from).collect();
		normalized.usage = result.usage.clone();
		normalized.warnings = result.warnings.clone();
		normalized.safety = result.safety.clone();
	}
	normalized
}

fn normalize_speech(operation: &Operation<SpeechSynthesisResult>) -> Normalized {
	let mut normalized = skeleton(operation);
	if let Some(result) = operation.result.as_ref() {
		normalized.artifacts = vec![WireArtifact::from(&result.artifact)];
		normalized.usage = result.usage.clone();
		normalized.warnings = result.warnings.clone();
	}
	normalized
}

/// Transcription is the one task whose payload is not an artifact.
fn normalize_transcription(operation: &Operation<TranscriptionResult>) -> Normalized {
	let mut normalized = skeleton(operation);
	if let Some(result) = operation.result.as_ref() {
		normalized.result = serde_json::to_value(WireTranscript {
			text: result.text.clone(),
			language: result.language.clone(),
			duration_secs: result.duration_secs,
			segments: result.segments.clone(),
			words: result.words.clone(),
		})
		.ok();
		normalized.usage = result.usage.clone();
		normalized.warnings = result.warnings.clone();
	}
	normalized
}

/// Transport controls from server config. Wait deadlines deliberately reuse
/// `server.upstream_timeout_secs` rather than growing a second timeout knob.
fn request_options(config: &Config, common: &MediaCommon) -> RequestOptions {
	RequestOptions {
		submit_timeout: Some(Duration::from_secs(config.media.submit_timeout_secs)),
		wait_timeout: Some(Duration::from_secs(config.server.upstream_timeout_secs)),
		polling_interval: Duration::from_secs(config.media.polling_interval_secs.max(1)),
		max_source_bytes: config.media.max_source_bytes,
		max_response_bytes: config.media.max_response_bytes,
		unsupported_parameter_policy: common.unsupported_parameters.into(),
		..RequestOptions::default()
	}
}

fn bad_request(message: impl std::fmt::Display) -> anyhow::Error {
	anyhow!(MediaError::InvalidRequest(message.to_string()))
}

fn media_status_is_provider_fault(status: u16) -> bool {
	matches!(status, 401..=403 | 408 | 429 | 500..=599)
}

fn is_media_provider_fault(error: &anyhow::Error) -> bool {
	let Some(media) = error.downcast_ref::<MediaError>() else {
		return matches!(
			error.downcast_ref::<ProxyTimeoutError>(),
			Some(ProxyTimeoutError::Upstream { .. })
		);
	};
	match media {
		MediaError::InvalidModelFormat(_)
		| MediaError::UnsupportedProvider(_)
		| MediaError::UnsupportedTask { .. }
		| MediaError::UnsupportedParameter { .. }
		| MediaError::InvalidRequest(_)
		| MediaError::WrongJobHandle { .. }
		| MediaError::LocalWaitCancelled { .. }
		| MediaError::SourceTooLarge { .. }
		| MediaError::Base64(_) => false,
		MediaError::Api { status, .. } => media_status_is_provider_fault(*status),
		MediaError::RemoteFailure(failure) => {
			!matches!(
				failure.category,
				FailureCategory::Unsupported
					| FailureCategory::ContentPolicy
					| FailureCategory::InvalidInput
					| FailureCategory::Cancelled
			) && match failure.http_status {
				Some(status) => media_status_is_provider_fault(status),
				None => true,
			}
		}
		MediaError::MissingApiKey(_)
		| MediaError::Authentication { .. }
		| MediaError::Permission { .. }
		| MediaError::InsufficientCredits { .. }
		| MediaError::RateLimit { .. }
		| MediaError::InvalidResponse { .. }
		| MediaError::WaitTimeout { .. }
		| MediaError::ArtifactTooLarge { .. }
		| MediaError::ResponseTooLarge { .. }
		| MediaError::Io(_)
		| MediaError::Transport(_)
		| MediaError::Json(_) => true,
	}
}

impl ProxyEngine {
	/// Submit a media job, persist it, and optionally wait for it inline.
	pub async fn process_media(&self, request: MediaRequest, api_key: &ApiKey) -> Result<MediaOutcome> {
		let config = self.config();
		let common = request.common();
		let wait = common.wait;
		let input_model = common.model.clone();
		let task = request.task()?;

		ensure_model_allowed(api_key, &input_model)?;
		reject_reserved_options(&common.provider_options).map_err(bad_request)?;

		// Held for the whole request — submit, wait and persist — so a tenant's
		// media work drains the same budget as its completions.
		let _owner_slot = self.acquire_owner_slot(api_key).await?;

		let mut candidates = config
			.media_model_candidates(&input_model)
			.with_context(|| format!("Failed to resolve media model '{input_model}'"))?;

		let cooldown = Duration::from_secs(config.server.provider_error_cooldown_secs);
		let limiter = self.limiter();
		let strategy = config.lane_strategy(&input_model);

		let (provider_name, resolved_model, mut normalized, upstream_duration) = loop {
			let mut available = candidates.clone();
			if strategy == LaneStrategy::Ordered {
				deprioritize_saturated(&mut available, &limiter);
			}
			let (provider_name, resolved_model) = pick_admitted(
				&self.rate_tracker,
				&self.provider_health,
				&config,
				available,
				&input_model,
			)?;
			tracing::Span::current().record("provider", provider_name.as_str());

			let attempt = async {
				let queue_timeout = Duration::from_secs(config.server.provider_queue_timeout_secs);
				// The permit covers the SUBMIT call only. A four-minute video
				// must not pin a provider slot for its whole runtime.
				let permit = tokio::time::timeout(queue_timeout, limiter.acquire(&provider_name))
					.await
					.map_err(|_| {
						anyhow!(ProxyTimeoutError::ProviderQueue {
							provider: provider_name.clone(),
							timeout: queue_timeout,
						})
					})?;

				let started = Instant::now();
				let operation = submit(&request, &config, &provider_name, &resolved_model).await?;
				drop(permit);
				anyhow::Ok((operation, started.elapsed()))
			}
			.await;

			match attempt {
				Ok((normalized, duration)) => {
					self.provider_health.record_success(&provider_name);
					break (provider_name, resolved_model, normalized, duration);
				}
				Err(err) => {
					let provider_fault = is_media_provider_fault(&err);
					if provider_fault {
						self.provider_health.record_failure(&provider_name, cooldown);
					}
					candidates.retain(|(p, _)| !p.eq_ignore_ascii_case(&provider_name));
					let failover = config.server.failover_on_error || strategy == LaneStrategy::Ordered;
					if !(failover && provider_fault) || candidates.is_empty() {
						return Err(err);
					}
					crate::metrics::record_failover(&provider_name);
					tracing::warn!(
						provider = %provider_name,
						error = %err,
						"media provider failed at submit — failing over to next candidate"
					);
				}
			}
		};

		let created_at = now_unix();
		let mut record = StoredMedia {
			id: format!("med_{}", Uuid::new_v4().simple()),
			api_key_id: api_key.id,
			task: task_name(task),
			status: status_name(normalized.status),
			input_model: input_model.clone(),
			resolved_model: resolved_model.clone(),
			provider: provider_name.clone(),
			request: request.redacted(),
			handle: normalized.handle.as_ref().and_then(|h| serde_json::to_value(h).ok()),
			result: result_json(&normalized),
			usage: usage_json(&normalized),
			warnings: warnings_json(&normalized),
			error: normalized.error.clone(),
			created_at,
			completed_at: is_terminal(normalized.status).then_some(created_at),
		};

		// Persist BEFORE waiting: from here the job is recoverable even if this
		// process dies mid-poll.
		let storage = self.storage.clone();
		let to_store = record.clone();
		tokio::task::spawn_blocking(move || storage.store_media(&to_store)).await??;

		let mut total = upstream_duration;
		if wait && !is_terminal(normalized.status) {
			let deadline = Duration::from_secs(config.server.upstream_timeout_secs);
			let started = Instant::now();
			let interval = Duration::from_secs(config.media.polling_interval_secs.max(1));
			while !is_terminal(normalized.status) && started.elapsed() < deadline {
				tokio::time::sleep(interval).await;
				let Some(handle) = normalized.handle.clone() else {
					break;
				};
				normalized = poll_once(&provider_name, &handle).await?;
			}
			total += started.elapsed();
			apply(&mut record, &normalized);
			let storage = self.storage.clone();
			let to_update = record.clone();
			// A sweep that settled this row first wins; the write is then a no-op.
			tokio::task::spawn_blocking(move || storage.update_media(&to_update)).await??;
		}

		Ok(self.finish(record, normalized, total))
	}

	/// Advance a stored job. A terminal row is returned as stored — no upstream
	/// call, so re-reading a finished record is free and never re-bills.
	pub async fn poll_media(&self, id: &str, api_key: &ApiKey) -> Result<Option<MediaOutcome>> {
		let storage = self.storage.clone();
		let (lookup_id, key_id) = (id.to_string(), api_key.id);
		let Some(mut record) = tokio::task::spawn_blocking(move || storage.get_media(&lookup_id, key_id)).await??
		else {
			return Ok(None);
		};

		if record.handle.is_none() || terminal_name(&record.status) {
			return Ok(Some(self.stored_outcome(record)));
		}

		let handle = stored_handle(&record)?;

		let started = Instant::now();
		let normalized = poll_once(&record.provider, &handle).await?;
		apply(&mut record, &normalized);

		let storage = self.storage.clone();
		let to_update = record.clone();
		let written = tokio::task::spawn_blocking(move || storage.update_media(&to_update)).await??;
		if !written {
			// Another poll or the sweep settled this job first. Its stored outcome
			// is the one billing reads, so serve that instead of ours.
			if let Some(stored) = self.finished_elsewhere(id, api_key.id).await? {
				return Ok(Some(self.stored_outcome(stored)));
			}
		}

		Ok(Some(self.finish(record, normalized, started.elapsed())))
	}

	/// Ask the provider to cancel. Cancellation is best-effort and provider
	/// dependent; the row is only marked cancelled once the provider confirms.
	pub async fn cancel_media(&self, id: &str, api_key: &ApiKey) -> Result<Option<MediaOutcome>> {
		let storage = self.storage.clone();
		let (lookup_id, key_id) = (id.to_string(), api_key.id);
		let Some(mut record) = tokio::task::spawn_blocking(move || storage.get_media(&lookup_id, key_id)).await??
		else {
			return Ok(None);
		};
		if terminal_name(&record.status) {
			return Ok(Some(self.stored_outcome(record)));
		}
		let handle = stored_handle(&record)?;

		cancel_once(&record.provider, &handle).await?;
		record.status = status_name(OperationStatus::Cancelled);
		record.completed_at = Some(now_unix());
		record.handle = None;

		let storage = self.storage.clone();
		let to_update = record.clone();
		let written = tokio::task::spawn_blocking(move || storage.update_media(&to_update)).await??;
		if !written {
			// The job finished while the cancel was in flight; the stored outcome stands.
			if let Some(stored) = self.finished_elsewhere(id, api_key.id).await? {
				return Ok(Some(self.stored_outcome(stored)));
			}
		}
		Ok(Some(self.stored_outcome(record)))
	}

	/// The stored row, when another writer has already taken it terminal.
	async fn finished_elsewhere(&self, id: &str, api_key_id: i64) -> Result<Option<StoredMedia>> {
		let storage = self.storage.clone();
		let lookup_id = id.to_string();
		let record = tokio::task::spawn_blocking(move || storage.get_media(&lookup_id, api_key_id)).await??;
		Ok(record.filter(|record| terminal_name(&record.status)))
	}

	/// One pass of the abandoned-job sweep: advance jobs whose client stopped
	/// polling, so a job the provider finished (and charged us for) still lands
	/// as a billable row. Returns how many jobs reached a terminal state.
	///
	/// A job that is still not terminal after `MEDIA_ABANDONED_AFTER_SECS` is
	/// failed. Terminal rows are write-once in storage, so this is safe against
	/// a client poll or another instance sweeping the same row.
	pub async fn sweep_media(&self) -> Result<usize> {
		let now = now_unix();
		let storage = self.storage.clone();
		let created_before = now.saturating_sub(MEDIA_SWEEP_MIN_AGE_SECS);
		let stale =
			tokio::task::spawn_blocking(move || storage.list_stale_media(created_before, MEDIA_SWEEP_BATCH)).await??;

		let mut settled = 0;
		for mut record in stale {
			let polled = match stored_handle(&record) {
				Ok(handle) => poll_once(&record.provider, &handle).await,
				Err(error) => Err(error),
			};
			match &polled {
				Ok(normalized) => apply(&mut record, normalized),
				// One bad row must not stop the pass; it is retried next tick
				// and abandoned once it is old enough.
				Err(error) => tracing::warn!(
					id = %record.id,
					provider = %record.provider,
					error = %error,
					"media sweep could not poll job"
				),
			}
			if !terminal_name(&record.status) && now.saturating_sub(record.created_at) >= MEDIA_ABANDONED_AFTER_SECS {
				abandon(&mut record, now);
			}

			let storage = self.storage.clone();
			let to_update = record.clone();
			let written = tokio::task::spawn_blocking(move || storage.update_media(&to_update)).await??;
			if written && terminal_name(&record.status) {
				settled += 1;
				tracing::info!(id = %record.id, status = %record.status, "media sweep settled job");
			}
		}
		Ok(settled)
	}

	/// Capability descriptors for every configured media candidate, plus the
	/// reference rate octolib would apply.
	pub fn media_models(&self) -> serde_json::Value {
		let config = self.config();
		let mut data = Vec::new();
		let mut aliases: Vec<&String> = config.media_models.keys().collect();
		aliases.sort();
		for alias in aliases {
			let Some(entries) = config.media_models.get(alias) else {
				continue;
			};
			for entry in entries {
				let Some((provider, model)) = entry.split_once(':') else {
					continue;
				};
				let descriptor = describe(provider, model);
				let price = octolib::media::get_reference_pricing(provider, model).map(|p| {
					serde_json::json!({
						"unit": p.unit,
						"usd_per_unit": p.usd_per_unit,
						"pattern": p.pattern,
					})
				});
				data.push(serde_json::json!({
					"alias": alias,
					"provider": provider,
					"model": model,
					"descriptor": descriptor,
					"price": price,
				}));
			}
		}
		serde_json::json!({ "object": "list", "data": data })
	}

	fn finish(&self, record: StoredMedia, normalized: Normalized, upstream_duration: Duration) -> MediaOutcome {
		let accepted = !is_terminal(normalized.status);
		MediaOutcome {
			response: MediaResponse {
				id: record.id.clone(),
				object: "media",
				task: parse_task(&record.task),
				status: normalized.status,
				model: record.resolved_model.clone(),
				provider: record.provider.clone(),
				progress: normalized.progress,
				artifacts: normalized.artifacts,
				result: normalized.result,
				usage: normalized.usage.as_ref().map(WireUsage::from),
				warnings: normalized.warnings,
				safety: normalized.safety,
				error: normalized.error,
				created_at: record.created_at,
				completed_at: record.completed_at,
			},
			accepted,
			finished_now: !accepted,
			alias: record.input_model,
			upstream_duration,
		}
	}

	/// Rebuild the wire envelope from a stored row without touching upstream.
	fn stored_outcome(&self, record: StoredMedia) -> MediaOutcome {
		let artifacts = record
			.result
			.as_ref()
			.and_then(|r| r.get("artifacts"))
			.and_then(|a| serde_json::from_value::<Vec<WireArtifact>>(a.clone()).ok())
			.unwrap_or_default();
		let transcript = record.result.as_ref().and_then(|r| r.get("transcript")).cloned();
		let usage = record
			.usage
			.as_ref()
			.and_then(|u| serde_json::from_value::<WireUsage>(u.clone()).ok());
		let warnings = record
			.warnings
			.as_ref()
			.and_then(|w| serde_json::from_value::<Vec<ProviderWarning>>(w.clone()).ok())
			.unwrap_or_default();
		let status = parse_status(&record.status);
		MediaOutcome {
			response: MediaResponse {
				id: record.id.clone(),
				object: "media",
				task: parse_task(&record.task),
				status,
				model: record.resolved_model.clone(),
				provider: record.provider.clone(),
				progress: is_terminal(status).then_some(1.0),
				artifacts,
				result: transcript,
				usage,
				warnings,
				safety: SafetyReport::default(),
				error: record.error.clone(),
				created_at: record.created_at,
				completed_at: record.completed_at,
			},
			accepted: !is_terminal(status),
			finished_now: false,
			alias: record.input_model,
			upstream_duration: Duration::ZERO,
		}
	}
}

fn apply(record: &mut StoredMedia, normalized: &Normalized) {
	record.status = status_name(normalized.status);
	record.handle = if is_terminal(normalized.status) {
		None
	} else {
		normalized.handle.as_ref().and_then(|h| serde_json::to_value(h).ok())
	};
	if let Some(result) = result_json(normalized) {
		record.result = Some(result);
	}
	if let Some(usage) = usage_json(normalized) {
		record.usage = Some(usage);
	}
	if let Some(warnings) = warnings_json(normalized) {
		record.warnings = Some(warnings);
	}
	if normalized.error.is_some() {
		record.error = normalized.error.clone();
	}
	if is_terminal(normalized.status) && record.completed_at.is_none() {
		record.completed_at = Some(now_unix());
	}
}

/// The resumable `JobHandle` a stored non-terminal row carries.
fn stored_handle(record: &StoredMedia) -> Result<JobHandle> {
	serde_json::from_value(
		record
			.handle
			.clone()
			.ok_or_else(|| anyhow!("media record has no resumable handle"))?,
	)
	.context("stored media handle is not a valid JobHandle")
}

/// Give up on a job the provider never finished.
fn abandon(record: &mut StoredMedia, now: u64) {
	record.status = status_name(OperationStatus::Failed);
	record.handle = None;
	record.completed_at = Some(now);
	record.error = Some(failure_json(&GenerationFailure {
		category: FailureCategory::Expired,
		message: "abandoned: provider never finished".to_string(),
		provider_code: None,
		http_status: None,
		request_id: None,
		provider_metadata: serde_json::Value::Null,
	}));
}

/// Background task: settle media jobs whose client stopped polling.
pub async fn media_sweep_loop(engine: Arc<ProxyEngine>) {
	let mut ticker = tokio::time::interval(MEDIA_SWEEP_INTERVAL);
	// A slow pass delays the next one rather than queueing a burst behind it.
	ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
	loop {
		ticker.tick().await;
		match engine.sweep_media().await {
			Ok(settled) if settled > 0 => tracing::info!(settled, "media sweep pass finished"),
			Ok(_) => {}
			Err(error) => tracing::error!(error = %error, "media sweep pass failed"),
		}
	}
}

fn result_json(normalized: &Normalized) -> Option<serde_json::Value> {
	if normalized.artifacts.is_empty() && normalized.result.is_none() {
		return None;
	}
	let mut value = serde_json::json!({
		"artifacts": serde_json::to_value(&normalized.artifacts).unwrap_or_default(),
	});
	if let (Some(map), Some(transcript)) = (value.as_object_mut(), normalized.result.clone()) {
		map.insert("transcript".to_string(), transcript);
	}
	Some(value)
}

/// The stored usage shape is the wire shape, so `usage.cost` is one
/// `json_extract` away for the admin aggregate.
fn usage_json(normalized: &Normalized) -> Option<serde_json::Value> {
	let usage = normalized.usage.as_ref()?;
	serde_json::to_value(WireUsage::from(usage)).ok()
}

fn warnings_json(normalized: &Normalized) -> Option<serde_json::Value> {
	if normalized.warnings.is_empty() {
		return None;
	}
	serde_json::to_value(&normalized.warnings).ok()
}

/// The endpoint family a task belongs to: the metrics `task` label, the same
/// one a submit records from its endpoint.
pub fn task_family(task: MediaTask) -> &'static str {
	match task {
		MediaTask::TextToImage | MediaTask::ImageEdit | MediaTask::Inpainting | MediaTask::ImageVariation => "image",
		MediaTask::TextToVideo
		| MediaTask::ImageToVideo
		| MediaTask::ReferenceToVideo
		| MediaTask::VideoExtend
		| MediaTask::VideoEdit => "video",
		MediaTask::TextToSpeech => "speech",
		MediaTask::SpeechToText | MediaTask::SpeechTranslation => "transcription",
		// octolib's task list is non-exhaustive; a task added there is no family yet.
		_ => "other",
	}
}

fn task_name(task: MediaTask) -> String {
	serde_json::to_value(task)
		.ok()
		.and_then(|v| v.as_str().map(str::to_string))
		.unwrap_or_else(|| "text_to_image".to_string())
}

fn parse_task(name: &str) -> MediaTask {
	serde_json::from_value(serde_json::Value::String(name.to_string())).unwrap_or(MediaTask::TextToImage)
}

fn status_name(status: OperationStatus) -> String {
	serde_json::to_value(status)
		.ok()
		.and_then(|v| v.as_str().map(str::to_string))
		.unwrap_or_else(|| "queued".to_string())
}

fn parse_status(name: &str) -> OperationStatus {
	serde_json::from_value(serde_json::Value::String(name.to_string())).unwrap_or(OperationStatus::Queued)
}

fn terminal_name(name: &str) -> bool {
	is_terminal(parse_status(name))
}

/// Build the octolib request for whichever task this is and hand it to the
/// adapter. The returned operation may already be terminal (ElevenLabs and
/// OpenRouter answer synchronously) or queued (fal, Replicate, Runway).
async fn submit(request: &MediaRequest, config: &Config, provider_name: &str, model: &str) -> Result<Normalized> {
	let common = request.common();
	let options = request_options(config, common);
	let provider_options = filter_options_for(&common.provider_options, provider_name);
	let max_bytes = options.max_source_bytes;

	match request {
		MediaRequest::Image(r) => {
			let mut octo = ImageGenerationRequest::new(r.prompt.clone()).with_model(model);
			octo.mode = parse_image_mode(r.mode.as_deref().unwrap_or("generate")).map_err(bad_request)?;
			for source in &r.source_images {
				octo.source_images
					.push(source.clone().into_source(max_bytes).map_err(bad_request)?);
			}
			if let Some(mask) = &r.mask {
				octo.mask = Some(mask.clone().into_source(max_bytes).map_err(bad_request)?);
			}
			octo.count = r.count;
			octo.seed = r.seed;
			if let Some(size) = &r.size {
				octo.geometry = Some(parse_geometry(size).map_err(bad_request)?);
			}
			octo.negative_prompt = r.negative_prompt.clone();
			if let Some(format) = &r.output_format {
				octo.output_format = Some(parse_image_format(format).map_err(bad_request)?);
			}
			octo.provider_options = provider_options;
			octo.request_options = options;
			let provider = image_provider(provider_name, model)?;
			let operation = provider.submit_image(octo).await.map_err(media_error)?;
			Ok(normalize_image(&operation))
		}
		MediaRequest::Video(r) => {
			let mut octo = VideoGenerationRequest::new(r.prompt.clone()).with_model(model);
			octo.mode = parse_video_mode(r.mode.as_deref().unwrap_or("text_to_video")).map_err(bad_request)?;
			if let Some(frame) = &r.first_frame {
				octo.first_frame = Some(frame.clone().into_source(max_bytes).map_err(bad_request)?);
			}
			if let Some(frame) = &r.last_frame {
				octo.last_frame = Some(frame.clone().into_source(max_bytes).map_err(bad_request)?);
			}
			for source in &r.reference_images {
				octo.reference_images
					.push(source.clone().into_source(max_bytes).map_err(bad_request)?);
			}
			if let Some(video) = &r.source_video {
				octo.source_video = Some(video.clone().into_source(max_bytes).map_err(bad_request)?);
			}
			octo.count = r.count;
			octo.seed = r.seed;
			octo.duration_secs = r.duration_secs;
			if let Some(size) = &r.size {
				octo.geometry = Some(parse_geometry(size).map_err(bad_request)?);
			}
			octo.negative_prompt = r.negative_prompt.clone();
			if let Some(format) = &r.output_format {
				octo.output_format = Some(parse_video_format(format).map_err(bad_request)?);
			}
			octo.provider_options = provider_options;
			octo.request_options = options;
			let provider = video_provider(provider_name, model)?;
			let operation = provider.submit_video(octo).await.map_err(media_error)?;
			Ok(normalize_video(&operation))
		}
		MediaRequest::Speech(r) => {
			let mut octo = SpeechSynthesisRequest::new(r.input.clone(), r.voice.clone()).with_model(model);
			octo.language = r.language.clone();
			octo.instructions = r.instructions.clone();
			octo.speed = r.speed;
			if let Some(output) = r.output.clone() {
				octo.output = output.into_spec().map_err(bad_request)?;
			}
			octo.provider_options = provider_options;
			octo.request_options = options;
			let provider = speech_provider(provider_name, model)?;
			let operation = provider.submit_speech(octo).await.map_err(media_error)?;
			Ok(normalize_speech(&operation))
		}
		MediaRequest::Transcription(r) => {
			let audio = r.audio.clone().into_source(max_bytes).map_err(bad_request)?;
			let mut octo = TranscriptionRequest::new(audio).with_model(model);
			octo.language = r.language.clone();
			octo.prompt = r.prompt.clone();
			for granularity in &r.timestamp_granularities {
				octo.timestamp_granularities
					.push(parse_granularity(granularity).map_err(bad_request)?);
			}
			octo.provider_options = provider_options;
			octo.request_options = options;
			let provider = transcription_provider(provider_name, model)?;
			let operation = provider.submit_transcription(octo).await.map_err(media_error)?;
			Ok(normalize_transcription(&operation))
		}
	}
}

/// The handle's own task selects the trait — a stored job knows what it is.
async fn poll_once(provider_name: &str, handle: &JobHandle) -> Result<Normalized> {
	match handle.task {
		MediaTask::TextToImage | MediaTask::ImageEdit | MediaTask::Inpainting | MediaTask::ImageVariation => {
			let provider = image_provider(provider_name, &handle.model)?;
			let operation = provider.poll_image(handle).await.map_err(media_error)?;
			Ok(normalize_image(&operation))
		}
		MediaTask::TextToVideo
		| MediaTask::ImageToVideo
		| MediaTask::ReferenceToVideo
		| MediaTask::VideoExtend
		| MediaTask::VideoEdit => {
			let provider = video_provider(provider_name, &handle.model)?;
			let operation = provider.poll_video(handle).await.map_err(media_error)?;
			Ok(normalize_video(&operation))
		}
		MediaTask::TextToSpeech => {
			let provider = speech_provider(provider_name, &handle.model)?;
			let operation = provider.poll_speech(handle).await.map_err(media_error)?;
			Ok(normalize_speech(&operation))
		}
		MediaTask::SpeechToText | MediaTask::SpeechTranslation => {
			let provider = transcription_provider(provider_name, &handle.model)?;
			let operation = provider.poll_transcription(handle).await.map_err(media_error)?;
			Ok(normalize_transcription(&operation))
		}
		other => Err(anyhow!("Invalid request: unsupported media task {other:?}")),
	}
}

async fn cancel_once(provider_name: &str, handle: &JobHandle) -> Result<()> {
	match handle.task {
		MediaTask::TextToImage | MediaTask::ImageEdit | MediaTask::Inpainting | MediaTask::ImageVariation => {
			image_provider(provider_name, &handle.model)?
				.cancel_image(handle)
				.await
				.map_err(media_error)
		}
		MediaTask::TextToVideo
		| MediaTask::ImageToVideo
		| MediaTask::ReferenceToVideo
		| MediaTask::VideoExtend
		| MediaTask::VideoEdit => video_provider(provider_name, &handle.model)?
			.cancel_video(handle)
			.await
			.map_err(media_error),
		MediaTask::TextToSpeech => speech_provider(provider_name, &handle.model)?
			.cancel_speech(handle)
			.await
			.map_err(media_error),
		MediaTask::SpeechToText | MediaTask::SpeechTranslation => transcription_provider(provider_name, &handle.model)?
			.cancel_transcription(handle)
			.await
			.map_err(media_error),
		other => Err(anyhow!("Invalid request: unsupported media task {other:?}")),
	}
}

fn describe(provider: &str, model: &str) -> serde_json::Value {
	// Any of the four traits answers `capabilities` for the same model; image
	// is used as the probe because every adapter implements it.
	match MediaProviderFactory::create_image_provider(provider) {
		Ok(adapter) => serde_json::to_value(adapter.capabilities(model)).unwrap_or(serde_json::Value::Null),
		Err(_) => serde_json::Value::Null,
	}
}

fn image_provider(name: &str, model: &str) -> Result<Box<dyn ImageGenerationProvider>> {
	let provider = MediaProviderFactory::create_image_provider(name).map_err(media_error)?;
	ensure_supports(provider.supports_model(model), name, model)?;
	Ok(provider)
}

fn video_provider(name: &str, model: &str) -> Result<Box<dyn VideoGenerationProvider>> {
	let provider = MediaProviderFactory::create_video_provider(name).map_err(media_error)?;
	ensure_supports(provider.supports_model(model), name, model)?;
	Ok(provider)
}

fn speech_provider(name: &str, model: &str) -> Result<Box<dyn SpeechSynthesisProvider>> {
	let provider = MediaProviderFactory::create_speech_provider(name).map_err(media_error)?;
	ensure_supports(provider.supports_model(model), name, model)?;
	Ok(provider)
}

fn transcription_provider(name: &str, model: &str) -> Result<Box<dyn TranscriptionProvider>> {
	let provider = MediaProviderFactory::create_transcription_provider(name).map_err(media_error)?;
	ensure_supports(provider.supports_model(model), name, model)?;
	Ok(provider)
}

fn ensure_supports(supported: bool, provider: &str, model: &str) -> Result<()> {
	if supported {
		return Ok(());
	}
	Err(anyhow!(MediaTaskUnsupported {
		provider: provider.to_string(),
		model: model.to_string(),
	}))
}

/// The selected provider cannot serve this model at all. Distinct from a
/// provider fault so failover treats it as a candidate filter, not an outage.
#[derive(Debug)]
pub struct MediaTaskUnsupported {
	pub provider: String,
	pub model: String,
}

impl std::fmt::Display for MediaTaskUnsupported {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(
			f,
			"provider '{}' does not support media model '{}'",
			self.provider, self.model
		)
	}
}

impl std::error::Error for MediaTaskUnsupported {}

/// Carry `MediaError` through `anyhow` without flattening it — the handler
/// downcasts to map each variant onto its HTTP status.
fn media_error(error: MediaError) -> anyhow::Error {
	anyhow!(error)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::auth::Credentials;
	use crate::proxy::limiter::ProviderLimiter;
	use crate::storage::sqlite::SqliteStorage;
	use crate::storage::Storage;
	use octolib::media::{ArtifactSource, MediaArtifact, MediaKind, UsageLineItem, UsageUnit};
	use std::sync::{Arc, Mutex, RwLock};

	struct EnvGuard(Vec<(&'static str, Option<std::ffi::OsString>)>);

	impl EnvGuard {
		fn set(values: &[(&'static str, String)]) -> Self {
			Self(
				values
					.iter()
					.map(|(name, value)| {
						let previous = std::env::var_os(name);
						std::env::set_var(name, value);
						(*name, previous)
					})
					.collect(),
			)
		}
	}

	impl Drop for EnvGuard {
		fn drop(&mut self) {
			for (name, value) in &self.0 {
				match value {
					Some(value) => std::env::set_var(name, value),
					None => std::env::remove_var(name),
				}
			}
		}
	}

	#[test]
	fn media_routing_fault_classifier_uses_typed_rejections_and_statuses() {
		let classify = |media, expected| {
			let error = media_error(media).context("media submit failed");
			assert_eq!(is_media_provider_fault(&error), expected, "{error:#}");
		};
		for status in [400, 404, 413, 422] {
			classify(
				MediaError::Api {
					provider: "fal".into(),
					status,
					message: "rejected".into(),
				},
				false,
			);
		}
		for status in [401, 402, 403, 408, 429, 500, 502, 503, 504, 529] {
			classify(
				MediaError::Api {
					provider: "fal".into(),
					status,
					message: "unavailable".into(),
				},
				true,
			);
		}
		for media in [
			MediaError::InvalidRequest("invalid input".into()),
			MediaError::UnsupportedParameter {
				provider: "fal".into(),
				parameter: "size".into(),
				reason: "unsupported".into(),
			},
			MediaError::UnsupportedTask {
				provider: "fal".into(),
				model: "audio-model".into(),
				task: MediaTask::TextToImage,
			},
			MediaError::SourceTooLarge { actual: 2, maximum: 1 },
		] {
			classify(media, false);
		}
		for media in [
			MediaError::Authentication {
				provider: "fal".into(),
				message: "revoked key".into(),
			},
			MediaError::Permission {
				provider: "fal".into(),
				message: "model denied".into(),
			},
			MediaError::InsufficientCredits {
				provider: "fal".into(),
				message: "quota".into(),
			},
			MediaError::RateLimit {
				provider: "fal".into(),
				message: "rate limit".into(),
				retry_after_secs: None,
			},
			MediaError::Io(std::io::Error::from(std::io::ErrorKind::TimedOut)),
			MediaError::WaitTimeout {
				handle: Box::new(normalized(OperationStatus::Running, None).handle.unwrap()),
			},
		] {
			classify(media, true);
		}
		for (category, http_status, expected) in [
			(FailureCategory::ContentPolicy, None, false),
			(FailureCategory::ContentPolicy, Some(403), false),
			(FailureCategory::InvalidInput, None, false),
			(FailureCategory::Unsupported, None, false),
			(FailureCategory::RemoteJob, Some(400), false),
			(FailureCategory::RemoteJob, Some(404), false),
			(FailureCategory::RemoteJob, Some(413), false),
			(FailureCategory::RemoteJob, Some(422), false),
			(FailureCategory::Authentication, Some(401), true),
			(FailureCategory::RateLimit, Some(429), true),
			(FailureCategory::Transport, None, true),
		] {
			classify(
				MediaError::RemoteFailure(octolib::media::GenerationFailure {
					category,
					message: "generation failed".into(),
					provider_code: None,
					http_status,
					request_id: None,
					provider_metadata: serde_json::Value::Null,
				}),
				expected,
			);
		}
		assert!(!is_media_provider_fault(&bad_request("invalid geometry")));
		assert!(!is_media_provider_fault(&anyhow!(MediaTaskUnsupported {
			provider: "fal".into(),
			model: "audio-model".into()
		})));
		assert!(!is_media_provider_fault(&anyhow!(ProxyTimeoutError::ProviderQueue {
			provider: "fal".into(),
			timeout: Duration::from_secs(1),
		})));
		assert!(is_media_provider_fault(&anyhow!(ProxyTimeoutError::Upstream {
			provider: "fal".into(),
			timeout: Duration::from_secs(1),
		})));
	}

	#[tokio::test]
	async fn media_routing_ordered_submit_prefers_primary_fails_over_and_polls_accepted_lane() {
		use http_body_util::{BodyExt, Full};
		use hyper::service::service_fn;
		use hyper_util::rt::TokioIo;

		const PRIMARY: &str = "/primary/fal-ai/flux/dev";
		const FALLBACK: &str = "/fallback/models/black-forest-labs/flux-1.1-pro/predictions";
		const POLL: &str = "/fallback/predictions/job-1";
		const THIRD: &str = "/third/text_to_image";

		struct Upstream {
			primary_status: u16,
			requests: Vec<String>,
			primary_barrier: Option<Arc<tokio::sync::Barrier>>,
		}

		let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
		let address = listener.local_addr().unwrap();
		let upstream = Arc::new(Mutex::new(Upstream {
			primary_status: 200,
			requests: Vec::new(),
			primary_barrier: None,
		}));
		let state = upstream.clone();
		let server = tokio::spawn(async move {
			loop {
				let (socket, _) = listener.accept().await.unwrap();
				let state = state.clone();
				tokio::spawn(async move {
					hyper::server::conn::http1::Builder::new()
						.serve_connection(
							TokioIo::new(socket),
							service_fn(move |req: hyper::Request<hyper::body::Incoming>| {
								let state = state.clone();
								async move {
									let path = req.uri().path().to_string();
									req.collect().await.unwrap();
									let barrier = if path == PRIMARY {
										state.lock().unwrap().primary_barrier.take()
									} else {
										None
									};
									if let Some(barrier) = barrier {
										barrier.wait().await;
										barrier.wait().await;
									}
									let (status, body) = {
										let mut state = state.lock().unwrap();
										match path.as_str() {
											"/catalog" => (
												200,
												serde_json::json!({"models": [{
													"endpoint_id": "fal-ai/flux/dev", "metadata": {"category": "text-to-image"}
												}]}),
											),
											"/fallback/models/black-forest-labs/flux-1.1-pro" => (
												200,
												serde_json::json!({
													"latest_version": {"openapi_schema": {"components": {"schemas": {"Input": {
														"type": "object", "properties": {"prompt": {"type": "string"}}
													}}}}}
												}),
											),
											PRIMARY => {
												state.requests.push(path);
												let body = if state.primary_status == 200 {
													serde_json::json!({"request_id": "primary-job", "queue_position": 0})
												} else {
													serde_json::json!({"detail": "rejected"})
												};
												(state.primary_status, body)
											}
											FALLBACK => {
												state.requests.push(path);
												(201, serde_json::json!({"id": "job-1", "status": "starting"}))
											}
											POLL => {
												state.requests.push(path);
												(
													200,
													serde_json::json!({"id": "job-1", "status": "succeeded",
													"output": ["https://cdn.example.test/image.png"]}),
												)
											}
											THIRD => {
												state.requests.push(path);
												(200, serde_json::json!({"id": "third-job"}))
											}
											_ => panic!("unexpected upstream path: {path}"),
										}
									};
									Ok::<_, std::convert::Infallible>(
										hyper::Response::builder()
											.status(status)
											.header("content-type", "application/json")
											.body(Full::new(bytes::Bytes::from(body.to_string())))
											.unwrap(),
									)
								}
							}),
						)
						.await
						.unwrap();
				});
			}
		});
		// No other proxy test sets these media adapter variables.
		let _env = EnvGuard::set(&[
			("FAL_API_URL", format!("http://{address}/primary")),
			("FAL_MODELS_URL", format!("http://{address}/catalog")),
			("FAL_API_KEY", "primary-key".into()),
			("REPLICATE_API_URL", format!("http://{address}/fallback")),
			("REPLICATE_API_KEY", "fallback-key".into()),
			("RUNWAY_API_URL", format!("http://{address}/third")),
			("RUNWAY_API_KEY", "third-key".into()),
		]);
		let config: Config = toml::from_str(
			r#"
			[server]
			api_key = "master"
			failover_on_error = false
			provider_queue_timeout_secs = 1
			provider_error_cooldown_secs = 60
			[media_models]
			flux = ["fal:fal-ai/flux/dev", "replicate:black-forest-labs/flux-1.1-pro", "runway:gen4_image"]
			[routing.models]
			flux = "ordered"
			[providers.fal]
			concurrency = 1
			[providers.replicate]
			concurrency = 1
			"#,
		)
		.unwrap();
		let storage = Arc::new(SqliteStorage::new(":memory:").unwrap());
		let key = storage.create_api_key("key", None, None, None).unwrap();
		let credentials = Arc::new(Credentials::new(storage.clone(), None).unwrap());
		let limiter = Arc::new(ProviderLimiter::from_config(&config));
		let engine = ProxyEngine::new(
			storage,
			Arc::new(RwLock::new(Arc::new(config))),
			Arc::new(RwLock::new(limiter.clone())),
			1,
			credentials,
		);
		let request = || {
			image_request(serde_json::json!({
				"model": "flux", "prompt": "a red panda", "wait": false, "size": "1024x1024"
			}))
		};
		let paths = || std::mem::take(&mut upstream.lock().unwrap().requests);

		for _ in 0..8 {
			let outcome = engine.process_media(request(), &key).await.unwrap();
			assert_eq!(outcome.response.provider, "fal");
			assert!(outcome.accepted);
		}
		assert_eq!(paths(), vec![PRIMARY; 8]);

		let busy = limiter.acquire("fal").await;
		let outcome = engine.process_media(request(), &key).await.unwrap();
		assert_eq!(outcome.response.provider, "replicate");
		assert_eq!(paths(), vec![FALLBACK]);
		drop(busy);

		for _ in 0..3 {
			engine.provider_health.record_failure("fal", Duration::from_secs(60));
		}
		let outcome = engine.process_media(request(), &key).await.unwrap();
		assert_eq!(outcome.response.provider, "replicate");
		assert_eq!(paths(), vec![FALLBACK]);
		engine.provider_health.record_success("fal");

		for status in [400, 404, 413, 422] {
			upstream.lock().unwrap().primary_status = status;
			for _ in 0..3 {
				let error = engine.process_media(request(), &key).await.err().unwrap();
				assert!(
					matches!(error.downcast_ref::<MediaError>(), Some(MediaError::Api { status: actual, .. }) if *actual == status)
				);
				assert_eq!(paths(), vec![PRIMARY]);
			}
			assert!(!engine.provider_health.is_cooling("fal"));
		}

		for status in [401, 429] {
			upstream.lock().unwrap().primary_status = status;
			let moved = engine.process_media(request(), &key).await.unwrap();
			assert_eq!(moved.response.provider, "replicate");
			assert_eq!(paths(), vec![PRIMARY, FALLBACK]);
			engine.provider_health.record_success("fal");
		}

		upstream.lock().unwrap().primary_status = 401;
		let barrier = Arc::new(tokio::sync::Barrier::new(2));
		upstream.lock().unwrap().primary_barrier = Some(barrier.clone());
		let saturate_second = async {
			barrier.wait().await;
			let busy = limiter.acquire("replicate").await;
			barrier.wait().await;
			busy
		};
		let (outcome, busy) = tokio::join!(engine.process_media(request(), &key), saturate_second);
		assert_eq!(outcome.unwrap().response.provider, "runway");
		assert_eq!(paths(), vec![PRIMARY, THIRD]);
		drop(busy);
		engine.provider_health.record_success("fal");

		let moved = engine.process_media(request(), &key).await.unwrap();
		assert_eq!(moved.response.provider, "replicate");
		assert!(moved.accepted);
		assert_eq!(paths(), vec![PRIMARY, FALLBACK]);

		upstream.lock().unwrap().primary_status = 200;
		let polled = engine.poll_media(&moved.response.id, &key).await.unwrap().unwrap();
		assert_eq!(polled.response.provider, "replicate");
		assert_eq!(polled.response.status, OperationStatus::Succeeded);
		assert_eq!(paths(), vec![POLL]);
		engine.poll_media(&moved.response.id, &key).await.unwrap().unwrap();
		assert!(paths().is_empty());
		server.abort();
	}

	fn image_request(json: serde_json::Value) -> MediaRequest {
		MediaRequest::Image(Box::new(serde_json::from_value(json).expect("valid image request")))
	}

	fn artifact() -> MediaArtifact {
		MediaArtifact {
			kind: MediaKind::Image,
			media_type: "image/png".to_string(),
			source: ArtifactSource::Url("https://cdn.example.test/a.png".to_string()),
			size_bytes: Some(10),
			dimensions: None,
			duration_secs: None,
			frame_rate: None,
			sample_rate_hz: None,
			channels: None,
			expires_at: None,
			metadata: serde_json::Value::Null,
		}
	}

	fn normalized(status: OperationStatus, usage: Option<MediaUsage>) -> Normalized {
		Normalized {
			status,
			progress: None,
			handle: Some(JobHandle {
				provider: "fal".to_string(),
				model: "fal-ai/flux/dev".to_string(),
				task: MediaTask::TextToImage,
				remote_id: "job-1".to_string(),
				cost_estimate: None,
				warnings: Vec::new(),
				expected_outputs: None,
			}),
			artifacts: vec![WireArtifact::from(&artifact())],
			result: None,
			usage,
			warnings: Vec::new(),
			safety: SafetyReport::default(),
			error: None,
		}
	}

	fn record() -> StoredMedia {
		StoredMedia {
			id: "med_1".to_string(),
			api_key_id: 1,
			task: task_name(MediaTask::TextToImage),
			status: status_name(OperationStatus::Queued),
			input_model: "flux".to_string(),
			resolved_model: "fal-ai/flux/dev".to_string(),
			provider: "fal".to_string(),
			request: serde_json::json!({}),
			handle: None,
			result: None,
			usage: None,
			warnings: None,
			error: None,
			created_at: 1_700_000_000,
			completed_at: None,
		}
	}

	#[test]
	fn task_names_round_trip_through_storage() {
		for task in [
			MediaTask::TextToImage,
			MediaTask::ImageEdit,
			MediaTask::TextToVideo,
			MediaTask::ImageToVideo,
			MediaTask::TextToSpeech,
			MediaTask::SpeechToText,
		] {
			assert_eq!(parse_task(&task_name(task)), task);
		}
	}

	#[test]
	fn status_names_round_trip_and_classify_terminality() {
		for status in [
			OperationStatus::Queued,
			OperationStatus::Running,
			OperationStatus::Succeeded,
			OperationStatus::Failed,
			OperationStatus::Cancelled,
			OperationStatus::Expired,
		] {
			assert_eq!(parse_status(&status_name(status)), status);
			assert_eq!(terminal_name(&status_name(status)), status.is_terminal());
		}
		assert!(!terminal_name("queued"));
		assert!(terminal_name("succeeded"));
	}

	#[test]
	fn image_mode_selects_the_task() {
		assert_eq!(
			image_request(serde_json::json!({"model":"flux","prompt":"x"}))
				.task()
				.unwrap(),
			MediaTask::TextToImage
		);
		assert_eq!(
			image_request(serde_json::json!({"model":"flux","prompt":"x","mode":"inpaint"}))
				.task()
				.unwrap(),
			MediaTask::Inpainting
		);
		assert!(
			image_request(serde_json::json!({"model":"flux","prompt":"x","mode":"upscale"}))
				.task()
				.is_err()
		);
	}

	/// The persisted request must stay useful for observability while never
	/// carrying a base64 blob into the database.
	#[test]
	fn redaction_drops_inline_payloads_but_keeps_the_request_legible() {
		let request = image_request(serde_json::json!({
			"model": "flux",
			"prompt": "a red panda",
			"count": 2,
			"size": "1024x1024",
			"source_images": [
				{"type": "base64", "data": "aGVsbG8=", "media_type": "image/png"},
				{"type": "url", "url": "https://example.test/b.png"}
			]
		}));
		let stored = request.redacted();
		assert_eq!(stored["prompt"], serde_json::json!("a red panda"));
		assert_eq!(stored["count"], serde_json::json!(2));
		assert_eq!(stored["size"], serde_json::json!("1024x1024"));

		let sources = stored["source_images"].as_array().unwrap();
		assert_eq!(
			sources[0]["inline_media_omitted"],
			serde_json::json!(true),
			"base64 payloads must never reach the database"
		);
		assert!(sources[0].get("data").is_none());
		assert_eq!(sources[1]["url"], serde_json::json!("https://example.test/b.png"));

		// Absent optional fields are omitted rather than stored as null noise.
		assert!(stored.get("seed").is_none());
		assert!(stored.get("mask").is_none());
	}

	#[test]
	fn labels_use_the_alias_not_the_resolved_model() {
		let request = image_request(serde_json::json!({"model":"flux","prompt":"x"}));
		assert_eq!(request.model_label(), "flux");
		assert_eq!(request.task_label(), "image");
	}

	/// A live job keeps its handle so it can be resumed; a terminal one drops it,
	/// which is also what stops `poll_media` re-billing a finished record.
	#[test]
	fn apply_keeps_the_handle_only_while_the_job_is_live() {
		let mut live = record();
		apply(&mut live, &normalized(OperationStatus::Running, None));
		assert_eq!(live.status, "running");
		assert!(live.handle.is_some());
		assert!(live.completed_at.is_none());

		let mut done = record();
		apply(&mut done, &normalized(OperationStatus::Succeeded, None));
		assert_eq!(done.status, "succeeded");
		assert!(done.handle.is_none());
		assert!(done.completed_at.is_some());
	}

	#[test]
	fn stored_usage_exposes_cost_where_the_admin_aggregate_reads_it() {
		let usage = MediaUsage {
			line_items: vec![UsageLineItem {
				unit: UsageUnit::Images,
				quantity: 2.0,
				cost: None,
				description: None,
			}],
			provider_reported_cost: Some(0.08),
			estimated_cost: None,
			..MediaUsage::default()
		};
		let mut done = record();
		apply(&mut done, &normalized(OperationStatus::Succeeded, Some(usage)));

		// `get_usage` extracts `$.cost` from this column; the key must be at the
		// top level of the stored object or every media row aggregates as unpriced.
		let stored = done.usage.expect("usage persisted");
		assert_eq!(stored["cost"], serde_json::json!(0.08));
		assert_eq!(stored["cost_source"], serde_json::json!("provider"));
	}

	#[test]
	fn an_unpriced_result_stores_no_cost_rather_than_zero() {
		let mut done = record();
		apply(
			&mut done,
			&normalized(OperationStatus::Succeeded, Some(MediaUsage::default())),
		);
		let stored = done.usage.expect("usage persisted");
		assert!(stored["cost"].is_null());
		assert_eq!(stored["cost_source"], serde_json::json!("unavailable"));
	}

	#[test]
	fn artifacts_survive_the_storage_round_trip() {
		let mut done = record();
		apply(&mut done, &normalized(OperationStatus::Succeeded, None));
		let stored = done.result.expect("result persisted");
		let artifacts: Vec<WireArtifact> =
			serde_json::from_value(stored["artifacts"].clone()).expect("artifacts decode");
		assert_eq!(artifacts.len(), 1);
		assert_eq!(artifacts[0].media_type, "image/png");
	}

	fn engine_with_storage() -> (ProxyEngine, Arc<SqliteStorage>) {
		let config: Config = toml::from_str("[server]\napi_key = \"master\"").unwrap();
		let storage = Arc::new(SqliteStorage::new(":memory:").unwrap());
		let credentials = Arc::new(Credentials::new(storage.clone(), None).unwrap());
		let limiter = Arc::new(ProviderLimiter::from_config(&config));
		let engine = ProxyEngine::new(
			storage.clone(),
			Arc::new(RwLock::new(Arc::new(config))),
			Arc::new(RwLock::new(limiter)),
			1,
			credentials,
		);
		(engine, storage)
	}

	/// The storage backends hard-code this list in SQL; it has to stay octolib's.
	#[test]
	fn storage_terminal_status_list_matches_octolib() {
		for status in [
			OperationStatus::Queued,
			OperationStatus::Running,
			OperationStatus::Succeeded,
			OperationStatus::Failed,
			OperationStatus::CancellationRequested,
			OperationStatus::Cancelled,
			OperationStatus::Expired,
		] {
			let listed = crate::storage::TERMINAL_MEDIA_STATUSES_SQL.contains(&format!("'{}'", status_name(status)));
			assert_eq!(listed, status.is_terminal(), "{status:?}");
		}
	}

	/// OpenRouter's video endpoint: `vid-done` has finished, `vid-slow` never does.
	async fn serve_video_jobs() -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
		use http_body_util::Full;
		use hyper::service::service_fn;
		use hyper_util::rt::TokioIo;

		let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
		let address = listener.local_addr().unwrap();
		let server = tokio::spawn(async move {
			loop {
				let (socket, _) = listener.accept().await.unwrap();
				tokio::spawn(async move {
					hyper::server::conn::http1::Builder::new()
						.serve_connection(
							TokioIo::new(socket),
							service_fn(|req: hyper::Request<hyper::body::Incoming>| async move {
								let body = match req.uri().path() {
									"/videos/vid-done" => serde_json::json!({
										"id": "vid-done", "status": "completed",
										"unsigned_urls": ["https://cdn.example.test/v.mp4"]
									}),
									"/videos/vid-slow" => {
										serde_json::json!({"id": "vid-slow", "status": "in_progress"})
									}
									other => panic!("unexpected upstream path: {other}"),
								};
								Ok::<_, std::convert::Infallible>(
									hyper::Response::builder()
										.status(200)
										.header("content-type", "application/json")
										.body(Full::new(bytes::Bytes::from(body.to_string())))
										.unwrap(),
								)
							}),
						)
						.await
						.unwrap();
				});
			}
		});
		(address, server)
	}

	fn live_video_row(id: &str, remote_id: &str, api_key_id: i64, age_secs: u64) -> StoredMedia {
		let mut row = record();
		row.id = id.to_string();
		row.api_key_id = api_key_id;
		row.task = task_name(MediaTask::TextToVideo);
		row.status = status_name(OperationStatus::Running);
		row.provider = "openrouter".to_string();
		row.resolved_model = "google/veo-3.1".to_string();
		row.created_at = now_unix() - age_secs;
		row.handle = Some(
			serde_json::to_value(JobHandle {
				provider: "openrouter".to_string(),
				model: "google/veo-3.1".to_string(),
				task: MediaTask::TextToVideo,
				remote_id: remote_id.to_string(),
				cost_estimate: None,
				warnings: Vec::new(),
				expected_outputs: None,
			})
			.unwrap(),
		);
		row
	}

	/// A caller that stopped polling must not leave a finished, already-charged
	/// job unbilled, and a job the provider never finishes must not stay open.
	#[tokio::test]
	async fn sweep_settles_stale_jobs_and_fails_abandoned_ones() {
		const HOUR_SECS: u64 = 60 * 60;

		let (address, server) = serve_video_jobs().await;
		// No other test sets these; the OpenRouter adapter reads them from the process environment.
		let _env = EnvGuard::set(&[
			("OPENROUTER_MEDIA_API_URL", format!("http://{address}")),
			("OPENROUTER_API_KEY", "sweep-key".into()),
		]);
		let (engine, storage) = engine_with_storage();
		let key = storage.create_api_key("key", None, None, None).unwrap();
		for row in [
			live_video_row("med_done", "vid-done", key.id, 10 * 60),
			live_video_row("med_fresh", "vid-done", key.id, 30),
			live_video_row("med_slow", "vid-slow", key.id, 10 * 60),
			live_video_row("med_abandoned", "vid-slow", key.id, 25 * HOUR_SECS),
		] {
			storage.store_media(&row).unwrap();
		}
		// A writer that read the row before the sweep settled it.
		let racer = storage.get_media("med_done", key.id).unwrap().unwrap();
		let get = |id: &str| storage.get_media(id, key.id).unwrap().unwrap();

		assert_eq!(engine.sweep_media().await.unwrap(), 2);

		let done = get("med_done");
		assert_eq!(done.status, "succeeded");
		assert!(done.handle.is_none());
		let completed_at = done
			.completed_at
			.expect("billing drains succeeded rows by completed_at");

		// Under five minutes old: still its client's to poll.
		let fresh = get("med_fresh");
		assert_eq!(fresh.status, "running");
		assert!(fresh.handle.is_some());
		// Polled, still running, not old enough to give up on.
		let slow = get("med_slow");
		assert_eq!(slow.status, "running");
		assert!(slow.handle.is_some());

		let abandoned = get("med_abandoned");
		assert_eq!(abandoned.status, "failed");
		assert!(abandoned.handle.is_none());
		let reason = abandoned.error.expect("failure recorded")["message"].to_string();
		assert!(reason.contains("abandoned: provider never finished"), "{reason}");

		// The stale copy cannot rewrite the settled row.
		let mut late = racer;
		late.status = status_name(OperationStatus::Failed);
		late.completed_at = Some(completed_at + 999);
		assert!(!storage.update_media(&late).unwrap());
		let kept = get("med_done");
		assert_eq!(kept.status, "succeeded");
		assert_eq!(kept.completed_at, Some(completed_at));

		// Settled rows leave the queue; the slow one is polled again and stays open.
		assert_eq!(engine.sweep_media().await.unwrap(), 0);
		assert_eq!(get("med_done").completed_at, Some(completed_at));
		assert_eq!(get("med_slow").status, "running");
		server.abort();
	}
}
