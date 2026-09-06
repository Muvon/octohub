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

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};
use octolib::media::{
    ImageGenerationProvider, ImageGenerationRequest, ImageGenerationResult, JobHandle, MediaError,
    MediaProviderFactory, MediaTask, MediaUsage, Operation, OperationStatus, ProviderWarning,
    RequestOptions, SafetyReport, SpeechSynthesisProvider, SpeechSynthesisRequest,
    SpeechSynthesisResult, TranscriptionProvider, TranscriptionRequest, TranscriptionResult,
    VideoGenerationProvider, VideoGenerationRequest, VideoGenerationResult,
};
use uuid::Uuid;

use crate::api::media_types::*;
use crate::config::Config;
use crate::proxy::engine::{ensure_model_allowed, is_provider_fault, pick_admitted, ProxyEngine};
use crate::storage::{ApiKey, StoredMedia};

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
    anyhow!("Invalid request: {message}")
}

impl ProxyEngine {
    /// Submit a media job, persist it, and optionally wait for it inline.
    pub async fn process_media(
        &self,
        request: MediaRequest,
        api_key: &ApiKey,
    ) -> Result<MediaOutcome> {
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

        let (provider_name, resolved_model, mut normalized, upstream_duration) = loop {
            let (provider_name, resolved_model) = pick_admitted(
                &self.rate_tracker,
                &self.provider_health,
                &config,
                candidates.clone(),
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
                        anyhow!(crate::proxy::engine::ProxyTimeoutError::ProviderQueue {
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
                    let provider_fault = is_provider_fault(&err);
                    if provider_fault {
                        self.provider_health
                            .record_failure(&provider_name, cooldown);
                    }
                    candidates.retain(|(p, _)| !p.eq_ignore_ascii_case(&provider_name));
                    if !(config.server.failover_on_error && provider_fault) || candidates.is_empty()
                    {
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
            handle: normalized
                .handle
                .as_ref()
                .and_then(|h| serde_json::to_value(h).ok()),
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
            tokio::task::spawn_blocking(move || storage.update_media(&to_update)).await??;
        }

        Ok(self.finish(record, normalized, total))
    }

    /// Advance a stored job. A terminal row is returned as stored — no upstream
    /// call, so re-reading a finished record is free and never re-bills.
    pub async fn poll_media(&self, id: &str, api_key: &ApiKey) -> Result<Option<MediaOutcome>> {
        let storage = self.storage.clone();
        let (lookup_id, key_id) = (id.to_string(), api_key.id);
        let Some(mut record) =
            tokio::task::spawn_blocking(move || storage.get_media(&lookup_id, key_id)).await??
        else {
            return Ok(None);
        };

        if record.handle.is_none() || terminal_name(&record.status) {
            return Ok(Some(self.from_record(record)));
        }

        let handle: JobHandle = serde_json::from_value(
            record
                .handle
                .clone()
                .ok_or_else(|| anyhow!("media record has no resumable handle"))?,
        )
        .context("stored media handle is not a valid JobHandle")?;

        let started = Instant::now();
        let normalized = poll_once(&record.provider, &handle).await?;
        apply(&mut record, &normalized);

        let storage = self.storage.clone();
        let to_update = record.clone();
        tokio::task::spawn_blocking(move || storage.update_media(&to_update)).await??;

        Ok(Some(self.finish(record, normalized, started.elapsed())))
    }

    /// Ask the provider to cancel. Cancellation is best-effort and provider
    /// dependent; the row is only marked cancelled once the provider confirms.
    pub async fn cancel_media(&self, id: &str, api_key: &ApiKey) -> Result<Option<MediaOutcome>> {
        let storage = self.storage.clone();
        let (lookup_id, key_id) = (id.to_string(), api_key.id);
        let Some(mut record) =
            tokio::task::spawn_blocking(move || storage.get_media(&lookup_id, key_id)).await??
        else {
            return Ok(None);
        };
        if terminal_name(&record.status) {
            return Ok(Some(self.from_record(record)));
        }
        let handle: JobHandle = serde_json::from_value(
            record
                .handle
                .clone()
                .ok_or_else(|| anyhow!("media record has no resumable handle"))?,
        )
        .context("stored media handle is not a valid JobHandle")?;

        cancel_once(&record.provider, &handle).await?;
        record.status = status_name(OperationStatus::Cancelled);
        record.completed_at = Some(now_unix());
        record.handle = None;

        let storage = self.storage.clone();
        let to_update = record.clone();
        tokio::task::spawn_blocking(move || storage.update_media(&to_update)).await??;
        Ok(Some(self.from_record(record)))
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

    fn finish(
        &self,
        record: StoredMedia,
        normalized: Normalized,
        upstream_duration: Duration,
    ) -> MediaOutcome {
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
            upstream_duration,
        }
    }

    /// Rebuild the wire envelope from a stored row without touching upstream.
    fn from_record(&self, record: StoredMedia) -> MediaOutcome {
        let artifacts = record
            .result
            .as_ref()
            .and_then(|r| r.get("artifacts"))
            .and_then(|a| serde_json::from_value::<Vec<WireArtifact>>(a.clone()).ok())
            .unwrap_or_default();
        let transcript = record
            .result
            .as_ref()
            .and_then(|r| r.get("transcript"))
            .cloned();
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
            upstream_duration: Duration::ZERO,
        }
    }
}

fn apply(record: &mut StoredMedia, normalized: &Normalized) {
    record.status = status_name(normalized.status);
    record.handle = if is_terminal(normalized.status) {
        None
    } else {
        normalized
            .handle
            .as_ref()
            .and_then(|h| serde_json::to_value(h).ok())
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

fn task_name(task: MediaTask) -> String {
    serde_json::to_value(task)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_else(|| "text_to_image".to_string())
}

fn parse_task(name: &str) -> MediaTask {
    serde_json::from_value(serde_json::Value::String(name.to_string()))
        .unwrap_or(MediaTask::TextToImage)
}

fn status_name(status: OperationStatus) -> String {
    serde_json::to_value(status)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_else(|| "queued".to_string())
}

fn parse_status(name: &str) -> OperationStatus {
    serde_json::from_value(serde_json::Value::String(name.to_string()))
        .unwrap_or(OperationStatus::Queued)
}

fn terminal_name(name: &str) -> bool {
    is_terminal(parse_status(name))
}

/// Build the octolib request for whichever task this is and hand it to the
/// adapter. The returned operation may already be terminal (ElevenLabs and
/// OpenRouter answer synchronously) or queued (fal, Replicate, Runway).
async fn submit(
    request: &MediaRequest,
    config: &Config,
    provider_name: &str,
    model: &str,
) -> Result<Normalized> {
    let common = request.common();
    let options = request_options(config, common);
    let provider_options = filter_options_for(&common.provider_options, provider_name);
    let max_bytes = options.max_source_bytes;

    match request {
        MediaRequest::Image(r) => {
            let mut octo = ImageGenerationRequest::new(r.prompt.clone()).with_model(model);
            octo.mode =
                parse_image_mode(r.mode.as_deref().unwrap_or("generate")).map_err(bad_request)?;
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
            octo.mode = parse_video_mode(r.mode.as_deref().unwrap_or("text_to_video"))
                .map_err(bad_request)?;
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
                octo.source_video =
                    Some(video.clone().into_source(max_bytes).map_err(bad_request)?);
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
            let mut octo =
                SpeechSynthesisRequest::new(r.input.clone(), r.voice.clone()).with_model(model);
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
            let audio = r
                .audio
                .clone()
                .into_source(max_bytes)
                .map_err(bad_request)?;
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
            let operation = provider
                .submit_transcription(octo)
                .await
                .map_err(media_error)?;
            Ok(normalize_transcription(&operation))
        }
    }
}

/// The handle's own task selects the trait — a stored job knows what it is.
async fn poll_once(provider_name: &str, handle: &JobHandle) -> Result<Normalized> {
    match handle.task {
        MediaTask::TextToImage
        | MediaTask::ImageEdit
        | MediaTask::Inpainting
        | MediaTask::ImageVariation => {
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
            let operation = provider
                .poll_transcription(handle)
                .await
                .map_err(media_error)?;
            Ok(normalize_transcription(&operation))
        }
        other => Err(anyhow!("Invalid request: unsupported media task {other:?}")),
    }
}

async fn cancel_once(provider_name: &str, handle: &JobHandle) -> Result<()> {
    match handle.task {
        MediaTask::TextToImage
        | MediaTask::ImageEdit
        | MediaTask::Inpainting
        | MediaTask::ImageVariation => image_provider(provider_name, &handle.model)?
            .cancel_image(handle)
            .await
            .map_err(media_error),
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
        MediaTask::SpeechToText | MediaTask::SpeechTranslation => {
            transcription_provider(provider_name, &handle.model)?
                .cancel_transcription(handle)
                .await
                .map_err(media_error)
        }
        other => Err(anyhow!("Invalid request: unsupported media task {other:?}")),
    }
}

fn describe(provider: &str, model: &str) -> serde_json::Value {
    // Any of the four traits answers `capabilities` for the same model; image
    // is used as the probe because every adapter implements it.
    match MediaProviderFactory::create_image_provider(provider) {
        Ok(adapter) => {
            serde_json::to_value(adapter.capabilities(model)).unwrap_or(serde_json::Value::Null)
        }
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
    let provider =
        MediaProviderFactory::create_transcription_provider(name).map_err(media_error)?;
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
    use octolib::media::{ArtifactSource, MediaArtifact, MediaKind, UsageLineItem, UsageUnit};

    fn image_request(json: serde_json::Value) -> MediaRequest {
        MediaRequest::Image(Box::new(
            serde_json::from_value(json).expect("valid image request"),
        ))
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
        assert_eq!(
            sources[1]["url"],
            serde_json::json!("https://example.test/b.png")
        );

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
        apply(
            &mut done,
            &normalized(OperationStatus::Succeeded, Some(usage)),
        );

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
}
