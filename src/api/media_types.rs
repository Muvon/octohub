//! Wire types for the media endpoints.
//!
//! These mirror octolib's `media` request structs one-to-one; the only
//! translations are the ones the wire forces: base64 for binary payloads,
//! a `size` string for `OutputGeometry`, and a single response envelope that
//! covers all four tasks in both terminal and in-flight states.

use octolib::media::{
    ArtifactSource, AudioFormat, AudioOutputSpec, ImageFormat, ImageGenerationMode, MediaArtifact,
    MediaKind, MediaSource, MediaTask, MediaUsage, OperationStatus, OutputGeometry,
    ProviderOptions, ProviderWarning, SafetyReport, TimestampGranularity, TranscriptSegment,
    TranscriptWord, UnsupportedParameterPolicy, VideoFormat, VideoGenerationMode,
};
use serde::{Deserialize, Serialize};

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;

/// The reserved provider-option key OctoHub refuses to accept from a client.
/// A caller who could supply their own rate could declare their own generation
/// free, so pricing stays entirely octolib's (see `media::reference_pricing`).
const RESERVED_OPTION: &str = "cost_estimate";

/// Fields shared by every media request.
#[derive(Debug, Clone, Deserialize)]
pub struct MediaCommon {
    /// Alias from `[media_models]`, or a literal `provider:model`.
    pub model: String,
    /// Wait for completion inline. `false` returns 202 with a job id.
    #[serde(default = "default_wait")]
    pub wait: bool,
    /// Provider-namespaced native options, filtered to the winning candidate.
    #[serde(default)]
    pub provider_options: ProviderOptions,
    /// Whether a parameter the adapter cannot honour fails the request or is
    /// dropped with a warning. Strict by default: money is at stake.
    #[serde(default)]
    pub unsupported_parameters: UnsupportedParameters,
}

fn default_wait() -> bool {
    true
}

#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnsupportedParameters {
    #[default]
    Error,
    WarnAndDrop,
}

impl From<UnsupportedParameters> for UnsupportedParameterPolicy {
    fn from(value: UnsupportedParameters) -> Self {
        match value {
            UnsupportedParameters::Error => Self::Error,
            UnsupportedParameters::WarnAndDrop => Self::WarnAndDrop,
        }
    }
}

/// A binary input as it arrives on the wire.
///
/// `base64` is OctoHub's encoding for `MediaSource::Bytes` — octolib has no
/// base64 variant because base64 is a wire encoding, not a source kind. The
/// `file`, `provider_file` and `object_storage` variants are deliberately
/// absent: a client-supplied path would read the *server's* filesystem.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WireMediaSource {
    Url {
        url: String,
        #[serde(default)]
        media_type: Option<String>,
    },
    Base64 {
        data: String,
        media_type: String,
    },
}

impl WireMediaSource {
    /// Decode into octolib's `MediaSource`, enforcing the byte cap before the
    /// payload is ever handed to a provider.
    pub fn into_source(self, max_bytes: usize) -> Result<MediaSource, String> {
        match self {
            Self::Url { url, media_type } => Ok(MediaSource::Url { url, media_type }),
            Self::Base64 { data, media_type } => {
                let data = BASE64
                    .decode(data.as_bytes())
                    .map_err(|error| format!("invalid base64 media payload: {error}"))?;
                if data.len() > max_bytes {
                    return Err(format!(
                        "media payload is {} bytes, which exceeds the {max_bytes}-byte limit",
                        data.len()
                    ));
                }
                Ok(MediaSource::Bytes { data, media_type })
            }
        }
    }

    /// The redacted form persisted in the `media` table. Base64 blobs must
    /// never reach the database — they would add megabytes per row for no
    /// observability gain.
    pub fn redacted(&self) -> serde_json::Value {
        match self {
            Self::Url { url, media_type } => {
                serde_json::json!({"type": "url", "url": url, "media_type": media_type})
            }
            Self::Base64 { data, media_type } => serde_json::json!({
                "type": "base64",
                "inline_media_omitted": true,
                "encoded_bytes": data.len(),
                "media_type": media_type,
            }),
        }
    }
}

/// `"1024x1024"` → dimensions, `"16:9"` → aspect ratio. The inverse of
/// `OutputGeometry::as_api_string()`.
pub fn parse_geometry(value: &str) -> Result<OutputGeometry, String> {
    let parse_pair = |sep: char| -> Option<(u32, u32)> {
        let (left, right) = value.split_once(sep)?;
        Some((left.trim().parse().ok()?, right.trim().parse().ok()?))
    };
    if let Some((width, height)) = parse_pair('x') {
        if width > 0 && height > 0 {
            return Ok(OutputGeometry::Dimensions { width, height });
        }
    }
    if let Some((width, height)) = parse_pair(':') {
        if width > 0 && height > 0 {
            return Ok(OutputGeometry::AspectRatio { width, height });
        }
    }
    Err(format!(
        "invalid size '{value}': expected WIDTHxHEIGHT (e.g. 1024x1024) or W:H (e.g. 16:9)"
    ))
}

#[derive(Debug, Clone, Deserialize)]
pub struct ImageRequest {
    #[serde(flatten)]
    pub common: MediaCommon,
    pub prompt: String,
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub source_images: Vec<WireMediaSource>,
    #[serde(default)]
    pub mask: Option<WireMediaSource>,
    #[serde(default)]
    pub count: Option<u32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub size: Option<String>,
    #[serde(default)]
    pub negative_prompt: Option<String>,
    #[serde(default)]
    pub output_format: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VideoRequest {
    #[serde(flatten)]
    pub common: MediaCommon,
    pub prompt: String,
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub first_frame: Option<WireMediaSource>,
    #[serde(default)]
    pub last_frame: Option<WireMediaSource>,
    #[serde(default)]
    pub reference_images: Vec<WireMediaSource>,
    #[serde(default)]
    pub source_video: Option<WireMediaSource>,
    #[serde(default)]
    pub count: Option<u32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub duration_secs: Option<f64>,
    #[serde(default)]
    pub size: Option<String>,
    #[serde(default)]
    pub negative_prompt: Option<String>,
    #[serde(default)]
    pub output_format: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SpeechRequest {
    #[serde(flatten)]
    pub common: MediaCommon,
    /// Named `input` to match OpenAI's `/v1/audio/speech` body.
    pub input: String,
    pub voice: String,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub speed: Option<f32>,
    #[serde(default)]
    pub output: Option<AudioOutput>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioOutput {
    #[serde(default)]
    pub format: Option<String>,
    #[serde(default)]
    pub sample_rate_hz: Option<u32>,
    #[serde(default)]
    pub channels: Option<u16>,
}

impl AudioOutput {
    pub fn into_spec(self) -> Result<AudioOutputSpec, String> {
        let format = match self.format.as_deref() {
            None => AudioFormat::Mp3,
            Some(value) => parse_audio_format(value)?,
        };
        Ok(AudioOutputSpec {
            format,
            sample_rate_hz: self.sample_rate_hz,
            channels: self.channels,
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct TranscriptionRequestBody {
    #[serde(flatten)]
    pub common: MediaCommon,
    pub audio: WireMediaSource,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub timestamp_granularities: Vec<String>,
}

pub fn parse_image_mode(value: &str) -> Result<ImageGenerationMode, String> {
    match value {
        "generate" => Ok(ImageGenerationMode::Generate),
        "edit" => Ok(ImageGenerationMode::Edit),
        "inpaint" => Ok(ImageGenerationMode::Inpaint),
        "variation" => Ok(ImageGenerationMode::Variation),
        other => Err(format!(
            "invalid mode '{other}': expected generate, edit, inpaint or variation"
        )),
    }
}

pub fn parse_video_mode(value: &str) -> Result<VideoGenerationMode, String> {
    match value {
        "text_to_video" => Ok(VideoGenerationMode::TextToVideo),
        "image_to_video" => Ok(VideoGenerationMode::ImageToVideo),
        "reference_to_video" => Ok(VideoGenerationMode::ReferenceToVideo),
        "extend" => Ok(VideoGenerationMode::Extend),
        "edit" => Ok(VideoGenerationMode::Edit),
        other => Err(format!(
            "invalid mode '{other}': expected text_to_video, image_to_video, reference_to_video, extend or edit"
        )),
    }
}

pub fn parse_image_format(value: &str) -> Result<ImageFormat, String> {
    match value {
        "png" => Ok(ImageFormat::Png),
        "jpeg" | "jpg" => Ok(ImageFormat::Jpeg),
        "webp" => Ok(ImageFormat::Webp),
        "svg" => Ok(ImageFormat::Svg),
        other => Err(format!(
            "invalid output_format '{other}': expected png, jpeg, webp or svg"
        )),
    }
}

pub fn parse_video_format(value: &str) -> Result<VideoFormat, String> {
    match value {
        "mp4" => Ok(VideoFormat::Mp4),
        "webm" => Ok(VideoFormat::Webm),
        "mov" => Ok(VideoFormat::Mov),
        other => Err(format!(
            "invalid output_format '{other}': expected mp4, webm or mov"
        )),
    }
}

pub fn parse_audio_format(value: &str) -> Result<AudioFormat, String> {
    match value {
        "mp3" => Ok(AudioFormat::Mp3),
        "pcm" => Ok(AudioFormat::Pcm),
        "wav" => Ok(AudioFormat::Wav),
        "flac" => Ok(AudioFormat::Flac),
        "aac" => Ok(AudioFormat::Aac),
        "ogg" => Ok(AudioFormat::Ogg),
        "m4a" => Ok(AudioFormat::M4a),
        other => Err(format!(
            "invalid audio format '{other}': expected mp3, pcm, wav, flac, aac, ogg or m4a"
        )),
    }
}

pub fn parse_granularity(value: &str) -> Result<TimestampGranularity, String> {
    match value {
        "segment" => Ok(TimestampGranularity::Segment),
        "word" => Ok(TimestampGranularity::Word),
        other => Err(format!(
            "invalid timestamp granularity '{other}': expected segment or word"
        )),
    }
}

/// Reject a client-supplied `cost_estimate` in any provider namespace.
pub fn reject_reserved_options(options: &ProviderOptions) -> Result<(), String> {
    for (namespace, value) in options {
        if value.get(RESERVED_OPTION).is_some() {
            return Err(format!(
                "provider_options.{namespace}.{RESERVED_OPTION} is reserved; pricing is resolved server-side"
            ));
        }
    }
    Ok(())
}

/// Keep only the namespace belonging to `provider`.
///
/// octolib's adapters reject a foreign namespace outright, so dropping the
/// others is what lets one alias fan out across providers: a client sends every
/// namespace once and the winning candidate reads its own.
pub fn filter_options_for(options: &ProviderOptions, provider: &str) -> ProviderOptions {
    options
        .iter()
        .filter(|(namespace, _)| namespace.eq_ignore_ascii_case(provider))
        .map(|(namespace, value)| (namespace.clone(), value.clone()))
        .collect()
}

// ── Response ──

#[derive(Debug, Clone, Serialize)]
pub struct MediaResponse {
    pub id: String,
    pub object: &'static str,
    pub task: MediaTask,
    pub status: OperationStatus,
    pub model: String,
    pub provider: String,
    pub progress: Option<f32>,
    pub artifacts: Vec<WireArtifact>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    pub usage: Option<WireUsage>,
    pub warnings: Vec<ProviderWarning>,
    pub safety: SafetyReport,
    pub error: Option<serde_json::Value>,
    pub created_at: u64,
    pub completed_at: Option<u64>,
}

/// An artifact as returned to the client.
///
/// `MediaArtifact` is serialized by hand rather than passed through because
/// `ArtifactSource::Inline(Vec<u8>)` serializes as a JSON array of integers —
/// megabytes of `[137,80,78,...]` where the caller expects base64.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireArtifact {
    pub kind: MediaKind,
    pub media_type: String,
    pub source: WireArtifactSource,
    pub size_bytes: Option<u64>,
    pub dimensions: Option<serde_json::Value>,
    pub duration_secs: Option<f64>,
    pub frame_rate: Option<f32>,
    pub sample_rate_hz: Option<u32>,
    pub channels: Option<u16>,
    pub expires_at: Option<u64>,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum WireArtifactSource {
    InlineBase64(String),
    Url(String),
    ProviderFileId(String),
    ObjectStorageUri(String),
    /// octolib grew an `ArtifactSource` variant this wire format cannot spell.
    /// Named rather than coerced into one of the above, so a client sees an
    /// honest "cannot represent" instead of, say, an empty URL.
    Unsupported,
}

impl From<&MediaArtifact> for WireArtifact {
    fn from(artifact: &MediaArtifact) -> Self {
        Self {
            kind: artifact.kind,
            media_type: artifact.media_type.clone(),
            source: match &artifact.source {
                ArtifactSource::Inline(bytes) => {
                    WireArtifactSource::InlineBase64(BASE64.encode(bytes))
                }
                ArtifactSource::Url(url) => WireArtifactSource::Url(url.clone()),
                ArtifactSource::ProviderFileId(id) => {
                    WireArtifactSource::ProviderFileId(id.clone())
                }
                ArtifactSource::ObjectStorageUri(uri) => {
                    WireArtifactSource::ObjectStorageUri(uri.clone())
                }
                _ => WireArtifactSource::Unsupported,
            },
            size_bytes: artifact.size_bytes,
            dimensions: artifact
                .dimensions
                .map(|d| serde_json::json!({"width": d.width, "height": d.height})),
            duration_secs: artifact.duration_secs,
            frame_rate: artifact.frame_rate,
            sample_rate_hz: artifact.sample_rate_hz,
            channels: artifact.channels,
            expires_at: artifact.expires_at,
            metadata: artifact.metadata.clone(),
        }
    }
}

/// Where the billed number came from. A provider-reported amount and a locally
/// computed one are different claims and must never be conflated.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CostSource {
    Provider,
    Estimate,
    Unavailable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireUsage {
    #[serde(flatten)]
    pub usage: MediaUsage,
    /// `MediaUsage::best_available_cost()` — the number OctoHub bills and
    /// aggregates. `null` when nothing could price the request.
    pub cost: Option<f64>,
    pub cost_source: CostSource,
}

impl From<&MediaUsage> for WireUsage {
    fn from(usage: &MediaUsage) -> Self {
        let cost_source = if usage.provider_reported_cost.is_some() {
            CostSource::Provider
        } else if usage.estimated_cost.is_some() {
            CostSource::Estimate
        } else {
            CostSource::Unavailable
        };
        Self {
            usage: usage.clone(),
            cost: usage.best_available_cost(),
            cost_source,
        }
    }
}

/// Transcript payload — the one task whose result is not an artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireTranscript {
    pub text: String,
    pub language: Option<String>,
    pub duration_secs: Option<f64>,
    pub segments: Vec<TranscriptSegment>,
    pub words: Vec<TranscriptWord>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options(json: serde_json::Value) -> ProviderOptions {
        serde_json::from_value(json).expect("valid provider options")
    }

    #[test]
    fn geometry_accepts_both_wire_spellings() {
        assert_eq!(
            parse_geometry("1024x1024").unwrap(),
            OutputGeometry::Dimensions {
                width: 1024,
                height: 1024
            }
        );
        assert_eq!(
            parse_geometry("16:9").unwrap(),
            OutputGeometry::AspectRatio {
                width: 16,
                height: 9
            }
        );
        // Round-trips against octolib's own formatter.
        assert_eq!(parse_geometry("16:9").unwrap().as_api_string(), "16:9");
        assert_eq!(
            parse_geometry("1024x1024").unwrap().as_api_string(),
            "1024x1024"
        );
    }

    #[test]
    fn geometry_rejects_nonsense_rather_than_defaulting() {
        for bad in ["", "big", "1024", "0x512", "16:0", "-1x-1", "1024xabc"] {
            assert!(parse_geometry(bad).is_err(), "{bad} should be rejected");
        }
    }

    #[test]
    fn base64_source_decodes_and_enforces_the_byte_cap() {
        let payload = WireMediaSource::Base64 {
            data: BASE64.encode(b"hello"),
            media_type: "image/png".to_string(),
        };
        let source = payload.clone().into_source(1024).unwrap();
        assert!(matches!(
            source,
            MediaSource::Bytes { ref data, .. } if data == b"hello"
        ));
        // The cap is checked on DECODED bytes, before anything reaches a provider.
        assert!(payload.into_source(4).is_err());
    }

    #[test]
    fn invalid_base64_is_a_request_error() {
        let payload = WireMediaSource::Base64 {
            data: "not!valid!base64".to_string(),
            media_type: "image/png".to_string(),
        };
        assert!(payload.into_source(1024).is_err());
    }

    /// A client-supplied local path would read the server's filesystem, so those
    /// variants have no wire spelling at all and must fail to deserialize.
    #[test]
    fn local_and_provider_file_sources_have_no_wire_form() {
        for body in [
            r#"{"type":"file","path":"/etc/passwd"}"#,
            r#"{"type":"provider_file","id":"file-123"}"#,
            r#"{"type":"object_storage","uri":"s3://bucket/key"}"#,
            r#"{"type":"bytes","data":[1,2,3],"media_type":"image/png"}"#,
        ] {
            assert!(
                serde_json::from_str::<WireMediaSource>(body).is_err(),
                "{body} must not deserialize"
            );
        }
    }

    #[test]
    fn redaction_keeps_the_shape_but_drops_the_payload() {
        let payload = WireMediaSource::Base64 {
            data: BASE64.encode(vec![0_u8; 4096]),
            media_type: "image/png".to_string(),
        };
        let redacted = payload.redacted();
        assert_eq!(redacted["inline_media_omitted"], serde_json::json!(true));
        assert_eq!(redacted["media_type"], serde_json::json!("image/png"));
        assert!(redacted.get("data").is_none(), "payload must not be stored");

        // URLs are not secrets and stay legible for observability.
        let url = WireMediaSource::Url {
            url: "https://example.test/a.png".to_string(),
            media_type: None,
        };
        assert_eq!(
            url.redacted()["url"],
            serde_json::json!("https://example.test/a.png")
        );
    }

    /// One alias can span providers, so a client sends every namespace once and
    /// only the winning candidate's survives — octolib rejects foreign ones.
    #[test]
    fn options_are_filtered_to_the_winning_provider() {
        let all = options(serde_json::json!({
            "fal": {"input": {"steps": 28}},
            "replicate": {"input": {"steps": 4}},
        }));
        let filtered = filter_options_for(&all, "fal");
        assert_eq!(filtered.len(), 1);
        assert!(filtered.contains_key("fal"));

        assert!(filter_options_for(&all, "FAL").contains_key("fal"));
        assert!(filter_options_for(&all, "runway").is_empty());
    }

    #[test]
    fn a_client_supplied_cost_estimate_is_rejected() {
        let sneaky = options(serde_json::json!({
            "fal": {"cost_estimate": {"unit": "images", "usd_per_unit": 0.0}}
        }));
        assert!(reject_reserved_options(&sneaky).is_err());

        let fine = options(serde_json::json!({"fal": {"input": {"steps": 28}}}));
        assert!(reject_reserved_options(&fine).is_ok());
    }

    /// Inline bytes must never reach the wire as octolib serializes them — a JSON
    /// array of integers instead of base64.
    #[test]
    fn inline_artifacts_are_base64_not_byte_arrays() {
        let artifact = MediaArtifact {
            kind: MediaKind::Image,
            media_type: "image/png".to_string(),
            source: ArtifactSource::Inline(b"hello".to_vec()),
            size_bytes: Some(5),
            dimensions: None,
            duration_secs: None,
            frame_rate: None,
            sample_rate_hz: None,
            channels: None,
            expires_at: None,
            metadata: serde_json::Value::Null,
        };
        let json = serde_json::to_value(WireArtifact::from(&artifact)).unwrap();
        assert_eq!(json["source"]["type"], serde_json::json!("inline_base64"));
        assert_eq!(
            json["source"]["value"],
            serde_json::json!(BASE64.encode(b"hello"))
        );
    }

    #[test]
    fn cost_source_distinguishes_billed_from_computed() {
        let billed = MediaUsage {
            provider_reported_cost: Some(0.42),
            estimated_cost: Some(0.10),
            ..MediaUsage::default()
        };
        let wire = WireUsage::from(&billed);
        assert_eq!(wire.cost, Some(0.42));
        assert_eq!(wire.cost_source, CostSource::Provider);

        let computed = MediaUsage {
            estimated_cost: Some(0.10),
            ..MediaUsage::default()
        };
        let wire = WireUsage::from(&computed);
        assert_eq!(wire.cost, Some(0.10));
        assert_eq!(wire.cost_source, CostSource::Estimate);

        // Unpriced stores null, never zero — a free request and an unknown one are
        // different facts and the usage aggregate must not conflate them.
        let wire = WireUsage::from(&MediaUsage::default());
        assert_eq!(wire.cost, None);
        assert_eq!(wire.cost_source, CostSource::Unavailable);
    }

    #[test]
    fn wait_defaults_to_blocking() {
        let common: MediaCommon =
            serde_json::from_str(r#"{"model":"flux"}"#).expect("minimal body parses");
        assert!(common.wait);
        assert!(matches!(
            common.unsupported_parameters,
            UnsupportedParameters::Error
        ));
    }

    #[test]
    fn enum_parsers_reject_unknown_values() {
        assert!(parse_image_mode("generate").is_ok());
        assert!(parse_image_mode("upscale").is_err());
        assert!(parse_video_mode("image_to_video").is_ok());
        assert!(parse_video_mode("morph").is_err());
        assert!(parse_image_format("jpg").is_ok());
        assert!(parse_image_format("gif").is_err());
        assert!(parse_video_format("mp4").is_ok());
        assert!(parse_video_format("avi").is_err());
        assert!(parse_audio_format("flac").is_ok());
        assert!(parse_audio_format("midi").is_err());
        assert!(parse_granularity("word").is_ok());
        assert!(parse_granularity("phoneme").is_err());
    }
}
