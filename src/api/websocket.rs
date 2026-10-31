//! Buffered completion requests over an authenticated, sequential WebSocket.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use base64::Engine;
use bytes::Bytes;
use futures_util::{
	stream::{SplitSink, SplitStream},
	SinkExt, StreamExt,
};
use http_body_util::{BodyExt, Full};
use hyper::{body::Incoming, upgrade::Upgraded, Request, Response, StatusCode, Version};
use hyper_util::rt::TokioIo;
use tokio::io::AsyncWriteExt;
use tokio::time::timeout;
use tokio_tungstenite::{
	tungstenite::{
		handshake::server::create_response,
		protocol::{frame::coding::CloseCode, CloseFrame, Role, WebSocketConfig},
		Error, Message,
	},
	WebSocketStream,
};
use tracing::Instrument;

use super::handler::{
	attribution_headers, auth_header, completion_auth, completion_from_bytes, error_response, model_purpose,
	MAX_REQUEST_BODY_BYTES,
};
use crate::{proxy::engine::ProxyEngine, server::UpgradeTasks, storage::Storage, BoxBody};

const TRANSPORT_TIMEOUT: Duration = Duration::from_secs(10);
// Bound the transport's single buffered reply as well as incoming requests.
const MAX_REPLY_BYTES: usize = 64 * 1024 * 1024;
type Socket = WebSocketStream<TokioIo<Upgraded>>;
type Writer = SplitSink<Socket, Message>;
type Reader = SplitStream<Socket>;

/// What the session does after handling an incoming frame.
#[derive(PartialEq)]
enum Flow {
	Continue,
	Stop,
	/// Stop, but the rejected frame's payload is still arriving: drain it first.
	Drain,
}

pub async fn handle_upgrade(
	mut req: Request<Incoming>,
	engine: Arc<ProxyEngine>,
	storage: Arc<dyn Storage>,
	remote: String,
) -> Response<BoxBody> {
	let Some(tasks) = req.extensions().get::<UpgradeTasks>().cloned() else {
		return error_response(StatusCode::SERVICE_UNAVAILABLE, "WebSocket serving unavailable");
	};
	if tasks.is_draining() {
		return error_response(StatusCode::SERVICE_UNAVAILABLE, "Server is draining");
	}
	// This bearer endpoint is for machine clients, not ambient browser credentials.
	if req.headers().contains_key("origin") {
		return error_response(StatusCode::FORBIDDEN, "Browser Origin is not allowed");
	}
	let mut handshake = Request::new(());
	*handshake.method_mut() = req.method().clone();
	*handshake.version_mut() = req.version();
	*handshake.headers_mut() = req.headers().clone();
	let key_valid = req
		.headers()
		.get("sec-websocket-key")
		.and_then(|key| base64::engine::general_purpose::STANDARD.decode(key.as_bytes()).ok())
		.is_some_and(|key| key.len() == 16);
	let singleton_headers = ["sec-websocket-key", "sec-websocket-version", "upgrade"];
	if req.version() != Version::HTTP_11
		|| !key_valid
		|| singleton_headers
			.iter()
			.any(|name| req.headers().get_all(*name).iter().count() != 1)
		|| req.headers().contains_key("transfer-encoding")
		|| req.headers().get("content-length").is_some_and(|value| value != "0")
	{
		return error_response(StatusCode::BAD_REQUEST, "Malformed WebSocket upgrade");
	}
	let response = match create_response(&handshake) {
		Ok(response) => response.map(|()| Full::new(Bytes::new())),
		Err(_) => return error_response(StatusCode::BAD_REQUEST, "Malformed WebSocket upgrade"),
	};
	let authorization = auth_header(&req);
	match timeout(
		TRANSPORT_TIMEOUT,
		completion_auth(authorization.clone(), storage.clone()),
	)
	.await
	{
		Ok(Ok(_)) => {}
		Ok(Err(response)) => return *response,
		Err(_) => return error_response(StatusCode::GATEWAY_TIMEOUT, "WebSocket authentication timed out"),
	}
	let purpose = model_purpose(&req);
	let attribution = attribution_headers(&req);
	let upgrade = hyper::upgrade::on(&mut req);
	let shutdown = tasks.subscribe();
	let session = Session {
		authorization,
		purpose,
		attribution,
		engine,
		storage,
		remote,
	};
	if !tasks.spawn(async move {
		match timeout(TRANSPORT_TIMEOUT, upgrade).await {
			Ok(Ok(upgraded)) => {
				let config = WebSocketConfig::default()
					.read_buffer_size(16 * 1024)
					.write_buffer_size(0)
					.max_write_buffer_size(MAX_REPLY_BYTES + 1024)
					.max_message_size(Some(MAX_REQUEST_BODY_BYTES))
					.max_frame_size(Some(MAX_REQUEST_BODY_BYTES));
				let socket = Socket::from_raw_socket(TokioIo::new(upgraded), Role::Server, Some(config)).await;
				session.serve(socket, shutdown).await;
			}
			Ok(Err(error)) => tracing::debug!(%error, "WebSocket upgrade failed"),
			Err(_) => tracing::warn!("WebSocket upgrade timed out"),
		}
	}) {
		return error_response(StatusCode::SERVICE_UNAVAILABLE, "Server is draining");
	}
	response
}

struct Session {
	authorization: Option<String>,
	purpose: Option<String>,
	attribution: Option<HashMap<String, String>>,
	engine: Arc<ProxyEngine>,
	storage: Arc<dyn Storage>,
	remote: String,
}

impl Session {
	async fn serve(self, socket: Socket, shutdown: tokio::sync::watch::Receiver<bool>) {
		let (mut writer, mut reader) = socket.split();
		if self.exchange(&mut writer, &mut reader, shutdown).await == Flow::Drain {
			drain(writer, reader).await;
		}
	}

	async fn exchange(
		&self,
		writer: &mut Writer,
		reader: &mut Reader,
		mut shutdown: tokio::sync::watch::Receiver<bool>,
	) -> Flow {
		loop {
			if *shutdown.borrow() {
				close(writer, CloseCode::Away, "Server is draining").await;
				return Flow::Stop;
			}
			let incoming = tokio::select! {
				biased;
				_ = shutdown.changed() => continue,
				incoming = reader.next() => incoming,
			};
			let text = match incoming {
				Some(Ok(Message::Text(text))) => text,
				other => match control(other, writer).await {
					Flow::Continue => continue,
					flow => return flow,
				},
			};
			if *shutdown.borrow() {
				close(writer, CloseCode::Away, "Server is draining").await;
				return Flow::Stop;
			}
			// The request future is owned here: closing the socket drops it and its permits.
			let request = self.complete(text.as_bytes());
			tokio::pin!(request);
			let mut draining = false;
			let response = loop {
				tokio::select! {
					_ = shutdown.changed(), if !draining => { draining = true; }
					incoming = reader.next() => {
						if matches!(incoming, Some(Ok(Message::Text(_)))) {
							close(writer, CloseCode::Policy, "Wait for the completion reply before sending another request").await;
							return Flow::Stop;
						}
						match control(incoming, writer).await {
							Flow::Continue => {}
							flow => return flow,
						}
					}
					response = &mut request => break response,
				}
			};
			let message = match envelope(response).await {
				Ok(message) => message,
				Err(error) => {
					tracing::error!(%error, "WebSocket reply encoding failed");
					close(writer, CloseCode::Error, "Reply encoding failed").await;
					return Flow::Stop;
				}
			};
			if message.len() > MAX_REPLY_BYTES {
				close(writer, CloseCode::Size, "Completion reply exceeds 64 MiB").await;
				return Flow::Stop;
			}
			match deliver(message, writer, reader).await {
				Flow::Continue => {}
				flow => return flow,
			}
		}
	}

	async fn complete(&self, body: &[u8]) -> Response<BoxBody> {
		let request_id = ulid::Ulid::new().to_string();
		let span = tracing::info_span!(
			"request", req_id = %request_id, method = "POST", route = "/v1/completions",
			path = "/v1/completions", remote = %self.remote, transport = "websocket",
			status = tracing::field::Empty, dur_ms = tracing::field::Empty,
			api_key_id = tracing::field::Empty, model = tracing::field::Empty,
			provider = tracing::field::Empty, queued_ms = tracing::field::Empty,
			chain_ms = tracing::field::Empty, upstream_ms = tracing::field::Empty,
			store_ms = tracing::field::Empty, tok_in = tracing::field::Empty, tok_out = tracing::field::Empty,
		);
		let start = Instant::now();
		let _flight = crate::metrics::in_flight_guard("/v1/completions");
		let response = async {
			let api_key = match completion_auth(self.authorization.clone(), self.storage.clone()).await {
				Ok(key) => key,
				Err(response) => return *response,
			};
			completion_from_bytes(
				body,
				&api_key,
				self.purpose.clone(),
				self.attribution.clone(),
				self.engine.clone(),
			)
			.await
		}
		.instrument(span.clone())
		.await;
		let elapsed = start.elapsed();
		span.record("status", response.status().as_u16());
		span.record("dur_ms", elapsed.as_millis() as u64);
		crate::metrics::record_request("/v1/completions", "POST", response.status().as_u16(), elapsed);
		tracing::info!(parent: &span, "request completed");
		crate::attach_request_id(response, &request_id)
	}
}

// Keep enforcing sequencing and servicing control frames until the reply is flushed.
async fn deliver(message: Message, writer: &mut Writer, reader: &mut Reader) -> Flow {
	enum Delivery {
		Finished(bool),
		Incoming(Option<Result<Message, Error>>),
	}
	let result = {
		let send = writer.send(message);
		tokio::pin!(send);
		let deadline = tokio::time::sleep(TRANSPORT_TIMEOUT);
		tokio::pin!(deadline);
		loop {
			tokio::select! {
				biased;
				_ = &mut deadline => break Delivery::Finished(false),
				incoming = reader.next() => match incoming {
					// Subsequent reads flush Tungstenite's automatic Pong even while sending.
					Some(Ok(Message::Ping(_) | Message::Pong(_))) => continue,
					other => break Delivery::Incoming(other),
				},
				result = &mut send => break Delivery::Finished(result.is_ok()),
			}
		}
	};
	match result {
		Delivery::Finished(true) => Flow::Continue,
		Delivery::Finished(false) => Flow::Stop,
		Delivery::Incoming(Some(Ok(Message::Text(_)))) => {
			close(
				writer,
				CloseCode::Policy,
				"Wait for the completion reply before sending another request",
			)
			.await;
			Flow::Stop
		}
		Delivery::Incoming(other) => match control(other, writer).await {
			Flow::Drain => Flow::Drain,
			_ => Flow::Stop,
		},
	}
}

async fn envelope(response: Response<BoxBody>) -> anyhow::Result<Message> {
	let (parts, body) = response.into_parts();
	let headers = parts
		.headers
		.iter()
		.map(|(name, value)| Ok((name.as_str().to_string(), value.to_str()?.to_string())))
		.collect::<anyhow::Result<HashMap<_, _>>>()?;
	let body: serde_json::Value = serde_json::from_slice(&body.collect().await?.to_bytes())?;
	anyhow::ensure!(body.is_object(), "Completion response must be a JSON object");
	Ok(Message::Text(
		serde_json::to_string(&serde_json::json!({
			"status": parts.status.as_u16(), "headers": headers, "body": body,
		}))?
		.into(),
	))
}

async fn close(writer: &mut Writer, code: CloseCode, reason: &'static str) {
	let _ = timeout(
		TRANSPORT_TIMEOUT,
		writer.send(Message::Close(Some(CloseFrame {
			code,
			reason: reason.into(),
		}))),
	)
	.await;
}

/// Tungstenite queues native Pong/Close replies while reading; flush without a work queue.
async fn control(incoming: Option<Result<Message, Error>>, writer: &mut Writer) -> Flow {
	match incoming {
		Some(Ok(Message::Ping(_))) => match timeout(TRANSPORT_TIMEOUT, writer.flush()).await {
			Ok(Ok(())) => Flow::Continue,
			_ => Flow::Stop,
		},
		Some(Ok(Message::Pong(_))) => Flow::Continue,
		Some(Ok(Message::Close(_))) => {
			let _ = timeout(TRANSPORT_TIMEOUT, writer.flush()).await;
			Flow::Stop
		}
		Some(Ok(Message::Binary(_))) => {
			close(writer, CloseCode::Unsupported, "Only JSON text requests are supported").await;
			Flow::Stop
		}
		// Both reject a frame by its header, before its payload has been read.
		Some(Err(Error::Capacity(_))) => {
			close(writer, CloseCode::Size, "Request exceeds 16 MiB").await;
			Flow::Drain
		}
		Some(Err(Error::Protocol(_))) => {
			close(writer, CloseCode::Protocol, "Malformed WebSocket frame").await;
			Flow::Drain
		}
		_ => Flow::Stop,
	}
}

/// Dropping a socket with unread input makes the OS answer with a TCP reset, and a
/// peer that receives a reset (always on Windows) discards the Close frame we just
/// sent before its application reads it. Half-close and consume the rejected
/// payload until the peer hangs up, so the Close code reaches the client.
async fn drain(writer: Writer, reader: Reader) {
	let mut socket = reader.reunite(writer).expect("halves split from one socket");
	let io = socket.get_mut();
	let _ = timeout(TRANSPORT_TIMEOUT, async {
		io.shutdown().await?;
		tokio::io::copy(io, &mut tokio::io::sink()).await
	})
	.await;
}

#[cfg(test)]
#[path = "websocket_tests.rs"]
mod tests;
