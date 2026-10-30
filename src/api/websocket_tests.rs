//! Real HTTP/1.1/WebSocket wire regressions; no provider traffic leaves loopback.
//!
//! Upstream cases re-exec exactly one test in a child process before overriding
//! OLLAMA_API_URL. A module-local mutex would NOT isolate other modules' env reads.
//! The registry-locked octolib 0.47.0 ollama.rs reads that variable as the complete
//! chat-completions URL (openai_compat::get_api_url), not a base to append to.

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use bytes::Bytes;
use futures_util::{SinkExt, Stream, StreamExt};
use http_body_util::{BodyExt, Full};
use hyper::{HeaderMap, Request, Response, Version};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, oneshot};
use tokio::task::{JoinHandle, JoinSet};
use tokio::time::timeout;
use tokio_tungstenite::tungstenite::{
	client::IntoClientRequest,
	protocol::frame::{
		coding::{CloseCode, Data, OpCode},
		Frame,
	},
	Error as WsError, Message,
};
use tokio_tungstenite::{connect_async, connect_async_with_config, MaybeTlsStream, WebSocketStream};

use crate::auth::Credentials;
use crate::config::Config;
use crate::proxy::engine::ProxyEngine;
use crate::proxy::limiter::ProviderLimiter;
use crate::storage::sqlite::SqliteStorage;
use crate::storage::{ApiKey, ListFilter, Storage};

const IO_WAIT: Duration = Duration::from_secs(5);
const MAX_MESSAGE: usize = 16 * 1024 * 1024;
const CHILD_ENV: &str = "OCTOHUB_WEBSOCKET_TEST_CHILD";
type Client = WebSocketStream<MaybeTlsStream<TcpStream>>;

fn config() -> Config {
	// Deserialize directly: developer credentials/config/env must not enter tests.
	toml::from_str("").unwrap()
}

struct Hub {
	addr: SocketAddr,
	storage: Arc<SqliteStorage>,
	engine: Arc<ProxyEngine>,
	key: ApiKey,
	shutdown: Option<oneshot::Sender<Duration>>,
	server: JoinHandle<anyhow::Result<()>>,
}

impl Hub {
	async fn new(config: Config) -> Self {
		let storage = Arc::new(SqliteStorage::new(":memory:").unwrap());
		let key = storage
			.create_api_key("wire", None, Some("wire-owner"), Some(1))
			.unwrap();
		let credentials = Arc::new(Credentials::new(storage.clone(), None).unwrap());
		let limiter = Arc::new(RwLock::new(Arc::new(ProviderLimiter::from_config(&config))));
		let engine = Arc::new(ProxyEngine::new(
			storage.clone(),
			Arc::new(RwLock::new(Arc::new(config))),
			limiter,
			1,
			credentials,
		));
		Self::with_engine(storage, engine, key).await
	}

	async fn with_engine(storage: Arc<SqliteStorage>, engine: Arc<ProxyEngine>, key: ApiKey) -> Self {
		let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
		let addr = listener.local_addr().unwrap();
		let (shutdown, stopped) = oneshot::channel();
		let route_engine = engine.clone();
		let route_storage: Arc<dyn Storage> = storage.clone();
		let server = tokio::spawn(crate::server::serve(
			listener,
			move |req, remote| crate::route(req, route_engine.clone(), route_storage.clone(), remote),
			async move { stopped.await.unwrap_or(Duration::ZERO) },
		));
		Self {
			addr,
			storage,
			engine,
			key,
			shutdown: Some(shutdown),
			server,
		}
	}

	fn upgrade_request(&self, token: Option<&str>) -> Request<()> {
		let mut request = format!("ws://{}/v1/completions", self.addr)
			.into_client_request()
			.unwrap();
		assert_eq!(request.method(), hyper::Method::GET);
		assert_eq!(request.version(), Version::HTTP_11);
		if let Some(token) = token {
			request
				.headers_mut()
				.insert("authorization", format!("Bearer {token}").parse().unwrap());
		}
		request
	}

	async fn connect(&self) -> Client {
		self.connect_key(&self.key).await
	}

	async fn connect_key(&self, key: &ApiKey) -> Client {
		let (socket, response) = timeout(IO_WAIT, connect_async(self.upgrade_request(Some(&key.key))))
			.await
			.unwrap()
			.unwrap();
		assert_eq!(response.status(), 101);
		assert_eq!(response.version(), Version::HTTP_11);
		assert_eq!(response.headers()["upgrade"], "websocket");
		socket
	}

	fn begin_shutdown(&mut self, bound: Duration) {
		self.shutdown.take().unwrap().send(bound).unwrap();
	}

	async fn stopped(&mut self) -> anyhow::Result<()> {
		timeout(IO_WAIT, &mut self.server)
			.await
			.expect("listener did not finish draining")
			.unwrap()
	}

	async fn listener_closed(&self) {
		timeout(IO_WAIT, async {
			loop {
				match TcpStream::connect(self.addr).await {
					Err(_) => break,
					Ok(stream) => drop(stream),
				}
				tokio::task::yield_now().await;
			}
		})
		.await
		.expect("shutdown did not close the listener");
	}
}

impl Drop for Hub {
	fn drop(&mut self) {
		self.server.abort();
	}
}

#[derive(Debug, serde::Deserialize)]
struct Reply {
	status: u16,
	headers: BTreeMap<String, String>,
	body: Value,
}

impl Reply {
	fn error(&self, status: u16, discriminator: &str) {
		assert_eq!(self.status, status, "{self:?}");
		assert_eq!(self.body["error"]["type"], discriminator, "{self:?}");
		assert!(self.body["error"]["message"].as_str().is_some_and(|m| !m.is_empty()));
	}

	fn completion(&self, text: &str) {
		assert_eq!(self.status, 200, "{self:?}");
		assert_eq!(self.body["object"], "completion");
		assert_eq!(self.body["provider"], "ollama");
		assert!(self.body["id"].as_str().unwrap().starts_with("cmpl_"));
		assert_eq!(self.body["output"][0]["type"], "message");
		assert_eq!(self.body["output"][0]["role"], "assistant");
		assert_eq!(self.body["output"][0]["content"][0]["text"], text);
		assert_eq!(self.body["usage"]["input_tokens"], 3);
		assert_eq!(self.body["usage"]["output_tokens"], 2);
		assert_eq!(self.body["usage"]["total_tokens"], 5);
	}
}

async fn frame<S>(socket: &mut S) -> Message
where
	S: Stream<Item = Result<Message, WsError>> + Unpin,
{
	timeout(IO_WAIT, socket.next())
		.await
		.expect("no WebSocket frame before deadline")
		.expect("socket ended without a frame")
		.expect("invalid WebSocket frame")
}

async fn reply(socket: &mut Client) -> Reply {
	let Message::Text(text) = frame(socket).await else {
		panic!("expected a JSON text reply")
	};
	let value: Value = serde_json::from_str(&text).unwrap();
	assert_eq!(value.as_object().unwrap().len(), 3, "{value}");
	assert!(value["status"].is_u64(), "status must be numeric, not a string");
	assert!(
		value["body"].is_object(),
		"the existing JSON body must not be double-encoded"
	);
	for (name, value) in value["headers"].as_object().unwrap() {
		assert_eq!(name, &name.to_ascii_lowercase());
		assert!(value.is_string(), "header {name} must be a string");
	}
	let reply: Reply = serde_json::from_value(value).unwrap();
	assert_eq!(reply.headers["content-type"], "application/json");
	let id = &reply.headers["x-request-id"];
	assert_eq!(id.len(), 26);
	assert_eq!(ulid::Ulid::from_string(id).unwrap().to_string(), *id);
	reply
}

async fn send(socket: &mut Client, body: Value) {
	timeout(IO_WAIT, socket.send(Message::Text(body.to_string().into())))
		.await
		.unwrap()
		.unwrap();
}

fn request(input: &str) -> Value {
	json!({"model":"wire", "input":input})
}

async fn close_code<S>(socket: &mut S, expected: CloseCode)
where
	S: Stream<Item = Result<Message, WsError>> + Unpin,
{
	let message = frame(socket).await;
	let Message::Close(Some(close)) = message else {
		panic!("expected Close({expected:?}), got {message:?}")
	};
	assert_eq!(close.code, expected);
}

async fn rejected_handshake(hub: &Hub, token: Option<&str>) {
	let error = timeout(IO_WAIT, connect_async(hub.upgrade_request(token)))
		.await
		.unwrap()
		.unwrap_err();
	let WsError::Http(response) = error else {
		panic!("expected HTTP auth rejection, got {error:?}")
	};
	assert_eq!(response.status(), 401);
	assert_eq!(response.headers()["content-type"], "application/json");
	assert!(response.headers().contains_key("x-request-id"));
	let body: Value = serde_json::from_slice(response.body().as_ref().unwrap()).unwrap();
	assert_eq!(body["error"]["type"], "invalid_request_error");
	assert!(body["error"]["message"].is_string());
}

#[tokio::test]
async fn upgrade_without_authorization_returns_http_401() {
	let hub = Hub::new(config()).await;
	rejected_handshake(&hub, None).await;
}

#[tokio::test]
async fn upgrade_with_master_key_instead_of_client_key_returns_http_401() {
	let mut config = config();
	config.server.api_key = "not-a-client-key".into();
	let hub = Hub::new(config).await;
	rejected_handshake(&hub, Some("not-a-client-key")).await;
	rejected_handshake(&hub, Some("unknown-client-key")).await;
}

#[tokio::test]
async fn upgrade_with_revoked_client_key_returns_http_401() {
	let hub = Hub::new(config()).await;
	assert!(hub.storage.revoke_api_key(hub.key.id).unwrap());
	rejected_handshake(&hub, Some(&hub.key.key)).await;
}

#[tokio::test]
async fn authenticated_get_upgrades_http_11_and_handles_native_idle_ping() {
	let hub = Hub::new(config()).await;
	let mut socket = hub.connect().await;
	socket
		.send(Message::Ping(Bytes::from_static(b"idle-probe")))
		.await
		.unwrap();
	assert_eq!(
		frame(&mut socket).await,
		Message::Pong(Bytes::from_static(b"idle-probe"))
	);
}

#[tokio::test]
async fn malformed_json_returns_existing_error_and_socket_recovers() {
	let hub = Hub::new(config()).await;
	let mut socket = hub.connect().await;
	for invalid in ["", "{", "null", "[]", "{\"model\":1,\"input\":\"hi\"}"] {
		socket.send(Message::Text(invalid.into())).await.unwrap();
		let response = reply(&mut socket).await;
		response.error(400, "invalid_request_error");
		assert!(response.body["error"]["message"]
			.as_str()
			.unwrap()
			.contains("Invalid request JSON"));
	}
	send(&mut socket, json!({"model":"missing-alias", "input":"合法 🦀"})).await;
	let response = reply(&mut socket).await;
	response.error(400, "invalid_request_error");
	assert!(!response.body["error"]["message"]
		.as_str()
		.unwrap()
		.contains("Invalid request JSON"));
}

#[tokio::test]
async fn messages_get_distinct_ulids_instead_of_reusing_handshake_request_id() {
	let hub = Hub::new(config()).await;
	assert!(hub.storage.update_api_key_models(hub.key.id, Some(&[])).unwrap());
	let mut request = hub.upgrade_request(Some(&hub.key.key));
	request
		.headers_mut()
		.insert("x-request-id", "handshake-id".parse().unwrap());
	let (mut socket, handshake) = connect_async(request).await.unwrap();
	assert_eq!(handshake.headers()["x-request-id"], "handshake-id");
	send(&mut socket, request_body_forbidden()).await;
	let first = reply(&mut socket).await;
	first.error(403, "invalid_request_error");
	send(&mut socket, request_body_forbidden()).await;
	let second = reply(&mut socket).await;
	second.error(403, "invalid_request_error");
	assert_ne!(first.headers["x-request-id"], second.headers["x-request-id"]);
	assert_ne!(first.headers["x-request-id"], "handshake-id");
	assert_ne!(second.headers["x-request-id"], "handshake-id");
}

fn request_body_forbidden() -> Value {
	json!({"model":"missing-alias", "input":"hello"})
}

#[tokio::test]
async fn live_socket_reloads_model_edits_without_rotating_the_key() {
	let hub = Hub::new(config()).await;
	let mut socket = hub.connect().await;
	assert!(hub.storage.update_api_key_models(hub.key.id, Some(&[])).unwrap());
	send(&mut socket, request_body_forbidden()).await;
	let denied = reply(&mut socket).await;
	denied.error(403, "invalid_request_error");
	assert!(!denied.body.to_string().contains("model_not_allowed_for_key"));

	let allowed = ["missing-alias".to_string()];
	assert!(hub.storage.update_api_key_models(hub.key.id, Some(&allowed)).unwrap());
	send(&mut socket, request_body_forbidden()).await;
	reply(&mut socket).await.error(400, "invalid_request_error");
	send(&mut socket, json!({"model":"Missing-Alias", "input":"hello"})).await;
	reply(&mut socket).await.error(403, "invalid_request_error");
	assert!(hub.storage.update_api_key_models(hub.key.id, None).unwrap());
	send(&mut socket, json!({"model":"other-missing-alias", "input":"hello"})).await;
	reply(&mut socket).await.error(400, "invalid_request_error");
	assert_eq!(
		hub.storage.get_api_key_by_key(&hub.key.key).unwrap().unwrap().id,
		hub.key.id
	);
}

#[tokio::test]
async fn live_socket_reauthenticates_and_rejects_revoked_key_before_parsing() {
	let hub = Hub::new(config()).await;
	let mut socket = hub.connect().await;
	send(&mut socket, request_body_forbidden()).await;
	reply(&mut socket).await.error(400, "invalid_request_error");
	assert!(hub.storage.revoke_api_key(hub.key.id).unwrap());
	// Auth wins even over invalid JSON: no stale handshake ApiKey may be used.
	socket.send(Message::Text("{".into())).await.unwrap();
	let response = reply(&mut socket).await;
	response.error(401, "invalid_request_error");
	assert!(response.body["error"]["message"].as_str().unwrap().contains("revoked"));
	send(&mut socket, request_body_forbidden()).await;
	reply(&mut socket).await.error(401, "invalid_request_error");
}

#[tokio::test]
async fn unsupported_video_preserves_modality_discriminator_without_upstream_work() {
	let hub = Hub::new(config()).await;
	let mut socket = hub.connect().await;
	send(
		&mut socket,
		json!({
			"model":"ollama:llama3.2", "input":[{"type":"message", "role":"user", "content":[
				{"type":"input_video", "video_url":"data:video/mp4;base64,AAAA"}
			]}]
		}),
	)
	.await;
	let response = reply(&mut socket).await;
	response.error(400, "modality_not_supported");
	assert!(response.body["error"]["message"].as_str().unwrap().contains("video"));
}

#[tokio::test]
async fn binary_request_closes_with_1003_not_an_application_error() {
	let hub = Hub::new(config()).await;
	let mut socket = hub.connect().await;
	socket.send(Message::Binary(Bytes::from_static(b"{}"))).await.unwrap();
	close_code(&mut socket, CloseCode::Unsupported).await;
}

#[tokio::test]
async fn text_exactly_16_mib_is_accepted_and_socket_remains_usable() {
	let hub = Hub::new(config()).await;
	let mut socket = hub.connect().await;
	let prefix = "{\"model\":\"missing-alias\",\"input\":\"";
	let suffix = "\"}";
	let body = format!(
		"{prefix}{}{suffix}",
		"x".repeat(MAX_MESSAGE - prefix.len() - suffix.len())
	);
	assert_eq!(body.len(), MAX_MESSAGE);
	socket.send(Message::Text(body.into())).await.unwrap();
	reply(&mut socket).await.error(400, "invalid_request_error");
	send(&mut socket, request_body_forbidden()).await;
	reply(&mut socket).await.error(400, "invalid_request_error");
}

async fn oversized_frames_close_with_1009(hub: &Hub, frames: Vec<Message>) {
	let socket = hub.connect().await;
	let (mut writer, mut reader) = socket.split();
	// Rejection can occur on the length header, before the sender finishes the
	// payload. Drive reads concurrently and do not mistake that write error for
	// the required, observable Close(1009).
	let (_, ()) = tokio::join!(
		timeout(IO_WAIT, async {
			for message in frames {
				if writer.send(message).await.is_err() {
					break;
				}
			}
		}),
		close_code(&mut reader, CloseCode::Size),
	);
}

#[tokio::test]
async fn text_one_byte_over_16_mib_closes_with_1009() {
	let hub = Hub::new(config()).await;
	// UTF-8 character count is below the limit; transport limits count bytes.
	let body = format!("{}xx", "界".repeat(MAX_MESSAGE / 3));
	assert_eq!(body.len(), MAX_MESSAGE + 1);
	oversized_frames_close_with_1009(&hub, vec![Message::Text(body.into())]).await;
}

#[tokio::test]
async fn fragmented_text_over_16_mib_closes_with_1009_even_when_each_frame_fits() {
	let hub = Hub::new(config()).await;
	let first = Frame::message(vec![b' '; MAX_MESSAGE / 2], OpCode::Data(Data::Text), false);
	let last = Frame::message(vec![b' '; MAX_MESSAGE / 2 + 1], OpCode::Data(Data::Continue), true);
	oversized_frames_close_with_1009(&hub, vec![Message::Frame(first), Message::Frame(last)]).await;
}

// Returns true only in the isolated child. The parent waits for the real test
// result (including assertion failures); it cannot pass by selecting zero tests.
async fn isolated(test: &str) -> bool {
	let module = module_path!().split_once("::").unwrap().1;
	let name = format!("{module}::{test}");
	if std::env::var(CHILD_ENV).as_deref() == Ok(name.as_str()) {
		return true;
	}
	let mut child = tokio::process::Command::new(std::env::current_exe().unwrap());
	child
		.args(["--exact", &name, "--nocapture", "--test-threads=1"])
		.env(CHILD_ENV, &name)
		.env("OLLAMA_API_KEY", "loopback-test-key")
		.env("NO_PROXY", "127.0.0.1,localhost,::1")
		.env("no_proxy", "127.0.0.1,localhost,::1")
		.kill_on_drop(true);
	for proxy in [
		"HTTP_PROXY",
		"HTTPS_PROXY",
		"ALL_PROXY",
		"http_proxy",
		"https_proxy",
		"all_proxy",
	] {
		child.env_remove(proxy);
	}
	let output = timeout(Duration::from_secs(90), child.output())
		.await
		.expect("isolated test timed out")
		.unwrap();
	let stdout = String::from_utf8_lossy(&output.stdout);
	let stderr = String::from_utf8_lossy(&output.stderr);
	assert!(output.status.success(), "isolated {name} failed:\n{stdout}\n{stderr}");
	assert!(
		stdout.contains("running 1 test"),
		"child did not run exactly {name}:\n{stdout}\n{stderr}"
	);
	false
}

struct Call {
	path: String,
	headers: HeaderMap,
	body: Value,
	finish: oneshot::Sender<String>,
}

impl Call {
	fn answer(self, text: &str) {
		self.finish
			.send(text.to_string())
			.expect("upstream request was cancelled before its response");
	}
}

struct Upstream {
	calls: mpsc::UnboundedReceiver<Call>,
	server: JoinHandle<()>,
}

impl Upstream {
	async fn new() -> Self {
		assert!(
			std::env::var_os(CHILD_ENV).is_some(),
			"env override requires an isolated child"
		);
		let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
		let address = listener.local_addr().unwrap();
		// Safe from other tests' reads/writes: only this exact test runs here.
		std::env::set_var("OLLAMA_API_URL", format!("http://{address}/v1/chat/completions"));
		let (sent, calls) = mpsc::unbounded_channel();
		let server = tokio::spawn(async move {
			let mut connections = JoinSet::new();
			loop {
				tokio::select! {
					accepted = listener.accept() => {
						let (socket, _) = accepted.unwrap();
						let sent = sent.clone();
						connections.spawn(async move {
							let service = hyper::service::service_fn(move |req: Request<hyper::body::Incoming>| {
								let sent = sent.clone();
								async move {
									assert_eq!(req.method(), hyper::Method::POST);
									let (parts, body) = req.into_parts();
									let body: Value = serde_json::from_slice(&body.collect().await.unwrap().to_bytes()).unwrap();
									let model = body["model"].clone();
									let (finish, released) = oneshot::channel();
									sent.send(Call { path: parts.uri.path().into(), headers: parts.headers, body, finish }).unwrap_or_else(|_| panic!("upstream observer dropped"));
									let answer = released.await.unwrap_or_else(|_| "abandoned".into());
									let body = json!({
										"id":"loopback", "object":"chat.completion", "created":1_700_000_000,
										"model":model, "choices":[{"index":0, "message":{"role":"assistant", "content":answer}, "finish_reason":"stop"}],
										"usage":{"prompt_tokens":3, "completion_tokens":2, "total_tokens":5, "cost":0.01}
									});
									Ok::<_, std::convert::Infallible>(Response::builder()
										.header("content-type", "application/json")
										.body(Full::new(Bytes::from(body.to_string()))).unwrap())
								}
							});
							// Disconnects deliberately cancel requests in several cases.
							let _ = hyper::server::conn::http1::Builder::new()
								.serve_connection(hyper_util::rt::TokioIo::new(socket), service).await;
						});
					}
					_ = connections.join_next(), if !connections.is_empty() => {}
				}
			}
		});
		Self { calls, server }
	}

	async fn next(&mut self) -> Call {
		let call = timeout(IO_WAIT, self.calls.recv())
			.await
			.expect("request did not reach loopback upstream")
			.expect("loopback upstream stopped");
		assert_eq!(call.path, "/v1/chat/completions");
		call
	}

	fn no_extra_calls(&mut self) {
		assert!(matches!(self.calls.try_recv(), Err(mpsc::error::TryRecvError::Empty)));
	}
}

impl Drop for Upstream {
	fn drop(&mut self) {
		self.server.abort();
	}
}

fn upstream_config() -> Config {
	let mut config: Config =
		toml::from_str("[models]\nwire = [\"ollama:wire-model\"]\n[providers.ollama]\nconcurrency = 1\n").unwrap();
	config.server.owner_queue_timeout_secs = 0;
	config.server.provider_queue_timeout_secs = 2;
	config.server.upstream_timeout_secs = 30;
	config
}

async fn reusable_capacity(hub: &Hub, upstream: &mut Upstream) {
	// A real second request must pass BOTH budgets while the cancelled call's
	// upstream gate is still held. Merely observing a Close cannot prove this.
	let mut socket = hub.connect().await;
	send(&mut socket, request("after-cancellation")).await;
	let call = upstream.next().await;
	assert_eq!(call.body["messages"][0]["content"], "after-cancellation");
	call.answer("capacity-reused");
	reply(&mut socket).await.completion("capacity-reused");
	let rows = hub
		.storage
		.list_completions(&ListFilter {
			limit: 100,
			..ListFilter::default()
		})
		.unwrap();
	assert_eq!(rows.len(), 1, "cancelled work must not produce a stored completion");
	upstream.no_extra_calls();
}

#[tokio::test]
async fn sequential_successes_reuse_socket_and_preserve_completion_chain() {
	if !isolated("sequential_successes_reuse_socket_and_preserve_completion_chain").await {
		return;
	}
	let mut upstream = Upstream::new().await;
	let hub = Hub::new(upstream_config()).await;
	let mut socket = hub.connect().await;
	send(&mut socket, request("first 🦀")).await;
	let first_call = upstream.next().await;
	assert_eq!(
		first_call.body["messages"],
		json!([{"role":"user", "content":"first 🦀"}])
	);
	first_call.answer("one");
	let first = reply(&mut socket).await;
	first.completion("one");
	send(
		&mut socket,
		json!({"model":"wire", "input":"second", "previous_completion_id":first.body["id"]}),
	)
	.await;
	let second_call = upstream.next().await;
	assert_eq!(
		second_call.body["messages"],
		json!([
			{"role":"user", "content":"first 🦀"},
			{"role":"assistant", "content":"one"},
			{"role":"user", "content":"second"}
		])
	);
	second_call.answer("two");
	let second = reply(&mut socket).await;
	second.completion("two");
	assert_ne!(first.body["id"], second.body["id"]);
	assert_ne!(first.headers["x-request-id"], second.headers["x-request-id"]);
	let stored = hub
		.storage
		.get_completion(second.body["id"].as_str().unwrap())
		.unwrap()
		.unwrap();
	assert_eq!(stored.previous_completion_id.as_deref(), first.body["id"].as_str());
	assert_eq!(stored.api_key_id, hub.key.id);
	assert_eq!(
		hub.storage
			.list_completions(&ListFilter {
				limit: 100,
				..ListFilter::default()
			})
			.unwrap()
			.len(),
		2
	);
	upstream.no_extra_calls();
}

#[tokio::test]
async fn handshake_purpose_and_attribution_are_preserved_for_every_message() {
	if !isolated("handshake_purpose_and_attribution_are_preserved_for_every_message").await {
		return;
	}
	let mut upstream = Upstream::new().await;
	let mut config = upstream_config();
	config
		.models
		.insert("chosen".into(), vec!["ollama:chosen-model".into()]);
	config.auto.insert("default".into(), "wire".into());
	config.auto.insert("writer".into(), "chosen".into());
	let hub = Hub::new(config).await;
	let mut handshake = hub.upgrade_request(Some(&hub.key.key));
	for (name, value) in [
		("x-model-purpose", "writer-rewrite"),
		("x-title", "Wire test app"),
		("http-referer", "https://client.invalid/app"),
		("x-openrouter-categories", "testing"),
	] {
		handshake.headers_mut().insert(name, value.parse().unwrap());
	}
	let (mut socket, _) = connect_async(handshake).await.unwrap();
	for input in ["one", "two"] {
		send(&mut socket, json!({"model":"auto", "input":input})).await;
		let call = upstream.next().await;
		assert_eq!(call.body["model"], "chosen-model", "handshake purpose was lost");
		assert_eq!(call.headers["x-title"], "Wire test app");
		assert_eq!(call.headers["http-referer"], "https://client.invalid/app");
		assert_eq!(call.headers["x-openrouter-categories"], "testing");
		assert_ne!(
			call.headers.get("authorization").unwrap().to_str().unwrap(),
			format!("Bearer {}", hub.key.key)
		);
		call.answer(input);
		let response = reply(&mut socket).await;
		response.completion(input);
		assert_eq!(response.body["model"], "chosen-model");
	}
	upstream.no_extra_calls();
}

#[tokio::test]
async fn exhausted_provider_window_preserves_numeric_429_and_retry_after_header() {
	if !isolated("exhausted_provider_window_preserves_numeric_429_and_retry_after_header").await {
		return;
	}
	let mut upstream = Upstream::new().await;
	let mut config = upstream_config();
	config.providers.get_mut("ollama").unwrap().requests_per_day = Some(1);
	let hub = Hub::new(config).await;
	let mut socket = hub.connect().await;
	send(&mut socket, request("admitted")).await;
	upstream.next().await.answer("one");
	reply(&mut socket).await.completion("one");
	send(&mut socket, request("limited")).await;
	let response = reply(&mut socket).await;
	response.error(429, "invalid_request_error");
	let retry: u64 = response.headers["retry-after"].parse().unwrap();
	assert!((1..=86_400).contains(&retry));
	upstream.no_extra_calls();
}

#[tokio::test]
async fn native_ping_receives_pong_while_completion_is_pending() {
	if !isolated("native_ping_receives_pong_while_completion_is_pending").await {
		return;
	}
	let mut upstream = Upstream::new().await;
	let hub = Hub::new(upstream_config()).await;
	let mut socket = hub.connect().await;
	send(&mut socket, request("held")).await;
	let held = upstream.next().await;
	socket
		.send(Message::Ping(Bytes::from_static(b"busy-probe")))
		.await
		.unwrap();
	assert_eq!(
		frame(&mut socket).await,
		Message::Pong(Bytes::from_static(b"busy-probe"))
	);
	let mut same_owner = hub.connect().await;
	send(&mut same_owner, request("owner-contender")).await;
	reply(&mut same_owner).await.error(429, "invalid_request_error");
	let other_key = hub
		.storage
		.create_api_key("other", None, Some("other-owner"), Some(1))
		.unwrap();
	let mut other_owner = hub.connect_key(&other_key).await;
	send(&mut other_owner, request("provider-contender")).await;
	assert!(
		timeout(Duration::from_millis(50), upstream.calls.recv()).await.is_err(),
		"Ping released the active provider permit"
	);
	held.answer("finished-after-pong");
	reply(&mut socket).await.completion("finished-after-pong");
	let queued = upstream.next().await;
	assert_eq!(queued.body["messages"][0]["content"], "provider-contender");
	queued.answer("next-owner");
	reply(&mut other_owner).await.completion("next-owner");
	upstream.no_extra_calls();
}

#[tokio::test]
async fn pipelined_request_closes_with_1008_without_dispatching_a_backlog() {
	if !isolated("pipelined_request_closes_with_1008_without_dispatching_a_backlog").await {
		return;
	}
	let mut upstream = Upstream::new().await;
	let hub = Hub::new(upstream_config()).await;
	let mut socket = hub.connect().await;
	send(&mut socket, request("held")).await;
	let held = upstream.next().await;
	send(&mut socket, request("must-not-be-queued")).await;
	close_code(&mut socket, CloseCode::Policy).await;
	reusable_capacity(&hub, &mut upstream).await;
	drop(held);
}

#[tokio::test]
async fn close_while_pending_cancels_work_and_releases_owner_and_provider() {
	if !isolated("close_while_pending_cancels_work_and_releases_owner_and_provider").await {
		return;
	}
	let mut upstream = Upstream::new().await;
	let hub = Hub::new(upstream_config()).await;
	let mut socket = hub.connect().await;
	send(&mut socket, request("held")).await;
	let held = upstream.next().await;
	socket.send(Message::Close(None)).await.unwrap();
	assert!(matches!(frame(&mut socket).await, Message::Close(_)));
	reusable_capacity(&hub, &mut upstream).await;
	drop(held);
}

#[tokio::test]
async fn tcp_disconnect_while_pending_releases_owner_and_provider() {
	if !isolated("tcp_disconnect_while_pending_releases_owner_and_provider").await {
		return;
	}
	let mut upstream = Upstream::new().await;
	let hub = Hub::new(upstream_config()).await;
	let mut socket = hub.connect().await;
	send(&mut socket, request("held")).await;
	let held = upstream.next().await;
	drop(socket); // No WebSocket Close handshake.
			   // A new handshake is not itself proof that the old reader has observed EOF.
			   // Wait on the actual provider permit rather than racing the zero-wait owner.
	let limiter = hub.engine.limiter();
	let permit = timeout(IO_WAIT, limiter.acquire("ollama"))
		.await
		.expect("disconnect leaked provider permit");
	drop(permit);
	reusable_capacity(&hub, &mut upstream).await;
	drop(held);
}

#[tokio::test]
async fn binary_while_pending_closes_with_1003_and_releases_both_budgets() {
	if !isolated("binary_while_pending_closes_with_1003_and_releases_both_budgets").await {
		return;
	}
	let mut upstream = Upstream::new().await;
	let hub = Hub::new(upstream_config()).await;
	let mut socket = hub.connect().await;
	send(&mut socket, request("held")).await;
	let held = upstream.next().await;
	socket.send(Message::Binary(Bytes::from_static(b"{}"))).await.unwrap();
	close_code(&mut socket, CloseCode::Unsupported).await;
	reusable_capacity(&hub, &mut upstream).await;
	drop(held);
}

#[tokio::test]
async fn live_socket_uses_edited_owner_budget_and_can_leave_saturated_owner() {
	if !isolated("live_socket_uses_edited_owner_budget_and_can_leave_saturated_owner").await {
		return;
	}
	let mut upstream = Upstream::new().await;
	let mut config = upstream_config();
	config.providers.get_mut("ollama").unwrap().concurrency = Some(2);
	let hub = Hub::new(config).await;
	let mut socket = hub.connect().await; // Capture the original owner at handshake.
	let peer = hub
		.storage
		.create_api_key("peer", None, Some("busy-owner"), Some(1))
		.unwrap();
	let mut peer_socket = hub.connect_key(&peer).await;
	send(&mut peer_socket, request("peer-holds-owner")).await;
	let peer_call = upstream.next().await;
	assert!(hub
		.storage
		.update_api_key_owner(hub.key.id, Some("busy-owner"), Some(1))
		.unwrap());
	send(&mut socket, request("must-use-edited-owner")).await;
	let denied = reply(&mut socket).await;
	denied.error(429, "invalid_request_error");
	assert!(!denied.body.to_string().contains("owner_concurrency_exhausted"));
	upstream.no_extra_calls();
	assert!(hub
		.storage
		.update_api_key_owner(hub.key.id, Some("free-owner"), Some(1))
		.unwrap());
	send(&mut socket, request("new-owner")).await;
	let free = upstream.next().await;
	assert_eq!(free.body["messages"][0]["content"], "new-owner");
	free.answer("free");
	reply(&mut socket).await.completion("free");
	peer_call.answer("peer");
	reply(&mut peer_socket).await.completion("peer");
	assert_eq!(
		hub.storage.get_api_key_by_key(&hub.key.key).unwrap().unwrap().id,
		hub.key.id
	);
}

#[tokio::test]
async fn idle_websocket_is_closed_with_1001_and_drain_completes() {
	let mut hub = Hub::new(config()).await;
	let mut socket = hub.connect().await;
	hub.begin_shutdown(IO_WAIT);
	close_code(&mut socket, CloseCode::Away).await;
	hub.stopped().await.unwrap();
	assert!(TcpStream::connect(hub.addr).await.is_err());
}

#[tokio::test]
async fn busy_drain_finishes_pending_reply_then_closes_without_accepting_more() {
	if !isolated("busy_drain_finishes_pending_reply_then_closes_without_accepting_more").await {
		return;
	}
	let mut upstream = Upstream::new().await;
	let mut hub = Hub::new(upstream_config()).await;
	let mut socket = hub.connect().await;
	send(&mut socket, request("held-during-drain")).await;
	let held = upstream.next().await;
	hub.begin_shutdown(IO_WAIT);
	hub.listener_closed().await;
	assert!(
		timeout(Duration::from_millis(50), &mut hub.server).await.is_err(),
		"busy request was detached or cancelled instead of drained"
	);
	socket
		.send(Message::Ping(Bytes::from_static(b"draining-probe")))
		.await
		.unwrap();
	assert_eq!(
		frame(&mut socket).await,
		Message::Pong(Bytes::from_static(b"draining-probe"))
	);
	held.answer("drained-answer");
	reply(&mut socket).await.completion("drained-answer");
	close_code(&mut socket, CloseCode::Away).await;
	hub.stopped().await.unwrap();
	assert_eq!(
		hub.storage
			.list_completions(&ListFilter {
				limit: 100,
				..ListFilter::default()
			})
			.unwrap()
			.len(),
		1
	);
	upstream.no_extra_calls();
}

#[tokio::test]
async fn drain_bound_aborts_pending_upgrade_and_releases_owner_and_provider() {
	if !isolated("drain_bound_aborts_pending_upgrade_and_releases_owner_and_provider").await {
		return;
	}
	let mut upstream = Upstream::new().await;
	let mut hub = Hub::new(upstream_config()).await;
	let mut socket = hub.connect().await;
	send(&mut socket, request("never-released")).await;
	let held = upstream.next().await;
	hub.begin_shutdown(Duration::from_millis(100));
	let error = hub
		.stopped()
		.await
		.expect_err("pending upgrade escaped the shutdown bound");
	assert!(error.to_string().contains("still open"), "{error:#}");
	let terminal = timeout(IO_WAIT, socket.next())
		.await
		.expect("aborted upgrade left a live socket");
	assert!(
		matches!(terminal, None | Some(Err(_)) | Some(Ok(Message::Close(_)))),
		"unexpected reply after abort: {terminal:?}"
	);
	// Reuse the SAME engine and limiters on a fresh listener. A fresh engine
	// would conceal leaked permits by constructing new semaphores.
	let next = Hub::with_engine(hub.storage.clone(), hub.engine.clone(), hub.key.clone()).await;
	reusable_capacity(&next, &mut upstream).await;
	drop(held);
}

async fn post(hub: &Hub, body: &str, token: Option<&str>) -> Reply {
	let mut socket = TcpStream::connect(hub.addr).await.unwrap();
	let auth = token
		.map(|key| format!("Authorization: Bearer {key}\r\n"))
		.unwrap_or_default();
	let request = format!(
		"POST /v1/completions HTTP/1.1\r\nHost: {}\r\n{auth}Content-Type: application/json\r\nContent-Length: {}\r\nX-Request-Id: post-id\r\nConnection: close\r\n\r\n{body}",
		hub.addr, body.len(),
	);
	socket.write_all(request.as_bytes()).await.unwrap();
	let mut bytes = Vec::new();
	timeout(IO_WAIT, socket.read_to_end(&mut bytes)).await.unwrap().unwrap();
	let split = bytes.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
	let head = std::str::from_utf8(&bytes[..split]).unwrap();
	let mut lines = head.split("\r\n");
	let status = lines
		.next()
		.unwrap()
		.split_whitespace()
		.nth(1)
		.unwrap()
		.parse()
		.unwrap();
	let headers = lines
		.map(|line| {
			let (name, value) = line.split_once(':').unwrap();
			(name.to_ascii_lowercase(), value.trim().to_string())
		})
		.collect::<BTreeMap<_, _>>();
	assert_eq!(headers["content-type"], "application/json");
	assert_eq!(headers["x-request-id"], "post-id");
	let body: Value = serde_json::from_slice(&bytes[split + 4..]).unwrap();
	assert!(body.get("status").is_none(), "POST must not gain a WebSocket envelope");
	Reply { status, headers, body }
}

#[tokio::test]
async fn post_keeps_http_auth_and_invalid_json_errors_without_websocket_envelope() {
	let hub = Hub::new(config()).await;
	post(&hub, "{", None).await.error(401, "invalid_request_error");
	post(&hub, "{", Some(&hub.key.key))
		.await
		.error(400, "invalid_request_error");
}

#[tokio::test]
async fn post_success_keeps_existing_json_response_and_persistence() {
	if !isolated("post_success_keeps_existing_json_response_and_persistence").await {
		return;
	}
	let mut upstream = Upstream::new().await;
	let hub = Hub::new(upstream_config()).await;
	let body = request("ordinary-post").to_string();
	let response = post(&hub, &body, Some(&hub.key.key));
	let release = async {
		let call = upstream.next().await;
		assert_eq!(call.body["messages"][0]["content"], "ordinary-post");
		call.answer("post-answer");
	};
	let (response, ()) = tokio::join!(response, release);
	response.completion("post-answer");
	let stored = hub
		.storage
		.get_completion(response.body["id"].as_str().unwrap())
		.unwrap()
		.unwrap();
	assert_eq!(stored.api_key_id, hub.key.id);
	assert_eq!(stored.input, json!("ordinary-post"));
	upstream.no_extra_calls();
}

// Send semantically malformed upgrades as raw HTTP: a WebSocket client may
// reject these before sending them, which would not exercise OctoHub's parser.
async fn raw_upgrade(hub: &Hub, websocket_headers: &str, token: Option<&str>) -> Reply {
	let mut socket = TcpStream::connect(hub.addr).await.unwrap();
	let auth = token
		.map(|key| format!("Authorization: Bearer {key}\r\n"))
		.unwrap_or_default();
	let request = format!(
		"GET /v1/completions HTTP/1.1\r\nHost: {}\r\nConnection: Upgrade, close\r\n{websocket_headers}{auth}X-Request-Id: parser-id\r\n\r\n",
		hub.addr,
	);
	timeout(IO_WAIT, socket.write_all(request.as_bytes()))
		.await
		.unwrap()
		.unwrap();
	let mut bytes = Vec::new();
	timeout(IO_WAIT, socket.read_to_end(&mut bytes)).await.unwrap().unwrap();
	let split = bytes.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
	let head = std::str::from_utf8(&bytes[..split]).unwrap();
	let mut lines = head.split("\r\n");
	let status = lines
		.next()
		.unwrap()
		.split_whitespace()
		.nth(1)
		.unwrap()
		.parse()
		.unwrap();
	let headers = lines
		.map(|line| {
			let (name, value) = line.split_once(':').unwrap();
			(name.to_ascii_lowercase(), value.trim().to_string())
		})
		.collect::<BTreeMap<_, _>>();
	assert_eq!(headers["content-type"], "application/json");
	assert_eq!(headers["x-request-id"], "parser-id");
	let body = serde_json::from_slice(&bytes[split + 4..]).unwrap();
	Reply { status, headers, body }
}

#[tokio::test]
async fn malformed_upgrade_returns_400_before_missing_bad_or_valid_authentication() {
	let hub = Hub::new(config()).await;
	let headers = "Upgrade: websocket\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Key: not-base64\r\n";
	for token in [None, Some("unknown-client-key"), Some(hub.key.key.as_str())] {
		let response = raw_upgrade(&hub, headers, token).await;
		response.error(400, "invalid_request_error");
		assert_eq!(response.body["error"]["message"], "Malformed WebSocket upgrade");
	}
	// With the same missing auth and a well-formed handshake, auth is now the
	// rejecting boundary, not the upgrade parser.
	rejected_handshake(&hub, None).await;
}

#[tokio::test]
async fn malformed_upgrade_fields_are_rejected_before_authentication() {
	let hub = Hub::new(config()).await;
	let key = "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n";
	let valid = format!("Upgrade: websocket\r\nSec-WebSocket-Version: 13\r\n{key}");
	let cases = [
		// Base64 decoding alone is insufficient: the nonce must be 16 bytes.
		"Upgrade: websocket\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Key: YQ==\r\n".to_string(),
		"Upgrade: websocket\r\nSec-WebSocket-Version: 13\r\n".to_string(),
		format!("Upgrade: websocket\r\n{key}"),
		format!("Upgrade: websocket\r\nSec-WebSocket-Version: 12\r\n{key}"),
		format!("Sec-WebSocket-Version: 13\r\n{key}"),
		format!("Upgrade: not-websocket\r\nSec-WebSocket-Version: 13\r\n{key}"),
		format!("{valid}{key}"),
		format!("{valid}Sec-WebSocket-Version: 13\r\n"),
		format!("{valid}Upgrade: websocket\r\n"),
	];
	for headers in cases {
		let response = raw_upgrade(&hub, &headers, None).await;
		response.error(400, "invalid_request_error");
		assert_eq!(response.body["error"]["message"], "Malformed WebSocket upgrade");
	}
}

#[tokio::test]
async fn reply_near_64_mib_is_delivered_but_larger_envelope_closes_with_1009() {
	if !isolated("reply_near_64_mib_is_delivered_but_larger_envelope_closes_with_1009").await {
		return;
	}
	assert_eq!(super::MAX_REPLY_BYTES, 64 * 1024 * 1024);
	let mut upstream = Upstream::new().await;
	let hub = Hub::new(upstream_config()).await;
	// Lift the CLIENT's defaults so a server regression that transmits an
	// oversized reply is observed as Text, not mistaken for a server Close.
	let client_limit = super::MAX_REPLY_BYTES + 1024 * 1024;
	let client_config = tokio_tungstenite::tungstenite::protocol::WebSocketConfig::default()
		.max_frame_size(Some(client_limit))
		.max_message_size(Some(client_limit));
	let (mut socket, _) = timeout(
		IO_WAIT,
		connect_async_with_config(hub.upgrade_request(Some(&hub.key.key)), Some(client_config), false),
	)
	.await
	.unwrap()
	.unwrap();
	let large_wait = Duration::from_secs(45);

	send(&mut socket, request("near-reply-limit")).await;
	let call = upstream.next().await;
	// Leave room for the completion/envelope metadata, all still on the wire.
	let accepted_payload_len = super::MAX_REPLY_BYTES - 4096;
	call.finish.send("x".repeat(accepted_payload_len)).unwrap();
	{
		let message = timeout(large_wait, socket.next()).await.unwrap().unwrap().unwrap();
		let Message::Text(text) = message else {
			panic!("reply below 64 MiB was not delivered as text");
		};
		assert!(text.len() > accepted_payload_len);
		assert!(text.len() <= super::MAX_REPLY_BYTES);
		let response: Reply = serde_json::from_str(&text).unwrap();
		assert_eq!(response.status, 200);
		assert_eq!(response.headers["content-type"], "application/json");
		assert!(ulid::Ulid::from_string(&response.headers["x-request-id"]).is_ok());
		let output = response.body["output"][0]["content"][0]["text"].as_str().unwrap();
		assert_eq!(output.len(), accepted_payload_len);
		assert!(output.bytes().all(|byte| byte == b'x'));
	}

	send(&mut socket, request("over-reply-limit")).await;
	let call = upstream.next().await;
	// The body text alone fills 64 MiB; the serialized completion/envelope
	// necessarily exceeds it. The cap applies to the whole serialized reply.
	call.finish.send("y".repeat(super::MAX_REPLY_BYTES)).unwrap();
	let message = timeout(large_wait, socket.next()).await.unwrap().unwrap().unwrap();
	let Message::Close(Some(close)) = message else {
		panic!("reply above 64 MiB was not rejected with a server Close");
	};
	assert_eq!(close.code, CloseCode::Size);
	assert_eq!(close.reason.as_str(), "Completion reply exceeds 64 MiB");

	// Rejecting delivery must not strand either admission budget.
	let mut next_socket = hub.connect().await;
	send(&mut next_socket, request("after-oversized-reply")).await;
	let call = upstream.next().await;
	assert_eq!(call.body["messages"][0]["content"], "after-oversized-reply");
	call.answer("still-serving");
	reply(&mut next_socket).await.completion("still-serving");
	upstream.no_extra_calls();
}

// Focused transport regressions complement the loopback route tests above.
// A bounded duplex transport makes a pending final flush deterministic without
// OS socket-buffer assumptions, sleeps, large completions, or production hooks.
struct DeliveryWire {
	socket: WebSocketStream<tokio::io::DuplexStream>,
	delivery: JoinHandle<(bool, Duration)>,
}

impl Drop for DeliveryWire {
	fn drop(&mut self) {
		self.delivery.abort();
	}
}

async fn pending_delivery() -> DeliveryWire {
	let (mut client, server) = tokio::io::duplex(64);
	let delivery = tokio::spawn(async move {
		let (upgrades, mut received) = mpsc::unbounded_channel();
		let service = hyper::service::service_fn(move |mut req: Request<hyper::body::Incoming>| {
			let upgrade = hyper::upgrade::on(&mut req);
			let response = super::create_response(&req.map(|_| ()))
				.unwrap()
				.map(|()| Full::new(Bytes::new()));
			upgrades.send(upgrade).unwrap();
			std::future::ready(Ok::<_, std::convert::Infallible>(response))
		});
		let connection = hyper::server::conn::http1::Builder::new()
			.serve_connection(hyper_util::rt::TokioIo::new(server), service)
			.with_upgrades();
		let (_, upgraded) = tokio::join!(async { connection.await.unwrap() }, async {
			received.recv().await.unwrap().await.unwrap()
		},);
		let config = tokio_tungstenite::tungstenite::protocol::WebSocketConfig::default().write_buffer_size(0);
		let socket = super::Socket::from_raw_socket(
			hyper_util::rt::TokioIo::new(upgraded),
			tokio_tungstenite::tungstenite::protocol::Role::Server,
			Some(config),
		)
		.await;
		let (mut writer, mut reader) = socket.split();
		let start = tokio::time::Instant::now();
		let delivered = super::deliver(Message::Text("d".repeat(4096).into()), &mut writer, &mut reader).await;
		(delivered, start.elapsed())
	});
	let request = b"GET /v1/completions HTTP/1.1\r\nHost: loopback\r\nConnection: Upgrade\r\nUpgrade: websocket\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n";
	timeout(IO_WAIT, client.write_all(request)).await.unwrap().unwrap();
	let mut headers = Vec::new();
	timeout(IO_WAIT, async {
		while !headers.ends_with(b"\r\n\r\n") {
			headers.push(client.read_u8().await.unwrap());
			assert!(headers.len() < 4096, "upgrade response headers did not end");
		}
	})
	.await
	.unwrap();
	assert!(headers.starts_with(b"HTTP/1.1 101 "));
	// Reading only the frame prefix proves deliver() has started sending. Its
	// 4096-byte payload cannot fit into the remaining 64-byte transport buffer.
	let mut prefix = vec![0; 2];
	timeout(IO_WAIT, client.read_exact(&mut prefix)).await.unwrap().unwrap();
	assert_eq!(prefix, [0x81, 126]);
	assert!(
		!delivery.is_finished(),
		"reply unexpectedly fit into the bounded transport"
	);
	let socket = WebSocketStream::from_partially_read(
		client,
		prefix,
		tokio_tungstenite::tungstenite::protocol::Role::Client,
		None,
	)
	.await;
	DeliveryWire { socket, delivery }
}

#[tokio::test]
async fn final_reply_flush_keeps_reading_native_ping_and_pong() {
	let mut wire = pending_delivery().await;
	// A masked 125-byte control frame itself exceeds the 64-byte incoming
	// buffer. send() cannot finish unless deliver() continues polling reads.
	let ping = Bytes::from(vec![b'p'; 125]);
	timeout(IO_WAIT, wire.socket.send(Message::Ping(ping.clone())))
		.await
		.unwrap()
		.unwrap();
	timeout(IO_WAIT, wire.socket.send(Message::Pong(Bytes::from(vec![b'q'; 125]))))
		.await
		.unwrap()
		.unwrap();
	let Message::Text(text) = frame(&mut wire.socket).await else {
		panic!("native control interrupted the buffered completion reply");
	};
	assert_eq!(text.as_str(), "d".repeat(4096));
	assert_eq!(frame(&mut wire.socket).await, Message::Pong(ping));
	let (delivered, _) = timeout(IO_WAIT, &mut wire.delivery).await.unwrap().unwrap();
	assert!(delivered, "native controls must not terminate final reply delivery");
}

#[tokio::test]
async fn second_text_during_final_reply_flush_closes_with_1008() {
	let mut wire = pending_delivery().await;
	let second = json!({"model":"wire", "input":"must-not-be-queued".repeat(8)});
	timeout(IO_WAIT, wire.socket.send(Message::Text(second.to_string().into())))
		.await
		.unwrap()
		.unwrap();
	// The first text frame may already be queued ahead of the Close. It must
	// remain intact; the next request must never receive an application reply.
	let message = frame(&mut wire.socket).await;
	let close = match message {
		Message::Text(text) => {
			assert_eq!(text.as_str(), "d".repeat(4096));
			frame(&mut wire.socket).await
		}
		other => other,
	};
	let Message::Close(Some(close)) = close else {
		panic!("pipelined text during final flush did not receive a Close");
	};
	assert_eq!(close.code, CloseCode::Policy);
	assert_eq!(
		close.reason.as_str(),
		"Wait for the completion reply before sending another request"
	);
	let (delivered, _) = timeout(IO_WAIT, &mut wire.delivery).await.unwrap().unwrap();
	assert!(!delivered, "a pipelined request must terminate the delivery session");
}

#[tokio::test]
async fn final_reply_flush_has_absolute_10_second_deadline_despite_control_traffic() {
	assert_eq!(super::TRANSPORT_TIMEOUT, Duration::from_secs(10));
	let mut wire = pending_delivery().await;
	let controls = async {
		loop {
			if wire
				.socket
				.send(Message::Pong(Bytes::from(vec![b'q'; 125])))
				.await
				.is_err()
			{
				// Peer closure races the task-result wakeup at the deadline.
				// Leave that result, not the traffic generator, to decide the test.
				std::future::pending::<()>().await;
			}
			tokio::time::sleep(Duration::from_millis(250)).await;
		}
	};
	// Never consume the reply. Keep supplying controls so a timeout restarted
	// for each read would not expire; the original absolute deadline must win.
	let (delivered, elapsed) = timeout(Duration::from_secs(13), async {
		tokio::select! {
			result = &mut wire.delivery => result.unwrap(),
			_ = controls => unreachable!("control traffic stopped without a deadline"),
		}
	})
	.await
	.expect("control traffic extended the absolute 10-second delivery deadline");
	assert!(!delivered, "a permanently backpressured reply cannot be delivered");
	assert!(
		elapsed >= Duration::from_millis(9990),
		"delivery timed out too early: {elapsed:?}"
	);
	assert!(
		elapsed < Duration::from_secs(13),
		"delivery deadline was extended: {elapsed:?}"
	);
}

#[tokio::test]
async fn non_upgrade_http_11_forces_connection_close_and_does_not_reuse_connection() {
	let hub = Hub::new(config()).await;
	// HTTP/1.1 defaults to persistence even without a Connection header. Test
	// both that default and an explicit keep-alive request; never ask to close.
	for connection in ["", "Connection: keep-alive\r\n"] {
		let mut socket = TcpStream::connect(hub.addr).await.unwrap();
		let first = format!(
			"GET /health HTTP/1.1\r\nHost: {}\r\n{connection}X-Request-Id: ordinary-first\r\n\r\n",
			hub.addr,
		);
		let second = format!(
			"GET /health HTTP/1.1\r\nHost: {}\r\n{connection}X-Request-Id: ordinary-second\r\n\r\n",
			hub.addr,
		);
		// Queue a valid second request on the SAME connection before the server
		// closes it. Exactly one complete response may come back, then EOF/reset.
		let requests = first + &second;
		timeout(IO_WAIT, socket.write_all(requests.as_bytes()))
			.await
			.unwrap()
			.unwrap();
		let mut bytes = Vec::new();
		let terminal = timeout(IO_WAIT, socket.read_to_end(&mut bytes))
			.await
			.expect("ordinary HTTP/1.1 connection remained open for persistent reuse");
		if let Err(error) = terminal {
			// Some platforms reset a connection closed with pipelined input
			// unread. The first response must still be complete below.
			assert_eq!(error.kind(), std::io::ErrorKind::ConnectionReset);
		}
		let split = bytes.windows(4).position(|window| window == b"\r\n\r\n").unwrap();
		let head = std::str::from_utf8(&bytes[..split]).unwrap();
		let mut lines = head.split("\r\n");
		assert_eq!(lines.next().unwrap(), "HTTP/1.1 200 OK");
		let headers = lines
			.map(|line| {
				let (name, value) = line.split_once(':').unwrap();
				(name.to_ascii_lowercase(), value.trim().to_string())
			})
			.collect::<BTreeMap<_, _>>();
		assert_eq!(headers["connection"], "close");
		assert_eq!(headers["x-request-id"], "ordinary-first");
		assert_eq!(headers["content-type"], "application/json");
		let body = &bytes[split + 4..];
		let content_length: usize = headers["content-length"].parse().unwrap();
		assert_eq!(
			body.len(),
			content_length,
			"connection served a second response or truncated the first"
		);
		let body: Value = serde_json::from_slice(body).unwrap();
		assert!(body.is_object());
	}
}

#[tokio::test]
async fn authenticated_upgrade_with_any_origin_returns_http_403() {
	let hub = Hub::new(config()).await;
	for origin in ["https://browser.example.invalid", "null", ""] {
		let mut request = hub.upgrade_request(Some(&hub.key.key));
		request.headers_mut().insert("origin", origin.parse().unwrap());
		let error = timeout(IO_WAIT, connect_async(request)).await.unwrap().unwrap_err();
		let WsError::Http(response) = error else {
			panic!("expected HTTP Origin rejection, got {error:?}");
		};
		assert_eq!(response.status(), 403, "Origin {origin:?} was not rejected");
		assert_eq!(response.headers()["content-type"], "application/json");
		assert!(response.headers().contains_key("x-request-id"));
		let body: Value = serde_json::from_slice(response.body().as_ref().unwrap()).unwrap();
		assert_eq!(body["error"]["type"], "invalid_request_error");
		assert_eq!(body["error"]["message"], "Browser Origin is not allowed");
	}
}

async fn malformed_frame_closes_with_1002(raw_frame: &[u8]) {
	let hub = Hub::new(config()).await;
	let mut socket = hub.connect().await;
	// Bypass the client encoder: it would otherwise mask/validate frames and
	// prevent deliberately malformed bytes from reaching the real server.
	timeout(IO_WAIT, socket.get_mut().write_all(raw_frame))
		.await
		.unwrap()
		.unwrap();
	let Message::Close(Some(close)) = frame(&mut socket).await else {
		panic!("malformed frame did not receive a protocol Close");
	};
	assert_eq!(close.code, CloseCode::Protocol);
	assert_eq!(close.reason.as_str(), "Malformed WebSocket frame");
}

#[tokio::test]
async fn unmasked_text_frame_closes_with_1002() {
	// FIN + Text, two payload bytes, MASK unset. Client frames MUST be masked.
	malformed_frame_closes_with_1002(&[0x81, 0x02, b'{', b'}']).await;
}

#[tokio::test]
async fn masked_text_frame_with_reserved_bit_closes_with_1002() {
	// RSV1 set without a negotiated extension; otherwise masked, empty Text.
	malformed_frame_closes_with_1002(&[0xc1, 0x80, 1, 2, 3, 4]).await;
}

#[tokio::test]
async fn masked_frame_with_reserved_opcode_closes_with_1002() {
	// Opcode 3 is reserved; the empty payload and masking are otherwise valid.
	malformed_frame_closes_with_1002(&[0x83, 0x80, 1, 2, 3, 4]).await;
}

#[tokio::test]
async fn masked_continuation_without_initial_fragment_closes_with_1002() {
	malformed_frame_closes_with_1002(&[0x80, 0x80, 1, 2, 3, 4]).await;
}

#[tokio::test]
async fn masked_fragmented_ping_closes_with_1002() {
	// Control frames may not be fragmented: FIN is unset on this masked Ping.
	malformed_frame_closes_with_1002(&[0x09, 0x80, 1, 2, 3, 4]).await;
}
