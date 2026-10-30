//! The client listener: binding, the accept loop, and the graceful drain on
//! SIGTERM/SIGINT.

use std::future::Future;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use hyper::body::Incoming;
use hyper::service::service_fn;
use hyper::{header, Request, Response, StatusCode, Version};
use hyper_util::rt::{TokioIo, TokioTimer};
use hyper_util::server::conn::auto;
use hyper_util::server::graceful::GracefulShutdown;
use tokio::net::{TcpListener, TcpSocket};
use tokio::sync::{watch, Notify};
use tokio::task::JoinSet;

use crate::BoxBody;

/// The backlog `tokio::net::TcpListener::bind` uses.
const LISTEN_BACKLOG: u32 = 128;
// Bounds every HTTP/1 request's headers, not its body or handler duration.
const HEADER_READ_TIMEOUT: Duration = Duration::from_secs(10);

/// Upgraded connections outlive Hyper's connection future, so they need their
/// own drain signal and task ownership.
#[derive(Clone)]
pub(crate) struct UpgradeTasks {
	tasks: Arc<Mutex<JoinSet<()>>>,
	draining: watch::Sender<bool>,
	changed: Arc<Notify>,
}

impl UpgradeTasks {
	fn new() -> Self {
		Self {
			tasks: Arc::new(Mutex::new(JoinSet::new())),
			draining: watch::channel(false).0,
			changed: Arc::new(Notify::new()),
		}
	}

	pub fn is_draining(&self) -> bool {
		*self.draining.borrow()
	}

	pub fn subscribe(&self) -> watch::Receiver<bool> {
		self.draining.subscribe()
	}

	/// Admission and the drain transition share the task lock: an upgrade
	/// cannot slip into the registry after shutdown has started.
	pub fn spawn(&self, task: impl Future<Output = ()> + Send + 'static) -> bool {
		let mut tasks = self.tasks.lock().unwrap();
		if self.is_draining() {
			return false;
		}
		tasks.spawn(task);
		self.changed.notify_one();
		true
	}

	fn begin_drain(&self) {
		let _tasks = self.tasks.lock().unwrap();
		self.draining.send_replace(true);
	}

	fn len(&self) -> usize {
		self.tasks.lock().unwrap().len()
	}

	fn abort_all(&self) {
		self.tasks.lock().unwrap().abort_all();
	}

	fn close(&self) {
		let tasks = {
			let mut tasks = self.tasks.lock().unwrap();
			self.draining.send_replace(true);
			std::mem::take(&mut *tasks)
		};
		// Drop outside the lock; JoinSet aborts all tasks and releases handles.
		drop(tasks);
	}

	async fn wait(&self) {
		// Lock only for a poll, never across an await: handlers must be able to
		// register another upgrade while existing tasks are still running.
		while let Some(result) = std::future::poll_fn(|cx| self.tasks.lock().unwrap().poll_join_next(cx)).await {
			if let Err(err) = result {
				if !err.is_cancelled() {
					tracing::error!(error = %err, "connection task failed");
				}
			}
		}
	}

	async fn reap(&self) {
		loop {
			self.wait().await;
			// One reaper per registry: notify_one stores a permit if spawn races
			// the empty check above, so this wait cannot miss that admission.
			self.changed.notified().await;
		}
	}
}

/// Task futures can hold registry clones themselves. A serve-owned guard breaks
/// that ownership cycle even if serve is cancelled before its normal drain.
struct ServeTasks {
	http: UpgradeTasks,
	upgrades: UpgradeTasks,
}

impl Drop for ServeTasks {
	fn drop(&mut self) {
		self.http.close();
		self.upgrades.close();
	}
}

/// Hyper executes HTTP/2 streams separately from the connection future. Keep
/// those futures owned too, rather than detaching them through TokioExecutor.
#[derive(Clone)]
struct HttpExecutor(UpgradeTasks);

impl<F> hyper::rt::Executor<F> for HttpExecutor
where
	F: Future + Send + 'static,
	F::Output: Send + 'static,
{
	fn execute(&self, future: F) {
		self.0.spawn(async move {
			future.await;
		});
	}
}

/// Protocol detection happens before Hyper's HTTP/1 header timer. Stop this
/// acceptance deadline at service entry, not when the response finishes.
async fn with_header_deadline<F: Future>(
	connection: F,
	received: &Notify,
	deadline: tokio::time::Instant,
) -> Option<F::Output> {
	let mut connection = std::pin::pin!(connection);
	tokio::select! {
		biased;
		_ = received.notified() => Some(connection.await),
		_ = tokio::time::sleep_until(deadline) => None,
		result = &mut connection => Some(result),
	}
}

/// Bind the client listener. `SO_REUSEPORT` lets another process bind the
/// same address while this one serves — a sibling deploy slot, or a load
/// balancer taking the port over — so a handover never leaves it unbound.
pub fn bind(addr: SocketAddr) -> std::io::Result<TcpListener> {
	let socket = if addr.is_ipv4() {
		TcpSocket::new_v4()?
	} else {
		TcpSocket::new_v6()?
	};
	// What `TcpListener::bind` sets on Unix: rebind at once while the previous
	// process's connections sit in TIME_WAIT. Never on Windows, where it lets
	// a socket hijack a port in use.
	#[cfg(unix)]
	socket.set_reuseaddr(true)?;
	#[cfg(unix)]
	socket.set_reuseport(true)?;
	socket.bind(addr)?;
	socket.listen(LISTEN_BACKLOG)
}

/// Resolves on the first SIGTERM or SIGINT (Ctrl-C on Windows). The handlers
/// are installed here, before serving, so an early signal is not lost — and
/// they must exist at all: octohub runs as PID 1 in its container, where the
/// kernel drops a signal that has no handler.
pub fn shutdown_signal() -> std::io::Result<impl Future<Output = ()>> {
	#[cfg(unix)]
	{
		use tokio::signal::unix::{signal, SignalKind};
		let mut term = signal(SignalKind::terminate())?;
		let mut int = signal(SignalKind::interrupt())?;
		Ok(async move {
			let name = tokio::select! {
				_ = term.recv() => "SIGTERM",
				_ = int.recv() => "SIGINT",
			};
			tracing::info!(signal = name, "shutdown requested");
		})
	}
	#[cfg(windows)]
	{
		let mut ctrl_c = tokio::signal::windows::ctrl_c()?;
		Ok(async move {
			ctrl_c.recv().await;
			tracing::info!(signal = "ctrl-c", "shutdown requested");
		})
	}
}

/// Serve `listener` until `shutdown` resolves, then drain. The listener is
/// closed at once, so new connections are refused here (or go to whoever
/// else holds the port); a connection serving a request finishes it and
/// closes; an idle one closes immediately. `shutdown` resolves to the drain
/// bound: connections still open after it are an error.
pub async fn serve<H, F>(
	listener: TcpListener,
	handler: H,
	shutdown: impl Future<Output = Duration>,
) -> anyhow::Result<()>
where
	H: Fn(Request<Incoming>, SocketAddr) -> F + Clone + Send + Sync + 'static,
	F: Future<Output = Response<BoxBody>> + Send + 'static,
{
	let graceful = GracefulShutdown::new();
	let upgrades = UpgradeTasks::new();
	let connections = UpgradeTasks::new();
	let _tasks = ServeTasks {
		http: connections.clone(),
		upgrades: upgrades.clone(),
	};
	// Every connection task holds a clone, so the strong count less this one
	// is the number of connections still open.
	let open = Arc::new(());
	let mut shutdown = std::pin::pin!(shutdown);

	let drain_timeout = loop {
		let accepted = tokio::select! {
			drain_timeout = &mut shutdown => break drain_timeout,
			_ = connections.reap() => unreachable!("HTTP reaper runs until shutdown"),
			_ = upgrades.reap() => unreachable!("upgrade reaper runs until shutdown"),
			accepted = listener.accept() => accepted,
		};
		// A failed accept (EMFILE, ECONNABORTED, ...) must not kill the server —
		// log, back off briefly, and keep accepting.
		let (stream, remote_addr) = match accepted {
			Ok(conn) => conn,
			Err(e) => {
				tracing::warn!(error = %e, "accept failed");
				tokio::time::sleep(Duration::from_millis(100)).await;
				continue;
			}
		};
		let header_deadline = tokio::time::Instant::now() + HEADER_READ_TIMEOUT;
		let io = TokioIo::new(stream);
		let handler = handler.clone();
		// Taken here, not in the task: a shutdown signalled before the task
		// first runs must still reach this connection.
		let watcher = graceful.watcher();
		let open = open.clone();
		let request_upgrades = upgrades.clone();
		let executor = HttpExecutor(connections.clone());

		connections.spawn(async move {
			let _open = open;
			let headers_received = Arc::new(Notify::new());
			let received = headers_received.clone();
			let service = service_fn(move |mut req: Request<Incoming>| {
				received.notify_one();
				req.extensions_mut().insert(request_upgrades.clone());
				let http1 = req.version() < Version::HTTP_2;
				let response = handler(req, remote_addr);
				async move {
					let mut response = response.await;
					// Disabling Hyper keep-alive also alters 101 upgrade headers.
					if http1 && response.status() != StatusCode::SWITCHING_PROTOCOLS {
						response
							.headers_mut()
							.insert(header::CONNECTION, header::HeaderValue::from_static("close"));
					}
					Ok::<_, hyper::Error>(response)
				}
			});

			// auto::Builder negotiates HTTP/1 or HTTP/2 from what the client speaks
			// (HTTP/2 prior-knowledge for plaintext, ALPN for TLS — TLS terminated
			// upstream). Ordinary HTTP/1 responses explicitly close above so clients
			// cannot reuse stale connections after long LLM gaps; upgrades stay open.
			let mut builder = auto::Builder::new(executor);
			builder
				.http1()
				.keep_alive(true)
				.timer(TokioTimer::new())
				.header_read_timeout(HEADER_READ_TIMEOUT);
			let connection = watcher.watch(builder.serve_connection_with_upgrades(io, service));
			match with_header_deadline(connection, &headers_received, header_deadline).await {
				Some(Ok(())) => {}
				Some(Err(err)) if is_idle_close(err.as_ref()) => {
					tracing::debug!(remote = %remote_addr, "idle connection closed for shutdown");
				}
				Some(Err(err)) => tracing::error!(remote = %remote_addr, error = %err, "connection error"),
				None => tracing::debug!(remote = %remote_addr, "connection header deadline expired"),
			}
		});
	};

	// Close the listening socket before draining: the port stops queueing
	// connections for this process now, not when the last request ends.
	drop(listener);
	upgrades.begin_drain();
	tracing::info!(
		connections = Arc::strong_count(&open) - 1,
		upgrades = upgrades.len(),
		timeout_secs = drain_timeout.as_secs(),
		"listener closed; draining open connections"
	);
	let drain = async {
		tokio::join!(graceful.shutdown(), connections.wait(), upgrades.wait());
	};
	if tokio::time::timeout(drain_timeout, drain).await.is_err() {
		let still_open = Arc::strong_count(&open) - 1 + upgrades.len();
		tracing::error!(connections = still_open, "drain timed out with connections still open");
		connections.begin_drain();
		connections.abort_all();
		upgrades.abort_all();
		// Join cancellation too: dropping JoinHandles would detach the work.
		tokio::join!(connections.wait(), upgrades.wait());
		anyhow::bail!("{still_open} connection(s) still open after the {drain_timeout:?} drain");
	}
	tracing::info!("drained; exiting");
	Ok(())
}

/// True for the error a connection ends with when shutdown closes it before
/// it sent a single request: hyper-util cancels the protocol sniff and yields
/// a bare `io::Error` of kind `Interrupted` ("Cancelled"). Expected during a
/// drain, unlike every other connection error, which hyper wraps in its own
/// error type.
fn is_idle_close(err: &(dyn std::error::Error + 'static)) -> bool {
	err.downcast_ref::<std::io::Error>()
		.is_some_and(|e| e.kind() == std::io::ErrorKind::Interrupted)
}

#[cfg(test)]
mod tests {
	use super::*;
	use bytes::Bytes;
	use http_body_util::Full;
	use hyper_util::rt::TokioExecutor;
	use std::time::Instant;
	use tokio::io::{AsyncReadExt, AsyncWriteExt};
	use tokio::net::TcpStream;
	use tokio::sync::{oneshot, Notify};

	const REQUEST: &[u8] = b"GET /slow HTTP/1.1\r\nHost: octohub\r\n\r\n";

	#[tokio::test]
	async fn http_executor_owns_and_aborts_hyper_tasks() {
		let tasks = UpgradeTasks::new();
		let dropped = Arc::new(Notify::new());
		let guard = DropSignal(dropped.clone());
		let executor = HttpExecutor(tasks.clone());
		hyper::rt::Executor::execute(&executor, async move {
			let _guard = guard;
			std::future::pending::<()>().await;
		});
		assert_eq!(tasks.len(), 1);
		tasks.begin_drain();
		tasks.abort_all();
		tasks.wait().await;
		dropped.notified().await;
		assert_eq!(tasks.len(), 0);
	}

	#[tokio::test]
	async fn admission_between_empty_check_and_notify_wait_is_not_lost() {
		let tasks = UpgradeTasks::new();
		tasks.wait().await;
		assert!(tasks.spawn(async {}));
		tokio::time::timeout(Duration::from_secs(2), tasks.changed.notified())
			.await
			.expect("admission must leave a notification permit even without a waiter");
		tasks.wait().await;
		assert_eq!(tasks.len(), 0);
	}

	#[tokio::test]
	async fn owner_drop_breaks_task_registry_cycles_and_rejects_new_tasks() {
		let tasks = UpgradeTasks::new();
		let dropped = Arc::new(Notify::new());
		let guard = DropSignal(dropped.clone());
		let task_registry = tasks.clone();
		assert!(tasks.spawn(async move {
			let _guard = guard;
			let _registry = task_registry;
			std::future::pending::<()>().await;
		}));
		let owner = ServeTasks {
			http: UpgradeTasks::new(),
			upgrades: tasks.clone(),
		};
		drop(owner);
		assert!(tasks.is_draining());
		assert_eq!(tasks.len(), 0, "owner drop must release every retained handle");
		assert!(!tasks.spawn(async {}));
		tokio::time::timeout(Duration::from_secs(2), dropped.notified())
			.await
			.expect("owner drop must abort a task even when it holds its own registry");
	}

	#[tokio::test]
	async fn cancelling_serve_aborts_busy_upgrades() {
		let started = Arc::new(Notify::new());
		let dropped = Arc::new(Notify::new());
		let (addr, _shutdown, server) = start(upgrade_handler(
			started.clone(),
			Some(Arc::new(Notify::new())),
			dropped.clone(),
		));
		let mut peer = upgrade_connection(addr).await;
		started.notified().await;
		server.abort();
		assert!(server.await.unwrap_err().is_cancelled());
		tokio::time::timeout(Duration::from_secs(2), dropped.notified())
			.await
			.expect("cancelling serve must not leave an upgraded task detached");
		assert_eq!(peer.read(&mut [0]).await.unwrap(), 0);
	}

	async fn assert_header_deadline_closes(client: &[u8]) {
		let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
		let mut peer = TcpStream::connect(listener.local_addr().unwrap()).await.unwrap();
		peer.write_all(client).await.unwrap();
		let (stream, _) = listener.accept().await.unwrap();
		let deadline = tokio::time::Instant::now() + Duration::from_millis(50);
		let headers_received = Arc::new(Notify::new());
		let received = headers_received.clone();
		let service = service_fn(move |_req: Request<Incoming>| {
			received.notify_one();
			async { Ok::<_, hyper::Error>(Response::new(Full::new(Bytes::new()))) }
		});
		let mut builder = auto::Builder::new(TokioExecutor::new());
		builder
			.http1()
			.keep_alive(false)
			.timer(TokioTimer::new())
			.header_read_timeout(HEADER_READ_TIMEOUT);
		let connection = builder.serve_connection_with_upgrades(TokioIo::new(stream), service);
		let result = tokio::time::timeout(
			Duration::from_secs(2),
			with_header_deadline(connection, &headers_received, deadline),
		)
		.await
		.unwrap();
		assert!(
			result.is_none(),
			"incomplete headers must exhaust the acceptance deadline"
		);
		assert_eq!(peer.read(&mut [0]).await.unwrap(), 0);
	}

	#[tokio::test]
	async fn idle_protocol_detection_times_out() {
		assert_header_deadline_closes(b"").await;
	}

	#[tokio::test]
	async fn matching_partial_http2_preface_times_out() {
		assert_header_deadline_closes(b"PRI * HTTP/2.0\r\n\r\nSM\r\n").await;
	}

	#[tokio::test]
	async fn incomplete_http1_headers_time_out() {
		assert_header_deadline_closes(b"GET /slow HTTP/1.1\r\nHost: octohub\r\n").await;
	}

	#[tokio::test]
	async fn header_deadline_does_not_limit_the_response() {
		let received = Notify::new();
		received.notify_one();
		let response = with_header_deadline(
			async { tokio::task::yield_now().await },
			&received,
			tokio::time::Instant::now(),
		)
		.await;
		assert_eq!(response, Some(()));
	}

	struct DropSignal(Arc<Notify>);

	impl Drop for DropSignal {
		fn drop(&mut self) {
			self.0.notify_one();
		}
	}

	fn upgrade_handler(
		started: Arc<Notify>,
		release: Option<Arc<Notify>>,
		dropped: Arc<Notify>,
	) -> impl Fn(Request<Incoming>, SocketAddr) -> std::future::Ready<Response<BoxBody>> + Clone {
		move |mut req, _remote| {
			let tasks = req
				.extensions()
				.get::<UpgradeTasks>()
				.expect("upgrade registry")
				.clone();
			assert!(!tasks.is_draining());
			let mut draining = tasks.subscribe();
			let upgrade = hyper::upgrade::on(&mut req);
			let started = started.clone();
			let release = release.clone();
			let dropped = DropSignal(dropped.clone());
			assert!(tasks.spawn(async move {
				let _dropped = dropped;
				let _io = upgrade.await.expect("HTTP upgrade must succeed");
				started.notify_one();
				draining.changed().await.unwrap();
				assert!(*draining.borrow());
				started.notify_one();
				if let Some(release) = release {
					release.notified().await;
				}
			}));
			std::future::ready(
				Response::builder()
					.status(101)
					.header("Connection", "upgrade")
					.header("Upgrade", "test")
					.body(Full::new(Bytes::new()))
					.unwrap(),
			)
		}
	}

	async fn upgrade_connection(addr: SocketAddr) -> TcpStream {
		let mut peer = TcpStream::connect(addr).await.unwrap();
		peer.write_all(b"GET /upgrade HTTP/1.1\r\nHost: octohub\r\nConnection: upgrade\r\nUpgrade: test\r\n\r\n")
			.await
			.unwrap();
		let mut response = Vec::new();
		tokio::time::timeout(Duration::from_secs(2), async {
			while !response.ends_with(b"\r\n\r\n") {
				let mut byte = [0];
				peer.read_exact(&mut byte).await.unwrap();
				response.push(byte[0]);
				assert!(response.len() < 8192);
			}
		})
		.await
		.unwrap();
		assert!(response.starts_with(b"HTTP/1.1 101 "), "{response:?}");
		peer
	}

	#[tokio::test]
	async fn finished_upgrade_tasks_are_reaped_without_new_requests() {
		let tasks = UpgradeTasks::new();
		let reaper_tasks = tasks.clone();
		let reaper = tokio::spawn(async move { reaper_tasks.reap().await });
		for _ in 0..64 {
			assert!(tasks.spawn(async {}));
		}
		tokio::time::timeout(Duration::from_secs(2), async {
			while tasks.len() != 0 {
				tokio::task::yield_now().await;
			}
		})
		.await
		.unwrap();
		reaper.abort();
		assert!(reaper.await.unwrap_err().is_cancelled());
	}

	#[tokio::test]
	async fn draining_rejects_and_drops_new_upgrade_tasks() {
		let tasks = UpgradeTasks::new();
		let mut draining = tasks.subscribe();
		tasks.begin_drain();
		draining.changed().await.unwrap();
		assert!(tasks.is_draining());
		assert!(*tasks.subscribe().borrow(), "late subscribers must see the drain");
		let dropped = Arc::new(Notify::new());
		let guard = DropSignal(dropped.clone());
		assert!(!tasks.spawn(async move {
			let _guard = guard;
			panic!("a rejected task must never run");
		}));
		dropped.notified().await;
		assert_eq!(tasks.len(), 0);
	}

	#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
	async fn upgrade_spawn_racing_drain_is_either_owned_or_rejected() {
		for _ in 0..64 {
			let tasks = UpgradeTasks::new();
			let barrier = Arc::new(tokio::sync::Barrier::new(3));
			let spawning_tasks = tasks.clone();
			let spawning_barrier = barrier.clone();
			let spawning = tokio::spawn(async move {
				spawning_barrier.wait().await;
				spawning_tasks.spawn(std::future::pending())
			});
			let draining_tasks = tasks.clone();
			let draining_barrier = barrier.clone();
			let draining = tokio::spawn(async move {
				draining_barrier.wait().await;
				draining_tasks.begin_drain();
			});
			barrier.wait().await;
			let accepted = spawning.await.unwrap();
			draining.await.unwrap();
			assert!(tasks.is_draining());
			assert_eq!(tasks.len(), usize::from(accepted));
			assert!(!tasks.spawn(async {}));
			tasks.abort_all();
			tasks.wait().await;
			assert_eq!(tasks.len(), 0);
		}
	}

	#[tokio::test]
	async fn drain_closes_idle_upgrades() {
		let started = Arc::new(Notify::new());
		let dropped = Arc::new(Notify::new());
		let (addr, shutdown, server) = start(upgrade_handler(started.clone(), None, dropped.clone()));
		let mut peer = upgrade_connection(addr).await;
		started.notified().await;
		shutdown.send(Duration::from_secs(60)).unwrap();
		tokio::time::timeout(Duration::from_secs(2), server)
			.await
			.unwrap()
			.unwrap()
			.unwrap();
		dropped.notified().await;
		assert_eq!(peer.read(&mut [0]).await.unwrap(), 0);
	}

	#[tokio::test]
	async fn drain_waits_for_busy_upgrades_to_finish() {
		let started = Arc::new(Notify::new());
		let release = Arc::new(Notify::new());
		let dropped = Arc::new(Notify::new());
		let (addr, shutdown, server) = start(upgrade_handler(started.clone(), Some(release.clone()), dropped));
		let mut peer = upgrade_connection(addr).await;
		started.notified().await;
		shutdown.send(Duration::from_secs(60)).unwrap();
		started.notified().await;
		assert!(!server.is_finished(), "busy upgrade must stay owned during the drain");
		release.notify_one();
		tokio::time::timeout(Duration::from_secs(2), server)
			.await
			.unwrap()
			.unwrap()
			.unwrap();
		assert_eq!(peer.read(&mut [0]).await.unwrap(), 0);
	}

	#[tokio::test]
	async fn drain_bound_aborts_upgrades_and_reports_them() {
		let started = Arc::new(Notify::new());
		let dropped = Arc::new(Notify::new());
		let (addr, shutdown, server) = start(upgrade_handler(
			started.clone(),
			Some(Arc::new(Notify::new())),
			dropped.clone(),
		));
		let mut peer = upgrade_connection(addr).await;
		started.notified().await;
		shutdown.send(Duration::from_millis(200)).unwrap();
		let err = tokio::time::timeout(Duration::from_secs(2), server)
			.await
			.unwrap()
			.unwrap()
			.expect_err("busy upgrade must not outlive the drain bound");
		assert_eq!(err.to_string(), "1 connection(s) still open after the 200ms drain");
		dropped.notified().await;
		assert_eq!(peer.read(&mut [0]).await.unwrap(), 0);
	}

	/// A handler that signals `started`, then answers `done` once `release` is
	/// notified: the test, not a timer, decides when the request ends.
	fn gated_handler(
		started: Arc<Notify>,
		release: Arc<Notify>,
	) -> impl Fn(Request<Incoming>, SocketAddr) -> std::pin::Pin<Box<dyn Future<Output = Response<BoxBody>> + Send>>
	       + Clone
	       + Send
	       + Sync
	       + 'static {
		move |_req, _remote| {
			let started = started.clone();
			let release = release.clone();
			Box::pin(async move {
				started.notify_one();
				release.notified().await;
				Response::new(Full::new(Bytes::from_static(b"done")))
			})
		}
	}

	/// Start `serve` on an ephemeral port; the returned sender triggers
	/// shutdown with the given drain bound.
	fn start<H, F>(
		handler: H,
	) -> (
		SocketAddr,
		oneshot::Sender<Duration>,
		tokio::task::JoinHandle<anyhow::Result<()>>,
	)
	where
		H: Fn(Request<Incoming>, SocketAddr) -> F + Clone + Send + Sync + 'static,
		F: Future<Output = Response<BoxBody>> + Send + 'static,
	{
		let listener = bind("127.0.0.1:0".parse().unwrap()).unwrap();
		let addr = listener.local_addr().unwrap();
		let (tx, rx) = oneshot::channel();
		let server = tokio::spawn(serve(listener, handler, async move { rx.await.unwrap() }));
		(addr, tx, server)
	}

	/// Wait until connecting to `addr` is refused, which proves the listener
	/// is closed (a connect that still succeeds raced the close; retry).
	#[cfg(unix)]
	async fn wait_refused(addr: SocketAddr, within: Duration) {
		let deadline = Instant::now() + within;
		loop {
			match TcpStream::connect(addr).await {
				Err(e) if e.kind() == std::io::ErrorKind::ConnectionRefused => return,
				// Linux resets a connect that was queued in the backlog as the
				// listener closed: it raced the close too.
				Err(e) if e.kind() != std::io::ErrorKind::ConnectionReset => panic!("unexpected connect error: {e}"),
				_ => {
					assert!(Instant::now() < deadline, "listener still accepting after shutdown");
					tokio::time::sleep(Duration::from_millis(5)).await;
				}
			}
		}
	}

	// unix-only: Windows answers a connect to a closed port only after ~2 s of
	// SYN retries, so the refusal probe is too slow to run there. The drain
	// itself is still covered there by the idle and bound tests below.
	#[cfg(unix)]
	#[tokio::test]
	async fn drain_refuses_new_connections_and_finishes_in_flight_request() {
		let started = Arc::new(Notify::new());
		let release = Arc::new(Notify::new());
		let (addr, shutdown, server) = start(gated_handler(started.clone(), release.clone()));

		let mut in_flight = TcpStream::connect(addr).await.unwrap();
		in_flight.write_all(REQUEST).await.unwrap();
		started.notified().await;

		shutdown.send(Duration::from_secs(10)).unwrap();

		// (a) New connections are refused while the request is still running.
		wait_refused(addr, Duration::from_millis(250)).await;
		assert!(
			!server.is_finished(),
			"server exited before the in-flight request finished"
		);

		// (b) The in-flight response arrives whole.
		release.notify_one();
		let mut response = Vec::new();
		in_flight.read_to_end(&mut response).await.unwrap();
		let response = String::from_utf8(response).unwrap();
		assert!(response.starts_with("HTTP/1.1 200 OK\r\n"), "{response}");
		assert!(response.ends_with("\r\n\r\ndone"), "{response}");

		// (c) serve() ends once it has.
		tokio::time::timeout(Duration::from_secs(2), server)
			.await
			.unwrap()
			.unwrap()
			.unwrap();
	}

	#[tokio::test]
	async fn drain_closes_idle_connections_at_once() {
		let started = Arc::new(Notify::new());
		let (addr, shutdown, server) = start(gated_handler(started, Arc::new(Notify::new())));

		// Connected but no request sent: nothing to finish.
		let mut idle = TcpStream::connect(addr).await.unwrap();
		// Let the accept loop take the connection before shutdown.
		tokio::time::sleep(Duration::from_millis(100)).await;

		let begun = Instant::now();
		shutdown.send(Duration::from_secs(60)).unwrap();
		tokio::time::timeout(Duration::from_secs(2), server)
			.await
			.unwrap()
			.unwrap()
			.unwrap();
		assert!(begun.elapsed() < Duration::from_secs(2));

		let mut buf = [0u8; 16];
		match idle.read(&mut buf).await {
			Ok(n) => assert_eq!(n, 0, "idle connection got data instead of EOF"),
			Err(e) => assert_eq!(e.kind(), std::io::ErrorKind::ConnectionReset),
		}
	}

	#[tokio::test]
	async fn drain_bound_reports_connections_left_open() {
		let started = Arc::new(Notify::new());
		let (addr, shutdown, server) = start(gated_handler(started.clone(), Arc::new(Notify::new())));

		let mut stuck = TcpStream::connect(addr).await.unwrap();
		stuck.write_all(REQUEST).await.unwrap();
		started.notified().await;

		shutdown.send(Duration::from_millis(200)).unwrap();
		let err = tokio::time::timeout(Duration::from_secs(2), server)
			.await
			.unwrap()
			.unwrap()
			.expect_err("drain past the bound must fail");
		assert_eq!(err.to_string(), "1 connection(s) still open after the 200ms drain");
		let closed = tokio::time::timeout(Duration::from_secs(2), stuck.read(&mut [0]))
			.await
			.expect("HTTP task must be cancelled before serve returns");
		match closed {
			Ok(n) => assert_eq!(n, 0),
			Err(err) => assert_eq!(err.kind(), std::io::ErrorKind::ConnectionReset),
		}
	}

	/// Serve one accepted connection the way `serve` does. `client` is what the
	/// peer writes first; `shutdown` signals a drain once the connection is up.
	async fn connection_error(client: &[u8], shutdown: bool) -> Box<dyn std::error::Error + Send + Sync> {
		let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
		let mut peer = TcpStream::connect(listener.local_addr().unwrap()).await.unwrap();
		peer.write_all(client).await.unwrap();
		let (stream, _) = listener.accept().await.unwrap();

		let service = service_fn(|_req: Request<Incoming>| async {
			Ok::<_, hyper::Error>(Response::new(Full::new(Bytes::new())))
		});
		let mut builder = auto::Builder::new(TokioExecutor::new());
		builder
			.http1()
			.keep_alive(false)
			.timer(TokioTimer::new())
			.header_read_timeout(HEADER_READ_TIMEOUT);
		let graceful = GracefulShutdown::new();
		let connection = graceful.watch(builder.serve_connection_with_upgrades(TokioIo::new(stream), service));
		let result = if shutdown {
			tokio::join!(connection, graceful.shutdown()).0
		} else {
			connection.await
		};
		result.expect_err("connection must end in an error")
	}

	#[tokio::test]
	async fn idle_close_on_shutdown_is_recognised() {
		let err = connection_error(b"", true).await;
		assert!(is_idle_close(err.as_ref()), "{err:?}");
	}

	#[tokio::test]
	async fn real_connection_errors_are_not_idle_closes() {
		let err = connection_error(b"NOT HTTP AT ALL\r\n\r\n", false).await;
		assert!(!is_idle_close(err.as_ref()), "{err:?}");
	}

	#[cfg(unix)]
	#[tokio::test]
	async fn reuseport_lets_two_listeners_share_a_port() {
		let first = bind("127.0.0.1:0".parse().unwrap()).unwrap();
		let addr = first.local_addr().unwrap();

		let second = bind(addr).expect("second SO_REUSEPORT listener on the same port");
		assert_eq!(second.local_addr().unwrap(), addr);

		// A socket without SO_REUSEPORT still cannot take the port.
		let err = std::net::TcpListener::bind(addr).expect_err("plain bind must fail");
		assert_eq!(err.kind(), std::io::ErrorKind::AddrInUse);
	}
}
