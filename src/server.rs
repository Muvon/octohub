//! The client listener: binding, the accept loop, and the graceful drain on
//! SIGTERM/SIGINT.

use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use hyper::body::Incoming;
use hyper::service::service_fn;
use hyper::{Request, Response};
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto;
use hyper_util::server::graceful::GracefulShutdown;
use tokio::net::{TcpListener, TcpSocket};

use crate::BoxBody;

/// The backlog `tokio::net::TcpListener::bind` uses.
const LISTEN_BACKLOG: u32 = 128;

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
	// Every connection task holds a clone, so the strong count less this one
	// is the number of connections still open.
	let open = Arc::new(());
	let mut shutdown = std::pin::pin!(shutdown);

	let drain_timeout = loop {
		let accepted = tokio::select! {
			drain_timeout = &mut shutdown => break drain_timeout,
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
		let io = TokioIo::new(stream);
		let handler = handler.clone();
		// Taken here, not in the task: a shutdown signalled before the task
		// first runs must still reach this connection.
		let watcher = graceful.watcher();
		let open = open.clone();

		tokio::task::spawn(async move {
			let _open = open;
			let service = service_fn(move |req: Request<Incoming>| {
				let response = handler(req, remote_addr);
				async move { Ok::<_, hyper::Error>(response.await) }
			});

			// auto::Builder negotiates HTTP/1 or HTTP/2 from what the client speaks
			// (HTTP/2 prior-knowledge for plaintext, ALPN for TLS — TLS terminated
			// upstream). HTTP/1 keep-alive stays disabled so each request closes
			// its connection — prevents clients from reusing a stale pooled
			// connection silently dropped by NAT/firewall during long LLM gaps.
			let mut builder = auto::Builder::new(TokioExecutor::new());
			builder.http1().keep_alive(false);
			match watcher.watch(builder.serve_connection(io, service)).await {
				Ok(()) => {}
				Err(err) if is_idle_close(err.as_ref()) => {
					tracing::debug!(remote = %remote_addr, "idle connection closed for shutdown");
				}
				Err(err) => tracing::error!(remote = %remote_addr, error = %err, "connection error"),
			}
		});
	};

	// Close the listening socket before draining: the port stops queueing
	// connections for this process now, not when the last request ends.
	drop(listener);
	tracing::info!(
		connections = Arc::strong_count(&open) - 1,
		timeout_secs = drain_timeout.as_secs(),
		"listener closed; draining open connections"
	);
	if tokio::time::timeout(drain_timeout, graceful.shutdown()).await.is_err() {
		let still_open = Arc::strong_count(&open) - 1;
		tracing::error!(connections = still_open, "drain timed out with connections still open");
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
	use std::time::Instant;
	use tokio::io::{AsyncReadExt, AsyncWriteExt};
	use tokio::net::TcpStream;
	use tokio::sync::{oneshot, Notify};

	const REQUEST: &[u8] = b"GET /slow HTTP/1.1\r\nHost: octohub\r\n\r\n";

	/// A handler that signals `started`, waits `delay`, then answers `done`.
	fn slow_handler(
		started: Arc<Notify>,
		delay: Duration,
	) -> impl Fn(Request<Incoming>, SocketAddr) -> std::pin::Pin<Box<dyn Future<Output = Response<BoxBody>> + Send>>
	       + Clone
	       + Send
	       + Sync
	       + 'static {
		move |_req, _remote| {
			let started = started.clone();
			Box::pin(async move {
				started.notify_one();
				tokio::time::sleep(delay).await;
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
	// SYN retries, longer than the 500 ms in-flight request, so the probe cannot
	// see the listener closed while that request still runs. The drain itself is
	// still covered there by the idle and bound tests below.
	#[cfg(unix)]
	#[tokio::test]
	async fn drain_refuses_new_connections_and_finishes_in_flight_request() {
		let started = Arc::new(Notify::new());
		let (addr, shutdown, server) = start(slow_handler(started.clone(), Duration::from_millis(500)));

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
		let (addr, shutdown, server) = start(slow_handler(started, Duration::from_secs(60)));

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
		let (addr, shutdown, server) = start(slow_handler(started.clone(), Duration::from_secs(60)));

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
		builder.http1().keep_alive(false);
		let graceful = GracefulShutdown::new();
		let connection = graceful.watch(builder.serve_connection(TokioIo::new(stream), service));
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
