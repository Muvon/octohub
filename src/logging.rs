use anyhow::Result;

use crate::config::{LogFormat, LoggingConfig};

pub fn init(cfg: &LoggingConfig) -> Result<()> {
    let level = cfg
        .level
        .as_deref()
        .map(|l| l.to_owned())
        .or_else(|| std::env::var("RUST_LOG").ok())
        .unwrap_or_else(|| "info".to_string());

    let filter = tracing_subscriber::EnvFilter::try_new(&level)?;

    // ANSI escape codes only when stdout is an interactive terminal — keeps
    // logs clean when piped to `tee`, journald, or a file even if the user
    // explicitly picked `format = "pretty"`.
    let use_ansi = {
        use std::io::IsTerminal;
        std::io::stdout().is_terminal()
    };

    match cfg.format {
        LogFormat::Json => {
            tracing_subscriber::fmt()
                .json()
                .flatten_event(true)
                .with_current_span(true)
                .with_span_list(false)
                .with_env_filter(filter)
                .init();
        }
        LogFormat::Pretty => {
            tracing_subscriber::fmt()
                .compact()
                .with_target(false)
                .with_ansi(use_ansi)
                .with_env_filter(filter)
                .init();
        }
        LogFormat::Auto => {
            if use_ansi {
                tracing_subscriber::fmt()
                    .compact()
                    .with_target(false)
                    .with_ansi(true)
                    .with_env_filter(filter)
                    .init();
            } else {
                tracing_subscriber::fmt()
                    .json()
                    .flatten_event(true)
                    .with_current_span(true)
                    .with_span_list(false)
                    .with_env_filter(filter)
                    .init();
            }
        }
    }

    Ok(())
}
