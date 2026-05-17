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
                .with_env_filter(filter)
                .init();
        }
        LogFormat::Auto => {
            use std::io::IsTerminal;
            if std::io::stdout().is_terminal() {
                tracing_subscriber::fmt()
                    .compact()
                    .with_target(false)
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
