use anyhow::{bail, Result};
use serde::Serialize;

#[derive(Clone, Serialize)]
pub struct Credential {
	pub provider: String,
	#[serde(skip)]
	pub secret: String,
	pub key_hint: String,
	pub position: i64,
	pub created_at: i64,
	pub last_used_at: Option<i64>,
	pub last_error: Option<CredentialError>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CredentialError {
	pub at: i64,
	pub status: u16,
	pub message: String,
}

pub enum CredentialChange {
	Put {
		provider: String,
		secret: String,
		key_hint: String,
	},
	Delete(String),
	Clear,
	Order(Vec<String>),
	Record {
		provider: String,
		secret: String,
		error: Option<CredentialError>,
	},
}

#[derive(Debug)]
pub enum CredentialInputError {
	NotFound,
	InvalidOrder,
}

impl std::fmt::Display for CredentialInputError {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.write_str(match self {
			Self::NotFound => "Credential not found",
			Self::InvalidOrder => "Order must be an exact permutation of the owner's providers",
		})
	}
}
impl std::error::Error for CredentialInputError {}

pub fn apply_change(rows: &mut Vec<Credential>, change: CredentialChange) -> Result<()> {
	match change {
		CredentialChange::Put {
			provider,
			secret,
			key_hint,
		} => {
			if let Some(row) = rows.iter_mut().find(|r| r.provider == provider) {
				row.secret = secret;
				row.key_hint = key_hint;
				row.last_error = None;
			} else {
				rows.push(Credential {
					provider,
					secret,
					key_hint,
					position: rows.len() as i64,
					created_at: super::now_unix() as i64,
					last_used_at: None,
					last_error: None,
				});
			}
		}
		CredentialChange::Delete(provider) => {
			let index = rows
				.iter()
				.position(|r| r.provider == provider)
				.ok_or(CredentialInputError::NotFound)?;
			rows.remove(index);
		}
		CredentialChange::Clear => rows.clear(),
		CredentialChange::Order(order) => {
			let unique: std::collections::HashSet<_> = order.iter().collect();
			if order.len() != rows.len()
				|| unique.len() != order.len()
				|| rows.iter().any(|r| !unique.contains(&r.provider))
			{
				bail!(CredentialInputError::InvalidOrder);
			}
			for (position, provider) in order.iter().enumerate() {
				let index = rows
					.iter()
					.position(|r| &r.provider == provider)
					.expect("permutation validated");
				rows.swap(position, index);
			}
		}
		CredentialChange::Record {
			provider,
			secret,
			error,
		} => {
			// An in-flight call must not stamp an admin's replacement or resurrect a deleted key.
			if let Some(row) = rows.iter_mut().find(|r| r.provider == provider && r.secret == secret) {
				if let Some(error) = error {
					row.last_error = Some(error);
				} else {
					row.last_used_at = Some(super::now_unix() as i64);
				}
			}
		}
	}
	for (position, row) in rows.iter_mut().enumerate() {
		row.position = position as i64;
	}
	Ok(())
}

pub fn decode_error(at: Option<i64>, status: Option<i64>, message: Option<String>) -> Result<Option<CredentialError>> {
	match (at, status, message) {
		(None, None, None) => Ok(None),
		(Some(at), Some(status), Some(message)) => Ok(Some(CredentialError {
			at,
			status: status.try_into()?,
			message,
		})),
		_ => bail!("Invalid owner credential last_error columns"),
	}
}
