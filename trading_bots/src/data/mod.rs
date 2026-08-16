pub mod account;
pub mod deep_daily;
pub mod earnings;
pub mod historical;
pub mod ingest;
pub mod macro_econ;
pub mod polygon;
pub mod universe;

pub use earnings::{get_cached_earnings_data_any, get_earnings_data_any, EarningsReport};

use std::path::Path;

/// Loads `<repo>/.env` and `trading_bots/.env`, without overriding variables already in the
/// process environment. Later files do not override earlier ones either.
pub fn load_dotenv() {
    let candidates = [
        Path::new(shared::paths::WORKSPACE_ROOT).join(".env"),
        Path::new(env!("CARGO_MANIFEST_DIR")).join(".env"),
    ];
    for path in candidates {
        let Ok(contents) = std::fs::read_to_string(&path) else {
            continue;
        };
        for line in contents.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let line = line.strip_prefix("export ").unwrap_or(line);
            let Some((key, value)) = line.split_once('=') else {
                continue;
            };
            let key = key.trim();
            if key.is_empty() || std::env::var_os(key).is_some() {
                continue;
            }
            std::env::set_var(key, unquote(value.trim()));
        }
    }
}

fn unquote(value: &str) -> &str {
    for quote in ['"', '\''] {
        if let Some(inner) = value
            .strip_prefix(quote)
            .and_then(|rest| rest.strip_suffix(quote))
        {
            return inner;
        }
    }
    value
}
