use std::fs::File;
use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result};
use ring::digest::{Context as DigestContext, SHA256};

/// Streaming SHA-256 of a file, returned as lowercase hex. Shared by the
/// world-model and planner checkpoint paths so their integrity guards agree.
pub(crate) fn file_sha256(path: impl AsRef<Path>) -> Result<String> {
    let path = path.as_ref();
    let mut file =
        File::open(path).with_context(|| format!("failed opening {}", path.display()))?;
    let mut context = DigestContext::new(&SHA256);
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("failed hashing {}", path.display()))?;
        if read == 0 {
            break;
        }
        context.update(&buffer[..read]);
    }
    Ok(context
        .finish()
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}
