use anyhow::Result;
use std::path::Path;
use std::process::{Command, Stdio};

pub fn copy_image_to_clipboard(path: &Path) -> Result<()> {
    let mime_type = match path.extension().and_then(|e| e.to_str()) {
        Some("png") => "image/png",
        Some("webp") => "image/webp",
        _ => "image/png",
    };

    Command::new("wl-copy")
        .arg("--type")
        .arg(mime_type)
        .stdin(Stdio::from(std::fs::File::open(path)?))
        .spawn()?
        .wait()?;

    Ok(())
}
