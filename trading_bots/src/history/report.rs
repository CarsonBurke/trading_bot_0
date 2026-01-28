pub use shared::report::*;

use std::io;

pub fn write_report(path: &str, report: &Report) -> io::Result<()> {
    let bytes = postcard::to_stdvec(report)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    std::fs::write(path, bytes)
}

pub fn read_report(path: &str) -> io::Result<Report> {
    let bytes = std::fs::read(path)?;
    postcard::from_bytes(&bytes).map_err(|e| io::Error::new(io::ErrorKind::Other, e))
}
