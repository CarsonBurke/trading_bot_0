//! Packed OHLCV bar corpus: `<dir>/<SYMBOL>.<res_secs>.bars`.
//!
//! Layout: a 64-byte header followed by `count` contiguous [`PackedBar`] records with
//! strictly increasing `ts_ms`. Reads are mmap-backed and zero-copy.

use anyhow::{anyhow, bail, ensure, Context, Result};
use bytemuck::{Pod, Zeroable};
use memmap2::Mmap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// UTC epoch millis of bar open, OHLC, volume, VWAP and trade count.
///
/// Packed to the on-disk record size, so fields must be read by value (`bar.close`); taking a
/// reference to a field is a compile error, and `ts_ms` needs [`PackedBar::ts`] in reference
/// positions such as `assert_eq!`/`format!` because `i64` outranks the struct alignment.
#[repr(C, packed(4))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct PackedBar {
    pub ts_ms: i64,
    pub open: f32,
    pub high: f32,
    pub low: f32,
    pub close: f32,
    pub volume: f32,
    pub vwap: f32,
    pub trades: u32,
}

impl PackedBar {
    /// `ts_ms` by value, usable where a reference to the packed field would be rejected.
    #[inline]
    pub fn ts(&self) -> i64 {
        self.ts_ms
    }
}

const _: () = assert!(std::mem::size_of::<PackedBar>() == 36);
const _: () = assert!(std::mem::align_of::<PackedBar>() == 4);

pub const MAGIC: [u8; 8] = *b"TBBARS01";
pub const VERSION: u32 = 1;
pub const HEADER_LEN: usize = 64;
pub const RECORD_LEN: usize = std::mem::size_of::<PackedBar>();
pub const SYMBOL_LEN: usize = 24;
pub const FILE_EXTENSION: &str = "bars";

const OFF_VERSION: usize = 8;
const OFF_RES_SECS: usize = 12;
const OFF_COUNT: usize = 16;
const OFF_FIRST_TS: usize = 24;
const OFF_LAST_TS: usize = 32;
const OFF_SYMBOL: usize = 40;

const _: () = assert!(OFF_SYMBOL + SYMBOL_LEN <= HEADER_LEN);

/// Canonical corpus path for a symbol at a given resolution.
pub fn bar_file_path(dir: impl AsRef<Path>, symbol: &str, res_secs: u32) -> PathBuf {
    dir.as_ref()
        .join(format!("{symbol}.{res_secs}.{FILE_EXTENSION}"))
}

/// Inverse of [`bar_file_path`]: recover `(symbol, res_secs)` from a corpus file name.
pub fn parse_bar_file_name(path: &Path) -> Result<(String, u32)> {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .with_context(|| format!("bar path {} has no UTF-8 file name", path.display()))?;
    let stem = name.strip_suffix(&format!(".{FILE_EXTENSION}")).with_context(|| {
        format!("bar file name {name:?} does not end in .{FILE_EXTENSION}")
    })?;
    let (symbol, res) = stem.rsplit_once('.').with_context(|| {
        format!("bar file name {name:?} is not <SYMBOL>.<res_secs>.{FILE_EXTENSION}")
    })?;
    let res_secs: u32 = res
        .parse()
        .with_context(|| format!("bar file name {name:?} has non-numeric resolution {res:?}"))?;
    ensure!(!symbol.is_empty(), "bar file name {name:?} has an empty symbol");
    ensure!(res_secs > 0, "bar file name {name:?} has resolution 0");
    Ok((symbol.to_string(), res_secs))
}

#[derive(Clone, Copy)]
struct Header {
    res_secs: u32,
    count: u64,
    first_ts_ms: i64,
    last_ts_ms: i64,
    symbol: [u8; SYMBOL_LEN],
}

impl Header {
    fn encode(&self) -> [u8; HEADER_LEN] {
        let mut out = [0u8; HEADER_LEN];
        out[..MAGIC.len()].copy_from_slice(&MAGIC);
        out[OFF_VERSION..OFF_VERSION + 4].copy_from_slice(&VERSION.to_le_bytes());
        out[OFF_RES_SECS..OFF_RES_SECS + 4].copy_from_slice(&self.res_secs.to_le_bytes());
        out[OFF_COUNT..OFF_COUNT + 8].copy_from_slice(&self.count.to_le_bytes());
        out[OFF_FIRST_TS..OFF_FIRST_TS + 8].copy_from_slice(&self.first_ts_ms.to_le_bytes());
        out[OFF_LAST_TS..OFF_LAST_TS + 8].copy_from_slice(&self.last_ts_ms.to_le_bytes());
        out[OFF_SYMBOL..OFF_SYMBOL + SYMBOL_LEN].copy_from_slice(&self.symbol);
        out
    }

    fn decode(bytes: &[u8; HEADER_LEN], path: &Path) -> Result<Self> {
        let magic = &bytes[..MAGIC.len()];
        ensure!(
            magic == MAGIC,
            "bar file {} has bad magic {:?}, expected {:?}",
            path.display(),
            String::from_utf8_lossy(magic),
            String::from_utf8_lossy(&MAGIC)
        );
        let version = u32::from_le_bytes(bytes[OFF_VERSION..OFF_VERSION + 4].try_into().unwrap());
        ensure!(
            version == VERSION,
            "bar file {} has version {version}, expected {VERSION}",
            path.display()
        );
        Ok(Self {
            res_secs: u32::from_le_bytes(
                bytes[OFF_RES_SECS..OFF_RES_SECS + 4].try_into().unwrap(),
            ),
            count: u64::from_le_bytes(bytes[OFF_COUNT..OFF_COUNT + 8].try_into().unwrap()),
            first_ts_ms: i64::from_le_bytes(
                bytes[OFF_FIRST_TS..OFF_FIRST_TS + 8].try_into().unwrap(),
            ),
            last_ts_ms: i64::from_le_bytes(
                bytes[OFF_LAST_TS..OFF_LAST_TS + 8].try_into().unwrap(),
            ),
            symbol: bytes[OFF_SYMBOL..OFF_SYMBOL + SYMBOL_LEN].try_into().unwrap(),
        })
    }

    fn expected_len(&self) -> u64 {
        HEADER_LEN as u64 + self.count * RECORD_LEN as u64
    }
}

fn encode_symbol(symbol: &str) -> Result<[u8; SYMBOL_LEN]> {
    let bytes = symbol.as_bytes();
    ensure!(!bytes.is_empty(), "bar file symbol must not be empty");
    ensure!(
        bytes.len() <= SYMBOL_LEN,
        "symbol {symbol:?} is {} bytes, exceeds the {SYMBOL_LEN}-byte header field",
        bytes.len()
    );
    ensure!(!bytes.contains(&0), "symbol {symbol:?} must not contain NUL");
    let mut out = [0u8; SYMBOL_LEN];
    out[..bytes.len()].copy_from_slice(bytes);
    Ok(out)
}

fn decode_symbol(field: &[u8; SYMBOL_LEN], path: &Path) -> Result<String> {
    let end = field.iter().position(|&b| b == 0).unwrap_or(SYMBOL_LEN);
    let symbol = std::str::from_utf8(&field[..end])
        .with_context(|| format!("bar file {} has a non-UTF-8 symbol field", path.display()))?;
    ensure!(
        !symbol.is_empty(),
        "bar file {} has an empty symbol field",
        path.display()
    );
    Ok(symbol.to_string())
}

fn validate_strictly_increasing(bars: &[PackedBar]) -> Result<()> {
    for (i, pair) in bars.windows(2).enumerate() {
        let (prev, next) = (pair[0].ts_ms, pair[1].ts_ms);
        if next <= prev {
            bail!(
                "bar timestamps must be strictly increasing: index {} has ts_ms {next} which does not exceed index {i} ts_ms {prev}",
                i + 1
            );
        }
    }
    Ok(())
}

/// Write a complete corpus file, replacing any existing one.
pub fn write_bar_file(
    path: &Path,
    symbol: &str,
    res_secs: u32,
    bars: &[PackedBar],
) -> Result<()> {
    ensure!(res_secs > 0, "bar resolution must be positive");
    let symbol_field = encode_symbol(symbol)?;
    validate_strictly_increasing(bars)?;
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating bar directory {}", parent.display()))?;
    }
    let header = Header {
        res_secs,
        count: bars.len() as u64,
        first_ts_ms: bars.first().map_or(0, |b| b.ts_ms),
        last_ts_ms: bars.last().map_or(0, |b| b.ts_ms),
        symbol: symbol_field,
    };
    let file =
        File::create(path).with_context(|| format!("creating bar file {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    writer.write_all(&header.encode())?;
    writer.write_all(bytemuck::cast_slice(bars))?;
    writer.flush()?;
    writer
        .into_inner()
        .map_err(|e| anyhow!("flushing bar file {}: {e}", path.display()))?
        .sync_data()
        .with_context(|| format!("syncing bar file {}", path.display()))?;
    Ok(())
}

/// Append only the records strictly newer than the file's `last_ts_ms`, updating the header
/// in place. Returns how many records were written. Creates the file if absent, taking
/// symbol and resolution from the `<SYMBOL>.<res_secs>.bars` file name.
pub fn append_bars(path: &Path, bars: &[PackedBar]) -> Result<usize> {
    validate_strictly_increasing(bars)?;
    let mut file = match OpenOptions::new().read(true).write(true).open(path) {
        Ok(file) => file,
        Err(e) if e.kind() == ErrorKind::NotFound => {
            let (symbol, res_secs) = parse_bar_file_name(path)?;
            write_bar_file(path, &symbol, res_secs, bars)?;
            return Ok(bars.len());
        }
        Err(e) => {
            return Err(e).with_context(|| format!("opening bar file {}", path.display()))?;
        }
    };

    let len = file
        .metadata()
        .with_context(|| format!("stat of bar file {}", path.display()))?
        .len();
    ensure!(
        len >= HEADER_LEN as u64,
        "bar file {} is {len} bytes, shorter than its {HEADER_LEN}-byte header",
        path.display()
    );
    let mut head = [0u8; HEADER_LEN];
    file.read_exact(&mut head)
        .with_context(|| format!("reading header of bar file {}", path.display()))?;
    let mut header = Header::decode(&head, path)?;
    ensure!(
        len == header.expected_len(),
        "bar file {} is {len} bytes but its header count {} implies {} bytes",
        path.display(),
        header.count,
        header.expected_len()
    );

    let fresh = if header.count == 0 {
        bars
    } else {
        let cutoff = header.last_ts_ms;
        &bars[bars.partition_point(|b| b.ts_ms <= cutoff)..]
    };
    let Some(last) = fresh.last() else {
        return Ok(0);
    };

    file.seek(SeekFrom::Start(header.expected_len()))?;
    file.write_all(bytemuck::cast_slice(fresh))
        .with_context(|| format!("appending {} bars to {}", fresh.len(), path.display()))?;
    if header.count == 0 {
        header.first_ts_ms = fresh[0].ts_ms;
    }
    header.last_ts_ms = last.ts_ms;
    header.count += fresh.len() as u64;
    file.seek(SeekFrom::Start(0))?;
    file.write_all(&header.encode())
        .with_context(|| format!("updating header of bar file {}", path.display()))?;
    file.sync_data()
        .with_context(|| format!("syncing bar file {}", path.display()))?;
    Ok(fresh.len())
}

/// Read-only mmap view over a corpus file.
pub struct BarFile {
    mmap: Mmap,
    path: PathBuf,
    symbol: String,
    res_secs: u32,
    count: usize,
}

impl std::fmt::Debug for BarFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BarFile")
            .field("path", &self.path)
            .field("symbol", &self.symbol)
            .field("res_secs", &self.res_secs)
            .field("count", &self.count)
            .finish()
    }
}

impl BarFile {
    pub fn open(path: &Path) -> Result<Self> {
        let file =
            File::open(path).with_context(|| format!("opening bar file {}", path.display()))?;
        let len = file
            .metadata()
            .with_context(|| format!("stat of bar file {}", path.display()))?
            .len();
        ensure!(
            len >= HEADER_LEN as u64,
            "bar file {} is {len} bytes, shorter than its {HEADER_LEN}-byte header",
            path.display()
        );
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("mapping bar file {}", path.display()))?;
        let header = Header::decode(mmap[..HEADER_LEN].try_into().unwrap(), path)?;
        ensure!(
            len == header.expected_len(),
            "bar file {} is {len} bytes but its header count {} implies {} bytes",
            path.display(),
            header.count,
            header.expected_len()
        );
        let count = header.count as usize;
        let bars: &[PackedBar] = bytemuck::try_cast_slice(&mmap[HEADER_LEN..]).map_err(|e| {
            anyhow!(
                "bar file {} record region is not a valid PackedBar slice: {e}",
                path.display()
            )
        })?;
        if let (Some(first), Some(last)) = (bars.first(), bars.last()) {
            let (first_ts, last_ts) = (first.ts_ms, last.ts_ms);
            ensure!(
                first_ts == header.first_ts_ms && last_ts == header.last_ts_ms,
                "bar file {} header span [{}, {}] disagrees with its records [{first_ts}, {last_ts}]",
                path.display(),
                header.first_ts_ms,
                header.last_ts_ms
            );
        }
        let symbol = decode_symbol(&header.symbol, path)?;
        Ok(Self {
            mmap,
            path: path.to_path_buf(),
            symbol,
            res_secs: header.res_secs,
            count,
        })
    }

    /// Zero-copy view of the mapped records.
    pub fn bars(&self) -> &[PackedBar] {
        bytemuck::cast_slice(&self.mmap[HEADER_LEN..HEADER_LEN + self.count * RECORD_LEN])
    }

    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    pub fn res_secs(&self) -> u32 {
        self.res_secs
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn first_ts_ms(&self) -> Option<i64> {
        self.bars().first().map(|b| b.ts_ms)
    }

    pub fn last_ts_ms(&self) -> Option<i64> {
        self.bars().last().map(|b| b.ts_ms)
    }

    /// Index of the first record with `ts_ms >= ts_ms`, or `len()` if none.
    pub fn index_at_or_after(&self, ts_ms: i64) -> usize {
        self.bars().partition_point(|b| b.ts_ms < ts_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static SEQ: AtomicU64 = AtomicU64::new(0);

    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let seq = SEQ.fetch_add(1, Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!(
                "trading-bot-bars-{}-{unique}-{seq}",
                std::process::id()
            ));
            fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        fn path(&self, symbol: &str, res_secs: u32) -> PathBuf {
            bar_file_path(&self.0, symbol, res_secs)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn bar(ts_ms: i64) -> PackedBar {
        let base = ts_ms as f32 / 1000.0;
        PackedBar {
            ts_ms,
            open: base,
            high: base + 2.0,
            low: base - 1.0,
            close: base + 1.0,
            volume: 1000.0 + base,
            vwap: base + 0.5,
            trades: (ts_ms % 997) as u32,
        }
    }

    fn ts_of(bars: &[PackedBar]) -> Vec<i64> {
        bars.iter().map(|b| b.ts_ms).collect()
    }

    #[test]
    fn bars_round_trip_through_write_and_open() {
        let dir = TempDir::new();
        let path = dir.path("AAPL", 300);
        let written: Vec<PackedBar> = [1_000, 2_000, 3_000, 4_000].map(bar).to_vec();
        write_bar_file(&path, "AAPL", 300, &written).unwrap();

        assert_eq!(
            fs::metadata(&path).unwrap().len(),
            (HEADER_LEN + 4 * RECORD_LEN) as u64
        );
        let file = BarFile::open(&path).unwrap();
        assert_eq!(file.symbol(), "AAPL");
        assert_eq!(file.res_secs(), 300);
        assert_eq!(file.len(), 4);
        assert!(!file.is_empty());
        assert_eq!(file.first_ts_ms(), Some(1_000));
        assert_eq!(file.last_ts_ms(), Some(4_000));
        assert_eq!(file.bars(), written.as_slice());
        assert_eq!(parse_bar_file_name(&path).unwrap(), ("AAPL".into(), 300));

        let read = file.bars()[2];
        let source = written[2];
        assert_eq!(read.ts(), source.ts());
        assert_eq!(read.open, source.open);
        assert_eq!(read.high, source.high);
        assert_eq!(read.low, source.low);
        assert_eq!(read.close, source.close);
        assert_eq!(read.volume, source.volume);
        assert_eq!(read.vwap, source.vwap);
        assert_eq!(read.trades, source.trades);
    }

    #[test]
    fn bars_slice_is_a_zero_copy_view_of_the_mapping() {
        let dir = TempDir::new();
        let path = dir.path("MSFT", 300);
        let written: Vec<PackedBar> = (0..2_000).map(|i| bar(1_000 + i * 300_000)).collect();
        write_bar_file(&path, "MSFT", 300, &written).unwrap();

        let file = BarFile::open(&path).unwrap();
        let first = file.bars();
        let second = file.bars();
        assert_eq!(first.as_ptr(), second.as_ptr());
        assert_eq!(first.len(), written.len());
        assert_eq!(std::mem::size_of_val(first), written.len() * RECORD_LEN);
        // The records live directly in the page-aligned mapping, right after the header.
        assert_eq!(first.as_ptr() as usize % 4096, HEADER_LEN);
    }

    #[test]
    fn bars_index_at_or_after_handles_every_boundary() {
        let dir = TempDir::new();
        let path = dir.path("NVDA", 60);
        write_bar_file(&path, "NVDA", 60, &[bar(1_000), bar(2_000), bar(3_000)]).unwrap();
        let file = BarFile::open(&path).unwrap();
        assert_eq!(file.index_at_or_after(i64::MIN), 0);
        assert_eq!(file.index_at_or_after(500), 0);
        assert_eq!(file.index_at_or_after(1_000), 0);
        assert_eq!(file.index_at_or_after(1_001), 1);
        assert_eq!(file.index_at_or_after(2_000), 1);
        assert_eq!(file.index_at_or_after(2_500), 2);
        assert_eq!(file.index_at_or_after(3_000), 2);
        assert_eq!(file.index_at_or_after(3_500), 3);
        assert_eq!(file.index_at_or_after(i64::MAX), 3);

        let empty_path = dir.path("EMPTY", 60);
        write_bar_file(&empty_path, "EMPTY", 60, &[]).unwrap();
        let empty = BarFile::open(&empty_path).unwrap();
        assert!(empty.is_empty());
        assert_eq!(empty.bars(), &[]);
        assert_eq!(empty.first_ts_ms(), None);
        assert_eq!(empty.last_ts_ms(), None);
        assert_eq!(empty.index_at_or_after(i64::MIN), 0);
        assert_eq!(empty.index_at_or_after(1_000), 0);
        assert_eq!(empty.index_at_or_after(i64::MAX), 0);
    }

    #[test]
    fn bars_reject_non_monotonic_timestamps() {
        let dir = TempDir::new();
        let path = dir.path("TSLA", 300);
        let duplicate = write_bar_file(&path, "TSLA", 300, &[bar(1_000), bar(1_000)])
            .unwrap_err()
            .to_string();
        assert!(
            duplicate.contains("index 1") && duplicate.contains("1000"),
            "{duplicate}"
        );
        let descending = write_bar_file(&path, "TSLA", 300, &[bar(1_000), bar(3_000), bar(2_000)])
            .unwrap_err()
            .to_string();
        assert!(
            descending.contains("index 2") && descending.contains("3000"),
            "{descending}"
        );
        assert!(append_bars(&path, &[bar(2_000), bar(2_000)]).is_err());
        assert!(!path.exists());
        assert!(write_bar_file(&path, "", 300, &[]).is_err());
        assert!(write_bar_file(&path, &"X".repeat(SYMBOL_LEN + 1), 300, &[]).is_err());
        assert!(write_bar_file(&path, "TSLA", 0, &[]).is_err());
    }

    #[test]
    fn bars_reject_truncated_and_corrupt_files() {
        let dir = TempDir::new();
        let path = dir.path("AMD", 300);
        write_bar_file(&path, "AMD", 300, &[bar(1_000), bar(2_000), bar(3_000)]).unwrap();
        let full = fs::read(&path).unwrap();

        let truncated = dir.path("TRUNC", 300);
        fs::write(&truncated, &full[..full.len() - 4]).unwrap();
        let err = BarFile::open(&truncated).unwrap_err().to_string();
        assert!(
            err.contains(&truncated.display().to_string())
                && err.contains(&format!("{} bytes", full.len() - 4))
                && err.contains(&format!("{} bytes", full.len())),
            "{err}"
        );
        assert!(append_bars(&truncated, &[bar(4_000)]).is_err());

        let short = dir.path("SHORT", 300);
        fs::write(&short, &full[..HEADER_LEN - 1]).unwrap();
        let err = BarFile::open(&short).unwrap_err().to_string();
        assert!(
            err.contains(&format!("{} bytes", HEADER_LEN - 1)) && err.contains("64-byte header"),
            "{err}"
        );

        let bad_magic = dir.path("MAGIC", 300);
        let mut bytes = full.clone();
        bytes[..8].copy_from_slice(b"TBBARSXX");
        fs::write(&bad_magic, &bytes).unwrap();
        assert!(BarFile::open(&bad_magic)
            .unwrap_err()
            .to_string()
            .contains("magic"));

        let bad_version = dir.path("VERSION", 300);
        let mut bytes = full.clone();
        bytes[OFF_VERSION..OFF_VERSION + 4].copy_from_slice(&(VERSION + 1).to_le_bytes());
        fs::write(&bad_version, &bytes).unwrap();
        assert!(BarFile::open(&bad_version)
            .unwrap_err()
            .to_string()
            .contains("version"));

        let inconsistent = dir.path("SPAN", 300);
        let mut bytes = full.clone();
        bytes[OFF_LAST_TS..OFF_LAST_TS + 8].copy_from_slice(&9_999i64.to_le_bytes());
        fs::write(&inconsistent, &bytes).unwrap();
        assert!(BarFile::open(&inconsistent)
            .unwrap_err()
            .to_string()
            .contains("disagrees"));
    }

    #[test]
    fn bars_append_dedupes_overlap_and_is_idempotent() {
        let dir = TempDir::new();
        let path = dir.path("GOOG", 300);
        write_bar_file(&path, "GOOG", 300, &[bar(1_000), bar(2_000)]).unwrap();

        let batch = [bar(1_500), bar(2_000), bar(3_000), bar(4_000)];
        assert_eq!(append_bars(&path, &batch).unwrap(), 2);
        assert_eq!(append_bars(&path, &batch).unwrap(), 0);
        assert_eq!(append_bars(&path, &[bar(500), bar(2_000)]).unwrap(), 0);

        let file = BarFile::open(&path).unwrap();
        assert_eq!(file.symbol(), "GOOG");
        assert_eq!(file.res_secs(), 300);
        assert_eq!(ts_of(file.bars()), vec![1_000, 2_000, 3_000, 4_000]);
        assert_eq!(file.last_ts_ms(), Some(4_000));
        assert_eq!(file.bars()[3], bar(4_000));
        drop(file);

        assert_eq!(append_bars(&path, &[bar(5_000)]).unwrap(), 1);
        assert_eq!(
            ts_of(BarFile::open(&path).unwrap().bars()),
            vec![1_000, 2_000, 3_000, 4_000, 5_000]
        );
    }

    #[test]
    fn bars_append_creates_missing_and_empty_files() {
        let dir = TempDir::new();
        let created = dir.path("META", 300);
        assert_eq!(append_bars(&created, &[bar(1_000), bar(2_000)]).unwrap(), 2);
        let file = BarFile::open(&created).unwrap();
        assert_eq!(file.symbol(), "META");
        assert_eq!(file.res_secs(), 300);
        assert_eq!(ts_of(file.bars()), vec![1_000, 2_000]);
        drop(file);

        let nested = dir.0.join("sub").join("dir");
        let deep = bar_file_path(&nested, "AMZN", 60);
        assert_eq!(append_bars(&deep, &[bar(7_000)]).unwrap(), 1);
        assert_eq!(BarFile::open(&deep).unwrap().first_ts_ms(), Some(7_000));

        let empty = dir.path("NFLX", 300);
        write_bar_file(&empty, "NFLX", 300, &[]).unwrap();
        assert_eq!(append_bars(&empty, &[]).unwrap(), 0);
        assert_eq!(append_bars(&empty, &[bar(-5_000), bar(6_000)]).unwrap(), 2);
        let file = BarFile::open(&empty).unwrap();
        assert_eq!(file.first_ts_ms(), Some(-5_000));
        assert_eq!(file.last_ts_ms(), Some(6_000));

        assert!(append_bars(&dir.0.join("no-extension"), &[bar(1)]).is_err());
        assert!(append_bars(&dir.0.join("noresolution.bars"), &[bar(1)]).is_err());
        assert!(append_bars(&dir.0.join("BAD.abc.bars"), &[bar(1)]).is_err());
    }
}
