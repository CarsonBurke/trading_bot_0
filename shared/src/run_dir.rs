use anyhow::{bail, Context, Result};
use chrono::Local;
use serde::{Deserialize, Serialize};
use std::fs;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// `meta.json`: a run's identity, and what it was HANDED rather than what a reader must infer.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RunMeta {
    pub commit: String,
    /// Absent on runs created before the record existed, which is exactly the gap it closes:
    /// their split instants are recoverable only by inferring the trainer's default at `commit`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<RunProvenance>,
}

/// Every value a run ASSUMED rather than RECORDED, without which a checkpoint cannot be
/// validated against the data it was trained on.
///
/// The distinction is not pedantic. `training/runs/bardist_v3_rfirst_1ep/meta.json` carried
/// `{"commit": ...}` and nothing else, so "this checkpoint was trained with `b0` =
/// 2025-10-07T12:10:00Z" was true, load-bearing for every economic number taken off that run,
/// and recoverable only by checking out the commit and reading the trainer's DEFAULT. That is a
/// fact about the code at a revision rather than a fact about the run, and it stops being
/// recoverable the moment the default moves.
///
/// The two contexts sit beside the bounds because the bounds alone do NOT pin what the model
/// conditioned on: a context reaches BACKWARD across `b0`, so a window at the start of the val
/// region legitimately reads train-side bars. "No SCORED bar is in train" and "no bar the model
/// SAW is in train" are different statements and only the first follows from the instants.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunProvenance {
    /// `(train|val, val|test)` split instants in epoch millis, as handed to the corpus loader.
    pub split_bounds_ms: [i64; 2],
    /// `false` when the run re-derived the instants from the live corpus, which makes them
    /// percentiles of whatever was on disk that day and comparable to nothing.
    pub split_bounds_pinned: bool,
    /// Deployment bar resolution. The split instants are resolution-specific.
    pub resolution_secs: u32,
    /// `BarCorpus::identity_fingerprint()`, taken after any symbol restriction.
    pub corpus_fingerprint: String,
    /// Corpus membership floor in bars.
    pub min_bars: usize,
    /// Liquidity floor for corpus membership; `0` loads every file on disk.
    pub min_dollar_volume: f64,
    /// Corpus the run read.
    pub data_dir: String,
    /// Context of the fixed across-run diagnostic evaluation.
    pub diagnostic_context_bars: i64,
    /// Context the run deploys and is promoted at.
    pub deployed_context_bars: i64,
    /// Pins the held-out window draw. A different value is a different bench.
    pub eval_window_seed: u64,
    /// Pins the training stream only, never the bench.
    pub train_seed: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunDir {
    pub root: PathBuf,
    pub gens: PathBuf,
    pub weights: PathBuf,
    pub log_file: PathBuf,
}

impl RunDir {
    pub fn create_fresh(runs_path: &str, name: Option<&str>) -> Result<Self> {
        let dir_name = match name {
            Some(n) => {
                validate_run_name(n)?;
                n.to_string()
            }
            None => Local::now().format("%Y-%m-%d_%H-%M-%S-%f").to_string(),
        };

        let runs = Path::new(runs_path);
        fs::create_dir_all(runs)?;

        let root = runs.join(&dir_name);
        let gens = root.join("gens");
        let weights = root.join("weights");
        let log_file = root.join("training.log");

        if root.exists() {
            let run_dir = Self {
                root: root.clone(),
                gens: gens.clone(),
                weights: weights.clone(),
                log_file: log_file.clone(),
            };
            if is_prepared_empty_run(&run_dir)? {
                run_dir.activate(runs_path)?;
                return Ok(run_dir);
            }
            bail!("run dir already exists: {}", root.display());
        }
        fs::create_dir(&root).context("failed to create run dir")?;
        fs::create_dir_all(&gens)?;
        fs::create_dir_all(&weights)?;
        fs::write(
            root.join("meta.json"),
            meta_bytes(&RunMeta {
                commit: current_git_commit().unwrap_or_default(),
                provenance: None,
            })?,
        )
        .context("failed to write run metadata")?;

        let run_dir = Self {
            root,
            gens,
            weights,
            log_file,
        };
        run_dir.activate(runs_path)?;
        Ok(run_dir)
    }

    /// `meta.json` as written. A run created before [`RunProvenance`] existed reads back with
    /// `provenance: None` rather than failing: the file is a historical record, not a schema.
    pub fn meta(&self) -> Result<RunMeta> {
        let path = self.root.join("meta.json");
        let bytes =
            fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
        serde_json::from_slice(&bytes)
            .with_context(|| format!("{} is not run metadata", path.display()))
    }

    /// Record what the run was HANDED, beside the commit [`Self::create_fresh`] wrote.
    ///
    /// Called once, at run start, the moment the corpus is open and its identity is known —
    /// before the first optimizer step and therefore before any artifact the record has to
    /// explain can exist. Writing it later would make the record conditional on the run
    /// surviving, which is the one case where a reader most needs it.
    pub fn record_provenance(&self, provenance: RunProvenance) -> Result<()> {
        let path = self.root.join("meta.json");
        let mut meta = self.meta()?;
        meta.provenance = Some(provenance);
        fs::write(&path, meta_bytes(&meta)?)
            .with_context(|| format!("failed to write {}", path.display()))
    }

    pub fn from_weights_path(path: &Path) -> Result<Self> {
        // path: runs/{name}/weights/{file}.ot
        let weights_dir = path.parent().context("weights path has no parent")?;
        if weights_dir.file_name().is_none_or(|name| name != "weights") {
            bail!(
                "weights path is not inside a run weights directory: {}",
                path.display()
            );
        }
        let root = weights_dir.parent().context("weights dir has no parent")?;
        Self::open(root)
    }

    pub fn from_weights_path_in(path: &Path, runs_path: impl AsRef<Path>) -> Result<Self> {
        let run = Self::from_weights_path(path)?;
        let expected_parent = canonical_or_original(runs_path.as_ref());
        let actual_parent = run
            .root
            .parent()
            .map(canonical_or_original)
            .context("run root has no parent")?;
        if actual_parent != expected_parent {
            bail!(
                "weights path {} is outside runs root {}",
                path.display(),
                runs_path.as_ref().display()
            );
        }
        Ok(run)
    }

    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let gens = root.join("gens");
        let weights = root.join("weights");
        let log_file = root.join("training.log");

        if !root.is_dir() {
            bail!("run root does not exist: {}", root.display());
        }
        if !gens.is_dir() {
            bail!("gens dir does not exist: {}", gens.display());
        }
        if !weights.is_dir() {
            bail!("weights dir does not exist: {}", weights.display());
        }

        Ok(Self {
            root,
            gens,
            weights,
            log_file,
        })
    }

    pub fn named(runs_path: impl AsRef<Path>, name: &str) -> Result<Self> {
        validate_run_name(name)?;
        Self::open(runs_path.as_ref().join(name))
    }

    pub fn select(runs_path: impl AsRef<Path>, name: &str) -> Result<Self> {
        if name == "latest" {
            let runs_path = runs_path.as_ref();
            return Self::latest(runs_path.to_string_lossy().as_ref());
        }
        Self::named(runs_path, name)
    }

    pub fn activate(&self, runs_path: impl AsRef<Path>) -> Result<()> {
        let runs = runs_path.as_ref();
        let run_parent = self
            .root
            .parent()
            .context("run root has no parent directory")?;
        if canonical_or_original(run_parent) != canonical_or_original(runs) {
            bail!(
                "run {} is not contained by runs root {}",
                self.root.display(),
                runs.display()
            );
        }
        let target = self
            .root
            .file_name()
            .context("run root has no directory name")?;
        fs::create_dir_all(runs)?;
        let latest = runs.join("latest");
        let temporary = runs.join(format!(".latest.tmp-{}", std::process::id()));
        if let Err(error) = fs::remove_file(&temporary) {
            if error.kind() != std::io::ErrorKind::NotFound {
                return Err(error).context("failed cleaning stale latest symlink temporary");
            }
        }
        symlink(target, &temporary).context("failed creating latest symlink temporary")?;
        if let Err(error) = fs::rename(&temporary, &latest) {
            let _ = fs::remove_file(&temporary);
            return Err(error).context("failed atomically activating run");
        }
        Ok(())
    }

    /// Scan runs newest-to-oldest, return the first that contains `filename` in its weights dir.
    pub fn find_with_weights(runs_path: &str, filename: &str) -> Option<(Self, PathBuf)> {
        let runs = Path::new(runs_path);
        let mut dirs: Vec<_> = fs::read_dir(runs)
            .ok()?
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().map_or(false, |ft| ft.is_dir()))
            .collect();
        sort_run_entries_newest_first(&mut dirs);

        for entry in dirs {
            let root = entry.path();
            let weights_file = root.join("weights").join(filename);
            if weights_file.exists() {
                let gens = root.join("gens");
                let weights = root.join("weights");
                let log_file = root.join("training.log");
                let run_dir = Self {
                    root,
                    gens,
                    weights,
                    log_file,
                };
                return Some((run_dir, weights_file));
            }
        }
        None
    }

    /// Scan runs newest-to-oldest, return the first whose gens dir is non-empty.
    pub fn latest_with_data(runs_path: &str) -> Option<Self> {
        if let Ok(run) = Self::latest(runs_path) {
            if has_generation_data(&run.gens) {
                return Some(run);
            }
        }

        let runs = Path::new(runs_path);
        let mut dirs: Vec<_> = fs::read_dir(runs)
            .ok()?
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().map_or(false, |ft| ft.is_dir()))
            .collect();
        sort_run_entries_newest_first(&mut dirs);

        for entry in dirs {
            let root = entry.path();
            let gens = root.join("gens");
            if has_generation_data(&gens) {
                let weights = root.join("weights");
                let log_file = root.join("training.log");
                return Some(Self {
                    root,
                    gens,
                    weights,
                    log_file,
                });
            }
        }
        None
    }

    pub fn latest(runs_path: &str) -> Result<Self> {
        let latest = Path::new(runs_path).join("latest");
        let target = fs::read_link(&latest).context("failed to read latest symlink")?;

        // Resolve relative symlink against runs_path
        let root = if target.is_relative() {
            Path::new(runs_path).join(&target)
        } else {
            target
        };

        Self::open(root).context("latest run is invalid")
    }
}

/// Pretty JSON with a trailing newline, so `meta.json` stays the readable, diffable,
/// `cat`-able file it has always been now that serde owns its shape.
fn meta_bytes(meta: &RunMeta) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec_pretty(meta).context("failed to encode run metadata")?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn canonical_or_original(path: &Path) -> PathBuf {
    fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

fn validate_run_name(name: &str) -> Result<()> {
    let path = Path::new(name);
    if name.is_empty()
        || name == "latest"
        || name.starts_with('.')
        || path.components().count() != 1
        || path.file_name().is_none()
    {
        bail!("run name must be one safe, non-reserved path component");
    }
    Ok(())
}

fn sort_run_entries_newest_first(entries: &mut [fs::DirEntry]) {
    entries.sort_by(|a, b| {
        let key = |entry: &fs::DirEntry| {
            let name = entry.file_name().to_string_lossy().to_string();
            let activity = newest_run_activity(&entry.path());
            (activity, name)
        };

        key(b).cmp(&key(a))
    });
}

fn newest_run_activity(path: &Path) -> Option<SystemTime> {
    let mut latest = fs::metadata(path).ok()?.modified().ok();

    for child in ["training.log", "gens", "weights"] {
        let modified = match fs::metadata(path.join(child))
            .ok()
            .and_then(|metadata| metadata.modified().ok())
        {
            Some(modified) => modified,
            None => continue,
        };
        latest = Some(latest.map_or(modified, |current| current.max(modified)));
    }

    latest
}

fn has_generation_data(gens: &Path) -> bool {
    fs::read_dir(gens)
        .ok()
        .map(|mut entries| entries.next().is_some())
        .unwrap_or(false)
}

fn is_prepared_empty_run(run_dir: &RunDir) -> Result<bool> {
    if !run_dir.root.is_dir() || !run_dir.gens.is_dir() || !run_dir.weights.is_dir() {
        return Ok(false);
    }
    if has_generation_data(&run_dir.gens) || has_generation_data(&run_dir.weights) {
        return Ok(false);
    }

    for entry in fs::read_dir(&run_dir.root)
        .with_context(|| format!("failed to read run dir {}", run_dir.root.display()))?
    {
        let name = entry?.file_name();
        let name = name.to_string_lossy();
        if !matches!(
            name.as_ref(),
            "gens" | "weights" | "meta.json" | "training.log"
        ) {
            return Ok(false);
        }
    }

    Ok(run_dir.root.join("meta.json").is_file())
}

fn current_git_commit() -> Option<String> {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent()?;
    let output = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(repo_root)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout)
        .ok()
        .map(|sha| sha.trim().to_string())
        .filter(|sha| !sha.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_runs() -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("trading-bot-runs-{}-{unique}", std::process::id()))
    }

    #[test]
    fn explicit_run_context_is_stable_and_latest_is_only_changed_by_activation() {
        let runs = temp_runs();
        let runs_str = runs.to_str().unwrap();
        let first = RunDir::create_fresh(runs_str, Some("first")).unwrap();
        let second = RunDir::create_fresh(runs_str, Some("second")).unwrap();
        assert_eq!(RunDir::latest(runs_str).unwrap(), second);
        assert_eq!(RunDir::select(&runs, "latest").unwrap(), second);
        assert_eq!(RunDir::named(&runs, "first").unwrap(), first);
        assert_eq!(RunDir::latest(runs_str).unwrap().root, second.root);

        first.activate(&runs).unwrap();
        assert_eq!(RunDir::latest(runs_str).unwrap().root, first.root);
        fs::remove_dir_all(runs).unwrap();
    }

    #[test]
    fn run_names_cannot_escape_or_replace_latest() {
        let runs = temp_runs();
        for name in ["../escape", ".", ".hidden", "latest", "a/b", ""] {
            assert!(RunDir::create_fresh(runs.to_str().unwrap(), Some(name)).is_err());
        }

        let external_runs = temp_runs().with_extension("external");
        let external =
            RunDir::create_fresh(external_runs.to_str().unwrap(), Some("source")).unwrap();
        assert!(RunDir::from_weights_path_in(&external.weights.join("ppo_ep1.ot"), &runs).is_err());
        fs::remove_dir_all(external_runs).unwrap();
    }

    fn provenance() -> RunProvenance {
        RunProvenance {
            split_bounds_ms: [1_759_839_000_000, 1_773_427_500_000],
            split_bounds_pinned: true,
            resolution_secs: 300,
            corpus_fingerprint: "5297:368222980:deadbeef".to_owned(),
            min_bars: 20_480,
            min_dollar_volume: 0.0,
            data_dir: "long_data/bars".to_owned(),
            diagnostic_context_bars: 896,
            deployed_context_bars: 2048,
            eval_window_seed: 0xE7A1_5E7D_0001,
            train_seed: 0x5EED,
        }
    }

    /// A fresh run states its commit and states that it has recorded no provenance YET, and
    /// what is then recorded reads back byte-for-byte equal to what the run was handed.
    ///
    /// The equality is the whole point. A record that merely EXISTS converts nothing: the claim
    /// being made is "these are the instants this run was handed", and only a round-trip against
    /// the handed value can support it.
    #[test]
    fn a_run_records_the_provenance_it_was_handed_and_reads_it_back_unchanged() {
        let runs = temp_runs();
        let run = RunDir::create_fresh(runs.to_str().unwrap(), Some("recorded")).unwrap();

        let fresh = run.meta().unwrap();
        assert!(
            fresh.provenance.is_none(),
            "a run that has recorded nothing must say so rather than imply a default"
        );

        let handed = provenance();
        run.record_provenance(handed.clone()).unwrap();
        let read_back = run.meta().unwrap();
        assert_eq!(read_back.provenance.as_ref(), Some(&handed));
        assert_eq!(
            read_back.commit, fresh.commit,
            "recording provenance must not disturb the run's identity"
        );

        // The instants specifically, spelled out: this is the field whose absence made
        // `bardist_v3_rfirst_1ep` unverifiable, and a reader must find it without deserializing
        // into this crate's types.
        let raw = fs::read_to_string(run.root.join("meta.json")).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(parsed["provenance"]["split_bounds_ms"][0], 1_759_839_000_000i64);
        assert_eq!(parsed["provenance"]["split_bounds_ms"][1], 1_773_427_500_000i64);
        assert_eq!(parsed["provenance"]["diagnostic_context_bars"], 896);
        assert_eq!(parsed["provenance"]["split_bounds_pinned"], true);

        fs::remove_dir_all(runs).unwrap();
    }

    /// Every run in `training/runs` predates the record, so `meta()` must read a bare
    /// `{"commit": ...}` rather than refuse it. A provenance field that made 700-odd historical
    /// runs unreadable would be a regression dressed as an improvement.
    #[test]
    fn a_run_written_before_the_record_existed_still_reads() {
        let runs = temp_runs();
        let run = RunDir::create_fresh(runs.to_str().unwrap(), Some("legacy")).unwrap();
        fs::write(
            run.root.join("meta.json"),
            "{\n  \"commit\": \"a0ff3b29330493c315a8afc4514d17d7d6b5995c\"\n}\n",
        )
        .unwrap();

        let meta = run.meta().unwrap();
        assert_eq!(meta.commit, "a0ff3b29330493c315a8afc4514d17d7d6b5995c");
        assert_eq!(meta.provenance, None);

        // And it can be upgraded in place without losing the commit it already carried.
        run.record_provenance(provenance()).unwrap();
        let upgraded = run.meta().unwrap();
        assert_eq!(upgraded.commit, meta.commit);
        assert_eq!(upgraded.provenance, Some(provenance()));

        fs::remove_dir_all(runs).unwrap();
    }
}
