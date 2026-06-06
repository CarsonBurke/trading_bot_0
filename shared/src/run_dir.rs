use anyhow::{bail, Context, Result};
use chrono::Local;
use std::fs;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

pub struct RunDir {
    pub root: PathBuf,
    pub gens: PathBuf,
    pub weights: PathBuf,
    pub log_file: PathBuf,
}

impl RunDir {
    pub fn create_fresh(runs_path: &str, name: Option<&str>) -> Result<Self> {
        let dir_name = match name {
            Some(n) => n.to_string(),
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
                return Ok(run_dir);
            }
            bail!("run dir already exists: {}", root.display());
        }
        fs::create_dir(&root).context("failed to create run dir")?;
        fs::create_dir_all(&gens)?;
        fs::create_dir_all(&weights)?;
        fs::write(
            root.join("meta.json"),
            format!(
                "{{\n  \"commit\": \"{}\"\n}}\n",
                current_git_commit().unwrap_or_default()
            ),
        )
        .context("failed to write run metadata")?;

        // Atomically update latest symlink (relative target)
        let latest = runs.join("latest");
        let _ = fs::remove_file(&latest);
        symlink(&dir_name, &latest).context("failed to create latest symlink")?;

        Ok(Self {
            root,
            gens,
            weights,
            log_file,
        })
    }

    pub fn from_weights_path(path: &Path) -> Result<Self> {
        // path: runs/{name}/weights/{file}.ot
        let weights_dir = path.parent().context("weights path has no parent")?;
        let root = weights_dir.parent().context("weights dir has no parent")?;

        let root = root.to_path_buf();
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

        let gens = root.join("gens");
        let weights = root.join("weights");
        let log_file = root.join("training.log");

        if !root.is_dir() {
            bail!("latest run dir does not exist: {}", root.display());
        }

        Ok(Self {
            root,
            gens,
            weights,
            log_file,
        })
    }
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
