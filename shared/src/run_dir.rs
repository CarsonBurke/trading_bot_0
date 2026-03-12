use anyhow::{bail, Context, Result};
use chrono::Local;
use std::fs;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};

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
            None => Local::now().format("%Y-%m-%d_%H-%M-%S").to_string(),
        };

        let runs = Path::new(runs_path);
        fs::create_dir_all(runs)?;

        let root = runs.join(&dir_name);
        let gens = root.join("gens");
        let weights = root.join("weights");
        let log_file = root.join("training.log");

        fs::create_dir_all(&gens)?;
        fs::create_dir_all(&weights)?;

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
