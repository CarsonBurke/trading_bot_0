use std::{
    fs::{self, File},
    io::{BufReader, BufWriter},
    path::{Path, PathBuf},
};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use tch::nn;

use super::{
    PLANNER_HEADS, PLANNER_LATENT_DIM, PLANNER_LAYERS, PLANNER_MODEL_DIM, PLANNER_OHLC_DIM,
    PLANNER_PORTFOLIO_DIM,
};

const FORMAT_VERSION: u32 = 1;
const ARCHITECTURE: &str = "world-model-planner-bidirectional-pma-v1";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlannerCheckpointMetadata {
    pub format_version: u32,
    pub architecture: String,
    pub world_model_sha256: String,
    pub horizon: usize,
    pub context_bars: usize,
    pub model_dim: i64,
    pub layers: usize,
    pub heads: i64,
    pub latent_dim: i64,
    pub ohlc_dim: i64,
    pub portfolio_dim: i64,
    pub optimizer_steps: u64,
}

impl PlannerCheckpointMetadata {
    pub fn new(
        world_model_sha256: impl Into<String>,
        horizon: usize,
        context_bars: usize,
        optimizer_steps: u64,
    ) -> Self {
        Self {
            format_version: FORMAT_VERSION,
            architecture: ARCHITECTURE.to_owned(),
            world_model_sha256: world_model_sha256.into(),
            horizon,
            context_bars,
            model_dim: PLANNER_MODEL_DIM,
            layers: PLANNER_LAYERS,
            heads: PLANNER_HEADS,
            latent_dim: PLANNER_LATENT_DIM,
            ohlc_dim: PLANNER_OHLC_DIM,
            portfolio_dim: PLANNER_PORTFOLIO_DIM,
            optimizer_steps,
        }
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        serde_json::from_reader(BufReader::new(File::open(path).with_context(|| {
            format!("failed opening planner metadata {}", path.display())
        })?))
        .with_context(|| format!("failed parsing planner metadata {}", path.display()))
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        self.validate_schema()?;
        let path = path.as_ref();
        if let Some(parent) = path.parent().filter(|path| !path.as_os_str().is_empty()) {
            fs::create_dir_all(parent).with_context(|| {
                format!(
                    "failed creating planner checkpoint directory {}",
                    parent.display()
                )
            })?;
        }
        serde_json::to_writer_pretty(
            BufWriter::new(
                File::create(path).with_context(|| {
                    format!("failed creating planner metadata {}", path.display())
                })?,
            ),
            self,
        )
        .with_context(|| format!("failed writing planner metadata {}", path.display()))
    }

    pub fn validate(
        &self,
        world_model_sha256: &str,
        expected_horizon: Option<usize>,
    ) -> Result<()> {
        self.validate_schema()?;
        if self.world_model_sha256 != world_model_sha256 {
            bail!(
                "planner/world-model mismatch: planner requires {}, loaded {}",
                self.world_model_sha256,
                world_model_sha256
            );
        }
        if let Some(horizon) = expected_horizon {
            if self.horizon != horizon {
                bail!(
                    "planner horizon mismatch: checkpoint={}, requested={horizon}",
                    self.horizon
                );
            }
        }
        Ok(())
    }

    fn validate_schema(&self) -> Result<()> {
        if self.format_version != FORMAT_VERSION || self.architecture != ARCHITECTURE {
            bail!("unsupported planner checkpoint metadata");
        }
        let expected = Self::new(
            self.world_model_sha256.clone(),
            self.horizon,
            self.context_bars,
            self.optimizer_steps,
        );
        if self.model_dim != expected.model_dim
            || self.layers != expected.layers
            || self.heads != expected.heads
            || self.latent_dim != expected.latent_dim
            || self.ohlc_dim != expected.ohlc_dim
            || self.portfolio_dim != expected.portfolio_dim
        {
            bail!("planner checkpoint architecture dimensions are incompatible");
        }
        if self.horizon == 0 || self.context_bars == 0 || self.world_model_sha256.is_empty() {
            bail!("planner checkpoint metadata contains empty required fields");
        }
        Ok(())
    }
}

pub fn planner_metadata_path(checkpoint: impl AsRef<Path>) -> PathBuf {
    checkpoint.as_ref().with_extension("metadata.json")
}

pub fn save_planner_checkpoint(
    var_store: &nn::VarStore,
    checkpoint: impl AsRef<Path>,
    metadata: &PlannerCheckpointMetadata,
) -> Result<()> {
    let checkpoint = checkpoint.as_ref();
    if let Some(parent) = checkpoint
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed creating planner checkpoint directory {}",
                parent.display()
            )
        })?;
    }
    var_store
        .save(checkpoint)
        .with_context(|| format!("failed saving planner weights {}", checkpoint.display()))?;
    metadata.save(planner_metadata_path(checkpoint))
}

pub fn load_planner_checkpoint(
    var_store: &mut nn::VarStore,
    checkpoint: impl AsRef<Path>,
    world_model_sha256: &str,
    expected_horizon: Option<usize>,
) -> Result<PlannerCheckpointMetadata> {
    let checkpoint = checkpoint.as_ref();
    let metadata = PlannerCheckpointMetadata::load(planner_metadata_path(checkpoint))?;
    metadata.validate(world_model_sha256, expected_horizon)?;
    var_store
        .load(checkpoint)
        .with_context(|| format!("failed loading planner weights {}", checkpoint.display()))?;
    Ok(metadata)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_rejects_wrong_world_model() {
        let metadata = PlannerCheckpointMetadata::new("wm-a", 100, 6_000, 10);
        assert!(metadata.validate("wm-b", Some(100)).is_err());
        assert!(metadata.validate("wm-a", Some(50)).is_err());
        metadata.validate("wm-a", Some(100)).unwrap();
    }

    #[test]
    fn metadata_path_tracks_checkpoint() {
        assert_eq!(
            planner_metadata_path("weights/planner.ot"),
            PathBuf::from("weights/planner.metadata.json")
        );
    }
}
