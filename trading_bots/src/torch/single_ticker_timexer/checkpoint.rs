use std::{
    collections::HashMap,
    fs,
    io::{Cursor, Read, Write},
    path::{Path, PathBuf},
};

use anyhow::{ensure, Context, Result};
use ring::digest::{digest, SHA256};
use serde::{Deserialize, Serialize};
use tch::{nn, Device, Tensor};

use super::{
    data::{Dataset, CONTEXT, DATA_CONTRACT, FEATURES, HORIZONS},
    model::ModelKind,
    support::Supports,
};
use crate::torch::hashing::file_sha256;

pub const FORMAT: &str = "single-ticker-direct-return-v1";
pub const PATCH_LAYOUT: [[usize; 3]; 3] = [[0, 1408, 32], [1408, 512, 8], [1920, 128, 1]];
pub const OBJECTIVE: &str =
    "mean-horizon-hard-categorical-nll-global-mean-one-100-bar-uniqueness-v1";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DataContract {
    pub ticker: String,
    pub corpus_sha256: String,
    pub source_bars: usize,
    pub split_bounds: [i64; 3],
    pub purge_bars: usize,
    pub context: usize,
    pub resolution_seconds: usize,
    pub patch_layout: [[usize; 3]; 3],
    pub features: Vec<String>,
    pub horizons: Vec<usize>,
    pub preprocessing: String,
    pub supports: Supports,
}

impl DataContract {
    pub fn from_dataset(dataset: &Dataset) -> Self {
        Self {
            ticker: dataset.ticker.clone(),
            corpus_sha256: dataset.fingerprint.clone(),
            source_bars: dataset.source_bars,
            split_bounds: dataset.split_bounds,
            purge_bars: 100,
            context: CONTEXT,
            resolution_seconds: 300,
            patch_layout: PATCH_LAYOUT,
            features: FEATURES.iter().map(|s| s.to_string()).collect(),
            horizons: HORIZONS.to_vec(),
            preprocessing: DATA_CONTRACT.to_owned(),
            supports: dataset.supports.clone(),
        }
    }

    pub fn sha256(&self) -> Result<String> {
        hash_serialized(self)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub format: String,
    pub architecture: String,
    pub model_kind: ModelKind,
    pub data: DataContract,
    pub objective: String,
    pub seed: u64,
    pub epoch: usize,
    pub step: usize,
    pub planned_epochs: usize,
    pub selection_epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub optimizer: String,
    pub weights_sha256: String,
    pub frozen: bool,
    pub manifest_sha256: String,
}

impl Manifest {
    pub fn seal(&mut self) -> Result<()> {
        self.manifest_sha256.clear();
        self.manifest_sha256 = hash_serialized(self)?;
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.format == FORMAT,
            "unsupported TimeXer checkpoint format"
        );
        ensure!(
            self.architecture == super::model::architecture_contract(self.model_kind),
            "checkpoint architecture differs from this executable"
        );
        ensure!(
            self.objective == objective(self.model_kind),
            "checkpoint objective differs"
        );
        let mut unsealed = self.clone();
        unsealed.manifest_sha256.clear();
        ensure!(
            hash_serialized(&unsealed)? == self.manifest_sha256,
            "checkpoint manifest authentication failed"
        );
        ensure!(
            self.data.context == CONTEXT
                && self.data.resolution_seconds == 300
                && self.data.purge_bars == 100
                && self.data.patch_layout == PATCH_LAYOUT
                && self.data.horizons == HORIZONS
                && self.data.preprocessing == DATA_CONTRACT
                && self.data.features == FEATURES,
            "checkpoint preprocessing contract differs"
        );
        self.data.supports.validate()?;
        ensure!(
            self.batch_size > 0
                && self.learning_rate.is_finite()
                && self.learning_rate > 0.0
                && self.epoch > 0
                && self.epoch <= self.planned_epochs
                && self.epoch.checked_mul(
                    self.data
                        .supports
                        .training_origin_count
                        .div_ceil(self.batch_size)
                ) == Some(self.step),
            "checkpoint training exposure is invalid"
        );
        ensure!(
            self.selection_epochs == 0 || self.selection_epochs == self.planned_epochs,
            "checkpoint selection did not complete the planned training protocol"
        );
        ensure!(
            !self.frozen || self.selection_epochs == self.planned_epochs,
            "frozen checkpoint has an incomplete selection protocol"
        );
        Ok(())
    }

    pub fn authenticate_dataset(&self, dataset: &Dataset) -> Result<()> {
        ensure!(
            self.data == DataContract::from_dataset(dataset),
            "checkpoint ticker, corpus, supports, or partition differs from input data"
        );
        Ok(())
    }
}

pub fn objective(kind: ModelKind) -> &'static str {
    if kind == ModelKind::RawTimeXer {
        "raw-timexer-point-mse-comparison-only-v1"
    } else {
        OBJECTIVE
    }
}

pub fn hash_serialized(value: &impl Serialize) -> Result<String> {
    Ok(digest(&SHA256, &serde_json::to_vec(value)?)
        .as_ref()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

pub fn read(path: &Path) -> Result<Manifest> {
    let manifest: Manifest = serde_json::from_slice(
        &fs::read(path.join("manifest.json"))
            .with_context(|| format!("read checkpoint {}", path.display()))?,
    )?;
    manifest.validate()?;
    ensure!(
        file_sha256(path.join("weights.ot"))? == manifest.weights_sha256,
        "checkpoint weights authentication failed"
    );
    Ok(manifest)
}

pub fn load_weights(path: &Path, store: &mut nn::VarStore) -> Result<()> {
    let manifest = read(path)?;
    let bytes = fs::read(path.join("weights.ot"))?;
    let sha256: String = digest(&SHA256, &bytes)
        .as_ref()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    ensure!(
        sha256 == manifest.weights_sha256,
        "checkpoint changed while loading"
    );
    let tensors = load_tensors(bytes, store.device())?;
    let mut variables = store.variables();
    ensure!(
        variables.len() == tensors.len(),
        "checkpoint parameter count differs"
    );
    for (name, destination) in &variables {
        let source = tensors
            .get(name)
            .with_context(|| format!("missing checkpoint parameter {name}"))?;
        ensure!(
            source.size() == destination.size() && source.kind() == destination.kind(),
            "checkpoint parameter shape or dtype differs: {name}"
        );
    }
    tch::no_grad(|| -> Result<()> {
        for (name, destination) in &mut variables {
            destination.f_copy_(&tensors[name])?;
        }
        Ok(())
    })
}

fn load_tensors(bytes: Vec<u8>, device: Device) -> Result<HashMap<String, Tensor>> {
    let mut archive = zip::ZipArchive::new(Cursor::new(bytes.as_slice()))?;
    let code_name = archive
        .file_names()
        .find(|name| name.ends_with("/code/__torch__.py"))
        .map(str::to_owned);
    let repaired = if let Some(code_name) = code_name {
        let mut code = String::new();
        archive.by_name(&code_name)?.read_to_string(&mut code)?;
        // Older libtorch archives emitted a Python keyword as a bare attribute annotation.
        if code.lines().any(|line| line == "  global : Tensor") {
            let code = code.replace(
                "  global : Tensor\n",
                "  __annotations__[\"global\"] = Tensor\n",
            );
            let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
            for i in 0..archive.len() {
                let entry = archive.by_index(i)?;
                if entry.name() == code_name {
                    writer.start_file(
                        &code_name,
                        zip::write::FileOptions::default().compression_method(entry.compression()),
                    )?;
                    writer.write_all(code.as_bytes())?;
                } else {
                    writer.raw_copy_file(entry)?;
                }
            }
            Some(writer.finish()?.into_inner())
        } else {
            None
        }
    } else {
        None
    };
    drop(archive);
    let tensors =
        Tensor::load_multi_from_stream_with_device(Cursor::new(repaired.unwrap_or(bytes)), device)?;
    let mut result = HashMap::with_capacity(tensors.len());
    for (name, tensor) in tensors {
        let name = if name == "global" {
            "global_token".to_owned()
        } else {
            name
        };
        ensure!(
            result.insert(name.clone(), tensor).is_none(),
            "duplicate checkpoint parameter: {name}"
        );
    }
    Ok(result)
}

pub fn save(path: &Path, store: &nn::VarStore, mut manifest: Manifest) -> Result<Manifest> {
    ensure!(
        !path.exists(),
        "checkpoint already exists: {}",
        path.display()
    );
    let parent = path.parent().context("checkpoint needs parent directory")?;
    fs::create_dir_all(parent)?;
    let staging = parent.join(format!(".timexer-{}", uuid::Uuid::new_v4()));
    fs::create_dir(&staging)?;
    let result = (|| {
        store.save(staging.join("weights.ot"))?;
        manifest.weights_sha256 = file_sha256(staging.join("weights.ot"))?;
        manifest.seal()?;
        manifest.validate()?;
        fs::write(
            staging.join("manifest.json"),
            serde_json::to_vec_pretty(&manifest)?,
        )?;
        fs::rename(&staging, path)?;
        Ok(manifest)
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(staging);
    }
    result
}

pub fn freeze(source: &Path, destination: &Path) -> Result<()> {
    let mut manifest = read(source)?;
    ensure!(
        manifest.selection_epochs == manifest.planned_epochs && manifest.selection_epochs > 0,
        "freeze requires the selected checkpoint from a completed training protocol"
    );
    ensure!(!destination.exists(), "freeze destination already exists");
    let parent = destination
        .parent()
        .context("freeze destination needs parent")?;
    fs::create_dir_all(parent)?;
    let staging = parent.join(format!(".timexer-freeze-{}", uuid::Uuid::new_v4()));
    fs::create_dir(&staging)?;
    let result = (|| {
        fs::copy(source.join("weights.ot"), staging.join("weights.ot"))?;
        ensure!(
            file_sha256(staging.join("weights.ot"))? == manifest.weights_sha256,
            "source changed while freezing"
        );
        manifest.frozen = true;
        manifest.seal()?;
        fs::write(
            staging.join("manifest.json"),
            serde_json::to_vec_pretty(&manifest)?,
        )?;
        fs::rename(&staging, destination)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(staging);
    }
    result
}

pub fn update_best(weights_dir: &Path, target: &str) -> Result<PathBuf> {
    ensure!(
        Path::new(target).components().count() == 1
            && matches!(
                Path::new(target).components().next(),
                Some(std::path::Component::Normal(_))
            ),
        "best checkpoint target must be a directory name"
    );
    let temporary = weights_dir.join(format!(".best-{}", uuid::Uuid::new_v4()));
    std::os::unix::fs::symlink(&target, &temporary)?;
    fs::rename(&temporary, weights_dir.join("best"))?;
    Ok(weights_dir.join(target))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind};

    struct Disposable(PathBuf);
    impl Disposable {
        fn new() -> Self {
            let root = std::env::var_os("TMPDIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("/var/tmp"));
            let path = root.join(format!("timexer-checkpoint-test-{}", uuid::Uuid::new_v4()));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Disposable {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn fixture() -> Manifest {
        let targets: Vec<_> = (0..1024)
            .map(|i| std::array::from_fn(|h| ((i as f64 + 0.31).ln() - 5.1) * (h + 1) as f64 / 3.7))
            .collect();
        Manifest {
            format: FORMAT.to_owned(),
            architecture: super::super::model::architecture_contract(
                ModelKind::SingleTickerTimeXer,
            ),
            model_kind: ModelKind::SingleTickerTimeXer,
            data: DataContract {
                ticker: "TEST".into(),
                corpus_sha256: "a".repeat(64),
                source_bars: 10000,
                split_bounds: [7000, 8000, 9000],
                purge_bars: 100,
                context: CONTEXT,
                resolution_seconds: 300,
                patch_layout: PATCH_LAYOUT,
                features: FEATURES.iter().map(|s| s.to_string()).collect(),
                horizons: HORIZONS.to_vec(),
                preprocessing: DATA_CONTRACT.into(),
                supports: Supports::fit(&targets).unwrap(),
            },
            objective: OBJECTIVE.into(),
            seed: 20260904,
            epoch: 1,
            step: 32,
            planned_epochs: 20,
            selection_epochs: 20,
            batch_size: 32,
            learning_rate: 0.0003,
            optimizer: "test".into(),
            weights_sha256: String::new(),
            frozen: false,
            manifest_sha256: String::new(),
        }
    }

    #[test]
    fn complete_model_weights_roundtrip_and_legacy_keyword_recovery() {
        tch::set_num_threads(1);
        let temporary = Disposable::new();
        for kind in [ModelKind::SingleTickerTimeXer, ModelKind::RawTimeXer] {
            let source = nn::VarStore::new(Device::Cpu);
            let _model = super::super::model::ForecastModel::new(&source.root(), kind);
            let mut target = nn::VarStore::new(Device::Cpu);
            let _model = super::super::model::ForecastModel::new(&target.root(), kind);
            let mut manifest = fixture();
            manifest.model_kind = kind;
            manifest.architecture = super::super::model::architecture_contract(kind);
            manifest.objective = objective(kind).to_owned();
            let checkpoint = temporary.0.join(format!("{kind:?}"));
            save(&checkpoint, &source, manifest).unwrap();
            // New archives must also remain directly readable by libtorch.
            target.load(checkpoint.join("weights.ot")).unwrap();
            load_weights(&checkpoint, &mut target).unwrap();
            for (name, parameter) in source.variables() {
                assert!(parameter.equal(&target.variables()[&name]), "{name}");
            }
            let legacy: Vec<_> = source
                .variables()
                .into_iter()
                .map(|(name, tensor)| {
                    (
                        if name == "global_token" {
                            "global".to_owned()
                        } else {
                            name
                        },
                        tensor,
                    )
                })
                .collect();
            let mut manifest = read(&checkpoint).unwrap();
            Tensor::save_multi(&legacy, checkpoint.join("weights.ot")).unwrap();
            manifest.weights_sha256 = file_sha256(checkpoint.join("weights.ot")).unwrap();
            manifest.seal().unwrap();
            fs::write(
                checkpoint.join("manifest.json"),
                serde_json::to_vec(&manifest).unwrap(),
            )
            .unwrap();
            let original = fs::read(checkpoint.join("weights.ot")).unwrap();
            assert!(target.load(checkpoint.join("weights.ot")).is_err());
            tch::no_grad(|| {
                for mut parameter in target.variables().into_values() {
                    let _ = parameter.zero_();
                }
            });
            load_weights(&checkpoint, &mut target).unwrap();
            for (name, parameter) in source.variables() {
                assert!(parameter.equal(&target.variables()[&name]), "legacy {name}");
            }
            assert_eq!(original, fs::read(checkpoint.join("weights.ot")).unwrap());
            assert_eq!(
                read(&checkpoint).unwrap().manifest_sha256,
                manifest.manifest_sha256
            );
        }
    }

    #[test]
    #[ignore = "set TIMEXER_CHECKPOINT_TEST to an existing authenticated checkpoint"]
    fn existing_checkpoint_loads_without_modification() {
        tch::set_num_threads(1);
        let path = PathBuf::from(
            std::env::var_os("TIMEXER_CHECKPOINT_TEST").expect("TIMEXER_CHECKPOINT_TEST"),
        );
        let before = read(&path).unwrap();
        let mut store = nn::VarStore::new(Device::Cpu);
        let _model = super::super::model::ForecastModel::new(&store.root(), before.model_kind);
        load_weights(&path, &mut store).unwrap();
        assert_eq!(before.manifest_sha256, read(&path).unwrap().manifest_sha256);
    }

    #[test]
    fn authenticates_weights_metadata_and_frozen_snapshot() {
        let temporary = Disposable::new();
        let checkpoint = temporary.0.join("epoch-0001");
        let store = nn::VarStore::new(Device::Cpu);
        let _weight = store.root().var("fixture", &[3], nn::Init::Const(0.31));
        let saved = save(&checkpoint, &store, fixture()).unwrap();
        let loaded = read(&checkpoint).unwrap();
        assert_eq!(saved.manifest_sha256, loaded.manifest_sha256);
        assert_eq!(saved.data, loaded.data);
        let frozen = temporary.0.join("frozen");
        freeze(&checkpoint, &frozen).unwrap();
        let sealed = read(&frozen).unwrap();
        assert!(sealed.frozen);
        assert_eq!(sealed.weights_sha256, loaded.weights_sha256);
        assert_ne!(sealed.manifest_sha256, loaded.manifest_sha256);
        assert!(save(&checkpoint, &store, fixture()).is_err());
        let mut damaged = loaded;
        damaged.data.horizons[0] = 2;
        fs::write(
            checkpoint.join("manifest.json"),
            serde_json::to_vec(&damaged).unwrap(),
        )
        .unwrap();
        assert!(read(&checkpoint).is_err());
        fs::write(frozen.join("weights.ot"), b"modified").unwrap();
        assert!(read(&frozen).is_err());
        assert_eq!(store.kind(), Kind::Float);
    }

    #[test]
    fn incomplete_training_cannot_unlock_test_and_best_tracks_final_selection() {
        let temporary = Disposable::new();
        let store = nn::VarStore::new(Device::Cpu);
        let _weight = store.root().var("fixture", &[1], nn::Init::Const(0.0));
        let mut intermediate = fixture();
        intermediate.selection_epochs = 0;
        let epoch = temporary.0.join("epoch-0001");
        save(&epoch, &store, intermediate).unwrap();
        update_best(&temporary.0, "epoch-0001").unwrap();
        assert!(freeze(&temporary.0.join("best"), &temporary.0.join("frozen")).is_err());
        assert!(!temporary.0.join("frozen").exists());
        save(&temporary.0.join("selected"), &store, fixture()).unwrap();
        update_best(&temporary.0, "selected").unwrap();
        assert_eq!(
            read(&temporary.0.join("best")).unwrap().selection_epochs,
            20
        );
        freeze(&temporary.0.join("best"), &temporary.0.join("frozen")).unwrap();
        assert!(read(&temporary.0.join("frozen")).unwrap().frozen);
        assert!(update_best(&temporary.0, "../elsewhere").is_err());
    }

    #[test]
    fn changing_support_moments_fails_even_after_manifest_resealing() {
        let mut manifest = fixture();
        manifest.data.supports.horizons[0].bins[0].mean += 0.1;
        manifest.seal().unwrap();
        assert!(manifest.validate().is_err());
    }
}
