use std::collections::HashMap;
use std::error::Error;
use std::io;
use std::path::Path;

use tch::{nn, no_grad, Device, TchError, Tensor};

pub struct ShapeMismatch {
    pub name: String,
    pub checkpoint_shape: Vec<i64>,
    pub model_shape: Vec<i64>,
}

pub struct LoadSummary {
    pub loaded: usize,
    pub missing: Vec<String>,
    pub shape_mismatches: Vec<ShapeMismatch>,
}

impl LoadSummary {
    pub fn require_complete(&self) -> Result<(), Box<dyn Error>> {
        if self.missing.is_empty() && self.shape_mismatches.is_empty() {
            return Ok(());
        }

        let missing_preview = self
            .missing
            .iter()
            .take(8)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        let shape_preview = self
            .shape_mismatches
            .iter()
            .take(8)
            .map(|mismatch| {
                format!(
                    "{} {:?} -> {:?}",
                    mismatch.name, mismatch.checkpoint_shape, mismatch.model_shape
                )
            })
            .collect::<Vec<_>>()
            .join(", ");

        let mut details = Vec::new();
        if !self.missing.is_empty() {
            details.push(format!(
                "{} missing ({})",
                self.missing.len(),
                missing_preview
            ));
        }
        if !self.shape_mismatches.is_empty() {
            details.push(format!(
                "{} shape-mismatched ({})",
                self.shape_mismatches.len(),
                shape_preview
            ));
        }

        Err(io::Error::other(format!(
            "checkpoint is incompatible with current model: {}",
            details.join("; ")
        ))
        .into())
    }
}

fn named_tensors<P: AsRef<Path>>(
    path: P,
    device: Device,
) -> Result<HashMap<String, Tensor>, TchError> {
    let tensors = match path.as_ref().extension().and_then(|ext| ext.to_str()) {
        Some("bin") | Some("pt") => Tensor::loadz_multi_with_device(path, device),
        Some("safetensors") => Tensor::read_safetensors(path),
        Some(_) | None => Tensor::load_multi_with_device(path, device),
    }?;
    Ok(tensors.into_iter().collect())
}

pub fn load_var_store_partial<P: AsRef<Path>>(
    vs: &mut nn::VarStore,
    path: P,
) -> Result<LoadSummary, Box<dyn Error>> {
    let checkpoint_tensors = named_tensors(&path, vs.device())?;
    let mut loaded = 0usize;
    let mut missing = Vec::new();
    let mut shape_mismatches = Vec::new();
    for (name, mut var) in vs.variables() {
        match checkpoint_tensors.get(&name) {
            Some(src) if src.size() == var.size() => {
                no_grad(|| {
                    var.set_data(&var.to_kind(src.kind()));
                    var.f_copy_(src)
                })?;
                loaded += 1;
            }
            Some(src) => shape_mismatches.push(ShapeMismatch {
                name,
                checkpoint_shape: src.size(),
                model_shape: var.size(),
            }),
            None => missing.push(name),
        }
    }
    println!(
        "Loaded weights: {} tensors copied, {} missing, {} shape-mismatched",
        loaded,
        missing.len(),
        shape_mismatches.len()
    );
    if !shape_mismatches.is_empty() {
        let preview = shape_mismatches
            .iter()
            .take(8)
            .map(|mismatch| {
                format!(
                    "{} {:?} -> {:?}",
                    mismatch.name, mismatch.checkpoint_shape, mismatch.model_shape
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        println!("Skipped shape-mismatched tensors: {}", preview);
    }
    Ok(LoadSummary {
        loaded,
        missing,
        shape_mismatches,
    })
}
