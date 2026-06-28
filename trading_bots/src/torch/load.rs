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
    pub migrated_legacy_value_head: Vec<String>,
}

fn format_shape_mismatch(mismatch: &ShapeMismatch) -> String {
    format!(
        "{} {:?} -> {:?}",
        mismatch.name, mismatch.checkpoint_shape, mismatch.model_shape
    )
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
            .map(format_shape_mismatch)
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

fn is_value_head_name(name: &str) -> bool {
    name == "value_proj.weight"
}

fn is_legacy_scalar_value_head_shape(
    name: &str,
    checkpoint_shape: &[i64],
    model_shape: &[i64],
) -> bool {
    match name {
        "value_proj.weight" => {
            checkpoint_shape.len() == 2
                && model_shape.len() == 2
                && checkpoint_shape[0] == 1
                && checkpoint_shape[1] == model_shape[1]
                && model_shape[0] > 1
        }
        _ => false,
    }
}

fn has_renamed_legacy_value_head(checkpoint_tensors: &HashMap<String, Tensor>) -> bool {
    checkpoint_tensors.keys().any(|name| {
        matches!(
            name.as_str(),
            "value_out.weight" | "value_out.bias" | "critic_out.weight" | "critic_out.bias"
        ) || name.starts_with("value_mlp")
    })
}

pub fn load_var_store_partial<P: AsRef<Path>>(
    vs: &mut nn::VarStore,
    path: P,
) -> Result<LoadSummary, Box<dyn Error>> {
    let checkpoint_tensors = named_tensors(&path, vs.device())?;
    let mut loaded = 0usize;
    let mut missing = Vec::new();
    let mut shape_mismatches = Vec::new();
    let mut migrated_legacy_value_head = Vec::new();
    let has_renamed_legacy_value_head = has_renamed_legacy_value_head(&checkpoint_tensors);
    let has_stale_value_proj_bias = checkpoint_tensors.contains_key("value_proj.bias");
    for (name, mut var) in vs.variables() {
        match checkpoint_tensors.get(&name) {
            Some(src)
                if name == "value_proj.weight"
                    && has_stale_value_proj_bias
                    && src.size() == var.size() =>
            {
                migrated_legacy_value_head.push(name);
            }
            Some(src) if src.size() == var.size() => {
                no_grad(|| {
                    var.set_data(&var.to_kind(src.kind()));
                    var.f_copy_(src)
                })?;
                loaded += 1;
            }
            Some(src) if is_legacy_scalar_value_head_shape(&name, &src.size(), &var.size()) => {
                migrated_legacy_value_head.push(name);
            }
            Some(src) => shape_mismatches.push(ShapeMismatch {
                name,
                checkpoint_shape: src.size(),
                model_shape: var.size(),
            }),
            None if has_renamed_legacy_value_head && is_value_head_name(&name) => {
                migrated_legacy_value_head.push(name);
            }
            None => missing.push(name),
        }
    }
    if has_stale_value_proj_bias {
        migrated_legacy_value_head.push("value_proj.bias".to_string());
    }
    println!(
        "Loaded weights: {} tensors copied, {} missing, {} shape-mismatched, {} legacy value-head migrated",
        loaded,
        missing.len(),
        shape_mismatches.len(),
        migrated_legacy_value_head.len()
    );
    if !shape_mismatches.is_empty() {
        let preview = shape_mismatches
            .iter()
            .take(8)
            .map(format_shape_mismatch)
            .collect::<Vec<_>>()
            .join(", ");
        println!("Skipped shape-mismatched tensors: {}", preview);
    }
    if !migrated_legacy_value_head.is_empty() {
        println!(
            "Skipped legacy value-head tensors: {}",
            migrated_legacy_value_head.join(", ")
        );
    }
    Ok(LoadSummary {
        loaded,
        missing,
        shape_mismatches,
        migrated_legacy_value_head,
    })
}
