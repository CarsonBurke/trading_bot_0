use tch::{nn, Tensor};
use std::path::Path;

use crate::torch::constants::TICKERS_COUNT;
use crate::torch::model::model;

/// Load a trained PPO model from a weights file
///
/// # Arguments
/// * `weight_path` - Path to the saved model weights (e.g., "weights/ppo_ep100.ot")
/// * `device` - Device to load the model on (CPU or CUDA)
///
/// # Returns
/// * VarStore with loaded weights and the model function
pub fn load_model<P: AsRef<Path>>(
    weight_path: P,
    device: tch::Device,
) -> Result<(nn::VarStore, Box<dyn Fn(&Tensor, &Tensor, bool) -> (Tensor, (Tensor, Tensor))>), Box<dyn std::error::Error>> {
    let mut vs = nn::VarStore::new(device);
    let model = model(&vs.root(), TICKERS_COUNT);

    vs.load(weight_path)?;

    println!("Model loaded successfully");
    println!("Device: {:?}", device);

    Ok((vs, model))
}

/// Run inference with a loaded model
///
/// # Arguments
/// * `model` - The loaded model function
/// * `price_deltas` - Price deltas tensor [batch, TICKERS_COUNT * PRICE_DELTAS_PER_TICKER]
/// * `static_obs` - Static observations tensor [batch, STATIC_OBSERVATIONS]
///
/// # Returns
/// * (critic_value, (action_mean, action_log_std))
pub fn infer(
    model: &dyn Fn(&Tensor, &Tensor, bool) -> (Tensor, (Tensor, Tensor)),
    price_deltas: &Tensor,
    static_obs: &Tensor,
) -> (Tensor, (Tensor, Tensor)) {
    tch::no_grad(|| model(price_deltas, static_obs, false))  // eval mode for inference
}

/// Sample actions from the model (for inference/evaluation)
///
/// # Arguments
/// * `model` - The loaded model function
/// * `price_deltas` - Price deltas tensor [batch, TICKERS_COUNT * PRICE_DELTAS_PER_TICKER]
/// * `static_obs` - Static observations tensor [batch, STATIC_OBSERVATIONS]
/// * `deterministic` - If true, use mean action; if false, sample with noise
///
/// # Returns
/// * Action tensor [batch, TICKERS_COUNT] in range approximately [-1, 1]
pub fn sample_actions(
    model: &dyn Fn(&Tensor, &Tensor, bool) -> (Tensor, (Tensor, Tensor)),
    price_deltas: &Tensor,
    static_obs: &Tensor,
    deterministic: bool,
) -> Tensor {
    let (_critic, (action_mean, action_log_std)) = tch::no_grad(|| {
        model(price_deltas, static_obs, false)  // eval mode for inference
    });

    if deterministic {
        // Use mean action (no exploration)
        (action_mean / 2.0).sigmoid() * 2.0 - 1.0
    } else {
        // Sample with noise (exploration)
        let action_std = action_log_std.exp();
        let noise = Tensor::randn_like(&action_mean);
        let z = action_mean + action_std * noise;
        (z / 2.0).sigmoid() * 2.0 - 1.0
    }
}
