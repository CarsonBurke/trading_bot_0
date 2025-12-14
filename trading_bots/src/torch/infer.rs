use tch::{nn, Device, Tensor};
use std::path::Path;
use std::time::Instant;

use crate::torch::constants::ACTION_COUNT;
use crate::torch::model::{model, TradingModel, StreamState};
use crate::torch::env::Env;

pub fn load_model<P: AsRef<Path>>(
    weight_path: P,
    device: tch::Device,
) -> Result<(nn::VarStore, Box<dyn Fn(&Tensor, &Tensor, bool) -> (Tensor, Tensor, (Tensor, Tensor, Tensor), Tensor)>), Box<dyn std::error::Error>> {
    let mut vs = nn::VarStore::new(device);
    let model_fn = model(&vs.root(), ACTION_COUNT);
    vs.load(weight_path)?;
    println!("Model loaded on {:?}", device);
    Ok((vs, model_fn))
}

pub fn sample_actions(
    model: &dyn Fn(&Tensor, &Tensor, bool) -> (Tensor, Tensor, (Tensor, Tensor, Tensor), Tensor),
    price_deltas: &Tensor,
    static_obs: &Tensor,
    deterministic: bool,
    temperature: f64,
) -> Tensor {
    let (_critic, _critic_logits, (action_mean, action_log_std, _divisor), _attn_weights) = tch::no_grad(|| {
        model(price_deltas, static_obs, false)
    });

    // Logistic-normal: model outputs K-1 dims, append 0 before softmax
    let batch_size = action_mean.size()[0];
    let zeros = Tensor::zeros([batch_size, 1], (tch::Kind::Float, action_mean.device()));

    if deterministic || temperature == 0.0 {
        let u_ext = Tensor::cat(&[action_mean, zeros], 1);
        u_ext.softmax(-1, tch::Kind::Float)
    } else {
        let action_std = action_log_std.exp() * temperature;
        let noise = Tensor::randn_like(&action_mean);
        let u = &action_mean + &action_std * noise;
        let u_ext = Tensor::cat(&[u, zeros], 1);
        u_ext.softmax(-1, tch::Kind::Float)
    }
}

pub fn run_inference<P: AsRef<Path>>(
    weight_path: P,
    num_episodes: usize,
    deterministic: bool,
    temperature: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting inference run...");
    println!("Loading model from: {:?}", weight_path.as_ref());

    let device = Device::cuda_if_available();
    println!("Using device: {:?}", device);

    let (_vs, model) = load_model(weight_path, device)?;

    let mut env = Env::new(false);

    let mut total_rewards = 0.0;
    let mut total_episodes = 0;

    for episode in 0..num_episodes {
        let episode_start = Instant::now();
        let mut episode_reward = 0.0;

        let (price_deltas_reset, static_obs_reset) = env.reset();
        let mut current_price_deltas = price_deltas_reset;
        let mut current_static_obs = static_obs_reset.to_device(device);

        env.episode = episode;

        for step in 0..env.max_step {
            env.step = step;

            let price_deltas_gpu = current_price_deltas.to_device(device);
            let actions = sample_actions(
                model.as_ref(),
                &price_deltas_gpu,
                &current_static_obs,
                deterministic,
                temperature,
            );

            let actions_flat = Vec::<f64>::try_from(actions.flatten(0, -1)).unwrap();
            let actions_vec: Vec<Vec<f64>> = actions_flat
                .chunks(ACTION_COUNT as usize)
                .map(|chunk| chunk.to_vec())
                .collect();

            let step_state = env.step(actions_vec);

            let reward = f64::try_from(step_state.reward.sum(tch::Kind::Float)).unwrap();
            episode_reward += reward;

            current_price_deltas = step_state.price_deltas;
            current_static_obs = step_state.static_obs.to_device(device);

            let is_done = f64::try_from(step_state.is_done.sum(tch::Kind::Float)).unwrap();
            if is_done > 0.0 {
                break;
            }
        }

        total_rewards += episode_reward;
        total_episodes += 1;

        let total_commissions = env.episode_history.total_commissions;
        env.record_inference(episode);

        let episode_time = Instant::now().duration_since(episode_start).as_secs_f32();
        println!(
            "Episode {}: Reward: {:.4}, Commissions: ${:.2}, Time: {:.2}s",
            episode, episode_reward, total_commissions, episode_time
        );
    }

    println!("\n=== Inference Summary ===");
    println!("Total episodes: {}", total_episodes);
    println!("Average reward: {:.4}", total_rewards / total_episodes as f64);

    Ok(())
}

/// Load TradingModel for streaming inference
pub fn load_trading_model<P: AsRef<Path>>(
    weight_path: P,
    device: tch::Device,
) -> Result<(nn::VarStore, TradingModel), Box<dyn std::error::Error>> {
    let mut vs = nn::VarStore::new(device);
    let model = TradingModel::new(&vs.root(), ACTION_COUNT);
    vs.load(weight_path)?;
    Ok((vs, model))
}

/// Run streaming inference with O(1) per-step SSM state
pub fn run_inference_streaming<P: AsRef<Path>>(
    weight_path: P,
    num_episodes: usize,
    deterministic: bool,
    temperature: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting streaming inference...");
    println!("Loading model from: {:?}", weight_path.as_ref());

    let device = Device::cuda_if_available();
    println!("Using device: {:?}", device);

    let (_vs, model) = load_trading_model(weight_path, device)?;
    let mut env = Env::new(false);
    let mut total_rewards = 0.0;
    let mut total_episodes = 0;

    for episode in 0..num_episodes {
        let episode_start = Instant::now();
        let mut episode_reward = 0.0;
        let mut state = model.init_stream_state();

        let (price_deltas_reset, static_obs_reset) = env.reset();
        let mut current_price_deltas = price_deltas_reset;
        let mut current_static_obs = static_obs_reset.to_device(device);
        env.episode = episode;

        for step in 0..env.max_step {
            env.step = step;

            let price_deltas_gpu = current_price_deltas.to_device(device);
            let (ready, (_, _, (action_mean, action_log_std, _), _)) = tch::no_grad(|| {
                model.step(&price_deltas_gpu.squeeze(), &current_static_obs, &mut state)
            });

            // Logistic-normal: append 0 before softmax
            let batch_size = action_mean.size()[0];
            let zeros = Tensor::zeros([batch_size, 1], (tch::Kind::Float, action_mean.device()));
            let actions = if deterministic || temperature == 0.0 {
                let u_ext = Tensor::cat(&[action_mean, zeros], 1);
                u_ext.softmax(-1, tch::Kind::Float)
            } else {
                let action_std = action_log_std.exp() * temperature;
                let noise = Tensor::randn_like(&action_mean);
                let u = &action_mean + &action_std * noise;
                let u_ext = Tensor::cat(&[u, zeros], 1);
                u_ext.softmax(-1, tch::Kind::Float)
            };

            let actions_flat = Vec::<f64>::try_from(actions.flatten(0, -1)).unwrap();
            let actions_vec: Vec<Vec<f64>> = actions_flat
                .chunks(ACTION_COUNT as usize)
                .map(|chunk| chunk.to_vec())
                .collect();

            let step_state = env.step(actions_vec);
            let reward = f64::try_from(step_state.reward.sum(tch::Kind::Float)).unwrap();
            episode_reward += reward;

            current_price_deltas = step_state.price_deltas;
            current_static_obs = step_state.static_obs.to_device(device);

            if f64::try_from(step_state.is_done.sum(tch::Kind::Float)).unwrap() > 0.0 {
                break;
            }
        }

        total_rewards += episode_reward;
        total_episodes += 1;
        env.record_inference(episode);

        println!(
            "Episode {} (streaming): Reward: {:.4}, Time: {:.2}s",
            episode, episode_reward, Instant::now().duration_since(episode_start).as_secs_f32()
        );
    }

    println!("\n=== Streaming Inference Summary ===");
    println!("Total episodes: {}", total_episodes);
    println!("Average reward: {:.4}", total_rewards / total_episodes as f64);
    Ok(())
}
