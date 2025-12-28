use tch::{nn, Device, Kind, Tensor};
use std::path::Path;
use std::time::Instant;

use crate::torch::load::load_var_store_partial;
use crate::torch::model::TradingModel;
use crate::torch::env::Env;

pub fn load_model<P: AsRef<Path>>(
    weight_path: P,
    device: Device,
) -> Result<(nn::VarStore, TradingModel), Box<dyn std::error::Error>> {
    let mut vs = nn::VarStore::new(device);
    let model = TradingModel::new(&vs.root());
    let _ = load_var_store_partial(&mut vs, weight_path)?;
    Ok((vs, model))
}

pub fn sample_actions_from_dist(
    action_logits: &Tensor,
    sde_latent: &Tensor,
    std_matrix: &Tensor,
    deterministic: bool,
    temperature: f64,
) -> Tensor {
    let u = if deterministic || temperature == 0.0 {
        action_logits.shallow_clone()
    } else {
        let eps = Tensor::randn(
            &[
                sde_latent.size()[0],
                std_matrix.size()[0],
                std_matrix.size()[1],
            ],
            (Kind::Float, sde_latent.device()),
        );
        let noise_mat = eps * std_matrix.unsqueeze(0);
        let noise_raw = (sde_latent * noise_mat.permute([0, 2, 1]))
            .sum_dim_intlist([-1].as_slice(), false, Kind::Float);
        action_logits + noise_raw * temperature
    };
    u.softmax(-1, Kind::Float)
}

pub fn run_inference<P: AsRef<Path>>(
    weight_path: P,
    num_episodes: usize,
    deterministic: bool,
    temperature: f64,
    tickers: Option<Vec<String>>,
    random_start: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting inference run...");
    println!("Loading model from: {:?}", weight_path.as_ref());

    let device = Device::cuda_if_available();
    println!("Using device: {:?}", device);

    let (_vs, model) = load_model(&weight_path, device)?;

    let mut env = match tickers {
        Some(t) => Env::new_with_tickers(t, random_start),
        None => Env::new(random_start),
    };
    println!("Backtesting tickers: {:?}", env.tickers);

    let mut total_rewards = 0.0;

    for episode in 0..num_episodes {
        let episode_start = Instant::now();
        let mut episode_reward = 0.0;
        let mut stream_state = model.init_stream_state();

        let (price_deltas, static_obs) = env.reset_single();
        let mut current_price_deltas = Tensor::from_slice(&price_deltas).to_device(device);
        let mut current_static_obs = Tensor::from_slice(&static_obs).to_device(device);

        env.episode = episode;

        for step in 0..env.max_step {
            env.step = step;

            // First call: model.step detects full obs and initializes stream state
            // Subsequent calls: model.step processes single delta per ticker
            let (action_logits, _action_log_std, sde_latent) = tch::no_grad(|| {
                let (_, (action_logits, action_log_std, sde_latent), _) =
                    model.step(&current_price_deltas, &current_static_obs, &mut stream_state);
                (action_logits, action_log_std, sde_latent)
            });

            let std_matrix = model.sde_std_matrix();
            let actions = sample_actions_from_dist(
                &action_logits,
                &sde_latent,
                &std_matrix,
                deterministic,
                temperature,
            );

            let actions_vec: Vec<f64> = Vec::<f64>::try_from(actions.flatten(0, -1)).unwrap();
            let step_result = env.step_step_single(actions_vec);
            episode_reward += step_result.reward;

            current_price_deltas = Tensor::from_slice(&step_result.step_deltas).to_device(device);
            current_static_obs = Tensor::from_slice(&step_result.static_obs).to_device(device);

            if step_result.is_done > 0.5 {
                break;
            }
        }

        total_rewards += episode_reward;
        let total_commissions = env.episode_history.total_commissions;
        env.record_inference(episode);

        println!(
            "Episode {}: Reward: {:.4}, Commissions: ${:.2}, Time: {:.2}s",
            episode, episode_reward, total_commissions,
            Instant::now().duration_since(episode_start).as_secs_f32()
        );
    }

    println!("\n=== Inference Summary ===");
    println!("Total episodes: {}", num_episodes);
    println!("Average reward: {:.4}", total_rewards / num_episodes as f64);

    Ok(())
}
