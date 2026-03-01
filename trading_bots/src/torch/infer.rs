use std::path::Path;
use std::time::Instant;
use tch::{nn, Device, Kind, Tensor};

use crate::torch::constants::{PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS, TICKERS_COUNT};
use crate::torch::env::Env;
use crate::torch::load::load_var_store_partial;
use crate::torch::model::{
    ModelVariant, TradingModel, TradingModelConfig,
};

pub fn load_model<P: AsRef<Path>>(
    weight_path: P,
    device: Device,
    model_variant: ModelVariant,
) -> Result<(nn::VarStore, TradingModel), Box<dyn std::error::Error>> {
    let mut vs = nn::VarStore::new(device);
    let model = TradingModel::new_with_config(
        &vs.root(),
        TradingModelConfig {
            variant: model_variant,
            ..TradingModelConfig::default()
        },
    );
    let _ = load_var_store_partial(&mut vs, weight_path)?;
    vs.bfloat16();
    Ok((vs, model))
}

pub fn sample_actions(
    action_mean: &Tensor,
    action_cov_chol: &Tensor,  // [batch, TICKERS_COUNT, TICKERS_COUNT] lower-triangular Cholesky
    deterministic: bool,
    temperature: f64,
) -> Tensor {
    let action_mean = action_mean.to_kind(Kind::Float);

    let u = if deterministic {
        action_mean
    } else {
        let chol = action_cov_chol.to_kind(Kind::Float);
        let batch = action_mean.size()[0];
        let z = Tensor::randn(&[batch, TICKERS_COUNT, 1], (Kind::Float, action_mean.device()));
        let ticker_noise = chol.bmm(&z).squeeze_dim(-1); // [B, TICKERS_COUNT]
        let zero_cash = Tensor::zeros(&[batch, 1], (Kind::Float, action_mean.device()));
        let noise = Tensor::cat(&[ticker_noise, zero_cash], -1);
        &action_mean + noise
    };

    let u = if temperature != 1.0 && temperature != 0.0 {
        &u / temperature
    } else {
        u
    };

    u.softmax(-1, Kind::Float)
}

pub fn run_inference<P: AsRef<Path>>(
    weight_path: P,
    num_episodes: usize,
    _deterministic: bool,
    _temperature: f64,
    tickers: Option<Vec<String>>,
    random_start: bool,
    model_variant: ModelVariant,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting inference run...");
    println!("Loading model from: {:?}", weight_path.as_ref());

    let device = Device::cuda_if_available();
    println!("Using device: {:?}", device);

    let deterministic = true;
    let temperature = 0.0;

    let (_vs, model) = load_model(&weight_path, device, model_variant)?;

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
        let mut price_deltas_full = Tensor::zeros(
            [TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64],
            (Kind::Float, device),
        );
        let mut price_deltas_incremental = Tensor::zeros([TICKERS_COUNT], (Kind::Float, device));
        let mut static_obs_tensor =
            Tensor::zeros([STATIC_OBSERVATIONS as i64], (Kind::Float, device));
        price_deltas_full.copy_(&Tensor::from_slice(&price_deltas));
        static_obs_tensor.copy_(&Tensor::from_slice(&static_obs));
        let mut use_full = true;

        env.episode = episode;

        for step in 0..env.max_step {
            env.step = step;

            // First call uses full history, then we switch to incremental per-ticker deltas.
            let (action_mean, action_cov_chol) = tch::no_grad(|| {
                let price_input = if use_full {
                    &price_deltas_full
                } else {
                    &price_deltas_incremental
                };
                let (_, _, _, action_mean, action_cov_chol) =
                    model.step_on_device(price_input, &static_obs_tensor, &mut stream_state);
                (action_mean, action_cov_chol)
            });
            let actions = sample_actions(
                &action_mean,
                &action_cov_chol,
                deterministic,
                temperature,
            );

            let actions_vec: Vec<f64> = Vec::<f64>::try_from(actions.flatten(0, -1)).unwrap();
            let step_result = env.step_step_single(&actions_vec);
            episode_reward += step_result.reward;

            price_deltas_incremental.copy_(&Tensor::from_slice(&step_result.step_deltas));
            static_obs_tensor.copy_(&Tensor::from_slice(&step_result.static_obs));
            use_full = false;

            if step_result.is_done > 0.5 {
                break;
            }
        }

        total_rewards += episode_reward;
        let total_commissions = env.episode_history.total_commissions;
        env.record_inference(episode);

        println!(
            "Episode {}: Reward: {:.4}, Commissions: ${:.2}, Time: {:.2}s",
            episode,
            episode_reward,
            total_commissions,
            Instant::now().duration_since(episode_start).as_secs_f32()
        );
    }

    println!("\n=== Inference Summary ===");
    println!("Total episodes: {}", num_episodes);
    println!("Average reward: {:.4}", total_rewards / num_episodes as f64);

    Ok(())
}
