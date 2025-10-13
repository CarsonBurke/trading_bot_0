use std::time::Instant;

/* Proximal Policy Optimization (PPO) model.

   Proximal Policy Optimization Algorithms, Schulman et al. 2017
   https://arxiv.org/abs/1707.06347

   See https://spinningup.openai.com/en/latest/algorithms/ppo.html for a
   reference python implementation.
*/
use tch::kind::FLOAT_CPU;
use tch::{nn, nn::OptimizerConfig, Kind, Tensor};

use crate::torch::constants::{TICKERS_COUNT, OBSERVATION_SPACE, PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS};
use crate::torch::step::Env;
use crate::torch::model::model;

pub const NPROCS: i64 = 1; // Parallel environments for better GPU utilization
const UPDATES: i64 = 1000000;
const OPTIM_BATCHSIZE: i64 = 2048; // Reduced due to larger model size (~96M params in FC layers)
const OPTIM_EPOCHS: i64 = 4; // More epochs to extract more learning per rollout

const LOG_2PI: f64 = 1.8378770664093453; // ln(2π)

pub fn train() {
    let mut env = Env::new();
    // n_steps is the episode length - use constant since all episodes are same length
    // Episodes are capped at 1500 steps, minus 2 for buffer = 1498
    let n_steps = 1498i64;
    let memory_size = n_steps * NPROCS;
    println!("action space: {}", TICKERS_COUNT);
    println!("observation space: {:?}", OBSERVATION_SPACE);

    println!("CUDA available: {}", tch::Cuda::is_available());
    println!("CUDA device count: {}", tch::Cuda::device_count());
    println!("cuDNN available: {}", tch::Cuda::cudnn_is_available());

    let device = tch::Device::cuda_if_available();
    println!("Using device {:?}", device);

    let vs = nn::VarStore::new(device);
    let model = model(&vs.root(), TICKERS_COUNT);
    // Reduced learning rate to prevent premature convergence
    let mut opt = nn::Adam::default().build(&vs, 1e-4).unwrap();

    // Create device-specific kind
    let float_kind = (Kind::Float, device);
    let int64_kind = (Kind::Int64, device);

    let mut sum_rewards = Tensor::zeros([NPROCS], float_kind);
    let mut total_rewards = 0f64;
    let mut total_episodes = 0f64;

    let (current_price_deltas_init, current_static_obs_init) = env.reset();
    let mut current_price_deltas = current_price_deltas_init.to_device(device);
    let mut current_static_obs = current_static_obs_init.to_device(device);

    // Separate storage for price deltas and static observations
    let s_price_deltas = Tensor::zeros(
        [
            n_steps + 1,
            NPROCS,
            TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64,
        ],
        float_kind,
    );
    let s_static_obs = Tensor::zeros(
        [
            n_steps + 1,
            NPROCS,
            STATIC_OBSERVATIONS as i64,
        ],
        float_kind,
    );

    for episode in 0..UPDATES {
        s_price_deltas.get(0).copy_(&s_price_deltas.get(-1));
        s_static_obs.get(0).copy_(&s_static_obs.get(-1));

        let s_values = Tensor::zeros([n_steps, NPROCS], float_kind);
        let s_rewards = Tensor::zeros([n_steps, NPROCS], float_kind);
        let s_actions = Tensor::zeros([n_steps, NPROCS, TICKERS_COUNT * 2], float_kind);
        let s_masks = Tensor::zeros([n_steps, NPROCS], float_kind);

        // Custom loop
        let (price_deltas_reset, static_obs_reset) = env.reset();
        current_price_deltas = price_deltas_reset.to_device(device);
        current_static_obs = static_obs_reset.to_device(device);
        env.episode = episode as usize;

        s_price_deltas.get(0).copy_(&current_price_deltas);
        s_static_obs.get(0).copy_(&current_static_obs);

        // Use a separate index (s) for tensor storage, starting from 0
        // Loop through the episode using relative steps (0 to max_step)
        let mut s: i64 = 0;
        for step in 0..env.max_step {
            env.step = step;

            let (critic, (action_mean, action_log_std)) = tch::no_grad(|| {
                model(
                    &s_price_deltas.get(s),
                    &s_static_obs.get(s),
                )
            });

            // Sample from Gaussian distribution
            let action_std = action_log_std.exp();
            let noise = Tensor::randn_like(&action_mean);
            let z = &action_mean + &action_std * noise;

            // Softer sigmoid squashing: action = 2 * sigmoid(z/2) - 1
            // Dividing z by 2 makes the squashing much gentler:
            // - sigmoid(z/2) changes more slowly than sigmoid(z)
            // - Reduces saturation at extremes
            // - With std max of 0.37, most samples stay in moderate range
            let actions = (z / 2.0).sigmoid() * 2.0 - 1.0;

            // Flatten the actions tensor [NPROCS, TICKERS_COUNT] to 1D before converting
            let actions_flat = Vec::<f64>::try_from(actions.flatten(0, -1)).unwrap();
            let actions_vec: Vec<Vec<f64>> = actions_flat
                .chunks(TICKERS_COUNT * 2)
                .map(|chunk| chunk.to_vec())
                .collect();

            // println!("Actions: {:?}", actions_vec);

            let step_state = env.step(actions_vec);

            // println!("Step reward: {:?}", step_state.reward);

            let reward = step_state.reward.to_device(device);
            let is_done = step_state.is_done.to_device(device);
            let price_deltas = step_state.price_deltas.to_device(device);
            let static_obs = step_state.static_obs.to_device(device);

            sum_rewards += &reward;
            total_rewards += f64::try_from((&sum_rewards * &is_done).sum(Kind::Float)).unwrap();
            total_episodes += f64::try_from(is_done.sum(Kind::Float)).unwrap();

            let masks = Tensor::from(1f32).to_device(device) - &is_done;
            sum_rewards *= &masks;

            s_actions.get(s).copy_(&actions);
            s_values.get(s).copy_(&critic.squeeze_dim(-1));
            s_price_deltas.get(s + 1).copy_(&price_deltas);
            s_static_obs.get(s + 1).copy_(&static_obs);
            s_rewards.get(s).copy_(&reward);
            s_masks.get(s).copy_(&masks);

            current_price_deltas = price_deltas;
            current_static_obs = static_obs;

            s += 1;  // Increment storage index
        }

        println!(
            "total rewards: {} sum rewards: {}",
            total_rewards, sum_rewards
        );

        let price_deltas_batch = s_price_deltas.narrow(0, 0, n_steps).view([
            memory_size,
            TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64,
        ]);
        let static_obs_batch = s_static_obs.narrow(0, 0, n_steps).view([
            memory_size,
            STATIC_OBSERVATIONS as i64,
        ]);

        let returns = {
            let r = Tensor::zeros([n_steps + 1, NPROCS], float_kind);
            let critic = tch::no_grad(|| {
                model(&s_price_deltas.get(-1), &s_static_obs.get(-1)).0
            });
            r.get(-1).copy_(&critic.view([NPROCS]));
            // Increased discount factor from 0.99 to 0.995 for longer-term thinking
            // This encourages holding profitable positions longer
            for s in (0..n_steps).rev() {
                let r_s = s_rewards.get(s) + r.get(s + 1) * s_masks.get(s) * 0.995;
                r.get(s).copy_(&r_s);
            }
            r.narrow(0, 0, n_steps).view([memory_size, 1])
        };
        let actions = s_actions.view([memory_size, TICKERS_COUNT * 2]);

        let opt_start = Instant::now();

        for _index in 0..OPTIM_EPOCHS {
            let batch_indexes = Tensor::randint(memory_size, [OPTIM_BATCHSIZE], int64_kind);
            let price_deltas_sample = price_deltas_batch.index_select(0, &batch_indexes);
            let static_obs_sample = static_obs_batch.index_select(0, &batch_indexes);
            let actions = actions.index_select(0, &batch_indexes);
            let returns = returns.index_select(0, &batch_indexes);

            let (critic, (action_mean, action_log_std)) = model(&price_deltas_sample, &static_obs_sample);

            // Recover the unsquashed Gaussian samples z from actions
            // action = 2 * sigmoid(z/2) - 1  =>  z = 2 * logit((action + 1) / 2)
            // Add small epsilon for numerical stability
            let eps = 1e-6;
            let action_01 = (&actions + 1.0) / 2.0;  // Transform from [-1, 1] to [0, 1]
            let action_01_clamped = action_01.clamp(eps, 1.0 - eps);  // Prevent log(0)
            let one_minus_action = Tensor::from(1.0) - &action_01_clamped;
            let z = (&action_01_clamped.log() - one_minus_action.log()) * 2.0;  // 2 * logit function

            // Compute log probability of z under Gaussian N(action_mean, action_std)
            let action_std = action_log_std.exp();
            let z_normalized = (&z - &action_mean) / &action_std;
            let z_squared = z_normalized.pow_tensor_scalar(2);
            let two_log_std = &action_log_std * 2.0;
            let log_prob_z = (&z_squared + two_log_std + LOG_2PI).g_mul_scalar(-0.5);

            // Compute log Jacobian correction for softer sigmoid squashing
            // action = 2 * sigmoid(z/2) - 1
            // |d_action/dz| = sigmoid(z/2) * (1 - sigmoid(z/2))
            // log|d_action/dz| = log(sigmoid(z/2)) + log(1 - sigmoid(z/2))
            let sigmoid_z_half = (&z / 2.0).sigmoid();
            let one_minus_sigmoid = Tensor::from(1.0) - &sigmoid_z_half;
            let log_jacobian = sigmoid_z_half.log() + one_minus_sigmoid.log();

            // Final log probability: log p(action) = log p(z) - log|d_action/dz|
            let action_log_probs = (log_prob_z - log_jacobian)
                .sum_dim_intlist(-1, false, Kind::Float);

            // Entropy approximation: Use base Gaussian entropy
            // Note: Exact entropy of sigmoid-squashed distribution is complex to compute
            // Using Gaussian entropy as approximation is standard practice for regularization
            let entropy_components: Tensor = 1.0 + LOG_2PI + 2.0 * action_log_std;
            let dist_entropy = entropy_components
                .g_mul_scalar(0.5)
                .sum_dim_intlist(-1, false, Kind::Float)
                .mean(Kind::Float);

            let advantages = returns - critic;

            // For policy gradient: detach advantages first, then normalize
            // This prevents wasting computation on gradients we immediately discard
            let advantages_detached = advantages.detach();
            let adv_mean = advantages_detached.mean(Kind::Float);
            let adv_std = advantages_detached.std(false);
            let advantages_normalized = (&advantages_detached - adv_mean) / (adv_std + 1e-8);

            let value_loss = (&advantages * &advantages).mean(Kind::Float);
            let action_loss = (-advantages_normalized * action_log_probs).mean(Kind::Float);
            // Increased entropy bonus from 0.01 to 0.05 to encourage more exploration
            let loss = value_loss * 0.5 + action_loss - dist_entropy * 0.05;

            // Moderately aggressive clip
            opt.backward_step_clip_norm(&loss, 2.0);
        }
        
        let opt_end = Instant::now();
        
        if episode > 0 && episode % 25 == 0 {
            // Debug: Check if exploration has collapsed or network diverged
            let (_, (debug_mean, debug_log_std)) = tch::no_grad(|| {
                model(&s_price_deltas.get(0), &s_static_obs.get(0))
            });

            // Show actual squashed actions, not raw mean
            let debug_actions = (&debug_mean / 2.0).sigmoid() * 2.0 - 1.0;
            let mean_squashed_action = f64::try_from(debug_actions.mean(Kind::Float)).unwrap();
            let mean_raw_action = f64::try_from(debug_mean.mean(Kind::Float)).unwrap();
            let mean_std = f64::try_from(debug_log_std.exp().mean(Kind::Float)).unwrap();
            let max_raw_action = f64::try_from(debug_mean.abs().max()).unwrap();

            println!(
                "[Ep {:6}] Episodes: {:.0}, Avg reward: {:.4}, Opt time: {:.2}s, Action (squashed): {:.3}, Action (raw): {:.1}, Max |raw|: {:.1}, Std: {:.4}",
                episode,
                total_episodes,
                total_rewards / total_episodes,
                opt_end.duration_since(opt_start).as_secs_f32(),
                mean_squashed_action,
                mean_raw_action,
                max_raw_action,
                mean_std
            );

            // Warn if network is diverging
            if max_raw_action > 100.0 {
                println!("WARNING: Network may be diverging! Raw action magnitude: {:.1}", max_raw_action);
            }

            total_rewards = 0.;
            total_episodes = 0.;
        }
        if episode > 0 && episode % 100 == 0 {
            std::fs::create_dir_all("weights").ok();
            if let Err(err) = vs.save(format!("weights/ppo_ep{}.ot", episode)) {
                println!("Error while saving weights: {}", err)
            } else {
                println!("Saved model weights: weights/ppo_ep{}.ot", episode);
            }
        }
    }
}

// Pretty sure I can ignore this, used for inference after training?
// Seems like it yes. Loads in weights and then infers a bunch
//
// pub fn sample<T: AsRef<std::path::Path>>(weight_file: T) -> cpython::PyResult<()> {
//     let env = Environment::new(false);
//     println!("action space: {}", env.action_space());
//     println!("observation space: {:?}", env.observation_space());

//     let mut vs = nn::VarStore::new(tch::Device::Cpu);
//     let model = model(&vs.root(), env.action_space());
//     vs.load(weight_file).unwrap();

//     let mut frame_stack = FrameStack::new(1, NSTACK);
//     let mut obs = frame_stack.update(&env.reset()?, None);

//     for _index in 0..5000 {
//         let (_critic, actor) = tch::no_grad(|| model(obs));
//         let probs = actor.softmax(-1, Kind::Float);
//         let actions = probs.multinomial(1, true).squeeze_dim(-1);
//         let step = env.step(Vec::<i64>::try_from(&actions).unwrap())?;

//         let masks = Tensor::from(1f32) - step.is_done;
//         obs = frame_stack.update(&step.obs, Some(&masks));
//     }
//     Ok(())
// }
