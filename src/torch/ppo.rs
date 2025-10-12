/* Proximal Policy Optimization (PPO) model.

   Proximal Policy Optimization Algorithms, Schulman et al. 2017
   https://arxiv.org/abs/1707.06347

   See https://spinningup.openai.com/en/latest/algorithms/ppo.html for a
   reference python implementation.
*/
use tch::kind::{FLOAT_CPU, INT64_CPU, INT64_CUDA};
use tch::{nn, nn::OptimizerConfig, Kind, Tensor};

use crate::torch::constants::{OBSERVATIONS_PER_TICKER, OBSERVATION_SPACE, TICKERS_COUNT};
use crate::torch::step::Env;

pub const NPROCS: i64 = 1; // DEFAULT 8 but disabled as unsure if step can handle multiple procs
const UPDATES: i64 = 1000000;
const OPTIM_BATCHSIZE: i64 = 2048;
const OPTIM_EPOCHS: i64 = 4;

const LOG_2PI: f64 = 1.8378770664093453; // ln(2π)

type Model = Box<dyn Fn(&Tensor) -> (Tensor, (Tensor, Tensor))>;

fn model(p: &nn::Path, nact: i64) -> Model {
    let stride = |s| nn::ConvConfig {
        stride: s,
        ..Default::default()
    };

    // 1D CNN for time series: input shape [batch, channels=TICKERS_COUNT, length=OBSERVATIONS_PER_TICKER]
    // With TICKERS_COUNT=5 and OBSERVATIONS_PER_TICKER=200: [batch, 5, 200]
    let seq = nn::seq()
        .add(nn::conv1d(p / "c1", TICKERS_COUNT, 64, 8, stride(4)))  // (200-8)/4+1 = 49
        .add_fn(|xs| xs.relu())
        .add(nn::conv1d(p / "c2", 64, 128, 5, stride(2)))            // (49-5)/2+1 = 23
        .add_fn(|xs| xs.relu())
        .add(nn::conv1d(p / "c3", 128, 256, 3, stride(2)))           // (23-3)/2+1 = 11
        .add_fn(|xs| xs.relu().flat_view())
        // Flattened size: 256 * 11 = 2816
        .add(nn::linear(p / "l1", 2816, 512, Default::default()))
        .add_fn(|xs| xs.relu());

    let critic = nn::linear(p / "cl", 512, 1, Default::default());
    let actor_mean = nn::linear(p / "al", 512, nact, Default::default());
    let actor_log_std = nn::linear(p / "al_log_std", 512, nact, Default::default());

    let device = p.device();
    Box::new(move |xs: &Tensor| {
        let xs = xs.to_device(device).apply(&seq);

        let critic_value = xs.apply(&critic);

        // Action mean passed through tanh to bound actions to [-1, 1]
        let action_mean = xs.apply(&actor_mean).tanh();

        // Log std for numerical stability, clamped to reasonable range
        let action_log_std = xs.apply(&actor_log_std).clamp(-20.0, 2.0);

        (critic_value, (action_mean, action_log_std))
    })
}

pub fn train() {
    let mut env = Env::new();
    let n_steps = env.max_step as i64;
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
    let mut opt = nn::Adam::default().build(&vs, 1e-4).unwrap();

    // Create device-specific kind
    let float_kind = (Kind::Float, device);
    let int64_kind = (Kind::Int64, device);

    let mut sum_rewards = Tensor::zeros([NPROCS], float_kind);
    let mut total_rewards = 0f64;
    let mut total_episodes = 0f64;

    let mut current_obs = env.reset().to_device(device);
    let s_states = Tensor::zeros([n_steps + 1, NPROCS, TICKERS_COUNT, OBSERVATIONS_PER_TICKER as i64], float_kind);

    for episode in 0..UPDATES {
        s_states.get(0).copy_(&s_states.get(-1));
        let s_values = Tensor::zeros([n_steps, NPROCS], float_kind);
        let s_rewards = Tensor::zeros([n_steps, NPROCS], float_kind);
        let s_actions = Tensor::zeros([n_steps, NPROCS, TICKERS_COUNT], float_kind);
        let s_masks = Tensor::zeros([n_steps, NPROCS], float_kind);

        // Custom loop

        let _ = env.reset();
        env.episode = episode as usize;

        s_states.get(0).copy_(&current_obs);
        
        for step in OBSERVATIONS_PER_TICKER..env.max_step {
            env.step = step;
            
            let (critic, (action_mean, action_log_std)) = tch::no_grad(|| model(&s_states.get(step as i64)));

            let action_std = action_log_std.exp();
            let noise = Tensor::randn_like(&action_mean);
            let actions = (action_mean + action_std * noise).clamp(-1.0, 1.0);

            // Flatten the actions tensor [NPROCS, TICKERS_COUNT] to 1D before converting
            let actions_flat = Vec::<f64>::try_from(actions.flatten(0, -1)).unwrap();
            let actions_vec: Vec<Vec<f64>> = actions_flat
                .chunks(TICKERS_COUNT as usize)
                .map(|chunk| chunk.to_vec())
                .collect();
            
            // println!("Actions: {:?}", actions_vec);

            let step_state = env.step(actions_vec);

            // println!("Step reward: {:?}", step_state.reward);

            let reward = step_state.reward.to_device(device);
            let is_done = step_state.is_done.to_device(device);
            let obs = step_state.obs.to_device(device);

            sum_rewards += &reward;
            total_rewards +=
                f64::try_from((&sum_rewards * &is_done).sum(Kind::Float)).unwrap();
            total_episodes += f64::try_from(is_done.sum(Kind::Float)).unwrap();

            let masks = Tensor::from(1f32).to_device(device) - &is_done;
            sum_rewards *= &masks;

            s_actions.get(step as i64).copy_(&actions);
            s_values.get(step as i64).copy_(&critic.squeeze_dim(-1));
            s_states.get(step as i64 + 1).copy_(&obs);
            s_rewards.get(step as i64).copy_(&reward);
            s_masks.get(step as i64).copy_(&masks);

            current_obs = obs;
            
            env.step += 1;
        }
        
        println!("total rewards: {} sum rewards: {}", total_rewards, sum_rewards);
        
        let states = s_states
            .narrow(0, 0, n_steps)
            .view([memory_size, TICKERS_COUNT, OBSERVATIONS_PER_TICKER as i64]);
        let returns = {
            let r = Tensor::zeros([n_steps + 1, NPROCS], float_kind);
            let critic = tch::no_grad(|| model(&s_states.get(-1)).0);
            r.get(-1).copy_(&critic.view([NPROCS]));
            for s in (0..n_steps).rev() {
                let r_s = s_rewards.get(s) + r.get(s + 1) * s_masks.get(s) * 0.99;
                r.get(s).copy_(&r_s);
            }
            r.narrow(0, 0, n_steps).view([memory_size, 1])
        };
        let actions = s_actions.view([memory_size, TICKERS_COUNT]);
        for _index in 0..OPTIM_EPOCHS {
            let batch_indexes = Tensor::randint(memory_size, [OPTIM_BATCHSIZE], int64_kind);
            let states = states.index_select(0, &batch_indexes);
            let actions = actions.index_select(0, &batch_indexes);
            let returns = returns.index_select(0, &batch_indexes);
            
            let (critic, (action_mean, action_log_std)) = model(&states);
            
            let action_std = action_log_std.exp();
            let actions_normalized = (&actions - &action_mean) / &action_std;
            
            let log_prob_components: Tensor = actions_normalized.pow_tensor_scalar(2) + 2.0 * &action_log_std + LOG_2PI;
            let action_log_probs = log_prob_components.g_mul_scalar(-0.5).sum_dim_intlist(-1, false, Kind::Float);
            
            let entropy_components: Tensor = 1.0 + LOG_2PI + 2.0 * action_log_std;
            let dist_entropy = entropy_components
                .g_mul_scalar(0.5)
                .sum_dim_intlist(-1, false, Kind::Float)
                .mean(Kind::Float);

            let advantages = returns - critic;
            let value_loss = (&advantages * &advantages).mean(Kind::Float);
            let action_loss = (-advantages.detach() * action_log_probs).mean(Kind::Float);
            let loss = value_loss * 0.5 + action_loss - dist_entropy * 0.01;
            
            // Moderately aggressive clip
            opt.backward_step_clip_norm(&loss, 1.0);
        }
        if episode > 0 && episode % 25 == 0 {
            println!(
                "{} {:.0} {}",
                episode,
                total_episodes,
                total_rewards / total_episodes
            );
            total_rewards = 0.;
            total_episodes = 0.;
        }
        if episode > 0 && episode % 1000 == 0 {
            if let Err(err) = vs.save(format!("trpo{episode}.ot")) {
                println!("error while saving {err}")
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
