/* Proximal Policy Optimization (PPO) model.

   Proximal Policy Optimization Algorithms, Schulman et al. 2017
   https://arxiv.org/abs/1707.06347

   See https://spinningup.openai.com/en/latest/algorithms/ppo.html for a
   reference python implementation.
*/
use tch::kind::{FLOAT_CPU, INT64_CPU, INT64_CUDA};
use tch::{nn, nn::OptimizerConfig, Kind, Tensor};

use crate::torch::constants::{TICKERS_COUNT, OBSERVATION_SPACE};
use crate::torch::step::Env;

pub const NPROCS: i64 = 1; // DEFAULT 8 but disabled as unsure if step can handle multiple procs
const NSTEPS: i64 = 256;
const MEMORY_SIZE: i64 = NPROCS * NSTEPS;
const UPDATES: i64 = 1000000;
const OPTIM_BATCHSIZE: i64 = 64;
const OPTIM_EPOCHS: i64 = 4;

const LOG_2PI: f64 = 1.8378770664093453; // ln(2π)

type Model = Box<dyn Fn(&Tensor) -> (Tensor, (Tensor, Tensor))>;

fn model(p: &nn::Path, nact: i64) -> Model {
    let stride = |s| nn::ConvConfig {
        stride: s,
        ..Default::default()
    };
    let seq = nn::seq()
        .add(nn::conv2d(p / "c1", TICKERS_COUNT, 64, 8, stride(4)))
        .add_fn(|xs| xs.relu())
        .add(nn::conv2d(p / "c2", 64, 128, 4, stride(2)))
        .add_fn(|xs| xs.relu())
        .add(nn::conv2d(p / "c3", 128, 256, 3, stride(1)))
        .add_fn(|xs| xs.relu().flat_view())
        // 256 * 61 = 15616
        .add(nn::linear(p / "l1", 15616, 512, Default::default()))
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

#[derive(Debug)]
struct FrameStack {
    data: Tensor,
    nprocs: i64,
    nstack: i64,
}

impl FrameStack {
    fn new(nprocs: i64, nstack: i64) -> FrameStack {
        FrameStack {
            data: Tensor::zeros([nprocs, nstack, 84, 84], FLOAT_CPU),
            nprocs,
            nstack,
        }
    }

    fn update<'a>(&'a mut self, img: &Tensor, masks: Option<&Tensor>) -> &'a Tensor {
        if let Some(masks) = masks {
            self.data *= masks.view([self.nprocs, 1, 1, 1])
        };
        let slice = |i| self.data.narrow(1, i, 1);
        for i in 1..self.nstack {
            slice(i - 1).copy_(&slice(i))
        }
        slice(self.nstack - 1).copy_(img);
        &self.data
    }
}

pub fn train() {
    let mut env = Env::new();
    println!("action space: {}", TICKERS_COUNT);
    println!("observation space: {:?}", OBSERVATION_SPACE);

    let device = tch::Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let model = model(&vs.root(), TICKERS_COUNT);
    let mut opt = nn::Adam::default().build(&vs, 1e-4).unwrap();

    let mut sum_rewards = Tensor::zeros([NPROCS], FLOAT_CPU);
    let mut total_rewards = 0f64;
    let mut total_episodes = 0f64;

    let mut obs_buffer = FrameStack::new(NPROCS, TICKERS_COUNT);
    let _ = obs_buffer.update(&env.reset(), None);
    let s_states = Tensor::zeros([NSTEPS + 1, NPROCS, TICKERS_COUNT, 84, 84], FLOAT_CPU);

    for update_index in 0..UPDATES {
        s_states.get(0).copy_(&s_states.get(-1));
        let s_values = Tensor::zeros([NSTEPS, NPROCS], FLOAT_CPU);
        let s_rewards = Tensor::zeros([NSTEPS, NPROCS], FLOAT_CPU);
        let s_actions = Tensor::zeros([NSTEPS, NPROCS], INT64_CPU);
        let s_masks = Tensor::zeros([NSTEPS, NPROCS], FLOAT_CPU);

        // Custom loop

        env.reset();

        for s in 0..NSTEPS {
            let (critic, (action_mean, action_log_std)) = tch::no_grad(|| model(&s_states.get(s)));
            
            let action_std = action_log_std.exp();
            let noise = Tensor::randn_like(&action_mean);
            let actions = (action_mean + action_std * noise).clamp(-1.0, 1.0);
            
            println!("actions {}", actions);
            // Make sure they are hyperbolized
            
            let actions_flat = Vec::<f64>::try_from(&actions).unwrap();
            let actions_vec: Vec<Vec<f64>> = actions_flat
                .chunks(TICKERS_COUNT as usize)
                .map(|chunk| chunk.to_vec())
                .collect();
            
            let step = env.step(actions_vec);

            sum_rewards += &step.reward;
            total_rewards +=
                f64::try_from((&sum_rewards * &step.is_done).sum(Kind::Float)).unwrap();
            total_episodes += f64::try_from(step.is_done.sum(Kind::Float)).unwrap();

            let masks = Tensor::from(1f32) - step.is_done;
            sum_rewards *= &masks;
            let obs = obs_buffer.update(&step.obs, Some(&masks));
            s_actions.get(s).copy_(&actions);
            s_values.get(s).copy_(&critic.squeeze_dim(-1));
            s_states.get(s + 1).copy_(obs);
            s_rewards.get(s).copy_(&step.reward);
            s_masks.get(s).copy_(&masks);
        }
        let states = s_states
            .narrow(0, 0, NSTEPS)
            .view([MEMORY_SIZE, TICKERS_COUNT, 84, 84]);
        let returns = {
            let r = Tensor::zeros([NSTEPS + 1, NPROCS], FLOAT_CPU);
            let critic = tch::no_grad(|| model(&s_states.get(-1)).0);
            r.get(-1).copy_(&critic.view([NPROCS]));
            for s in (0..NSTEPS).rev() {
                let r_s = s_rewards.get(s) + r.get(s + 1) * s_masks.get(s) * 0.99;
                r.get(s).copy_(&r_s);
            }
            r.narrow(0, 0, NSTEPS).view([MEMORY_SIZE, 1])
        };
        let actions = s_actions.view([MEMORY_SIZE]);
        for _index in 0..OPTIM_EPOCHS {
            let batch_indexes = Tensor::randint(MEMORY_SIZE, [OPTIM_BATCHSIZE], INT64_CUDA);
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
            
            let advantages = returns.to_device(device) - critic;
            let value_loss = (&advantages * &advantages).mean(Kind::Float);
            let action_loss = (-advantages.detach() * action_log_probs).mean(Kind::Float);
            let loss = value_loss * 0.5 + action_loss - dist_entropy * 0.01;
            
            // Moderately aggressive clip
            opt.backward_step_clip_norm(&loss, 1.0);
        }
        if update_index > 0 && update_index % 25 == 0 {
            println!(
                "{} {:.0} {}",
                update_index,
                total_episodes,
                total_rewards / total_episodes
            );
            total_rewards = 0.;
            total_episodes = 0.;
        }
        if update_index > 0 && update_index % 1000 == 0 {
            if let Err(err) = vs.save(format!("trpo{update_index}.ot")) {
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
