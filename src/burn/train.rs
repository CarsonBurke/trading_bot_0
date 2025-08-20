use burn::{
    backend::{Autodiff, NdArray, Vulkan, Wgpu},
    module::Module,
    nn::{Initializer, Linear, LinearConfig},
    optim::AdamWConfig,
    prelude::Backend,
    tensor::{
        activation::{gelu, relu, silu, softmax},
        backend::AutodiffBackend,
        Tensor,
    },
};

use crate::{
    burn::{
        agent::{
            base::{Action, Agent, ElemType, Environment, Memory, Model, State},
            ppo::{
                agent::PPO,
                config::PPOTrainingConfig,
                model::{PPOModel, PPOOutput},
            },
        },
        env::Env,
    },
    data::historical::get_historical_data,
    history::{episode_tickers_separate::EpisodeHistory, meta_tickers_separate::MetaHistory},
    utils::get_price_deltas,
};

const MEMORY_SIZE: usize = 32_768;
// 512;
const DENSE_SIZE: usize = 256;
const MAX_EPISODES: usize = 1000;

#[derive(Module, Debug)]
pub struct Net<B: Backend> {
    linear: Linear<B>,
    linear_actor: Linear<B>,
    linear_critic: Linear<B>,
}

impl<B: Backend> Net<B> {
    #[allow(unused)]
    pub fn new(input_size: usize, dense_size: usize, output_size: usize) -> Self {
        Self {
            linear: LinearConfig::new(input_size, dense_size).init(&Default::default()),
            linear_actor: LinearConfig::new(dense_size, output_size).init(&Default::default()),
            linear_critic: LinearConfig::new(dense_size, 1).init(&Default::default()),
        }
    }
}

impl<B: Backend> Model<B, Tensor<B, 2>, PPOOutput<B>, Tensor<B, 2>> for Net<B> {
    fn forward(&self, input: Tensor<B, 2>) -> PPOOutput<B> {
        let layer_0_output = silu(self.linear.forward(input));
        let policies = softmax(self.linear_actor.forward(layer_0_output.clone()), 1);
        let values = self.linear_critic.forward(layer_0_output);

        // println!("polcies {} values {}", policies, values);

        PPOOutput::<B>::new(policies, values)
    }

    fn infer(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let layer_0_output = silu(self.linear.forward(input));
        softmax(self.linear_actor.forward(layer_0_output.clone()), 1)
    }
}

impl<B: Backend> PPOModel<B> for Net<B> {}

type MyAgent<E, B> = PPO<E, B, Net<B>>;

pub fn run_training() {
    train::<Env, Autodiff<Vulkan<ElemType>>>();
}

pub fn train<E: Environment, B: AutodiffBackend>() -> impl Agent<E> {
    let mut env = E::new(false);
    let mut model = Net::<B>::new(
        <<E as Environment>::StateType as State>::size(),
        DENSE_SIZE,
        <<E as Environment>::ActionType as Action>::size(),
    );
    let agent = MyAgent::default();
    let config = PPOTrainingConfig {
        batch_size: 2048,
        ..Default::default()
    };
    let mut optimizer = AdamWConfig::new()
        .with_grad_clipping(config.clip_grad.clone())
        .init();
    let mut memory = Memory::<E, B, MEMORY_SIZE>::default();

    for _ in 0..MAX_EPISODES {
        env.reset();
        let mut done = false;

        while !done {
            let state = env.state();

            if let Some(action) = MyAgent::<E, _>::react_with_model(&state, &model) {
                let snapshot = env.step(action);

                memory.push(
                    state,
                    *snapshot.state(),
                    action,
                    snapshot.reward().clone(),
                    snapshot.done(),
                );

                if memory.len() >= MEMORY_SIZE {
                    println!("Memory limit reached - training model");
                    model = MyAgent::train::<MEMORY_SIZE>(model, &memory, &mut optimizer, &config);
                    memory.clear();
                }

                done = snapshot.done();
            } else {
                println!("No action selected");
                println!("state {:?}", state);
                panic!();
            }
        }

        // model = MyAgent::train::<MEMORY_SIZE>(model, &memory, &mut optimizer, &config);
        // memory.clear();
    }

    agent.valid(model)
}
