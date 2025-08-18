use burn::{
    backend::{Autodiff, NdArray, Wgpu}, module::Module, nn::{Initializer, Linear, LinearConfig}, optim::AdamWConfig, prelude::Backend, tensor::{
        activation::{relu, softmax},
        backend::AutodiffBackend,
        Tensor,
    }
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
    history::{episode_tickers_separate::EpisodeHistory, meta_tickers_separate::MetaHistory}, utils::get_price_deltas,
};

const MEMORY_SIZE: usize = 512;
const DENSE_SIZE: usize = 128;
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
        let initializer = Initializer::KaimingUniform {
            gain: 1.0,
            fan_out_only: true,
        };
        Self {
            linear: LinearConfig::new(input_size, dense_size)
                .with_initializer(initializer.clone())
                .init(&Default::default()),
            linear_actor: LinearConfig::new(dense_size, output_size)
                .with_initializer(initializer.clone())
                .init(&Default::default()),
            linear_critic: LinearConfig::new(dense_size, 1)
                .with_initializer(initializer)
                .init(&Default::default()),
        }
    }
}

impl<B: Backend> Model<B, Tensor<B, 2>, PPOOutput<B>, Tensor<B, 2>> for Net<B> {
    fn forward(&self, input: Tensor<B, 2>) -> PPOOutput<B> {
        let layer_0_output = relu(self.linear.forward(input));
        let policies = softmax(self.linear_actor.forward(layer_0_output.clone()), 1);
        let values = self.linear_critic.forward(layer_0_output);

        PPOOutput::<B>::new(policies, values)
    }

    fn infer(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let layer_0_output = relu(self.linear.forward(input));
        softmax(self.linear_actor.forward(layer_0_output.clone()), 1)
    }
}

impl<B: Backend> PPOModel<B> for Net<B> {}

type MyAgent<E, B> = PPO<E, B, Net<B>>;

pub fn run_training() {
    println!("start run");
    train::<Env, Autodiff<NdArray<ElemType>>>();
}

pub fn train<E: Environment, B: AutodiffBackend>() -> impl Agent<E> {
    println!("start train");
    let mut env = E::new(false);

    let mut model = Net::<B>::new(
        <<E as Environment>::StateType as State>::size(),
        DENSE_SIZE,
        <<E as Environment>::ActionType as Action>::size(),
    );
    let agent = MyAgent::default();
    let config = PPOTrainingConfig::default();

    let mut optimizer = AdamWConfig::new()
        .with_grad_clipping(config.clip_grad.clone())
        .init();
    let mut memory = Memory::<E, B, MEMORY_SIZE>::default();
    println!("made things");
    for _ in 0..MAX_EPISODES {
        println!("start episode");
        env.reset();
        let mut done = false;
        println!("pre steps");
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
                
                done = snapshot.done();
            }
        }
        println!("pre train");
        model = MyAgent::train::<MEMORY_SIZE>(model, &memory, &mut optimizer, &config);
        memory.clear();
    }

    agent.valid(model)
}
