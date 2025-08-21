use burn::{
    backend::{Autodiff, NdArray, Vulkan, Wgpu}, grad_clipping::GradientClippingConfig, module::Module, nn::{Initializer, Linear, LinearConfig}, optim::AdamWConfig, prelude::Backend, tensor::{
        activation::{gelu, relu, silu, softmax}, backend::AutodiffBackend, bf16, f16, DType, Tensor
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
        }, conv::{ConvAgent, ConvNet}, env::Env
    },
    data::historical::get_historical_data,
    history::{episode_tickers_separate::EpisodeHistory, meta_tickers_separate::MetaHistory},
    utils::get_price_deltas,
};

// const MEMORY_SIZE: usize = 4_096;
// const MEMORY_SIZE: usize = 16_384;
const MEMORY_SIZE: usize = 512;
const DENSE_SIZE: usize = 8;
const MAX_EPISODES: usize = 1000;
const KERNEL_SIZE: usize = 3;
const TICKER_COUNT: usize = 1;

pub fn run_training() {
    train::<Env, Autodiff<Wgpu<ElemType>>>();
}

pub fn train<E: Environment, B: AutodiffBackend>() -> impl Agent<E> {
    
    let mut env = E::new(false);
    let mut model = ConvNet::<B>::new(
        <<E as Environment>::StateType as State>::size(),
        DENSE_SIZE,
        <<E as Environment>::ActionType as Action>::size(),
        KERNEL_SIZE,
        TICKER_COUNT
    );
    let agent = ConvAgent::default();
    let config = PPOTrainingConfig {
        batch_size: 64,
        // batch_size: 512,
        // batch_size: 2048,
        entropy_weight: 0.01,
        learning_rate: 3e-4,
        epochs: 5,
        clip_grad: Some(GradientClippingConfig::Norm(0.5)),
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

            if let Some(action) = ConvAgent::<E, _>::react_with_model(&state, &model) {
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
                    model = ConvAgent::train::<MEMORY_SIZE>(model, &memory, &mut optimizer, &config);
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
