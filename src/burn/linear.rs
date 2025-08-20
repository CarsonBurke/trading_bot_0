use burn::{
    backend::{Autodiff, NdArray, Vulkan, Wgpu}, grad_clipping::GradientClippingConfig, module::Module, nn::{Initializer, Linear, LinearConfig}, optim::AdamWConfig, prelude::Backend, tensor::{
        activation::{gelu, relu, silu, softmax},
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
    history::{episode_tickers_separate::EpisodeHistory, meta_tickers_separate::MetaHistory},
    utils::get_price_deltas,
};

#[derive(Module, Debug)]
pub struct LinearNet<B: Backend> {
    linear: Linear<B>,
    linear_actor: Linear<B>,
    linear_critic: Linear<B>,
}

impl<B: Backend> LinearNet<B> {
    #[allow(unused)]
    pub fn new(input_size: usize, dense_size: usize, output_size: usize) -> Self {
        let initializer = Initializer::XavierUniform { gain: 1.0 };
        Self {
            linear: LinearConfig::new(input_size, dense_size).with_initializer(initializer.clone()).init(&Default::default()),
            linear_actor: LinearConfig::new(dense_size, output_size).with_initializer(initializer.clone()).init(&Default::default()),
            linear_critic: LinearConfig::new(dense_size, 1).with_initializer(initializer).init(&Default::default()),
        }
    }
}

impl<B: Backend> Model<B, Tensor<B, 2>, PPOOutput<B>, Tensor<B, 2>> for LinearNet<B> {
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

impl<B: Backend> PPOModel<B> for LinearNet<B> {}

pub type LinearAgent<E, B> = PPO<E, B, LinearNet<B>>;