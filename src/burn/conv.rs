use burn::{
    backend::{Autodiff, NdArray, Vulkan, Wgpu}, grad_clipping::GradientClippingConfig, module::Module, nn::{conv::{Conv1d, Conv1dConfig}, Initializer, Linear, LinearConfig}, optim::AdamWConfig, prelude::Backend, tensor::{
        activation::{gelu, relu, silu, softmax},
        backend::AutodiffBackend,
        Tensor,
    }
};

use crate::{
    burn::{
        agent::{
            base::{Action, ElemType, Environment, Memory, Model, State},
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
pub struct ConvNet<B: Backend> {
    conv: Conv1d<B>,
    linear_actor: Linear<B>,
    linear_critic: Linear<B>,
}

impl<B: Backend> ConvNet<B> {
    #[allow(unused)]
    pub fn new(input_size: usize, dense_size: usize, output_size: usize, kernel_size: usize) -> Self {
        let initializer = Initializer::KaimingUniform { gain: 1.0, fan_out_only: true };
        Self {
            conv: Conv1dConfig::new(1, dense_size, kernel_size)
                .init(&Default::default()),
            linear_actor: LinearConfig::new(dense_size, output_size).with_initializer(initializer.clone()).init(&Default::default()),
            linear_critic: LinearConfig::new(dense_size, 1).with_initializer(initializer).init(&Default::default()),
        }
    }
}

impl<B: Backend> Model<B, Tensor<B, 2>, PPOOutput<B>, Tensor<B, 2>> for ConvNet<B> {
    fn forward(&self, input: Tensor<B, 2>) -> PPOOutput<B> {
        // (B, F) -> (B, 1, F)
        let x = input.unsqueeze_dim(1);
        // (B, C, L_out)
        let x = silu(self.conv.forward(x));
        // Global average over the temporal axis -> (B, C)
        let x = x.mean_dim(2).squeeze(2);

        let policies = softmax(self.linear_actor.forward(x.clone()), 1);
        let values = self.linear_critic.forward(x);
        PPOOutput::<B>::new(policies, values)
    }

    fn infer(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = input.unsqueeze_dim(1);
        let x = silu(self.conv.forward(x));
        let x = x.mean_dim(2).squeeze(2);
        softmax(self.linear_actor.forward(x), 1)
    }
}

impl<B: Backend> PPOModel<B> for ConvNet<B> {}

pub type ConvAgent<E, B> = PPO<E, B, ConvNet<B>>;