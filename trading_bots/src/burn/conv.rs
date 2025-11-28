use burn::{
    backend::{Autodiff, NdArray, Vulkan, Wgpu}, grad_clipping::GradientClippingConfig, module::Module, nn::{conv::{Conv1d, Conv1dConfig}, Initializer, Linear, LinearConfig, PaddingConfig1d}, optim::AdamWConfig, prelude::Backend, tensor::{
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

const CONV_LAYERS: usize = 3;

#[derive(Module, Debug)]
pub struct ConvNet<B: Backend> {
    conv_in: Conv1d<B>,
    channels_in: usize,
    conv_size: usize,
    conv_layers: Vec<Conv1d<B>>,
    linear_actor: Linear<B>,
    linear_critic: Linear<B>,
}

impl<B: Backend> ConvNet<B> {
    #[allow(unused)]
    pub fn new(input_size: usize, dense_size: usize, output_size: usize, mut kernel_size: usize, ticker_count: usize) -> Self {
        
        let initializer = Initializer::KaimingUniform { gain: 1.0, fan_out_only: true };
        let conv_size = input_size / ticker_count;
        
        let conv_layers = (0..CONV_LAYERS).map(|i| {
            let dilation = 2_usize.pow(i as u32);
            if dilation > conv_size * 2 {
                panic!("Dilation {} exceeds convolution size {}", dilation, conv_size);
            }
            
            // Calculate padding to maintain same output size
            let padding = dilation * (kernel_size - 1) / 2;
            
            Conv1dConfig::new(dense_size, dense_size, kernel_size)
                .with_dilation(dilation)
                .with_groups(dense_size)
                .with_padding(PaddingConfig1d::Explicit(padding))
                .init(&Default::default())
        }).collect();

        Self {
            conv_in: Conv1dConfig::new(ticker_count, dense_size, 1)
                .with_padding(PaddingConfig1d::Valid)
                .init(&Default::default()),
            channels_in: ticker_count,
            conv_size,
            conv_layers,
            linear_actor: LinearConfig::new(dense_size, output_size).with_initializer(initializer.clone()).init(&Default::default()),
            linear_critic: LinearConfig::new(dense_size, 1).with_initializer(initializer).init(&Default::default()),
        }
    }
    
    #[inline]
    fn to_ncl(&self, x: Tensor<B, 2>) -> Tensor<B, 3> {
        // (B, F) -> (B, cin, len)
        let b = x.dims()[0];
        x.reshape([b, self.channels_in, self.conv_size])
    }
}

impl<B: Backend> Model<B, Tensor<B, 2>, PPOOutput<B>, Tensor<B, 2>> for ConvNet<B> {
    fn forward(&self, input: Tensor<B, 2>) -> PPOOutput<B> {
        // Shape to (B, C_in, L)
        let x = self.to_ncl(input);

        // Project to hidden channels once
        let mut h = self.conv_in.forward(x);

        // Residual dilated stack: h <- h + SiLU(Conv_dilated(h))
        for conv in &self.conv_layers {
            let z = silu(conv.forward(h.clone()));
            h = h + z;
        }

        // Global average pool over time -> (B, H)
        let h = h.mean_dim(2).squeeze(2);

        let policies = softmax(self.linear_actor.forward(h.clone()), 1);
        let values = self.linear_critic.forward(h);
        PPOOutput::<B>::new(policies, values)
    }

    fn infer(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.to_ncl(input);
        let mut h = self.conv_in.forward(x);
        
        for conv in &self.conv_layers {
            let z = silu(conv.forward(h.clone()));
            h = h + z;
        }
        let h = h.mean_dim(2).squeeze(2);
        
        softmax(self.linear_actor.forward(h), 1)
    }
}

impl<B: Backend> PPOModel<B> for ConvNet<B> {}

pub type ConvAgent<E, B> = PPO<E, B, ConvNet<B>>;