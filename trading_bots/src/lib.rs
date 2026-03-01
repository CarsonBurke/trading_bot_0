// For burn Wgpu
#![recursion_limit = "256"]
#![feature(f16)]
#![feature(stdarch_x86_avx512_bf16)]

pub mod agent;
pub mod charts;
pub mod constants;
pub mod data;
pub mod history;
pub mod neural_net;
pub mod strategies;
pub mod torch;
pub mod types;
pub mod utils;
