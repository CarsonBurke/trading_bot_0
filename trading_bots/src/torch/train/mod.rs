mod advantages;
pub(crate) mod config;
pub(crate) mod gae;
mod geometry;
mod log;
mod loop_;
mod numeric_debug;
mod optimizer_glue;
pub mod pretrain;
mod rollout;
mod sample;
mod trainer;
mod update;
pub(crate) mod value_loss;

pub use loop_::train;
pub use pretrain::{pretrain, PretrainArgs, PretrainObjective};
