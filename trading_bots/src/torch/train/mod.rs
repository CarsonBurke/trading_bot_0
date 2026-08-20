mod advantages;
pub mod bar_family;
pub(crate) mod config;
pub(crate) mod gae;
mod geometry;
pub mod growth;
pub mod horizon;
mod log;
pub mod lr_disentangle;
mod loop_;
pub mod mem_probe;
pub(crate) mod numeric_debug;
pub(crate) mod optimizer_glue;
pub mod portfolio;
pub mod portfolio_cost;
pub mod pretrain;
pub mod pretrain_aux;
pub mod pretrain_reports;
pub mod pretrain_stats;
mod rollout;
mod sample;
pub mod skill;
pub mod split_seams;
pub mod support_moments;
pub mod trade_bench;
mod trainer;
pub(crate) mod update;
pub(crate) mod value_loss;

pub use loop_::train;
pub use pretrain::{
    pretrain, pretrain_calibration, pretrain_candles, pretrain_trade, CalibrationArgs, CandleArgs,
    CorpusFlags, PretrainArgs, TradeArgs,
};
