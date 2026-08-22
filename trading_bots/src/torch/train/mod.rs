mod advantages;
pub mod bar_family;
pub(crate) mod config;
pub(crate) mod gae;
mod geometry;
pub mod eval_budget;
pub mod growth;
pub mod horizon;
mod log;
mod loop_;
pub mod lr_disentangle;
pub mod mem_probe;
pub(crate) mod numeric_debug;
pub(crate) mod optimizer_glue;
pub mod portfolio;
pub mod portfolio_cost;
pub mod pretrain;
pub mod recirculate;
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

pub use horizon::{
    run_receding_evaluation, RecedingArgs, RecedingBench, DEFAULT_FORECAST_HORIZON,
    FORECAST_HORIZONS,
};
pub use loop_::train;
pub use pretrain::{
    pretrain, pretrain_calibration, pretrain_candles, pretrain_trade, CalibrationArgs, CandleArgs,
    CorpusFlags, PretrainArgs, TradeArgs,
};
