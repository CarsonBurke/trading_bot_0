use enum_map::{enum_map, EnumMap};
use hashbrown::HashMap;
use ibapi::market_data::historical;
use rust_neural_network::neural_network::{Input, NeuralNetwork};

use crate::{constants::agent::TARGET_AGENT_COUNT, types::MappedHistorical, utils::{convert_historical, ema, ema_diff_percent, get_macd, get_rsi_percents, get_rsi_values, get_stochastic_oscillator, get_w_percent_range}};

pub fn create_networks(inputs: &[Input], output_count: usize) -> HashMap<u32, NeuralNetwork> {
    let mut neural_nets = HashMap::new();

    for i in 0..TARGET_AGENT_COUNT {
        let mut neural_net = NeuralNetwork::new();
        neural_net.build(inputs, output_count);
        neural_net.mutate();
        neural_nets.insert(neural_net.id, neural_net);
    }

    neural_nets
}

pub fn create_mapped_indicators(mapped_data: &MappedHistorical) -> HashMap<String, Indicators> {
    let mut indicators = HashMap::new();

    for (ticker, bars) in mapped_data.iter() {
        let data = convert_historical(bars);
        indicators.insert(ticker.to_string(), create_indicators(data, bars));
    }

    indicators
}

fn create_indicators(data: Vec<f64>, bars: &[historical::Bar]) -> Indicators {
    enum_map! {
        Indicator::EMADiff7 => ema_diff_percent(&data, 1. / 7.),
        Indicator::EMADiff14 => ema_diff_percent(&data, 1. / 14.),
        Indicator::EMADiff50 => ema_diff_percent(&data, 1. / 50.),
        Indicator::EMADiff100 => ema_diff_percent(&data, 1. / 100.),
        Indicator::EMADiff1000 => ema_diff_percent(&data, 1. / 1000.),
        Indicator::RSI7 => get_rsi_percents(&data, 1. / 7.),
        Indicator::RSI14 => get_rsi_percents(&data, 1. / 14.),
        Indicator::RSI28 => get_rsi_percents(&data, 1. / 28.),
        Indicator::RSI50 => get_rsi_percents(&data, 1. / 50.),
        Indicator::RSI100 => get_rsi_percents(&data, 1. / 100.),
        Indicator::StochasticOscillator => get_stochastic_oscillator(bars),
        Indicator::MACDDiff => get_macd(&data),
        Indicator::WilliamsPercentRange => get_w_percent_range(bars),
    }
}

// pub struct Indicators {
//     pub ema_14: Vec<f64>,
//     pub ema_50: Vec<f64>,
//     pub ema_100: Vec<f64>,
//     pub rsi_14: Vec<f64>,
//     pub rsi_50: Vec<f64>,
//     pub rsi_100: Vec<f64>,
//     pub stoch_oscillator: Vec<f64>,
//     pub macd: Vec<f64>,
//     pub williams_percent_range: Vec<f64>,
// }

// impl Indicators {
//     pub fn new(data: &[f64]) -> Self {
//         Indicators {
//             ema_14: ema(data, 1. / 14.),
//             ema_50: ema(data, 1. / 50.),
//             ema_100: ema(data, 1. / 100.),
//             rsi_14: get_rsi_values(data, 1. / 14.),
//             rsi_50: get_rsi_values(data, 1. / 50.),
//             rsi_100: get_rsi_values(data, 1. / 100.),
//             stoch_oscillator: get_rsi_values(data, 1. / 14.),
//             macd: get_rsi_values(data, 1. / 14.),
//             williams_percent_range: get_rsi_values(data, 1. / 14.),
//         }
//     }
// }

pub type Indicators = EnumMap<Indicator, Vec<f64>>;

#[derive(enum_map::Enum, Clone, Copy, )]
pub enum Indicator {
    EMADiff7,
    EMADiff14,
    EMADiff50,
    EMADiff100,
    EMADiff1000,
    RSI7,
    RSI14,
    RSI28,
    RSI50,
    RSI100,
    StochasticOscillator,
    /// Difference between the 12 peroid EMA and 26 period EMA
    MACDDiff,
    WilliamsPercentRange
}

