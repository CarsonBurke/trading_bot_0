use std::fs;

use enum_map::{enum_map, EnumMap};
use rand::Rng;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::constants::agent::LEARNING_RATE;

pub mod create;
pub mod runner;
pub mod train;

#[derive(Default, Clone)]
pub struct Agent {
    pub weights: Weights,
    pub id: Uuid,
}

impl Agent {
    pub fn from_weights_file() -> Self {

        let file = fs::read("weights/weights.bin").unwrap();
        let weights: Weights = postcard::from_bytes(&file).unwrap();

        Self {
            id: Uuid::default(),
            weights,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Weights {
    pub map: WeightMap,
}

impl Weights {
    fn mutate(&mut self) {
        let mut rng = rand::thread_rng();

        for weight in self.map.values_mut() {
            *weight += rng.gen_range(-LEARNING_RATE..LEARNING_RATE);
            // could clamp it. I think I will because it is similar to RELU
            *weight = weight.clamp(0.0, 1.0);
        }
    }
}

impl Default for Weights {
    fn default() -> Self {
        Self {
            map: enum_map! {
                Weight::MinRsiSell => 0.6,
                Weight::MaxRsiBuy => 0.4,
                Weight::RsiBuyAmountWeight => 0.8,
                Weight::RsiSellAmountWeight => 0.3,
                Weight::MaxChange => 0.1,
                Weight::DiffToBuy => 0.05,
                Weight::DiffToSell => 0.05,
                Weight::RsiEmaAlpha => 0.2,
                Weight::ReboundSellThreshold => 0.05,
                Weight::ReboundBuyThreshold => 0.05,
            },
        }
    }
}

impl ToString for Weights {
    fn to_string(&self) -> String {
        format!("{:?}", self)
    }
}

pub type WeightMap = EnumMap<Weight, f64>;

#[derive(Clone, Copy, enum_map::Enum, Debug, Serialize, Deserialize)]
pub enum Weight {
    MinRsiSell,
    MaxRsiBuy,
    RsiBuyAmountWeight,
    RsiSellAmountWeight,
    MaxChange,
    DiffToBuy,
    DiffToSell,
    RsiEmaAlpha,
    // What RSI amount / 100 the position needs to go back up before it can be purchased
    ReboundSellThreshold,
    // What RSI amount / 100 the position needs to go down before it can be sold
    ReboundBuyThreshold,
}
