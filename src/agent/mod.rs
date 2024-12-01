use std::fs;

use enum_map::{enum_map, EnumMap};
use rand::Rng;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::constants::agent::{LEARNING_RATE, MAX_WEIGHT, MIN_WEIGHT};

pub mod create;
pub mod runner;
pub mod train;

#[derive(Default)]
pub struct Agent {
    pub weights: Weights,
    pub id: Uuid,
}

impl Agent {
    pub fn from_weights_file() -> Self {

        let file = fs::read("weights/weights.bin").unwrap();
        let weights: Weights = postcard::from_bytes(&file).unwrap();

        Self {
            id: Uuid::new_v4(),
            weights,
        }
    }
}

impl Clone for Agent {
    // Clones the agent, creating a new id for it
    fn clone(&self) -> Self {
        Self {
            id: Uuid::new_v4(),
            weights: self.weights,
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
            // consider adjusting learning rate as the generations go on, to slowly get closer to local minima gradient descent
            *weight += rng.gen_range(-LEARNING_RATE..LEARNING_RATE);
            // could clamp it. I think I will because it is similar to RELU
            *weight = weight.clamp(-1.0, 1.0);
        }
    }
}

impl Default for Weights {
    fn default() -> Self {

        let mut rng = rand::thread_rng();

        Self {
            /* map: enum_map! {
                Weight::MinRsiSell => 0.7,
                Weight::MaxRsiBuy => 0.3,
                Weight::RsiBuyAmountWeight => rng.gen_range(MIN_WEIGHT..MAX_WEIGHT),
                Weight::RsiSellAmountWeight => rng.gen_range(MIN_WEIGHT..MAX_WEIGHT),
                Weight::DiffToBuy => rng.gen_range(MIN_WEIGHT..MAX_WEIGHT),
                Weight::DiffToSell => rng.gen_range(MIN_WEIGHT..MAX_WEIGHT),
                Weight::DeciderRsiEmaAlpha => rng.gen_range(MIN_WEIGHT..MAX_WEIGHT),
                Weight::AmountRsiEmaAlpha => rng.gen_range(MIN_WEIGHT..MAX_WEIGHT),
                Weight::ReboundSellThreshold => rng.gen_range(MIN_WEIGHT..MAX_WEIGHT),
                Weight::ReboundBuyThreshold => rng.gen_range(MIN_WEIGHT..MAX_WEIGHT),
            }, */
            map: enum_map! {
                Weight::MinRsiSell => 0.7,
                Weight::MaxRsiBuy => 0.3,
                Weight::RsiBuyAmountWeight => 0.5,
                Weight::RsiSellAmountWeight => 0.5,
                Weight::DiffToBuy => 0.05,
                Weight::DiffToSell => 0.05,
                Weight::DeciderRsiEmaAlpha => 0.02/* 15 */,
                Weight::AmountRsiEmaAlpha => 0.02,
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
    DiffToBuy,
    DiffToSell,
    DeciderRsiEmaAlpha,
    AmountRsiEmaAlpha,
    // What RSI amount / 100 the position needs to go back up before it can be purchased
    ReboundSellThreshold,
    // What RSI amount / 100 the position needs to go down before it can be sold
    ReboundBuyThreshold,
}
