use burn::prelude::*;

use crate::{burn::{agent::base::{ElemType, State}, constants::OBSERVATION_SIZE}, types::Account};

pub type ObservationData = [ElemType; OBSERVATION_SIZE];

/// Observation data to feed into the neural network.
#[derive(Debug, Clone, Copy)]
pub struct ObservationState {
    data: ObservationData
}

impl ObservationState {
    pub fn new(step: usize, account: &Account, current_prices: &[f64], ticker_price_deltas: &[Vec<f64>]) -> Self {
        let mut data = Vec::new();

        data.push(account.cash_cost_basis_ratio());
        data.extend(account.position_percents(current_prices));

        for i in 0..(OBSERVATION_SIZE - data.len()) {
            for price_deltas in ticker_price_deltas.iter() {
                let price_diff = match price_deltas.get(step - i) {
                    Some(price_diff) => *price_diff,
                    None => 0.0,
                };
                data.push(price_diff)
            }

        }

        Self { data: data.try_into().unwrap() }
    }

    pub fn new_empty() -> Self {
        Self { data: [0.0; OBSERVATION_SIZE] }
    }
}

impl State for ObservationState {
    type Data = ObservationData;

    fn size() -> usize {
        OBSERVATION_SIZE
    }

    fn to_tensor<B: Backend>(&self) -> Tensor<B, 1> {
        Tensor::<B, 1>::from_floats(self.data, &Default::default())
    }
}
