use burn::prelude::*;

use crate::burn::{agent::base::{ElemType, State}, constants::OBSERVATION_SIZE};

/// Observation data to feed into the neural network.
#[derive(Debug, Clone, Copy)]
pub struct ObservationState {
    data: [ElemType; OBSERVATION_SIZE]
}

impl ObservationState {
    pub fn new() -> Self {
        let data = [ElemType::default(); 2];
        Self { data }
    }
}

impl State for ObservationState {
    type Data = [ElemType; OBSERVATION_SIZE];
    
    fn size() -> usize {
        OBSERVATION_SIZE
    }
    
    fn to_tensor<B: Backend>(&self) -> Tensor<B, 1> {
        Tensor::<B, 1>::from_floats(self.data, &Default::default())
    }
}