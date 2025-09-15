use crate::{
    gym::base::State, torch::constants::OBSERVATION_SIZE, types::Account
};

pub type ObservationData = [f16; OBSERVATION_SIZE];

/// Observation data to feed into the neural network.
#[derive(Debug, Clone, Copy)]
pub struct ObservationState {
    data: ObservationData,
}

impl ObservationState {
    pub fn new(
        step: usize,
        account: &Account,
        prices: &[Vec<f64>],
        ticker_price_obs: &[Vec<f64>],
    ) -> Self {
        let mut data: Vec<f16> = Vec::with_capacity(OBSERVATION_SIZE);

        data.push((account.cash / account.total_assets) as f16);
        data.extend(
            account
                .position_percents(prices, step)
                .iter()
                .map(|percent| *percent as f16)
                .collect::<Vec<f16>>(),
        );

        for (ticker_index, position) in account.positions.iter().enumerate() {
            data.push(position.appreciation(prices[ticker_index][step]) as f16);
        }

        // Simple moving averages - normalized

        // for (ticker_index, _) in account.positions.iter().enumerate() {

        //     let moving_average_times = vec![5, 10, 20, 50, 100, 200, 400, 800];
        //     for &time in &moving_average_times {
        //         let ma = prices[ticker_index][step.saturating_sub(time)..=step]
        //             .iter()
        //             .sum::<f64>()
        //             / time as f64;

        //         data.push((ma / 100.0) as f32);
        //     }
        // }

        // for prices in prices.iter() {
        //     for i in 0..(OBSERVATION_SIZE - data.len()) {
        //         if let Some(price) = prices.get(step - i) {
        //             data.push((*price / 100.0) as ElemType);
        //             continue;
        //         }

        //         data.push(0.0)
        //     }
        // }

        for price_deltas in ticker_price_obs.iter() {
            for i in 0..(OBSERVATION_SIZE - data.len()) {
                if let Some(price_diff) = price_deltas.get(step - i) {
                    data.push(*price_diff as f16);
                    continue;
                }

                data.push(0.0)
            }
        }

        // while data.len() < OBSERVATION_SIZE {
        //     data.push(0.0);
        // }

        Self {
            data: data.try_into().unwrap(),
        }
    }

    pub fn new_random() -> Self {
        Self {
            data: (0..OBSERVATION_SIZE)
                .map(|_| rand::random_range(-0.1..0.1) as f16)
                .collect::<Vec<f16>>()
                .try_into()
                .unwrap(),
        }
    }
}

impl State for ObservationState {
    type Data = ObservationData;

    fn size() -> usize {
        OBSERVATION_SIZE
    }

    // fn to_tensor<B: Backend>(&self) -> Tensor<B, 1> {
    //     Tensor::<B, 1>::from_floats(
    //         self.data
    //             .iter()
    //             .map(|&x| x as ElemType)
    //             .collect::<Vec<ElemType>>()
    //             .as_slice(),
    //         &Default::default(),
    //     )
    // }
}
