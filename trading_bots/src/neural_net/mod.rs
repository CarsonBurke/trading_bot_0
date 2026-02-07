use ndarray::Array2;

pub mod create;
pub mod train_genetic;

pub struct Replay {
    // The inputs provided to the network at a certain point
    pub state: Array2<f32>,
    // How much percent profit was made from this action
    pub profit_percent: f64,
    // How much the network was rewarded for this action for the coinciding state
    pub reward: f32,
    // The amount of stock that was bought or sold (sold would be negative)
    // Maybe this should be the output parameters instead
    pub action: f64,
}
