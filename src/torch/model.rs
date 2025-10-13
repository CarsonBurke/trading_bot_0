use tch::nn::Init;
use tch::{nn, Tensor};

use crate::torch::constants::{
    PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS, TICKERS_COUNT
};

// Model returns (critic_value, (action_mean, action_log_std))
// Actions are sampled from Gaussian then sigmoid-squashed to approximately [-1, 1]
pub type Model = Box<dyn Fn(&Tensor, &Tensor) -> (Tensor, (Tensor, Tensor))>;

/// Creates the PPO neural network model (Actor-Critic architecture)
///
/// The model consists of:
/// 1. Convolutional layers for processing price deltas (temporal patterns)
/// 2. Fully connected layers combining conv features with static observations
/// 3. Actor head: Outputs action mean and log_std for continuous actions
/// 4. Critic head: Outputs state value estimate
///
/// # Arguments
/// * `p` - Neural network path for parameter storage
/// * `nact` - Number of actions (tickers to trade)
///
/// # Returns
/// A boxed function that takes (price_deltas, static_obs) and returns (critic_value, (action_mean, action_log_std))
pub fn model(p: &nn::Path, nact: i64) -> Model {
    let stride = |s| nn::ConvConfig {
        stride: s,
        ..Default::default()
    };

    // Convolutional layers for price deltas only
    let c1 = nn::conv1d(p / "c1", 1, 64, 8, stride(2));
    let bn1 = nn::batch_norm1d(p / "bn1", 64, Default::default());
    let c2 = nn::conv1d(p / "c2", 64, 128, 5, stride(2));
    let bn2 = nn::batch_norm1d(p / "bn2", 128, Default::default());
    let c3 = nn::conv1d(p / "c3", 128, 256, 3, stride(2));
    let bn3 = nn::batch_norm1d(p / "bn3", 256, Default::default());

    // FC layers: conv features + static observations
    // Conv output: ~92160 (5 tickers * 256 channels * ~72 timesteps)
    // Static obs: 32
    // Total: ~92192
    // Use gradual compression: 92192 -> 2048 -> 512 -> 256 to preserve more information
    let fc1 = nn::linear(p / "l1", 92192, 2048, Default::default());
    let fc2 = nn::linear(p / "l2", 2048, 512, Default::default());
    let fc3 = nn::linear(p / "l3", 512, 256, Default::default());

    let critic = nn::linear(p / "cl", 256, 1, Default::default());

    // Actor outputs mean and log_std for a Gaussian distribution
    // Actions will be sigmoid-squashed: action = 2 * sigmoid(z) - 1 where z ~ N(mean, std)
    // Initialize mean with small weights for moderate initial actions (around 0 after squashing)
    let actor_mean_cfg = nn::LinearConfig {
        ws_init: Init::Uniform {
            lo: -0.01,
            up: 0.01,
        },
        bs_init: Some(Init::Const(0.0)), // sigmoid(0) = 0.5, so action = 0
        bias: true,
    };
    let actor_mean = nn::linear(p / "al_mean", 256, nact, actor_mean_cfg);
    let actor_log_std = nn::linear(p / "al_log_std", 256, nact, Default::default());

    let device = p.device();
    Box::new(move |price_deltas: &Tensor, static_features: &Tensor| {
        let price_deltas = price_deltas.to_device(device);
        let static_features = static_features.to_device(device);
        let batch_size = price_deltas.size()[0];

        // Reshape price deltas for conv: [batch, TICKERS * PRICE_DELTAS] -> [batch*TICKERS, 1, PRICE_DELTAS]
        let price_deltas_reshaped = price_deltas
            .reshape(&[batch_size, TICKERS_COUNT, PRICE_DELTAS_PER_TICKER as i64])
            .reshape(&[batch_size * TICKERS_COUNT, 1, PRICE_DELTAS_PER_TICKER as i64]);

        // Apply CNN layers with batch norm in training mode
        // Note: Using true for training mode allows batch norm to update running stats
        // Using SiLU (Swish) for smoother gradients and better performance in deep networks
        let x = price_deltas_reshaped.apply(&c1).apply_t(&bn1, true).silu();
        let x = x.apply(&c2).apply_t(&bn2, true).silu();
        let x = x.apply(&c3).apply_t(&bn3, true).silu();

        // Flatten conv outputs: [batch*TICKERS, 256, L] -> [batch*TICKERS, 256*L]
        let cnn_flat = x.flat_view();

        // Reshape to combine all ticker features: [batch*TICKERS, features] -> [batch, TICKERS*features]
        let conv_features = cnn_flat.view([batch_size, -1]);

        // Concatenate conv features with static observations
        let combined = Tensor::cat(&[conv_features, static_features], 1);

        // Apply FC layers to combined features with gradual compression
        // Using SiLU for smoother gradient flow through deep network
        let features = combined.apply(&fc1).silu().apply(&fc2).silu().apply(&fc3).silu();

        let critic_value = features.apply(&critic);

        // Gaussian distribution parameters (before sigmoid squashing)
        let action_mean = features.apply(&actor_mean);

        // Clamp log_std to allow sufficient exploration
        // Range [-2.0, 0.0] gives std in [0.135, 1.0]
        // This allows more exploration while preventing extreme saturation
        let action_log_std = features.apply(&actor_log_std).clamp(-2.0, 0.0);

        (critic_value, (action_mean, action_log_std))
    })
}
