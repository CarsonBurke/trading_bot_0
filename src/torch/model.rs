use tch::nn::Init;
use tch::{nn, Tensor};

use crate::torch::constants::{
    PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS, TICKERS_COUNT
};

// Model returns (critic_value, (action_mean, action_log_std))
// Actions are sampled from Gaussian then sigmoid-squashed to approximately [-1, 1]
pub type Model = Box<dyn Fn(&Tensor, &Tensor, bool) -> (Tensor, (Tensor, Tensor))>;

/// # Returns
/// A boxed function that takes (price_deltas, static_obs, train) and returns (critic_value, (action_mean, action_log_std))
pub fn model(p: &nn::Path, nact: i64) -> Model {
    let stride = |s| nn::ConvConfig {
        stride: s,
        ..Default::default()
    };

    // ConvNeXt-style architecture: depthwise conv + pointwise (1x1) expansion/contraction
    // Pre-norm: GroupNorm applied BEFORE conv, no running stats, more stable than BatchNorm

    // Initial stem: regular conv to expand from 1 -> 64 channels (can't do depthwise with 1 channel)
    // No GroupNorm on input - preserves price delta scale information
    let c1 = nn::conv1d(p / "c1", 1, 64, 8, stride(2));  // Regular conv for stem

    // Block 2: 64 -> 128 channels (depthwise + pointwise)
    let gn2 = nn::group_norm(p / "gn2", 32, 64, Default::default());
    let c2_dw = nn::conv1d(p / "c2_dw", 64, 64, 5, nn::ConvConfig { stride: 2, groups: 64, ..Default::default() });  // Depthwise
    let c2_pw = nn::conv1d(p / "c2_pw", 64, 128, 1, Default::default());  // Pointwise 1x1 to expand

    // Block 3: 128 -> 256 channels (depthwise + pointwise)
    let gn3 = nn::group_norm(p / "gn3", 32, 128, Default::default());
    let c3_dw = nn::conv1d(p / "c3_dw", 128, 128, 3, nn::ConvConfig { stride: 2, groups: 128, ..Default::default() });  // Depthwise
    let c3_pw = nn::conv1d(p / "c3_pw", 128, 256, 1, Default::default());  // Pointwise 1x1 to expand

    // Block 4: 256 -> 256 channels (depthwise + pointwise)
    let gn4 = nn::group_norm(p / "gn4", 32, 256, Default::default());
    let c4_dw = nn::conv1d(p / "c4_dw", 256, 256, 3, nn::ConvConfig { stride: 2, groups: 256, ..Default::default() });  // Depthwise
    let c4_pw = nn::conv1d(p / "c4_pw", 256, 256, 1, Default::default());  // Pointwise 1x1 (maintain)

    // FC layers: conv features + static observations
    // Calculate expected conv output size with GAP:
    // After conv4: [batch*TICKERS, 256, 148]
    // After GAP: [batch*TICKERS, 256] -> [batch, TICKERS*256]
    // Per ticker: 256 features (after global average pooling)
    let conv_output_size = TICKERS_COUNT * 256;
    let fc1_input_size = conv_output_size + STATIC_OBSERVATIONS as i64;

    // More efficient compression with GAP - dramatically reduced parameter count
    // GAP reduces conv features from 37,888 to 256, so fc1 goes from ~39M to ~153K params
    let fc1 = nn::linear(p / "l1", fc1_input_size, 512, Default::default());
    let ln_fc1 = nn::layer_norm(p / "ln_fc1", vec![512], Default::default());
    let fc2 = nn::linear(p / "l2", 512, 256, Default::default());
    let ln_fc2 = nn::layer_norm(p / "ln_fc2", vec![256], Default::default());

    // Separate paths for actor and critic after shared features
    let fc3_actor = nn::linear(p / "l3_actor", 256, 256, Default::default());
    let ln_fc3_actor = nn::layer_norm(p / "ln_fc3_actor", vec![256], Default::default());
    let fc3_critic = nn::linear(p / "l3_critic", 256, 256, Default::default());
    let ln_fc3_critic = nn::layer_norm(p / "ln_fc3_critic", vec![256], Default::default());

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
    Box::new(move |price_deltas: &Tensor, static_features: &Tensor, train: bool| {
        let price_deltas = price_deltas.to_device(device);
        let static_features = static_features.to_device(device);
        let batch_size = price_deltas.size()[0];

        // Reshape price deltas for conv: [batch, TICKERS * PRICE_DELTAS] -> [batch*TICKERS, 1, PRICE_DELTAS]
        let price_deltas_reshaped = price_deltas
            .reshape(&[batch_size, TICKERS_COUNT, PRICE_DELTAS_PER_TICKER as i64])
            .reshape(&[batch_size * TICKERS_COUNT, 1, PRICE_DELTAS_PER_TICKER as i64]);

        // Apply ConvNeXt-style blocks with depthwise-pointwise convolutions
        // One SiLU per block after pointwise conv

        // Block 1: Regular Conv stem (no pre-norm to preserve input scale)
        let x = price_deltas_reshaped.apply(&c1).silu();

        // Block 2: Pre-GN → Depthwise → Pointwise → SiLU
        let x = x.apply(&gn2).apply(&c2_dw).apply(&c2_pw).silu();

        // Block 3: Pre-GN → Depthwise → Pointwise → SiLU
        let x = x.apply(&gn3).apply(&c3_dw).apply(&c3_pw).silu();

        // Block 4: Pre-GN → Depthwise → Pointwise → SiLU
        let x = x.apply(&gn4).apply(&c4_dw).apply(&c4_pw).silu();

        // Apply Global Average Pooling per ticker: [batch*TICKERS, 256, 148] -> [batch*TICKERS, 256]
        // This reduces spatial dimension from 148 to 1 by averaging across time
        let gap_output = x.mean_dim(&[2i64][..], false, x.kind());

        // Reshape to combine all ticker features: [batch*TICKERS, 256] -> [batch, TICKERS*256]
        let conv_features = gap_output.view([batch_size, -1]);

        // Concatenate conv features with static observations
        let combined = Tensor::cat(&[conv_features, static_features], 1);

        // Apply shared FC layers with Pre-LayerNorm and residual connections
        // Using SiLU for smooth gradient flow
        let fc1_out = combined.apply(&fc1);
        let fc1_norm = fc1_out.apply(&ln_fc1).silu();

        let fc2_out = fc1_norm.apply(&fc2);
        let shared_features = fc2_out.apply(&ln_fc2).silu();

        // Separate actor and critic paths with Pre-LayerNorm and residual connections
        let actor_out = shared_features.apply(&fc3_actor);
        let actor_features = (&actor_out + &shared_features).apply(&ln_fc3_actor).silu();  // Residual connection

        let critic_out = shared_features.apply(&fc3_critic);
        let critic_features = (&critic_out + &shared_features).apply(&ln_fc3_critic).silu();  // Residual connection

        let critic_value = critic_features.apply(&critic);

        // Gaussian distribution parameters (before sigmoid squashing)
        let action_mean = actor_features.apply(&actor_mean);

        // Clamp log_std to allow sufficient exploration
        // Range [-1.0, 1.0] gives std in [0.368, 2.718]
        // With sigmoid squashing by /2 in ppo.rs, this provides adequate exploration
        // without saturating actions too quickly
        let action_log_std = actor_features.apply(&actor_log_std).clamp(-1.0, 1.0);

        (critic_value, (action_mean, action_log_std))
    })
}
