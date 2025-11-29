use tch::nn::Init;
use tch::{nn, Tensor};

use crate::torch::constants::{
    PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS, TICKERS_COUNT
};

// Temporal length after all conv layers: 
// 3400 -> 1697 -> 849 -> 425 -> 425 -> 425
// 4400 -> 2197 -> 1099 -> 550 -> 550 -> 550
const CONV_TEMPORAL_LEN: i64 = 550;

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
    let gn2 = nn::group_norm(p / "gn2", 8, 64, Default::default());
    let c2_dw = nn::conv1d(p / "c2_dw", 64, 64, 5, nn::ConvConfig { stride: 2, padding: 2, groups: 64, ..Default::default() });  // Depthwise
    let c2_pw = nn::conv1d(p / "c2_pw", 64, 128, 1, Default::default());  // Pointwise 1x1 to expand

    // Block 3: 128 -> 256 channels (depthwise + pointwise)
    let gn3 = nn::group_norm(p / "gn3", 8, 128, Default::default());
    let c3_dw = nn::conv1d(p / "c3_dw", 128, 128, 3, nn::ConvConfig { stride: 2, padding: 1, groups: 128, ..Default::default() });  // Depthwise
    let c3_pw = nn::conv1d(p / "c3_pw", 128, 256, 1, Default::default());  // Pointwise 1x1 to expand

    // Block 4: 256 -> 256 channels (depthwise + pointwise)
    let gn4 = nn::group_norm(p / "gn4", 8, 256, Default::default());
    let c4_dw = nn::conv1d(p / "c4_dw", 256, 256, 3, nn::ConvConfig { stride: 1, padding: 1, groups: 256, ..Default::default() });  // Depthwise
    let c4_pw = nn::conv1d(p / "c4_pw", 256, 256, 1, Default::default());  // Pointwise 1x1 (maintain)

    // Block 5: 256 -> 256 channels (depthwise + pointwise) with residual
    let gn5 = nn::group_norm(p / "gn5", 8, 256, Default::default());
    let c5_dw = nn::conv1d(p / "c5_dw", 256, 256, 3, nn::ConvConfig { stride: 1, padding: 1, groups: 256, ..Default::default() });  // Depthwise
    let c5_pw = nn::conv1d(p / "c5_pw", 256, 256, 1, Default::default());  // Pointwise 1x1 (maintain)

    // Cross-ticker attention for learning inter-asset relationships
    // Simple multi-head self-attention over ticker dimension
    let num_heads = 4;
    let head_dim = 64;  // 256 / 4
    let attn_qkv = nn::linear(p / "attn_qkv", 256, 256 * 3, Default::default());
    let attn_out = nn::linear(p / "attn_out", 256, 256, Default::default());
    let ln_attn = nn::layer_norm(p / "ln_attn", vec![256], Default::default());

    // FC layers: conv features + static observations
    // After attention: [batch, TICKERS, 256] -> [batch, TICKERS*256]
    let conv_output_size = TICKERS_COUNT * 256;
    let fc1_input_size = conv_output_size + STATIC_OBSERVATIONS as i64;

    // More efficient compression with GAP - dramatically reduced parameter count
    // GAP reduces conv features from 37,888 to 256, so fc1 goes from ~39M to ~153K params
    let fc1 = nn::linear(p / "l1", fc1_input_size, 512, Default::default());
    let ln_fc1 = nn::layer_norm(p / "ln_fc1", vec![512], Default::default());

    // Split actor/critic paths early (at FC2) for better specialization
    let fc2_actor = nn::linear(p / "l2_actor", 512, 256, Default::default());
    let ln_fc2_actor = nn::layer_norm(p / "ln_fc2_actor", vec![256], Default::default());
    let fc2_critic = nn::linear(p / "l2_critic", 512, 256, Default::default());
    let ln_fc2_critic = nn::layer_norm(p / "ln_fc2_critic", vec![256], Default::default());

    // Final FC layers for actor and critic with scaled-down init
    let fc3_actor_cfg = nn::LinearConfig {
        ws_init: Init::Uniform { lo: -0.02, up: 0.02 },
        ..Default::default()
    };
    let fc3_actor = nn::linear(p / "l3_actor", 256, 256, fc3_actor_cfg);
    let ln_fc3_actor = nn::layer_norm(p / "ln_fc3_actor", vec![256], Default::default());
    let fc3_critic = nn::linear(p / "l3_critic", 256, 256, Default::default());
    let ln_fc3_critic = nn::layer_norm(p / "ln_fc3_critic", vec![256], Default::default());

    // Critic head: small weights, zero bias for initial values near 0
    let critic_cfg = nn::LinearConfig {
        ws_init: Init::Uniform { lo: -0.01, up: 0.01 },
        bs_init: Some(Init::Const(0.0)),
        bias: true,
    };
    let critic = nn::linear(p / "cl", 256, 1, critic_cfg);

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

    // State-independent log_std parameter for more stable exploration
    let log_std_param = p.var("log_std", &[nact], Init::Const(0.0));

    // Learnable positional embedding for temporal dimension before GAP
    // Allows model to weight recent vs old patterns differently
    let pos_embedding = p.var("pos_emb", &[1, 256, CONV_TEMPORAL_LEN], Init::Uniform { lo: -0.01, up: 0.01 });

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

        // Block 5: Pre-GN → Depthwise → Pointwise → SiLU + Residual
        let x5_input = x.shallow_clone();  // Save for residual
        let x = x.apply(&gn5).apply(&c5_dw).apply(&c5_pw);
        let x = (x + x5_input).silu();  // Residual connection + activation

        // Add positional embedding before GAP to preserve recency information
        let x = x + &pos_embedding;

        // Apply Global Average Pooling per ticker using adaptive pooling
        // Shape-agnostic: [batch*TICKERS, 256, T] -> [batch*TICKERS, 256, 1] -> [batch*TICKERS, 256]
        let gap_output = x.adaptive_avg_pool1d(1).squeeze_dim(2);

        // Reshape for cross-ticker attention: [batch*TICKERS, 256] -> [batch, TICKERS, 256]
        let ticker_features = gap_output.view([batch_size, TICKERS_COUNT, 256]);

        // Multi-head self-attention over tickers
        let qkv = ticker_features.apply(&attn_qkv);  // [batch, TICKERS, 768]
        let qkv = qkv.view([batch_size, TICKERS_COUNT, 3, num_heads, head_dim]);
        let qkv = qkv.permute(&[2, 0, 3, 1, 4]);  // [3, batch, heads, TICKERS, head_dim]

        let q = qkv.get(0);  // [batch, heads, TICKERS, head_dim]
        let k = qkv.get(1);
        let v = qkv.get(2);

        // Scaled dot-product attention
        let scale = (head_dim as f64).sqrt();
        let attn_scores = q.matmul(&k.transpose(-2, -1)) / scale;  // [batch, heads, TICKERS, TICKERS]
        let attn_weights = attn_scores.softmax(-1, attn_scores.kind());
        let attn_output = attn_weights.matmul(&v);  // [batch, heads, TICKERS, head_dim]

        // Reshape and project back
        let attn_output = attn_output.permute(&[0, 2, 1, 3]).contiguous();
        let attn_output = attn_output.view([batch_size, TICKERS_COUNT, 256]);
        let attn_output = attn_output.apply(&attn_out);

        // Residual connection + LayerNorm
        let ticker_features = (ticker_features + attn_output).apply(&ln_attn);

        // Flatten ticker features: [batch, TICKERS, 256] -> [batch, TICKERS*256]
        let conv_features = ticker_features.view([batch_size, -1]);

        // Concatenate conv features with static observations
        let combined = Tensor::cat(&[conv_features, static_features], 1);

        // Shared FC1 layer with Pre-LayerNorm
        let fc1_out = combined.apply(&fc1);
        let fc1_norm = fc1_out.apply(&ln_fc1).silu();

        // Split into separate actor and critic paths at FC2
        let fc2_actor_out = fc1_norm.apply(&fc2_actor);
        let actor_fc2_norm = fc2_actor_out.apply(&ln_fc2_actor).silu();

        let fc2_critic_out = fc1_norm.apply(&fc2_critic);
        let critic_fc2_norm = fc2_critic_out.apply(&ln_fc2_critic).silu();

        // Final actor and critic layers with residual connections
        let actor_out = actor_fc2_norm.apply(&fc3_actor);
        let actor_features = (&actor_out + &actor_fc2_norm).apply(&ln_fc3_actor).silu();

        let critic_out = critic_fc2_norm.apply(&fc3_critic);
        let critic_features = (&critic_out + &critic_fc2_norm).apply(&ln_fc3_critic).silu();

        let critic_value = critic_features.apply(&critic);

        // Gaussian distribution parameters (before sigmoid squashing)
        let action_mean = actor_features.apply(&actor_mean);

        // State-independent log_std with smooth bounded range via tanh
        // tanh gives range [-2.0, 2.0], so std in [0.135, 7.39]
        // Smoother gradients than clamp, no dead zones
        let action_log_std = log_std_param.tanh() * 2.0;

        (critic_value, (action_mean, action_log_std))
    })
}
