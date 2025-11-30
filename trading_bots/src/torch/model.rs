use tch::nn::Init;
use tch::{nn, Tensor};

use crate::torch::constants::{
    GLOBAL_STATIC_OBS, PER_TICKER_STATIC_OBS, PRICE_DELTAS_PER_TICKER, TICKERS_COUNT,
};

// Temporal length after all conv layers: 
// 3400 -> 1697 -> 849 -> 425 -> 425 -> 425
// 4400 -> 2197 -> 1099 -> 550 -> 550 -> 550
const CONV_TEMPORAL_LEN: i64 = 425;

// Model returns (critic_value, (action_mean, action_log_std), static_attn_weights)
pub type Model = Box<dyn Fn(&Tensor, &Tensor, bool) -> (Tensor, (Tensor, Tensor), Tensor)>;

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

    // Projection to combine conv features (256) with per-ticker static features
    let static_proj = nn::linear(
        p / "static_proj",
        256 + PER_TICKER_STATIC_OBS as i64,
        256,
        Default::default(),
    );
    let ln_static_proj = nn::layer_norm(p / "ln_static_proj", vec![256], Default::default());

    // Cross-ticker attention for learning inter-asset relationships
    let num_heads = 4;
    let head_dim = 64; // 256 / 4
    let attn_qkv = nn::linear(p / "attn_qkv", 256, 256 * 3, Default::default());
    let attn_out = nn::linear(p / "attn_out", 256, 256, Default::default());
    let ln_attn = nn::layer_norm(p / "ln_attn", vec![256], Default::default());

    // PMA (Pooling by Multihead Attention) - permutation-invariant set pooling
    let pma_num_seeds = 2;
    let pma_seeds = p.var("pma_seeds", &[pma_num_seeds, 256], Init::Uniform { lo: -0.1, up: 0.1 });
    let pma_kv = nn::linear(p / "pma_kv", 256, 256 * 2, Default::default()); // K, V for ticker features
    let pma_q = nn::linear(p / "pma_q", 256, 256, Default::default()); // Q for seeds
    let pma_out = nn::linear(p / "pma_out", 256, 256, Default::default());
    let ln_pma = nn::layer_norm(p / "ln_pma", vec![256], Default::default());

    // (B) Condition PMA seeds on global static features
    let global_to_seed = nn::linear(p / "global_to_seed", GLOBAL_STATIC_OBS as i64, 256, Default::default());

    // FC layers: PMA output (2 seeds * 256 = 512)
    // Global static now flows through PMA seed conditioning, no need to concat here
    let fc1_input_size = 512;
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

    let log_std_param = p.var("log_std", &[nact], Init::Const(0.0));

    // Learnable positional embedding for temporal dimension
    let pos_embedding = p.var("pos_emb", &[1, 256, CONV_TEMPORAL_LEN], Init::Uniform { lo: -0.01, up: 0.01 });

    // (A) Condition temporal pooling on per-ticker statics
    // Project per-ticker statics to 64 dims to modulate temporal attention
    let static_to_temporal = nn::linear(p / "static_to_temporal", PER_TICKER_STATIC_OBS as i64, 64, Default::default());
    let ln_static_temporal = nn::layer_norm(p / "ln_static_temporal", vec![64], Default::default());

    // Channel-grouped temporal attention pooling (4 groups of 64 channels)
    // Each group: concat [64 conv channels, 64 static proj] = 128 -> 1
    let temporal_pool_0 = nn::linear(p / "temporal_pool_0", 128, 1, Default::default());
    let temporal_pool_1 = nn::linear(p / "temporal_pool_1", 128, 1, Default::default());
    let temporal_pool_2 = nn::linear(p / "temporal_pool_2", 128, 1, Default::default());
    let temporal_pool_3 = nn::linear(p / "temporal_pool_3", 128, 1, Default::default());

    let device = p.device();
    Box::new(move |price_deltas: &Tensor, static_features: &Tensor, train: bool| {
        let price_deltas = price_deltas.to_device(device);
        let static_features = static_features.to_device(device);
        let batch_size = price_deltas.size()[0];

        // === Parse static features into global and per-ticker ===
        // Format: [global (7), per_ticker_0 (44), per_ticker_1 (44), ...]
        let global_static = static_features.narrow(1, 0, GLOBAL_STATIC_OBS as i64);
        let per_ticker_static = static_features
            .narrow(1, GLOBAL_STATIC_OBS as i64, TICKERS_COUNT * PER_TICKER_STATIC_OBS as i64)
            .reshape([batch_size, TICKERS_COUNT, PER_TICKER_STATIC_OBS as i64]);

        // === Conv processing (shared weights across tickers) ===
        let price_deltas_reshaped = price_deltas
            .view([batch_size, TICKERS_COUNT, PRICE_DELTAS_PER_TICKER as i64])
            .view([batch_size * TICKERS_COUNT, 1, PRICE_DELTAS_PER_TICKER as i64]);

        let x = price_deltas_reshaped.apply(&c1).silu();
        let x = x.apply(&gn2).apply(&c2_dw).apply(&c2_pw).silu();
        let x = x.apply(&gn3).apply(&c3_dw).apply(&c3_pw).silu();
        let x = x.apply(&gn4).apply(&c4_dw).apply(&c4_pw).silu();

        let x5_input = x.shallow_clone();
        let x = x.apply(&gn5).apply(&c5_dw).apply(&c5_pw);
        let x = (x + x5_input).silu();

        let x = x + &pos_embedding;
        let temporal_len = x.size()[2];

        // (A) Project per-ticker statics to condition temporal attention
        // per_ticker_static: [batch, TICKERS, S] -> [batch*TICKERS, 64]
        let static_temporal_proj = per_ticker_static
            .reshape([batch_size * TICKERS_COUNT, PER_TICKER_STATIC_OBS as i64])
            .apply(&static_to_temporal)
            .apply(&ln_static_temporal);
        // Broadcast over time: [batch*TICKERS, 64] -> [batch*TICKERS, 64, T]
        let static_temporal_broadcast = static_temporal_proj
            .unsqueeze(2)
            .expand(&[-1, -1, temporal_len], false);

        // Channel-grouped attention pooling: [batch*TICKERS, 256, T] -> [batch*TICKERS, 256]
        // Each group concats [64 conv, 64 static] = 128 before scoring
        let x_groups = x.chunk(4, 1); // 4 x [batch*TICKERS, 64, T]
        let temporal_pools = [&temporal_pool_0, &temporal_pool_1, &temporal_pool_2, &temporal_pool_3];

        let mut pooled_groups = Vec::with_capacity(4);
        let mut attn_sum = Tensor::zeros(&[batch_size * TICKERS_COUNT, temporal_len], (x.kind(), x.device()));

        for (group, pool) in x_groups.iter().zip(temporal_pools.iter()) {
            // Concat conv features with static projection: [batch*TICKERS, 128, T]
            let group_with_static = Tensor::cat(&[group.shallow_clone(), static_temporal_broadcast.shallow_clone()], 1);
            let logits = group_with_static
                .permute(&[0, 2, 1])    // [batch*TICKERS, T, 128]
                .apply(*pool)           // [batch*TICKERS, T, 1]
                .squeeze_dim(-1);       // [batch*TICKERS, T]
            let attn = logits.softmax(-1, logits.kind());
            attn_sum = attn_sum + &attn;
            // Pool original conv features (not the concat)
            let pooled_group = group.bmm(&attn.unsqueeze(-1)).squeeze_dim(-1); // [batch*TICKERS, 64]
            pooled_groups.push(pooled_group);
        }

        let pooled = Tensor::cat(&pooled_groups, 1); // [batch*TICKERS, 256]

        // Average temporal attention across groups and tickers for visualization
        let temporal_attn_avg = (attn_sum / 4.0)
            .reshape([batch_size, TICKERS_COUNT, -1])
            .mean_dim(1, false, x.kind());

        // Reshape: [batch*TICKERS, 256] -> [batch, TICKERS, 256]
        let conv_features = pooled.view([batch_size, TICKERS_COUNT, 256]);

        // === Combine conv features with per-ticker static features ===
        // [batch, TICKERS, 256] + [batch, TICKERS, 44] -> [batch, TICKERS, 300]
        let combined_ticker = Tensor::cat(&[conv_features, per_ticker_static], 2);

        // Project back to 256 dims: [batch, TICKERS, 300] -> [batch, TICKERS, 256]
        let combined_ticker = combined_ticker
            .reshape([batch_size * TICKERS_COUNT, 256 + PER_TICKER_STATIC_OBS as i64])
            .apply(&static_proj)
            .apply(&ln_static_proj)
            .silu()
            .reshape([batch_size, TICKERS_COUNT, 256]);

        // === Cross-ticker attention ===
        let qkv = combined_ticker.apply(&attn_qkv);
        let qkv = qkv.reshape([batch_size, TICKERS_COUNT, 3, num_heads, head_dim]);
        let qkv = qkv.permute(&[2, 0, 3, 1, 4]);

        let q = qkv.get(0);
        let k = qkv.get(1);
        let v = qkv.get(2);

        let scale = (head_dim as f64).sqrt();
        let attn_scores = q.matmul(&k.transpose(-2, -1)) / scale;
        let attn_weights = attn_scores.softmax(-1, attn_scores.kind());
        let attn_output = attn_weights.matmul(&v);

        let attn_output = attn_output.permute(&[0, 2, 1, 3]).contiguous();
        let attn_output = attn_output.view([batch_size, TICKERS_COUNT, 256]);
        let attn_output = attn_output.apply(&attn_out);

        let ticker_features = (combined_ticker + attn_output).apply(&ln_attn);

        // === PMA: Pooling by Multihead Attention ===
        // (B) Condition seeds on global static features
        // global_static: [batch, G] -> [batch, 256] -> [batch, 1, 256]
        let global_seed_bias = global_static.apply(&global_to_seed).unsqueeze(1);
        // Seeds: [num_seeds, 256] + global bias -> [batch, num_seeds, 256]
        let seeds_expanded = pma_seeds.unsqueeze(0) + global_seed_bias;

        // Project seeds to Q, ticker_features to K,V
        let pma_q_proj = seeds_expanded.apply(&pma_q); // [batch, num_seeds, 256]
        let pma_kv_proj = ticker_features.apply(&pma_kv); // [batch, TICKERS, 512]
        let pma_kv_split = pma_kv_proj.chunk(2, 2);
        let pma_k = &pma_kv_split[0]; // [batch, TICKERS, 256]
        let pma_v = &pma_kv_split[1]; // [batch, TICKERS, 256]

        // Reshape for multi-head attention: [batch, seq, 256] -> [batch, heads, seq, head_dim]
        let pma_q_heads = pma_q_proj.reshape([batch_size, pma_num_seeds, num_heads, head_dim]).permute(&[0, 2, 1, 3]);
        let pma_k_heads = pma_k.reshape([batch_size, TICKERS_COUNT, num_heads, head_dim]).permute(&[0, 2, 1, 3]);
        let pma_v_heads = pma_v.reshape([batch_size, TICKERS_COUNT, num_heads, head_dim]).permute(&[0, 2, 1, 3]);

        // Scaled dot-product attention: seeds attend to tickers
        let pma_scale = (head_dim as f64).sqrt();
        let pma_scores = pma_q_heads.matmul(&pma_k_heads.transpose(-2, -1)) / pma_scale;
        let pma_weights = pma_scores.softmax(-1, pma_scores.kind());
        let pma_attn_out = pma_weights.matmul(&pma_v_heads); // [batch, heads, num_seeds, head_dim]

        // Reshape back and project
        let pma_attn_out = pma_attn_out.permute(&[0, 2, 1, 3]).contiguous();
        let pma_attn_out = pma_attn_out.view([batch_size, pma_num_seeds, 256]);
        let pma_attn_out = pma_attn_out.apply(&pma_out);

        // Residual + LayerNorm, then flatten
        let pma_out_final = (seeds_expanded + pma_attn_out).apply(&ln_pma);
        let pooled_features = pma_out_final.reshape([batch_size, pma_num_seeds * 256]); // [batch, 512]

        // Shared FC1 layer with Pre-LayerNorm
        let fc1_out = pooled_features.apply(&fc1);
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

        let action_log_std = log_std_param.tanh() * 4.0 - 1.5;

        (critic_value, (action_mean, action_log_std), temporal_attn_avg)
    })
}
