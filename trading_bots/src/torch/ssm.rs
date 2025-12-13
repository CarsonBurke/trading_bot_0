//! Mamba 2 Selective State Space Model
//!
//! Implements architecture from "Transformers are SSMs" (Dao & Gu, 2024):
//! - Multi-head structure (like attention)
//! - Per-head A parameter (not per-channel)
//! - Chunked parallel scan for numerical stability
//! - Fused RMSNorm with gating
//! - Single projection for z, x, B, C, dt

use tch::nn::Init;
use tch::{nn, Kind, Tensor};

const D_STATE: i64 = 64; // 128 is overkill for most tasks, 16-64 is typical
const D_CONV: i64 = 4;
const EXPAND: i64 = 2;
const HEADDIM: i64 = 32; // d_inner/headdim = nheads, want >=2 heads
const NGROUPS: i64 = 1;
const CHUNK_SIZE: i64 = 64; // smaller chunks = less memory, tune based on seq length
const DT_MIN: f64 = 0.001;
const DT_MAX: f64 = 0.1;
const DT_INIT_FLOOR: f64 = 1e-4;

pub struct Mamba2Config {
    pub d_model: i64,
    pub d_state: i64,
    pub d_conv: i64,
    pub expand: i64,
    pub headdim: i64,
    pub ngroups: i64,
    pub chunk_size: i64,
    pub dt_min: f64,
    pub dt_max: f64,
    pub rmsnorm: bool,
    pub norm_before_gate: bool,
    /// If true, D skip connection is per-channel [nheads, headdim] instead of per-head [nheads]
    pub d_has_hdim: bool,
}

impl Default for Mamba2Config {
    fn default() -> Self {
        Self {
            d_model: 64,
            d_state: D_STATE,
            d_conv: D_CONV,
            expand: EXPAND,
            headdim: HEADDIM,
            ngroups: NGROUPS,
            chunk_size: CHUNK_SIZE,
            dt_min: DT_MIN,
            dt_max: DT_MAX,
            rmsnorm: true,
            norm_before_gate: false,
            d_has_hdim: false,
        }
    }
}

impl Mamba2Config {
    pub fn new(d_model: i64) -> Self {
        Self {
            d_model,
            ..Default::default()
        }
    }

    pub fn d_inner(&self) -> i64 {
        self.d_model * self.expand
    }

    pub fn nheads(&self) -> i64 {
        self.d_inner() / self.headdim
    }
}

/// RMSNorm with optional gating fusion and group normalization
struct RMSNormGated {
    weight: Tensor,
    eps: f64,
    norm_before_gate: bool,
    group_size: i64,
}

impl RMSNormGated {
    fn new(p: &nn::Path, dim: i64, eps: f64, norm_before_gate: bool, ngroups: i64) -> Self {
        let weight = p.var("weight", &[dim], Init::Const(1.0));
        let group_size = dim / ngroups;
        Self {
            weight,
            eps,
            norm_before_gate,
            group_size,
        }
    }

    fn forward(&self, x: &Tensor, gate: Option<&Tensor>) -> Tensor {
        let x_f32 = x.to_kind(Kind::Float);
        let orig_shape = x_f32.size();
        let dim = *orig_shape.last().unwrap();
        let ngroups = dim / self.group_size;

        // Reshape for group normalization: [..., ngroups, group_size]
        let reshape_for_norm = |t: &Tensor| -> Tensor {
            let mut shape = t.size();
            shape.pop();
            shape.push(ngroups);
            shape.push(self.group_size);
            t.view(&shape[..])
        };

        let group_rms = |t: &Tensor| -> Tensor {
            let grouped = reshape_for_norm(t);
            // RMS over last dim (group_size), keep dims for broadcasting
            let rms = (grouped.pow_tensor_scalar(2).mean_dim(-1, true, Kind::Float) + self.eps).sqrt();
            (&grouped / rms).flatten(-2, -1)
        };

        let normed = if self.norm_before_gate {
            let normed = group_rms(&x_f32);
            match gate {
                Some(g) => normed * g.silu(),
                None => normed,
            }
        } else {
            match gate {
                Some(g) => {
                    let gated = &x_f32 * g.silu();
                    group_rms(&gated)
                }
                None => group_rms(&x_f32),
            }
        };

        (normed * &self.weight).to_kind(x.kind())
    }
}

/// Inference state for O(1) per-step computation
pub struct Mamba2State {
    /// Conv buffer: [batch, conv_dim, d_conv]
    pub conv_state: Tensor,
    /// SSM hidden state: [batch, nheads, headdim, d_state]
    pub ssm_state: Tensor,
}

impl Mamba2State {
    pub fn reset(&mut self) {
        let _ = self.conv_state.zero_();
        let _ = self.ssm_state.zero_();
    }
}

pub struct Mamba2 {
    config: Mamba2Config,
    in_proj: nn::Linear,
    conv1d: nn::Conv1D,
    dt_bias: Tensor,
    a_log: Tensor,
    d_param: Tensor,
    norm: Option<RMSNormGated>,
    out_proj: nn::Linear,
}

impl Mamba2 {
    pub fn new(p: &nn::Path, config: Mamba2Config) -> Self {
        let d_inner = config.d_inner();
        let nheads = config.nheads();
        let d_state = config.d_state;
        let ngroups = config.ngroups;

        // Single projection: [z, x, B, C, dt]
        // z: d_inner, x: d_inner, B: ngroups*d_state, C: ngroups*d_state, dt: nheads
        let d_in_proj = 2 * d_inner + 2 * ngroups * d_state + nheads;
        let in_proj = nn::linear(
            p / "in_proj",
            config.d_model,
            d_in_proj,
            nn::LinearConfig {
                bias: false,
                ..Default::default()
            },
        );

        // Conv over [x, B, C] for local context on selectivity
        let conv_dim = d_inner + 2 * ngroups * d_state;
        let conv1d = nn::conv1d(
            p / "conv1d",
            conv_dim,
            conv_dim,
            config.d_conv,
            nn::ConvConfig {
                padding: config.d_conv - 1,
                groups: conv_dim,
                bias: true,
                ..Default::default()
            },
        );

        // dt_bias: per-head, initialized via inverse softplus
        let dt_init = Tensor::empty(&[nheads], (Kind::Float, p.device()))
            .uniform_(config.dt_min.ln(), config.dt_max.ln())
            .exp()
            .clamp(DT_INIT_FLOOR, f64::INFINITY);
        // Inverse softplus: x = dt + log(-expm1(-dt))
        let inv_dt = &dt_init + (-&dt_init).expm1().neg().log();
        let dt_bias = p.var_copy("dt_bias", &inv_dt);

        // A: per-head scalar (massive param reduction vs per-channel)
        let a_init =
            Tensor::empty(&[nheads], (Kind::Float, p.device())).uniform_(1.0_f64.ln(), 16.0_f64.ln());
        let a_log = p.var_copy("A_log", &a_init);

        // D: skip connection - per-head or per-channel depending on d_has_hdim
        let d_param = if config.d_has_hdim {
            p.var("D", &[nheads, config.headdim], Init::Const(1.0))
        } else {
            p.var("D", &[nheads], Init::Const(1.0))
        };

        // Optional RMSNorm with gating and group normalization
        let norm = if config.rmsnorm {
            Some(RMSNormGated::new(
                &(p / "norm"),
                d_inner,
                1e-5,
                config.norm_before_gate,
                ngroups,
            ))
        } else {
            None
        };

        let out_proj = nn::linear(
            p / "out_proj",
            d_inner,
            config.d_model,
            nn::LinearConfig {
                bias: false,
                ..Default::default()
            },
        );

        Self {
            config,
            in_proj,
            conv1d,
            dt_bias,
            a_log,
            d_param,
            norm,
            out_proj,
        }
    }

    pub fn forward(&self, u: &Tensor, _train: bool) -> Tensor {
        let (batch, seqlen, _) = u.size3().unwrap();
        let d_inner = self.config.d_inner();
        let nheads = self.config.nheads();
        let headdim = self.config.headdim;
        let d_state = self.config.d_state;
        let ngroups = self.config.ngroups;

        // Project: [z, x, B, C, dt]
        let zxbcdt = u.apply(&self.in_proj);

        let z_dim = d_inner;
        let x_dim = d_inner;
        let b_dim = ngroups * d_state;
        let c_dim = ngroups * d_state;
        let dt_dim = nheads;

        let z = zxbcdt.narrow(-1, 0, z_dim);
        let x = zxbcdt.narrow(-1, z_dim, x_dim);
        let b_raw = zxbcdt.narrow(-1, z_dim + x_dim, b_dim);
        let c_raw = zxbcdt.narrow(-1, z_dim + x_dim + b_dim, c_dim);
        let dt_raw = zxbcdt.narrow(-1, z_dim + x_dim + b_dim + c_dim, dt_dim);

        // Concatenate x, B, C for convolution
        let xbc = Tensor::cat(&[&x, &b_raw, &c_raw], -1);

        // Causal conv1d
        let xbc_conv = xbc
            .transpose(1, 2)
            .apply(&self.conv1d)
            .narrow(2, 0, seqlen)
            .transpose(1, 2)
            .silu();

        // Split back
        let x_conv = xbc_conv.narrow(-1, 0, d_inner);
        let b = xbc_conv.narrow(-1, d_inner, b_dim);
        let c = xbc_conv.narrow(-1, d_inner + b_dim, c_dim);

        // dt: add bias and softplus
        let dt = (&dt_raw + &self.dt_bias).softplus().clamp(self.config.dt_min, self.config.dt_max);

        // A: negative exp for stability
        let a = self.a_log.exp().neg();

        // Reshape for multi-head SSM
        // x: [batch, seq, nheads, headdim]
        let x_heads = x_conv.view([batch, seqlen, nheads, headdim]);
        // B: [batch, seq, ngroups, d_state]
        let b_groups = b.view([batch, seqlen, ngroups, d_state]);
        // C: [batch, seq, ngroups, d_state]
        let c_groups = c.view([batch, seqlen, ngroups, d_state]);
        // dt: [batch, seq, nheads]
        // A: [nheads]

        // Run chunked SSM scan
        let y = self.chunked_ssm_scan(&x_heads, &dt, &a, &b_groups, &c_groups);

        // Add skip connection with D: [batch, seq, nheads, headdim]
        let y_skip = if self.config.d_has_hdim {
            // D is [nheads, headdim] -> [1, 1, nheads, headdim]
            let d_expanded = self.d_param.view([1, 1, nheads, headdim]);
            &y + &x_heads * d_expanded
        } else {
            // D is [nheads] -> [1, 1, nheads, 1]
            let d_expanded = self.d_param.view([1, 1, nheads, 1]);
            &y + &x_heads * d_expanded
        };

        // Flatten heads: [batch, seq, d_inner]
        let y_flat = y_skip.view([batch, seqlen, d_inner]);

        // Norm and gate
        let y_out = match &self.norm {
            Some(norm) => {
                let z_for_gate = z.view([batch, seqlen, d_inner]);
                norm.forward(&y_flat, Some(&z_for_gate))
            }
            None => &y_flat * z.silu(),
        };

        y_out.apply(&self.out_proj)
    }

    /// Initialize inference state for a given batch size
    pub fn init_state(&self, batch_size: i64, device: tch::Device) -> Mamba2State {
        let d_inner = self.config.d_inner();
        let nheads = self.config.nheads();
        let headdim = self.config.headdim;
        let d_state = self.config.d_state;
        let ngroups = self.config.ngroups;
        let d_conv = self.config.d_conv;

        let conv_dim = d_inner + 2 * ngroups * d_state;

        Mamba2State {
            conv_state: Tensor::zeros(&[batch_size, conv_dim, d_conv], (Kind::Float, device)),
            ssm_state: Tensor::zeros(&[batch_size, nheads, headdim, d_state], (Kind::Float, device)),
        }
    }

    /// Single-step inference with state caching - O(1) per step
    /// Input: u [batch, 1, d_model] or [batch, d_model]
    /// Returns: output [batch, d_model], updated state in-place
    pub fn step(&self, u: &Tensor, state: &mut Mamba2State) -> Tensor {
        let u = if u.dim() == 2 { u.unsqueeze(1) } else { u.shallow_clone() };
        let batch = u.size()[0];
        let d_inner = self.config.d_inner();
        let nheads = self.config.nheads();
        let headdim = self.config.headdim;
        let d_state = self.config.d_state;
        let ngroups = self.config.ngroups;
        let heads_per_group = nheads / ngroups;

        // Project: [z, x, B, C, dt]
        let zxbcdt = u.squeeze_dim(1).apply(&self.in_proj); // [batch, d_in_proj]

        let z_dim = d_inner;
        let x_dim = d_inner;
        let b_dim = ngroups * d_state;
        let c_dim = ngroups * d_state;

        let z = zxbcdt.narrow(-1, 0, z_dim);
        let x = zxbcdt.narrow(-1, z_dim, x_dim);
        let b_raw = zxbcdt.narrow(-1, z_dim + x_dim, b_dim);
        let c_raw = zxbcdt.narrow(-1, z_dim + x_dim + b_dim, c_dim);
        let dt_raw = zxbcdt.narrow(-1, z_dim + x_dim + b_dim + c_dim, nheads);

        // Concatenate x, B, C for conv: [batch, conv_dim]
        let xbc = Tensor::cat(&[&x, &b_raw, &c_raw], -1);

        // Update conv state: shift left and add new input
        // conv_state: [batch, conv_dim, d_conv]
        let _ = state.conv_state.narrow(2, 0, self.config.d_conv - 1)
            .copy_(&state.conv_state.narrow(2, 1, self.config.d_conv - 1));
        let _ = state.conv_state.narrow(2, self.config.d_conv - 1, 1)
            .copy_(&xbc.unsqueeze(-1));

        // Manual depthwise conv1d: sum(conv_state * weight) + bias
        // weight: [conv_dim, 1, d_conv] -> [conv_dim, d_conv]
        let conv_weight = self.conv1d.ws.squeeze_dim(1);
        let xbc_conv = (&state.conv_state * &conv_weight).sum_dim_intlist(-1, false, Kind::Float);
        let xbc_conv = match &self.conv1d.bs {
            Some(bias) => (&xbc_conv + bias).silu(),
            None => xbc_conv.silu(),
        };

        // Split back
        let x_conv = xbc_conv.narrow(-1, 0, d_inner);
        let b = xbc_conv.narrow(-1, d_inner, b_dim);
        let c = xbc_conv.narrow(-1, d_inner + b_dim, c_dim);

        // dt: add bias and softplus
        let dt = (&dt_raw + &self.dt_bias).softplus().clamp(self.config.dt_min, self.config.dt_max);

        // A: negative exp for stability
        let a = self.a_log.exp().neg(); // [nheads]

        // Discretize: dA = exp(dt * A)
        let da = (dt.unsqueeze(-1) * a.view([1, nheads, 1])).exp(); // [batch, nheads, 1]

        // Reshape x: [batch, nheads, headdim]
        let x_heads = x_conv.view([batch, nheads, headdim]);

        // Expand B to heads: [batch, nheads, d_state]
        let b_heads = if ngroups == 1 {
            b.view([batch, 1, d_state]).expand([batch, nheads, d_state], false)
        } else {
            b.view([batch, ngroups, 1, d_state])
                .expand([batch, ngroups, heads_per_group, d_state], false)
                .reshape([batch, nheads, d_state])
        };

        // Expand C to heads: [batch, nheads, d_state]
        let c_heads = if ngroups == 1 {
            c.view([batch, 1, d_state]).expand([batch, nheads, d_state], false)
        } else {
            c.view([batch, ngroups, 1, d_state])
                .expand([batch, ngroups, heads_per_group, d_state], false)
                .reshape([batch, nheads, d_state])
        };

        // dB*x: [batch, nheads, headdim, d_state]
        // dB = dt * B: [batch, nheads, d_state]
        let db = dt.unsqueeze(-1) * &b_heads;
        // dBx = dB * x: [batch, nheads, headdim] x [batch, nheads, d_state] -> outer product
        let dbx = x_heads.unsqueeze(-1) * db.unsqueeze(2); // [batch, nheads, headdim, d_state]

        // SSM update: h_new = dA * h_old + dB * x
        // ssm_state: [batch, nheads, headdim, d_state]
        let new_ssm = &state.ssm_state * da.unsqueeze(2) + dbx;
        state.ssm_state.copy_(&new_ssm);

        // Output: y = C * h
        // c_heads: [batch, nheads, d_state]
        // ssm_state: [batch, nheads, headdim, d_state]
        // y = sum over d_state: [batch, nheads, headdim]
        let y = (&state.ssm_state * c_heads.unsqueeze(2)).sum_dim_intlist(-1, false, Kind::Float);

        // Add skip connection with D
        let y_skip = if self.config.d_has_hdim {
            let d_expanded = self.d_param.view([1, nheads, headdim]);
            &y + &x_heads * d_expanded
        } else {
            let d_expanded = self.d_param.view([1, nheads, 1]);
            &y + &x_heads * d_expanded
        };

        // Flatten heads: [batch, d_inner]
        let y_flat = y_skip.view([batch, d_inner]);

        // Norm and gate
        let y_out = match &self.norm {
            Some(norm) => {
                let z_for_gate = z.view([batch, d_inner]);
                norm.forward(&y_flat.unsqueeze(1), Some(&z_for_gate.unsqueeze(1))).squeeze_dim(1)
            }
            None => &y_flat * z.silu(),
        };

        y_out.apply(&self.out_proj)
    }

    /// Chunked SSM scan - parallel within chunks, sequential across chunks
    fn chunked_ssm_scan(
        &self,
        x: &Tensor,  // [batch, seq, nheads, headdim]
        dt: &Tensor, // [batch, seq, nheads]
        a: &Tensor,  // [nheads]
        b: &Tensor,  // [batch, seq, ngroups, d_state]
        c: &Tensor,  // [batch, seq, ngroups, d_state]
    ) -> Tensor {
        let (batch, nheads, headdim, d_state) = {
            let s = x.size4().unwrap();
            (s.0, self.config.nheads(), self.config.headdim, self.config.d_state)
        };
        let device = x.device();
        let kind = x.kind();
        let initial_state = Tensor::zeros(&[batch, nheads, headdim, d_state], (kind, device));
        let (output, _) = self.chunked_ssm_scan_with_state(x, dt, a, b, c, &initial_state);
        output
    }

    /// Chunked SSM scan with external state - GPU efficient with memory across calls
    fn chunked_ssm_scan_with_state(
        &self,
        x: &Tensor,  // [batch, seq, nheads, headdim]
        dt: &Tensor, // [batch, seq, nheads]
        a: &Tensor,  // [nheads]
        b: &Tensor,  // [batch, seq, ngroups, d_state]
        c: &Tensor,  // [batch, seq, ngroups, d_state]
        initial_state: &Tensor, // [batch, nheads, headdim, d_state]
    ) -> (Tensor, Tensor) {
        let (batch, seqlen, nheads, headdim) = x.size4().unwrap();
        let d_state = self.config.d_state;
        let ngroups = self.config.ngroups;
        let chunk_size = self.config.chunk_size.min(seqlen);
        let heads_per_group = nheads / ngroups;

        let device = x.device();

        let log_da = dt * a.view([1, 1, nheads]);

        let b_exp = if ngroups == 1 {
            b.expand([batch, seqlen, nheads, d_state], false)
        } else {
            b.unsqueeze(3)
                .expand([batch, seqlen, ngroups, heads_per_group, d_state], false)
                .reshape([batch, seqlen, nheads, d_state])
        };

        let c_exp = if ngroups == 1 {
            c.expand([batch, seqlen, nheads, d_state], false)
        } else {
            c.unsqueeze(3)
                .expand([batch, seqlen, ngroups, heads_per_group, d_state], false)
                .reshape([batch, seqlen, nheads, d_state])
        };

        let db = dt.unsqueeze(-1) * &b_exp;

        let mut state = initial_state.to_device(device);
        let mut outputs = Vec::new();
        let num_chunks = (seqlen + chunk_size - 1) / chunk_size;

        for chunk_idx in 0..num_chunks {
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(seqlen);
            let chunk_len = end - start;

            let x_chunk = x.narrow(1, start, chunk_len);
            let log_da_chunk = log_da.narrow(1, start, chunk_len);
            let db_chunk = db.narrow(1, start, chunk_len);
            let c_chunk = c_exp.narrow(1, start, chunk_len);

            let (y_chunk, new_state) = parallel_scan_chunk(
                &x_chunk,
                &log_da_chunk,
                &db_chunk,
                &c_chunk,
                &state,
                chunk_len,
            );

            outputs.push(y_chunk);
            state = new_state;
        }

        (Tensor::cat(&outputs, 1), state)
    }

    /// Forward with external state - GPU efficient chunked scan with state carry
    pub fn forward_with_state(&self, u: &Tensor, state: &mut Mamba2State) -> Tensor {
        let (batch, seqlen, _) = u.size3().unwrap();
        let d_inner = self.config.d_inner();
        let nheads = self.config.nheads();
        let headdim = self.config.headdim;
        let d_state = self.config.d_state;
        let ngroups = self.config.ngroups;

        let zxbcdt = u.apply(&self.in_proj);

        let z_dim = d_inner;
        let x_dim = d_inner;
        let b_dim = ngroups * d_state;
        let c_dim = ngroups * d_state;

        let z = zxbcdt.narrow(-1, 0, z_dim);
        let x = zxbcdt.narrow(-1, z_dim, x_dim);
        let b_raw = zxbcdt.narrow(-1, z_dim + x_dim, b_dim);
        let c_raw = zxbcdt.narrow(-1, z_dim + x_dim + b_dim, c_dim);
        let dt_raw = zxbcdt.narrow(-1, z_dim + x_dim + b_dim + c_dim, nheads);

        let xbc = Tensor::cat(&[&x, &b_raw, &c_raw], -1);
        let xbc_conv = xbc
            .transpose(1, 2)
            .apply(&self.conv1d)
            .narrow(2, 0, seqlen)
            .transpose(1, 2)
            .silu();

        let x_conv = xbc_conv.narrow(-1, 0, d_inner);
        let b = xbc_conv.narrow(-1, d_inner, b_dim);
        let c = xbc_conv.narrow(-1, d_inner + b_dim, c_dim);

        let dt = (&dt_raw + &self.dt_bias).softplus().clamp(self.config.dt_min, self.config.dt_max);
        let a = self.a_log.exp().neg();

        let x_heads = x_conv.view([batch, seqlen, nheads, headdim]);
        let b_groups = b.view([batch, seqlen, ngroups, d_state]);
        let c_groups = c.view([batch, seqlen, ngroups, d_state]);

        let (y, new_ssm_state) = self.chunked_ssm_scan_with_state(&x_heads, &dt, &a, &b_groups, &c_groups, &state.ssm_state);
        state.ssm_state.copy_(&new_ssm_state);

        let y_skip = if self.config.d_has_hdim {
            let d_expanded = self.d_param.view([1, 1, nheads, headdim]);
            &y + &x_heads * d_expanded
        } else {
            let d_expanded = self.d_param.view([1, 1, nheads, 1]);
            &y + &x_heads * d_expanded
        };

        let y_flat = y_skip.view([batch, seqlen, d_inner]);

        let y_out = match &self.norm {
            Some(norm) => {
                let z_for_gate = z.view([batch, seqlen, d_inner]);
                norm.forward(&y_flat, Some(&z_for_gate))
            }
            None => &y_flat * z.silu(),
        };

        y_out.apply(&self.out_proj)
    }
}

/// Parallel scan within a chunk using cumsum trick
/// h_t = cum_a[t] * (h_0 + cumsum(B*x / cum_a)[t])
fn parallel_scan_chunk(
    x: &Tensor,       // [batch, chunk, nheads, headdim]
    log_da: &Tensor,  // [batch, chunk, nheads]
    db: &Tensor,      // [batch, chunk, nheads, d_state]
    c: &Tensor,       // [batch, chunk, nheads, d_state]
    h0: &Tensor,      // [batch, nheads, headdim, d_state]
    chunk_len: i64,
) -> (Tensor, Tensor) {
    // Cumulative log(A) for parallel scan: [batch, chunk, nheads]
    let log_cum_a = log_da.cumsum(1, Kind::Float);

    // Compute cum_a and inv_cum_a together, expanded for broadcasting
    // [batch, chunk, nheads, 1, 1]
    let log_cum_a_exp = log_cum_a.unsqueeze(-1).unsqueeze(-1);
    let cum_a = log_cum_a_exp.exp();
    let inv_cum_a = (-&log_cum_a_exp).exp();

    // B*x scaled by inv_cum_a, then cumsum: [batch, chunk, nheads, headdim, d_state]
    let scaled_bx = (db.unsqueeze(3) * x.unsqueeze(-1)) * &inv_cum_a;
    let cumsum_scaled = scaled_bx.cumsum(1, Kind::Float);

    // h0: [batch, nheads, headdim, d_state] -> [batch, 1, nheads, headdim, d_state]
    let h0_exp = h0.unsqueeze(1);

    // c: [batch, chunk, nheads, d_state] -> [batch, chunk, nheads, 1, d_state]
    let c_exp = c.unsqueeze(3);

    // Fused: y = sum(cum_a * (h0 + cumsum_scaled) * C, dim=-1)
    // Avoids materializing full h tensor separately
    let y = (&cum_a * (&h0_exp + &cumsum_scaled) * &c_exp).sum_dim_intlist(-1, false, Kind::Float);

    // h_final from last timestep only: [batch, nheads, headdim, d_state]
    let cumsum_last = cumsum_scaled.select(1, chunk_len - 1);
    let cum_a_last = cum_a.select(1, chunk_len - 1).squeeze_dim(1);
    let h_final = cum_a_last * (h0 + cumsum_last);

    (y, h_final)
}

pub type MambaLayer = Box<dyn Fn(&Tensor, bool) -> Tensor>;

/// Stateful Mamba block for O(1) streaming inference
pub struct StatefulMamba {
    mamba: Mamba2,
}

impl StatefulMamba {
    pub fn new(p: &nn::Path, config: Mamba2Config) -> Self {
        Self { mamba: Mamba2::new(p, config) }
    }

    /// Batch forward pass (for training)
    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        self.mamba.forward(x, train)
    }

    /// GPU-efficient forward with state carry - uses chunked parallel scan
    pub fn forward_with_state(&self, x: &Tensor, state: &mut Mamba2State) -> Tensor {
        self.mamba.forward_with_state(x, state)
    }

    /// Initialize state for streaming inference
    pub fn init_state(&self, batch_size: i64, device: tch::Device) -> Mamba2State {
        self.mamba.init_state(batch_size, device)
    }

    /// Single step inference - O(1) per step
    pub fn step(&self, x: &Tensor, state: &mut Mamba2State) -> Tensor {
        self.mamba.step(x, state)
    }
}

/// Create a Mamba2 block with default config
pub fn mamba_block(p: &nn::Path, d_model: i64) -> MambaLayer {
    mamba_block_cfg(p, Mamba2Config::new(d_model))
}

/// Create a Mamba2 block with custom config
pub fn mamba_block_cfg(p: &nn::Path, config: Mamba2Config) -> MambaLayer {
    let mamba = Mamba2::new(p, config);
    Box::new(move |x: &Tensor, train: bool| mamba.forward(x, train))
}

/// Create a stateful Mamba2 block for streaming inference
pub fn stateful_mamba_block(p: &nn::Path, d_model: i64) -> StatefulMamba {
    stateful_mamba_block_cfg(p, Mamba2Config::new(d_model))
}

/// Create a stateful Mamba2 block with custom config
pub fn stateful_mamba_block_cfg(p: &nn::Path, config: Mamba2Config) -> StatefulMamba {
    StatefulMamba::new(p, config)
}

/// Stack of Mamba2 blocks with residual connections
pub fn mamba_stack(p: &nn::Path, d_model: i64, n_layers: i64) -> MambaLayer {
    mamba_stack_cfg(p, Mamba2Config::new(d_model), n_layers)
}

pub fn mamba_stack_cfg(p: &nn::Path, config: Mamba2Config, n_layers: i64) -> MambaLayer {
    let layers: Vec<_> = (0..n_layers)
        .map(|i| {
            let cfg = Mamba2Config {
                d_model: config.d_model,
                d_state: config.d_state,
                d_conv: config.d_conv,
                expand: config.expand,
                headdim: config.headdim,
                ngroups: config.ngroups,
                chunk_size: config.chunk_size,
                dt_min: config.dt_min,
                dt_max: config.dt_max,
                rmsnorm: config.rmsnorm,
                norm_before_gate: config.norm_before_gate,
                d_has_hdim: config.d_has_hdim,
            };
            (
                Mamba2::new(&(p / format!("layer_{}", i)), cfg),
                nn::layer_norm(
                    p / format!("ln_{}", i),
                    vec![config.d_model],
                    Default::default(),
                ),
            )
        })
        .collect();

    Box::new(move |x: &Tensor, train: bool| {
        let mut out = x.shallow_clone();
        for (mamba, ln) in &layers {
            let normed = out.apply(ln);
            out = &out + mamba.forward(&normed, train);
        }
        out
    })
}

