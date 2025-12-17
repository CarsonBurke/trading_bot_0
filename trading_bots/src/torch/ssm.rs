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

const D_STATE: i64 = 128;
const D_CONV: i64 = 4;
const EXPAND: i64 = 2;
const HEADDIM: i64 = 64;
const NGROUPS: i64 = 1;
const CHUNK_SIZE: i64 = 256;
const DT_MIN: f64 = 0.001;
const DT_MAX: f64 = 0.1;
const DT_INIT_FLOOR: f64 = 1e-4;

#[derive(Clone, Debug)]
pub struct Mamba2Config {
    pub d_model: i64,
    pub d_state: i64,
    pub d_conv: i64,
    pub expand: i64,
    pub headdim: i64,
    /// If set, SSM is applied only on this many inner dims; the remainder uses gated MLP.
    pub d_ssm: Option<i64>,
    pub ngroups: i64,
    pub chunk_size: i64,
    pub dt_min: f64,
    pub dt_max: f64,
    pub dt_limit: (f64, f64),
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
            d_ssm: None,
            ngroups: NGROUPS,
            chunk_size: CHUNK_SIZE,
            dt_min: DT_MIN,
            dt_max: DT_MAX,
            dt_limit: (0.0, f64::INFINITY),
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
    d_ssm: i64,
    d_inner: i64,
    nheads: i64,
}

impl Mamba2 {
    pub fn new(p: &nn::Path, config: Mamba2Config) -> Self {
        let mut config = config;
        let d_inner = config.d_inner();
        let d_ssm = config.d_ssm.unwrap_or(d_inner);
        assert!(d_ssm > 0 && d_ssm <= d_inner);
        assert_eq!(d_ssm % config.headdim, 0);
        let nheads = d_ssm / config.headdim;
        let d_state = config.d_state;
        let mut ngroups = config.ngroups.clamp(1, nheads);
        while ngroups > 1 && (nheads % ngroups != 0) {
            ngroups -= 1;
        }
        config.ngroups = ngroups;

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
        let conv_dim = d_ssm + 2 * ngroups * d_state;
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

        // A: per-head scalar
        let a = Tensor::empty(&[nheads], (Kind::Float, p.device())).uniform_(1.0, 16.0);
        let a_log = p.var_copy("A_log", &a.log());

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
                d_ssm,
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
            d_ssm,
            d_inner,
            nheads,
        }
    }

    /// Forward pass with optional dt_scale for variable patch sizes
    /// dt_scale: [1, seq, 1] or None - multiplies dt to account for patch size
    pub fn forward(&self, u: &Tensor, _train: bool) -> Tensor {
        self.forward_with_dt_scale(u, None)
    }

    pub fn forward_with_dt_scale(&self, u: &Tensor, dt_scale: Option<&Tensor>) -> Tensor {
        let (batch, seqlen, _) = u.size3().unwrap();
        let d_inner = self.d_inner;
        let d_ssm = self.d_ssm;
        let nheads = self.nheads;
        let headdim = self.config.headdim;
        let d_state = self.config.d_state;
        let ngroups = self.config.ngroups;
        let d_mlp = d_inner - d_ssm;

        let zxbcdt = u.apply(&self.in_proj);

        let z0_dim = d_mlp;
        let x0_dim = d_mlp;
        let z_dim = d_ssm;
        let xbc_dim = d_ssm + 2 * ngroups * d_state;
        let b_dim = ngroups * d_state;
        let c_dim = ngroups * d_state;
        let dt_dim = nheads;

        let z0 = if d_mlp > 0 {
            zxbcdt.narrow(-1, 0, z0_dim)
        } else {
            Tensor::zeros(&[batch, seqlen, 0], (zxbcdt.kind(), zxbcdt.device()))
        };
        let x0 = if d_mlp > 0 {
            zxbcdt.narrow(-1, z0_dim, x0_dim)
        } else {
            Tensor::zeros(&[batch, seqlen, 0], (zxbcdt.kind(), zxbcdt.device()))
        };
        let z = zxbcdt.narrow(-1, z0_dim + x0_dim, z_dim);
        let xbc = zxbcdt.narrow(-1, z0_dim + x0_dim + z_dim, xbc_dim);
        let dt_raw = zxbcdt.narrow(-1, z0_dim + x0_dim + z_dim + xbc_dim, dt_dim);

        let xbc_conv = xbc
            .transpose(1, 2)
            .apply(&self.conv1d)
            .narrow(2, 0, seqlen)
            .transpose(1, 2)
            .silu();

        let x_conv = xbc_conv.narrow(-1, 0, d_ssm);
        let b = xbc_conv.narrow(-1, d_ssm, b_dim);
        let c = xbc_conv.narrow(-1, d_ssm + b_dim, c_dim);

        // dt: bias + softplus, then scale by patch_size if provided
        let dt = (&dt_raw.to_kind(Kind::Float) + &self.dt_bias).softplus();
        let dt = match dt_scale {
            Some(scale) => &dt * scale,
            None => dt,
        };
        let dt = if self.config.dt_limit == (0.0, f64::INFINITY) {
            dt
        } else {
            dt.clamp(self.config.dt_limit.0, self.config.dt_limit.1)
        };

        let a = self.a_log.to_kind(Kind::Float).exp().neg();

        let x_heads = x_conv.view([batch, seqlen, nheads, headdim]);
        let b_groups = b.view([batch, seqlen, ngroups, d_state]);
        let c_groups = c.view([batch, seqlen, ngroups, d_state]);

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
        let y_flat = y_skip.view([batch, seqlen, d_ssm]);

        // Norm and gate
        let mut y_out = match &self.norm {
            Some(norm) => {
                let z_for_gate = z.view([batch, seqlen, d_ssm]);
                norm.forward(&y_flat, Some(&z_for_gate))
            }
            None => &y_flat * z.silu(),
        };

        if d_mlp > 0 {
            let mlp = &x0 * z0.silu();
            y_out = Tensor::cat(&[mlp, y_out], -1);
        }

        y_out.apply(&self.out_proj)
    }

    /// Initialize inference state for a given batch size
    pub fn init_state(&self, batch_size: i64, device: tch::Device) -> Mamba2State {
        let d_inner = self.d_inner;
        let d_ssm = self.d_ssm;
        let nheads = self.nheads;
        let headdim = self.config.headdim;
        let d_state = self.config.d_state;
        let ngroups = self.config.ngroups;
        let d_conv = self.config.d_conv;

        let conv_dim = d_ssm + 2 * ngroups * d_state;

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
        let d_inner = self.d_inner;
        let d_ssm = self.d_ssm;
        let nheads = self.nheads;
        let headdim = self.config.headdim;
        let d_state = self.config.d_state;
        let ngroups = self.config.ngroups;
        let heads_per_group = nheads / ngroups;
        let d_mlp = d_inner - d_ssm;

        // Project: [z, x, B, C, dt]
        let zxbcdt = u.squeeze_dim(1).apply(&self.in_proj); // [batch, d_in_proj]

        let z0_dim = d_mlp;
        let x0_dim = d_mlp;
        let z_dim = d_ssm;
        let b_dim = ngroups * d_state;
        let c_dim = ngroups * d_state;

        let z0 = if d_mlp > 0 {
            zxbcdt.narrow(-1, 0, z0_dim)
        } else {
            Tensor::zeros(&[batch, 0], (zxbcdt.kind(), zxbcdt.device()))
        };
        let x0 = if d_mlp > 0 {
            zxbcdt.narrow(-1, z0_dim, x0_dim)
        } else {
            Tensor::zeros(&[batch, 0], (zxbcdt.kind(), zxbcdt.device()))
        };
        let z = zxbcdt.narrow(-1, z0_dim + x0_dim, z_dim);
        let xbc_in = zxbcdt.narrow(-1, z0_dim + x0_dim + z_dim, d_ssm + 2 * ngroups * d_state);
        let dt_raw = zxbcdt.narrow(-1, z0_dim + x0_dim + z_dim + d_ssm + 2 * ngroups * d_state, nheads);

        // Concatenate x, B, C for conv: [batch, conv_dim]
        let xbc = xbc_in;

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
        let x_conv = xbc_conv.narrow(-1, 0, d_ssm);
        let b = xbc_conv.narrow(-1, d_ssm, b_dim);
        let c = xbc_conv.narrow(-1, d_ssm + b_dim, c_dim);

        // dt: add bias and softplus
        let dt = (&dt_raw.to_kind(Kind::Float) + &self.dt_bias).softplus();
        let dt = if self.config.dt_limit == (0.0, f64::INFINITY) {
            dt
        } else {
            dt.clamp(self.config.dt_limit.0, self.config.dt_limit.1)
        };

        // A: negative exp for stability
        let a = self.a_log.to_kind(Kind::Float).exp().neg(); // [nheads]

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
        let y_flat = y_skip.view([batch, d_ssm]);

        // Norm and gate
        let mut y_out = match &self.norm {
            Some(norm) => {
                let z_for_gate = z.view([batch, d_ssm]);
                norm.forward(&y_flat.unsqueeze(1), Some(&z_for_gate.unsqueeze(1))).squeeze_dim(1)
            }
            None => &y_flat * z.silu(),
        };

        if d_mlp > 0 {
            let mlp = x0 * z0.silu();
            y_out = Tensor::cat(&[mlp, y_out], -1);
        }

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
        let (batch, _seqlen, nheads, headdim) = x.size4().unwrap();
        let d_state = self.config.d_state;
        let device = x.device();
        let kind = x.kind();
        let initial_state = Tensor::zeros(&[batch, nheads, headdim, d_state], (kind, device));
        let (output, _) = self.chunked_ssm_scan_with_state(x, dt, a, b, c, &initial_state);
        output
    }

    /// Chunked SSM scan with external state - SSD algorithm from Mamba2 paper
    /// Uses attention-like within-chunk computation: Y = (L ◦ CB^T) @ X
    /// Avoids materializing [B,H,L,N,P] outer product tensors
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
        let x_kind = x.kind();

        let log_da = dt.to_kind(Kind::Float) * a.view([1, 1, nheads]);
        let dt_f = dt.to_kind(Kind::Float);
        let x_f = x.to_kind(Kind::Float);

        let b_exp = if ngroups == 1 {
            b.expand([batch, seqlen, nheads, d_state], false)
        } else {
            b.unsqueeze(3)
                .expand([batch, seqlen, ngroups, heads_per_group, d_state], false)
                .reshape([batch, seqlen, nheads, d_state])
        }
        .to_kind(Kind::Float);

        let c_exp = if ngroups == 1 {
            c.expand([batch, seqlen, nheads, d_state], false)
        } else {
            c.unsqueeze(3)
                .expand([batch, seqlen, ngroups, heads_per_group, d_state], false)
                .reshape([batch, seqlen, nheads, d_state])
        }
        .to_kind(Kind::Float);

        let mut state = initial_state.to_device(device).to_kind(Kind::Float);
        let num_chunks = (seqlen + chunk_size - 1) / chunk_size;
        let mut outputs: Vec<Tensor> = Vec::with_capacity(num_chunks as usize);

        for chunk_idx in 0..num_chunks {
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(seqlen);
            let chunk_len = end - start;

            let x_chunk = x_f.narrow(1, start, chunk_len);
            let dt_chunk = dt_f.narrow(1, start, chunk_len);
            let log_da_chunk = log_da.narrow(1, start, chunk_len);
            let b_chunk = b_exp.narrow(1, start, chunk_len);
            let c_chunk = c_exp.narrow(1, start, chunk_len);

            // Scale x by dt: [B, L, H, P]
            let x_dt = &x_chunk * dt_chunk.unsqueeze(-1);

            // Transpose to [B, H, L, *] for matmuls (strided ok for matmul)
            let x_t = x_dt.permute([0, 2, 1, 3]);  // [B, H, L, P]
            let b_t = b_chunk.permute([0, 2, 1, 3]); // [B, H, L, N]
            let c_t = c_chunk.permute([0, 2, 1, 3]); // [B, H, L, N]
            let log_da_t = log_da_chunk.permute([0, 2, 1]); // [B, H, L]

            // Cumulative log(A) for decay computations
            let a_cumsum = log_da_t.cumsum(-1, Kind::Float); // [B, H, L]

            // === Within-chunk: SSD attention-like form ===
            // Y_diag = (L ◦ C @ B^T) @ X  where L_ij = exp(cumsum_i - cumsum_j) for i >= j
            // Compute via: for each head, attention scores = C @ B^T, masked by L

            // L matrix: segsum gives log(L), exp for L
            let l_mat = segsum_stable(&log_da_t).exp(); // [B, H, L, L]

            // SSD quadratic form: scores = C @ B^T, then (L ◦ scores) @ X
            // scores: [B, H, L, L] via [B, H, L, N] @ [B, H, N, L]
            let scores = c_t.matmul(&b_t.transpose(-1, -2)); // [B, H, L, L]
            let masked_scores = &l_mat * &scores; // [B, H, L, L]

            // y_diag = masked_scores @ x: [B, H, L, L] @ [B, H, L, P] -> [B, H, L, P]
            let y_diag = masked_scores.matmul(&x_t); // [B, H, L, P]

            // === Cross-chunk: contribution from previous state ===
            // y_off = decay * (state @ C^T)
            // state: [B, H, P, N], C: [B, H, L, N] -> [B, H, L, P]
            let decay = a_cumsum.exp().unsqueeze(-1); // [B, H, L, 1]
            // state @ C^T: [B, H, P, N] @ [B, H, N, L] -> [B, H, P, L] -> transpose -> [B, H, L, P]
            let state_contrib = state.matmul(&c_t.transpose(-1, -2)).transpose(-1, -2); // [B, H, L, P]
            let y_off = &decay * &state_contrib; // [B, H, L, P]

            let y_chunk = (&y_diag + &y_off).permute([0, 2, 1, 3]); // [B, L, H, P]
            outputs.push(y_chunk);

            // === Update state for next chunk ===
            // state_new = decay_total * state + sum_t(decay_from_t * B_t * x_t)
            let a_last = a_cumsum.select(-1, chunk_len - 1); // [B, H]
            let decay_total = a_last.unsqueeze(-1).unsqueeze(-1).exp(); // [B, H, 1, 1]

            // Decay from each position to end of chunk
            let decay_states = (a_last.unsqueeze(-1) - &a_cumsum).exp(); // [B, H, L]
            // B weighted by decay: [B, H, L, N] * [B, H, L, 1]
            let b_decay = &b_t * decay_states.unsqueeze(-1); // [B, H, L, N]
            // state_add = X^T @ B_decay: [B, H, P, L] @ [B, H, L, N] -> [B, H, P, N]
            let state_add = x_t.transpose(-1, -2).matmul(&b_decay); // [B, H, P, N]

            state = &decay_total * &state + state_add;
        }

        (Tensor::cat(&outputs, 1).to_kind(x_kind), state.to_kind(x_kind))
    }

    /// Forward with external state - GPU efficient chunked scan with state carry
    pub fn forward_with_state(&self, u: &Tensor, state: &mut Mamba2State) -> Tensor {
        self.forward_with_state_dt_scale(u, state, None)
    }

    pub fn forward_with_state_dt_scale(&self, u: &Tensor, state: &mut Mamba2State, dt_scale: Option<&Tensor>) -> Tensor {
        let (batch, seqlen, _) = u.size3().unwrap();
        let d_inner = self.d_inner;
        let d_ssm = self.d_ssm;
        let d_mlp = d_inner - d_ssm;
        let nheads = self.nheads;
        let headdim = self.config.headdim;
        let d_state = self.config.d_state;
        let ngroups = self.config.ngroups;

        let zxbcdt = u.apply(&self.in_proj);

        let z0_dim = d_mlp;
        let x0_dim = d_mlp;
        let z_dim = d_ssm;
        let xbc_dim = d_ssm + 2 * ngroups * d_state;
        let b_dim = ngroups * d_state;
        let c_dim = ngroups * d_state;

        let z0 = if d_mlp > 0 {
            zxbcdt.narrow(-1, 0, z0_dim)
        } else {
            Tensor::zeros(&[batch, seqlen, 0], (zxbcdt.kind(), zxbcdt.device()))
        };
        let x0 = if d_mlp > 0 {
            zxbcdt.narrow(-1, z0_dim, x0_dim)
        } else {
            Tensor::zeros(&[batch, seqlen, 0], (zxbcdt.kind(), zxbcdt.device()))
        };
        let z = zxbcdt.narrow(-1, z0_dim + x0_dim, z_dim);
        let xbc = zxbcdt.narrow(-1, z0_dim + x0_dim + z_dim, xbc_dim);
        let dt_raw = zxbcdt.narrow(-1, z0_dim + x0_dim + z_dim + xbc_dim, nheads);
        let xbc_conv = xbc
            .transpose(1, 2)
            .apply(&self.conv1d)
            .narrow(2, 0, seqlen)
            .transpose(1, 2)
            .silu();

        let x_conv = xbc_conv.narrow(-1, 0, d_ssm);
        let b = xbc_conv.narrow(-1, d_ssm, b_dim);
        let c = xbc_conv.narrow(-1, d_ssm + b_dim, c_dim);

        let dt = (&dt_raw.to_kind(Kind::Float) + &self.dt_bias).softplus();
        let dt = match dt_scale {
            Some(scale) => &dt * scale,
            None => dt,
        };
        let dt = if self.config.dt_limit == (0.0, f64::INFINITY) {
            dt
        } else {
            dt.clamp(self.config.dt_limit.0, self.config.dt_limit.1)
        };
        let a = self.a_log.to_kind(Kind::Float).exp().neg();

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

        let y_flat = y_skip.view([batch, seqlen, d_ssm]);

        let mut y_out = match &self.norm {
            Some(norm) => {
                let z_for_gate = z.view([batch, seqlen, d_ssm]);
                norm.forward(&y_flat, Some(&z_for_gate))
            }
            None => &y_flat * z.silu(),
        };

        if d_mlp > 0 {
            let mlp = &x0 * z0.silu();
            y_out = Tensor::cat(&[mlp, y_out], -1);
        }

        y_out.apply(&self.out_proj)
    }

    /// Batched forward with initial states - processes multiple sequences in parallel
    pub fn forward_batched_init(&self, u: &Tensor, initial_states: &Tensor) -> (Tensor, Tensor) {
        self.forward_batched_init_dt_scale(u, initial_states, None)
    }

    pub fn forward_batched_init_dt_scale(&self, u: &Tensor, initial_states: &Tensor, dt_scale: Option<&Tensor>) -> (Tensor, Tensor) {
        let (batch, seqlen, _) = u.size3().unwrap();
        let d_inner = self.d_inner;
        let d_ssm = self.d_ssm;
        let d_mlp = d_inner - d_ssm;
        let nheads = self.nheads;
        let headdim = self.config.headdim;
        let d_state = self.config.d_state;
        let ngroups = self.config.ngroups;

        let zxbcdt = u.apply(&self.in_proj);

        let z0_dim = d_mlp;
        let x0_dim = d_mlp;
        let z_dim = d_ssm;
        let xbc_dim = d_ssm + 2 * ngroups * d_state;
        let b_dim = ngroups * d_state;
        let c_dim = ngroups * d_state;

        let z0 = if d_mlp > 0 {
            zxbcdt.narrow(-1, 0, z0_dim)
        } else {
            Tensor::zeros(&[batch, seqlen, 0], (zxbcdt.kind(), zxbcdt.device()))
        };
        let x0 = if d_mlp > 0 {
            zxbcdt.narrow(-1, z0_dim, x0_dim)
        } else {
            Tensor::zeros(&[batch, seqlen, 0], (zxbcdt.kind(), zxbcdt.device()))
        };
        let z = zxbcdt.narrow(-1, z0_dim + x0_dim, z_dim);
        let xbc = zxbcdt.narrow(-1, z0_dim + x0_dim + z_dim, xbc_dim);
        let dt_raw = zxbcdt.narrow(-1, z0_dim + x0_dim + z_dim + xbc_dim, nheads);
        let xbc_conv = xbc
            .transpose(1, 2)
            .apply(&self.conv1d)
            .narrow(2, 0, seqlen)
            .transpose(1, 2)
            .silu();

        let x_conv = xbc_conv.narrow(-1, 0, d_ssm);
        let b = xbc_conv.narrow(-1, d_ssm, b_dim);
        let c = xbc_conv.narrow(-1, d_ssm + b_dim, c_dim);

        let dt = (&dt_raw.to_kind(Kind::Float) + &self.dt_bias).softplus();
        let dt = match dt_scale {
            Some(scale) => &dt * scale,
            None => dt,
        };
        let dt = dt
            .clamp(self.config.dt_min, self.config.dt_max)
            .clamp(self.config.dt_limit.0, self.config.dt_limit.1);
        let a = self.a_log.to_kind(Kind::Float).exp().neg();

        let x_heads = x_conv.view([batch, seqlen, nheads, headdim]);
        let b_groups = b.view([batch, seqlen, ngroups, d_state]);
        let c_groups = c.view([batch, seqlen, ngroups, d_state]);

        let (y, final_states) = self.chunked_ssm_scan_with_state(&x_heads, &dt, &a, &b_groups, &c_groups, initial_states);

        let y_skip = if self.config.d_has_hdim {
            let d_expanded = self.d_param.view([1, 1, nheads, headdim]);
            &y + &x_heads * d_expanded
        } else {
            let d_expanded = self.d_param.view([1, 1, nheads, 1]);
            &y + &x_heads * d_expanded
        };

        let y_flat = y_skip.view([batch, seqlen, d_ssm]);

        let mut y_out = match &self.norm {
            Some(norm) => {
                let z_for_gate = z.view([batch, seqlen, d_ssm]);
                norm.forward(&y_flat, Some(&z_for_gate))
            }
            None => &y_flat * z.silu(),
        };

        if d_mlp > 0 {
            let mlp = &x0 * z0.silu();
            y_out = Tensor::cat(&[mlp, y_out], -1);
        }

        (y_out.apply(&self.out_proj), final_states)
    }
}

fn segsum_stable(a: &Tensor) -> Tensor {
    debug_assert_eq!(a.dim(), 3);
    let b = a.size()[0];
    let h = a.size()[1];
    let t = a.size()[2];
    let device = a.device();

    let a = a.unsqueeze(-1).expand(&[b, h, t, t], false);
    let mask_strict = Tensor::ones(&[t, t], (Kind::Bool, device)).tril(-1);
    let a = a.masked_fill(&mask_strict.logical_not(), 0.0);
    let a = a.cumsum(-2, Kind::Float);
    let mask = Tensor::ones(&[t, t], (Kind::Bool, device)).tril(0);
    a.masked_fill(&mask.logical_not(), f64::NEG_INFINITY)
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

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        self.mamba.forward(x, train)
    }

    pub fn forward_with_dt_scale(&self, x: &Tensor, dt_scale: Option<&Tensor>) -> Tensor {
        self.mamba.forward_with_dt_scale(x, dt_scale)
    }

    pub fn forward_with_state(&self, x: &Tensor, state: &mut Mamba2State) -> Tensor {
        self.mamba.forward_with_state(x, state)
    }

    pub fn forward_with_state_dt_scale(&self, x: &Tensor, state: &mut Mamba2State, dt_scale: Option<&Tensor>) -> Tensor {
        self.mamba.forward_with_state_dt_scale(x, state, dt_scale)
    }

    pub fn init_state(&self, batch_size: i64, device: tch::Device) -> Mamba2State {
        self.mamba.init_state(batch_size, device)
    }

    pub fn step(&self, x: &Tensor, state: &mut Mamba2State) -> Tensor {
        self.mamba.step(x, state)
    }

    pub fn forward_batched_init(&self, x: &Tensor, initial_states: &Tensor) -> (Tensor, Tensor) {
        self.mamba.forward_batched_init(x, initial_states)
    }

    pub fn forward_batched_init_dt_scale(&self, x: &Tensor, initial_states: &Tensor, dt_scale: Option<&Tensor>) -> (Tensor, Tensor) {
        self.mamba.forward_batched_init_dt_scale(x, initial_states, dt_scale)
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
                d_ssm: config.d_ssm,
                ngroups: config.ngroups,
                chunk_size: config.chunk_size,
                dt_min: config.dt_min,
                dt_max: config.dt_max,
                dt_limit: config.dt_limit,
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

#[cfg(test)]
mod ssd_attention_shape_tests {
    use super::*;

    #[test]
    fn test_ssd_attention_shaped_equivalence() {
        let device = tch::Device::Cpu;
        let b = 2i64;
        let h = 3i64;
        let l = 8i64;
        let n = 5i64;
        let p = 4i64;

        let log_da = Tensor::randn(&[b, h, l], (Kind::Float, device)).clamp(-5.0, 0.0);
        let l_mat = segsum_stable(&log_da).exp(); // [B,H,L,L]

        let k = Tensor::randn(&[b, h, l, n], (Kind::Float, device));
        let q = Tensor::randn(&[b, h, l, n], (Kind::Float, device));
        let v = Tensor::randn(&[b, h, l, p], (Kind::Float, device));

        let kv = k.unsqueeze(-1) * v.unsqueeze(-2); // [B,H,L,N,P]
        let kv_flat = kv.reshape([b * h, l, n * p]);
        let l_batched = l_mat.reshape([b * h, l, l]);
        let tmp = l_batched.bmm(&kv_flat).reshape([b, h, l, n, p]);
        let y1 = (tmp * q.unsqueeze(-1)).sum_dim_intlist(3, false, Kind::Float); // [B,H,L,P]

        let scores = q.matmul(&k.transpose(-1, -2)); // [B,H,L,L]
        let weights = scores * l_mat; // [B,H,L,L]
        let v_flat = v.reshape([b * h, l, p]);
        let y2 = weights.reshape([b * h, l, l]).bmm(&v_flat).reshape([b, h, l, p]);

        let max_diff: f64 = (&y1 - &y2).abs().max().double_value(&[]);
        assert!(max_diff < 1e-4, "Max diff {} too large", max_diff);
    }
}
