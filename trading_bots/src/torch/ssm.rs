//! Mamba 2 Selective State Space Model
//!
//! Implements architecture from "Transformers are SSMs" (Dao & Gu, 2024):
//! - Multi-head structure (like attention)
//! - Per-head A parameter (not per-channel)
//! - Chunked parallel scan for numerical stability
//! - Fused RMSNorm with gating
//! - Single projection for z, x, B, C, dt

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::LazyLock;
use tch::nn::Init;
use tch::{nn, Kind, Tensor};

use crate::torch::mamba_fused;

static DEBUG_MEM: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("MAMBA_DEBUG_MEM").ok().as_deref() == Some("1")
});

static USE_CUDA_GRAPH: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("MAMBA_USE_CUDA_GRAPH").ok().as_deref() == Some("1")
});

const D_STATE: i64 = 128;
const D_CONV: i64 = 4;
const EXPAND: i64 = 2;
const HEADDIM: i64 = 64;
const NGROUPS: i64 = 1;
const CHUNK_SIZE: i64 = 256;
const DT_MIN: f64 = 0.001;
const DT_MAX: f64 = 0.1;
const DT_INIT_FLOOR: f64 = 1e-4;

static MAMBA_CALL_ID: AtomicUsize = AtomicUsize::new(0);

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
    pub has_conv_state: bool,
}

impl Mamba2State {
    pub fn reset(&mut self) {
        let _ = self.conv_state.zero_();
        let _ = self.ssm_state.zero_();
        self.has_conv_state = false;
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
    conv1d_w_f32: Tensor,
    conv1d_b_f32: Tensor,
    dt_bias_f32: Tensor,
    // Pre-allocated empty tensors for Option::None cases
    empty_f32: Tensor,
    empty_i64: Tensor,
}

struct RMSNormSimple {
    weight: Tensor,
    eps: f64,
}

impl RMSNormSimple {
    fn new(p: &nn::Path, dim: i64, eps: f64) -> Self {
        let weight = p.var("weight", &[dim], Init::Const(1.0));
        Self { weight, eps }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let x_f32 = x.to_kind(Kind::Float);
        let rms = (x_f32.pow_tensor_scalar(2).mean_dim(-1, true, Kind::Float) + self.eps).sqrt();
        (x_f32 / rms * &self.weight).to_kind(x.kind())
    }
}

impl Mamba2 {
    fn empty_tensor(&self, kind: Kind, device: tch::Device) -> Tensor {
        if self.empty_f32.device() == device {
            if kind == Kind::Int64 {
                self.empty_i64.shallow_clone()
            } else if kind == Kind::Float {
                self.empty_f32.shallow_clone()
            } else {
                self.empty_f32.to_kind(kind)
            }
        } else {
            Tensor::zeros(&[0], (kind, device))
        }
    }

    pub fn new(p: &nn::Path, config: Mamba2Config) -> Self {
        let mut config = config;
        let d_inner = config.d_inner();
        let d_ssm = config.d_ssm.unwrap_or(d_inner);
        assert!(d_ssm > 0 && d_ssm <= d_inner);
        assert_eq!(d_ssm % config.headdim, 0);
        let nheads = d_ssm / config.headdim;
        let d_state = config.d_state;
        let ngroups = config.ngroups;
        assert!(
            ngroups >= 1 && ngroups <= nheads,
            "ngroups must be in [1, nheads], got {}",
            ngroups
        );
        assert!(
            nheads % ngroups == 0,
            "ngroups must divide nheads ({}), got {}",
            nheads,
            ngroups
        );
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

        // A: per-head scalar (Mamba2-style initialization)
        // a_log = log(uniform(1, 8)), so a_log in [0, ~2.08]
        // Then A = -exp(a_log) in [-8, -1] for moderate-to-fast decay
        let a_init = Tensor::empty(&[nheads], (Kind::Float, p.device())).uniform_(1.0, 8.0);
        let a_log = p.var_copy("A_log", &a_init.log());

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

        let conv1d_w_f32 = conv1d.ws.squeeze_dim(1).to_kind(Kind::Float);
        let conv1d_b_f32 = match &conv1d.bs {
            Some(b) => b.to_kind(Kind::Float),
            None => Tensor::zeros(&[0], (Kind::Float, p.device())),
        };
        let dt_bias_f32 = dt_bias.to_kind(Kind::Float);

        let device = in_proj.ws.device();
        let empty_f32 = Tensor::zeros(&[0], (Kind::Float, device));
        let empty_i64 = Tensor::zeros(&[0], (Kind::Int64, device));

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
            conv1d_w_f32,
            conv1d_b_f32,
            dt_bias_f32,
            empty_f32,
            empty_i64,
        }
    }

    /// Forward pass with optional dt_scale for variable patch sizes
    /// dt_scale: [1, seq, 1] or None - scales dt in fused path
    pub fn forward(&self, u: &Tensor, _train: bool) -> Tensor {
        self.forward_with_dt_scale(u, None)
    }

    pub fn forward_with_dt_scale(&self, u: &Tensor, dt_scale: Option<&Tensor>) -> Tensor {
        self.forward_with_dt_scale_seq_idx(u, dt_scale, None)
    }

    pub fn forward_with_dt_scale_seq_idx(
        &self,
        u: &Tensor,
        dt_scale: Option<&Tensor>,
        seq_idx: Option<&Tensor>,
    ) -> Tensor {
        let call_id = MAMBA_CALL_ID.fetch_add(1, Ordering::Relaxed);
        let (batch, seqlen, _) = u.size3().unwrap();
        let nheads = self.nheads;
        let headdim = self.config.headdim;
        let d_state = self.config.d_state;
        let ngroups = self.config.ngroups;

        let debug_mem = *DEBUG_MEM;
        if debug_mem {
            let stats = mamba_fused::cuda_memory_stats();
            eprintln!(
                "SSM#{} pre-in_proj: alloc={}MB batch={} seqlen={}",
                call_id,
                stats.get(0).unwrap_or(&0) / (1024 * 1024),
                batch,
                seqlen
            );
        }

        let zxbcdt = u.apply(&self.in_proj);

        if debug_mem {
            let stats = mamba_fused::cuda_memory_stats();
            eprintln!(
                "SSM#{} post-in_proj: alloc={}MB zxbcdt={:?}",
                call_id,
                stats.get(0).unwrap_or(&0) / (1024 * 1024),
                zxbcdt.size()
            );
        }

        let y_out = self.forward_fused_from_zxbcdt(
            &zxbcdt,
            batch,
            seqlen,
            dt_scale,
            seq_idx,
            call_id,
            debug_mem,
        );

        if debug_mem {
            let stats = mamba_fused::cuda_memory_stats();
            eprintln!(
                "SSM#{} post-fused: alloc={}MB peak={}MB y_out={:?}",
                call_id,
                stats.get(0).unwrap_or(&0) / (1024 * 1024),
                stats.get(3).unwrap_or(&0) / (1024 * 1024),
                y_out.size()
            );
        }

        y_out
    }

    pub fn forward_with_pre_norm_seq_idx(
        &self,
        u: &Tensor,
        norm_weight: &Tensor,
        norm_eps: f64,
        dt_scale: Option<&Tensor>,
        seq_idx: Option<&Tensor>,
    ) -> Tensor {
        let call_id = MAMBA_CALL_ID.fetch_add(1, Ordering::Relaxed);
        let (batch, seqlen, _) = u.size3().unwrap();
        let debug_mem = *DEBUG_MEM;
        let device = u.device();
        let kind = u.kind();

        let zxbcdt = if matches!(device, tch::Device::Cuda(_)) {
            let bias = match &self.in_proj.bs {
                Some(b) => b.to_kind(kind).to_device(device),
                None => self.empty_tensor(kind, device),
            };
            mamba_fused::rmsnorm_linear(
                u,
                &norm_weight.to_kind(kind).to_device(device),
                norm_eps,
                &self.in_proj.ws.to_kind(kind).to_device(device),
                &bias,
            )
        } else {
            let x_f32 = u.to_kind(Kind::Float);
            let rms =
                (x_f32.pow_tensor_scalar(2).mean_dim(-1, true, Kind::Float) + norm_eps).sqrt();
            let normed = (x_f32 / rms * norm_weight.to_kind(Kind::Float)).to_kind(u.kind());
            normed.apply(&self.in_proj)
        };

        let y_out = self.forward_fused_from_zxbcdt(&zxbcdt, batch, seqlen, dt_scale, seq_idx, call_id, debug_mem);

        y_out
    }

    /// Initialize inference state for a given batch size
    pub fn init_state(&self, batch_size: i64, device: tch::Device) -> Mamba2State {
        let d_ssm = self.d_ssm;
        let nheads = self.nheads;
        let headdim = self.config.headdim;
        let d_state = self.config.d_state;
        let ngroups = self.config.ngroups;
        let d_conv = self.config.d_conv;

        let conv_dim = d_ssm + 2 * ngroups * d_state;

        Mamba2State {
            conv_state: Tensor::zeros(&[batch_size, conv_dim, d_conv], (self.conv1d.ws.kind(), device)),
            ssm_state: Tensor::zeros(&[batch_size, nheads, headdim, d_state], (Kind::Float, device)),
            has_conv_state: false,
        }
    }

    fn step_impl(&self, u: &Tensor, state: &mut Mamba2State, dt_scale: Option<f64>) -> Tensor {
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

        let zxbcdt = u.squeeze_dim(1).apply(&self.in_proj);

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

        let xbc = xbc_in;

        let _ = state.conv_state.narrow(2, 0, self.config.d_conv - 1)
            .copy_(&state.conv_state.narrow(2, 1, self.config.d_conv - 1));
        let _ = state.conv_state.narrow(2, self.config.d_conv - 1, 1)
            .copy_(&xbc.unsqueeze(-1));
        state.has_conv_state = true;

        let conv_weight = self.conv1d.ws.squeeze_dim(1).to_kind(Kind::Float);
        let xbc_conv = (state.conv_state.to_kind(Kind::Float) * &conv_weight).sum_dim_intlist(-1, false, Kind::Float);
        let xbc_conv = match &self.conv1d.bs {
            Some(bias) => (&xbc_conv + &bias.to_kind(Kind::Float)).silu(),
            None => xbc_conv.silu(),
        };

        let x_conv = xbc_conv.narrow(-1, 0, d_ssm);
        let b = xbc_conv.narrow(-1, d_ssm, b_dim);
        let c = xbc_conv.narrow(-1, d_ssm + b_dim, c_dim);

        if matches!(zxbcdt.device(), tch::Device::Cuda(_)) {
            let x_heads = x_conv.view([batch, nheads, headdim]);
            let b_groups = b.view([batch, ngroups, d_state]);
            let c_groups = c.view([batch, ngroups, d_state]);
            let z_gate = if self.config.rmsnorm {
                self.empty_tensor(Kind::Float, zxbcdt.device())
            } else {
                z.view([batch, nheads, headdim])
            };
            let apply_dt_limit = if self.config.dt_limit == (0.0, f64::INFINITY) { 0 } else { 1 };
            let (dt_input, dt_bias_input, use_bias) = match dt_scale {
                Some(scale) => {
                    let dt_pre = (&dt_raw.to_kind(Kind::Float) + &self.dt_bias.to_kind(Kind::Float)).softplus() * scale;
                    let dt_bias = Tensor::zeros(&[nheads], (Kind::Float, zxbcdt.device()));
                    (dt_pre, dt_bias, false)
                }
                None => (dt_raw.shallow_clone(), self.dt_bias.shallow_clone(), true),
            };
            let (y_heads, new_state) = mamba_fused::selective_state_update(
                &state.ssm_state,
                &x_heads,
                &dt_input,
                &self.a_log,
                &b_groups,
                &c_groups,
                &self.d_param,
                &z_gate,
                &dt_bias_input,
                use_bias,
                self.config.dt_limit.0,
                self.config.dt_limit.1,
                ngroups,
                headdim,
                apply_dt_limit,
            );
            state.ssm_state.copy_(&new_state);
            let y_flat = y_heads.to_kind(zxbcdt.kind()).view([batch, d_ssm]);
            let mut y_out = match &self.norm {
                Some(norm) => {
                    let z_for_gate = z.view([batch, d_ssm]);
                    norm.forward(&y_flat.unsqueeze(1), Some(&z_for_gate.unsqueeze(1))).squeeze_dim(1)
                }
                None => y_flat,
            };
            if d_mlp > 0 {
                let mlp = x0 * z0.silu();
                y_out = Tensor::cat(&[mlp, y_out], -1);
            }
            return y_out.apply(&self.out_proj);
        }

        let mut dt = (&dt_raw.to_kind(Kind::Float) + &self.dt_bias.to_kind(Kind::Float)).softplus();
        if let Some(scale) = dt_scale {
            dt = dt * scale;
        }
        let dt = if self.config.dt_limit == (0.0, f64::INFINITY) {
            dt
        } else {
            dt.clamp(self.config.dt_limit.0, self.config.dt_limit.1)
        };

        let a = self.a_log.to_kind(Kind::Float).exp().neg();
        let da = (dt.unsqueeze(-1) * a.view([1, nheads, 1])).exp();
        let x_heads = x_conv.view([batch, nheads, headdim]);

        let b_heads = if ngroups == 1 {
            b.view([batch, 1, d_state]).expand([batch, nheads, d_state], false)
        } else {
            b.view([batch, ngroups, 1, d_state])
                .expand([batch, ngroups, heads_per_group, d_state], false)
                .reshape([batch, nheads, d_state])
        };

        let c_heads = if ngroups == 1 {
            c.view([batch, 1, d_state]).expand([batch, nheads, d_state], false)
        } else {
            c.view([batch, ngroups, 1, d_state])
                .expand([batch, ngroups, heads_per_group, d_state], false)
                .reshape([batch, nheads, d_state])
        };

        let db = dt.unsqueeze(-1) * &b_heads;
        let dbx = x_heads.unsqueeze(-1) * db.unsqueeze(2);
        let new_ssm = &state.ssm_state * da.unsqueeze(2) + dbx;
        state.ssm_state.copy_(&new_ssm);

        let y = (&state.ssm_state * c_heads.unsqueeze(2)).sum_dim_intlist(-1, false, Kind::Float);
        let y_skip = if self.config.d_has_hdim {
            let d_expanded = self.d_param.view([1, nheads, headdim]);
            &y + &x_heads * d_expanded
        } else {
            let d_expanded = self.d_param.view([1, nheads, 1]);
            &y + &x_heads * d_expanded
        };

        let y_flat = y_skip.view([batch, d_ssm]);
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

    pub fn step(&self, u: &Tensor, state: &mut Mamba2State) -> Tensor {
        self.step_impl(u, state, None)
    }

    pub fn step_with_dt_scale(&self, u: &Tensor, state: &mut Mamba2State, dt_scale: f64) -> Tensor {
        self.step_impl(u, state, Some(dt_scale))
    }

    pub fn step_with_pre_norm_dt_scale(
        &self,
        u: &Tensor,
        norm_weight: &Tensor,
        norm_eps: f64,
        state: &mut Mamba2State,
        dt_scale: f64,
    ) -> Tensor {
        let device = u.device();
        let kind = u.kind();

        let u_normed = if matches!(device, tch::Device::Cuda(_)) {
            let weight = norm_weight.to_kind(kind).to_device(device);
            mamba_fused::rmsnorm_forward(u, &weight, norm_eps)
        } else {
            let x_f32 = u.to_kind(Kind::Float);
            let rms = (x_f32.pow_tensor_scalar(2).mean_dim(-1, true, Kind::Float) + norm_eps).sqrt();
            (x_f32 / rms * norm_weight.to_kind(Kind::Float)).to_kind(u.kind())
        };

        self.step_impl(&u_normed, state, Some(dt_scale))
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

    fn forward_fused_from_zxbcdt(
        &self,
        zxbcdt: &Tensor,
        batch: i64,
        seqlen: i64,
        dt_scale: Option<&Tensor>,
        seq_idx: Option<&Tensor>,
        call_id: usize,
        debug_mem: bool,
    ) -> Tensor {
        let nheads = self.nheads;
        let headdim = self.config.headdim;
        let d_state = self.config.d_state;
        let ngroups = self.config.ngroups;
        let device = zxbcdt.device();
        let kind = zxbcdt.kind();

        let dt_scale_tensor = match dt_scale {
            Some(scale) => {
                if scale.kind() == kind && scale.device() == device {
                    scale.shallow_clone()
                } else {
                    scale.to_kind(kind).to_device(device)
                }
            }
            None => self.empty_tensor(kind, device),
        };

        let conv_w = self.conv1d.ws.squeeze_dim(1);
        let conv_w = if conv_w.kind() == kind && conv_w.device() == device {
            conv_w
        } else {
            conv_w.to_kind(kind).to_device(device)
        };

        let conv_b = match &self.conv1d.bs {
            Some(bias) => {
                if bias.kind() == kind && bias.device() == device {
                    bias.shallow_clone()
                } else {
                    bias.to_kind(kind).to_device(device)
                }
            }
            None => self.empty_tensor(kind, device),
        };

        let dt_bias = if self.dt_bias.kind() == kind && self.dt_bias.device() == device {
            self.dt_bias.shallow_clone()
        } else {
            self.dt_bias.to_kind(kind).to_device(device)
        };

        let a_log = if self.a_log.kind() == kind && self.a_log.device() == device {
            self.a_log.shallow_clone()
        } else {
            self.a_log.to_kind(kind).to_device(device)
        };

        let d_param = if self.d_param.kind() == kind && self.d_param.device() == device {
            self.d_param.shallow_clone()
        } else {
            self.d_param.to_kind(kind).to_device(device)
        };

        let initial_state = Tensor::zeros(&[batch, nheads, headdim, d_state], (kind, device));
        let seq_idx = match seq_idx {
            Some(idx) => idx.to_device(device),
            None => self.empty_tensor(Kind::Int64, device),
        };

        let (rmsnorm_weight, rmsnorm_eps, norm_before_gate) = match &self.norm {
            Some(norm) => {
                let weight = if norm.weight.kind() == kind && norm.weight.device() == device {
                    norm.weight.shallow_clone()
                } else {
                    norm.weight.to_kind(kind).to_device(device)
                };
                (weight, norm.eps, norm.norm_before_gate)
            }
            None => (self.empty_tensor(kind, device), 1e-5, false),
        };

        let outproj_w = if self.out_proj.ws.kind() == kind && self.out_proj.ws.device() == device {
            self.out_proj.ws.shallow_clone()
        } else {
            self.out_proj.ws.to_kind(kind).to_device(device)
        };

        let outproj_b = match &self.out_proj.bs {
            Some(bias) => {
                if bias.kind() == kind && bias.device() == device {
                    bias.shallow_clone()
                } else {
                    bias.to_kind(kind).to_device(device)
                }
            }
            None => self.empty_tensor(kind, device),
        };

        if !matches!(zxbcdt.device(), tch::Device::Cuda(_)) {
            panic!("mamba fused op requires CUDA");
        }

        if debug_mem {
            let stats = mamba_fused::cuda_memory_stats();
            eprintln!(
                "SSM#{} pre-fused: alloc={}MB initial_state={:?}",
                call_id,
                stats.get(0).unwrap_or(&0) / (1024 * 1024),
                initial_state.size()
            );
        }

        let use_graph = *USE_CUDA_GRAPH && !zxbcdt.requires_grad();
        let (y_out, _final_state) = if use_graph {
            mamba_fused::fused_conv_scan_full_graph(
                zxbcdt,
                &conv_w,
                &conv_b,
                &dt_bias,
                &a_log,
                &d_param,
                &dt_scale_tensor,
                &initial_state,
                &seq_idx,
                self.config.chunk_size,
                ngroups,
                headdim,
                self.config.dt_limit.0,
                self.config.dt_limit.1,
                &rmsnorm_weight,
                rmsnorm_eps,
                norm_before_gate,
                &outproj_w,
                &outproj_b,
            )
        } else {
            mamba_fused::fused_conv_scan_full(
                zxbcdt,
                &conv_w,
                &conv_b,
                &dt_bias,
                &a_log,
                &d_param,
                &dt_scale_tensor,
                &initial_state,
                &seq_idx,
                self.config.chunk_size,
                ngroups,
                headdim,
                self.config.dt_limit.0,
                self.config.dt_limit.1,
                &rmsnorm_weight,
                rmsnorm_eps,
                norm_before_gate,
                &outproj_w,
                &outproj_b,
            )
        };

        if debug_mem {
            let stats = mamba_fused::cuda_memory_stats();
            eprintln!(
                "SSM#{} post-fused: alloc={}MB peak={}MB y_out={:?}",
                call_id,
                stats.get(0).unwrap_or(&0) / (1024 * 1024),
                stats.get(3).unwrap_or(&0) / (1024 * 1024),
                y_out.size()
            );
        }

        y_out
    }

    /// Forward with external state - GPU efficient chunked scan with state carry
    pub fn forward_with_state(&self, u: &Tensor, state: &mut Mamba2State) -> Tensor {
        self.forward_with_state_dt_scale(u, state, None)
    }

    pub fn forward_with_state_pre_norm_dt_scale(
        &self,
        u: &Tensor,
        norm_weight: &Tensor,
        norm_eps: f64,
        state: &mut Mamba2State,
        dt_scale: Option<&Tensor>,
    ) -> Tensor {
        let device = u.device();
        let kind = u.kind();

        let zxbcdt = if matches!(device, tch::Device::Cuda(_)) {
            let bias = match &self.in_proj.bs {
                Some(b) => b.to_kind(kind).to_device(device),
                None => self.empty_tensor(kind, device),
            };
            mamba_fused::rmsnorm_linear(
                u,
                &norm_weight.to_kind(kind).to_device(device),
                norm_eps,
                &self.in_proj.ws.to_kind(kind).to_device(device),
                &bias,
            )
        } else {
            let x_f32 = u.to_kind(Kind::Float);
            let rms = (x_f32.pow_tensor_scalar(2).mean_dim(-1, true, Kind::Float) + norm_eps).sqrt();
            let normed = (x_f32 / rms * norm_weight.to_kind(Kind::Float)).to_kind(u.kind());
            normed.apply(&self.in_proj)
        };

        self.forward_with_state_from_zxbcdt(&zxbcdt, state, dt_scale)
    }

    pub fn forward_with_state_dt_scale(&self, u: &Tensor, state: &mut Mamba2State, dt_scale: Option<&Tensor>) -> Tensor {
        let zxbcdt = u.apply(&self.in_proj);
        self.forward_with_state_from_zxbcdt(&zxbcdt, state, dt_scale)
    }

    fn forward_with_state_from_zxbcdt(&self, zxbcdt: &Tensor, state: &mut Mamba2State, dt_scale: Option<&Tensor>) -> Tensor {
        let nheads = self.nheads;
        let headdim = self.config.headdim;
        let ngroups = self.config.ngroups;
        let d_ssm = self.d_ssm;
        let device = zxbcdt.device();
        let kind = zxbcdt.kind();
        let seqlen = zxbcdt.size()[1];

        let dt_scale_tensor = match dt_scale {
            Some(scale) => {
                if scale.kind() == kind && scale.device() == device {
                    scale.shallow_clone()
                } else {
                    scale.to_kind(kind).to_device(device)
                }
            }
            None => self.empty_tensor(kind, device),
        };

        let conv_w = self.conv1d.ws.squeeze_dim(1);
        let conv_w = if conv_w.kind() == kind && conv_w.device() == device {
            conv_w
        } else {
            conv_w.to_kind(kind).to_device(device)
        };

        let conv_b = match &self.conv1d.bs {
            Some(bias) => {
                if bias.kind() == kind && bias.device() == device {
                    bias.shallow_clone()
                } else {
                    bias.to_kind(kind).to_device(device)
                }
            }
            None => self.empty_tensor(kind, device),
        };

        let dt_bias = if self.dt_bias.kind() == kind && self.dt_bias.device() == device {
            self.dt_bias.shallow_clone()
        } else {
            self.dt_bias.to_kind(kind).to_device(device)
        };

        let a_log = if self.a_log.kind() == kind && self.a_log.device() == device {
            self.a_log.shallow_clone()
        } else {
            self.a_log.to_kind(kind).to_device(device)
        };

        let d_param = if self.d_param.kind() == kind && self.d_param.device() == device {
            self.d_param.shallow_clone()
        } else {
            self.d_param.to_kind(kind).to_device(device)
        };

        let initial_state = if state.ssm_state.kind() == kind && state.ssm_state.device() == device {
            state.ssm_state.shallow_clone()
        } else {
            state.ssm_state.to_kind(kind).to_device(device)
        };

        let seq_idx = self.empty_tensor(Kind::Int64, device);

        let (rmsnorm_weight, rmsnorm_eps, norm_before_gate) = match &self.norm {
            Some(norm) => {
                let weight = if norm.weight.kind() == kind && norm.weight.device() == device {
                    norm.weight.shallow_clone()
                } else {
                    norm.weight.to_kind(kind).to_device(device)
                };
                (weight, norm.eps, norm.norm_before_gate)
            }
            None => (self.empty_tensor(kind, device), 1e-5, false),
        };

        let outproj_w = if self.out_proj.ws.kind() == kind && self.out_proj.ws.device() == device {
            self.out_proj.ws.shallow_clone()
        } else {
            self.out_proj.ws.to_kind(kind).to_device(device)
        };

        let outproj_b = match &self.out_proj.bs {
            Some(bias) => {
                if bias.kind() == kind && bias.device() == device {
                    bias.shallow_clone()
                } else {
                    bias.to_kind(kind).to_device(device)
                }
            }
            None => self.empty_tensor(kind, device),
        };

        if !matches!(zxbcdt.device(), tch::Device::Cuda(_)) {
            panic!("forward_with_state_from_zxbcdt requires CUDA device");
        }

        tch::no_grad(|| {
            let conv_state = state.conv_state.to_kind(kind).to_device(device);
            let (y_out, new_ssm_state, new_conv_state) = mamba_fused::fused_conv_scan_stateful(
                &zxbcdt,
                &conv_w,
                &conv_b,
                &dt_bias,
                &a_log,
                &d_param,
                &dt_scale_tensor,
                &initial_state,
                &conv_state,
                &seq_idx,
                self.config.chunk_size,
                ngroups,
                headdim,
                self.config.dt_limit.0,
                self.config.dt_limit.1,
                &rmsnorm_weight,
                rmsnorm_eps,
                norm_before_gate,
                &outproj_w,
                &outproj_b,
            );
            state.ssm_state.copy_(&new_ssm_state.to_kind(state.ssm_state.kind()));
            state
                .conv_state
                .copy_(&new_conv_state.to_kind(state.conv_state.kind()));
            state.has_conv_state = seqlen > 0;
            y_out
        })
    }

    /// Batched forward with external state - processes multiple sequences in parallel
    pub fn forward_batched_init(&self, u: &Tensor, state: &mut Mamba2State) -> Tensor {
        self.forward_batched_init_dt_scale(u, state, None)
    }

    pub fn forward_batched_init_dt_scale(
        &self,
        u: &Tensor,
        state: &mut Mamba2State,
        dt_scale: Option<&Tensor>,
    ) -> Tensor {
        self.forward_with_state_dt_scale(u, state, dt_scale)
    }

    fn update_conv_state_from_xbc(&self, state: &mut Mamba2State, xbc_t: &Tensor, include_prev: bool) -> Tensor {
        let d_conv = self.config.d_conv;
        let conv_input = if include_prev && d_conv > 1 {
            let prev = state
                .conv_state
                .narrow(2, 0, d_conv - 1)
                .to_kind(xbc_t.kind())
                .to_device(xbc_t.device());
            Tensor::cat(&[prev, xbc_t.shallow_clone()], 2)
        } else {
            xbc_t.shallow_clone()
        };
        let total = conv_input.size()[2];
        let updated = if total >= d_conv {
            conv_input.narrow(2, total - d_conv, d_conv)
        } else {
            let pad = Tensor::zeros(
                &[conv_input.size()[0], conv_input.size()[1], d_conv - total],
                (conv_input.kind(), conv_input.device()),
            );
            Tensor::cat(&[pad, conv_input.shallow_clone()], 2)
        };
        state.conv_state.copy_(&updated);
        state.has_conv_state = true;
        conv_input
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

    pub fn forward_with_dt_scale_seq_idx(
        &self,
        x: &Tensor,
        dt_scale: Option<&Tensor>,
        seq_idx: Option<&Tensor>,
    ) -> Tensor {
        self.mamba.forward_with_dt_scale_seq_idx(x, dt_scale, seq_idx)
    }

    pub fn forward_with_pre_norm_seq_idx(
        &self,
        x: &Tensor,
        norm_weight: &Tensor,
        norm_eps: f64,
        dt_scale: Option<&Tensor>,
        seq_idx: Option<&Tensor>,
    ) -> Tensor {
        self.mamba
            .forward_with_pre_norm_seq_idx(x, norm_weight, norm_eps, dt_scale, seq_idx)
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

    pub fn step_with_dt_scale(&self, x: &Tensor, state: &mut Mamba2State, dt_scale: f64) -> Tensor {
        self.mamba.step_with_dt_scale(x, state, dt_scale)
    }

    pub fn step_with_pre_norm_dt_scale(
        &self,
        x: &Tensor,
        norm_weight: &Tensor,
        norm_eps: f64,
        state: &mut Mamba2State,
        dt_scale: f64,
    ) -> Tensor {
        self.mamba.step_with_pre_norm_dt_scale(x, norm_weight, norm_eps, state, dt_scale)
    }

    pub fn forward_with_state_pre_norm_dt_scale(
        &self,
        x: &Tensor,
        norm_weight: &Tensor,
        norm_eps: f64,
        state: &mut Mamba2State,
        dt_scale: Option<&Tensor>,
    ) -> Tensor {
        self.mamba.forward_with_state_pre_norm_dt_scale(x, norm_weight, norm_eps, state, dt_scale)
    }

    pub fn forward_batched_init(&self, x: &Tensor, state: &mut Mamba2State) -> Tensor {
        self.mamba.forward_batched_init(x, state)
    }

    pub fn forward_batched_init_dt_scale(
        &self,
        x: &Tensor,
        state: &mut Mamba2State,
        dt_scale: Option<&Tensor>,
    ) -> Tensor {
        self.mamba.forward_batched_init_dt_scale(x, state, dt_scale)
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
                RMSNormSimple::new(&(p / format!("ln_{}", i)), config.d_model, 1e-5),
            )
        })
        .collect();

    Box::new(move |x: &Tensor, train: bool| {
        let mut out = x.shallow_clone();
        for (mamba, ln) in &layers {
            let normed = ln.forward(&out);
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
