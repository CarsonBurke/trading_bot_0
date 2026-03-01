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
pub use crate::torch::ssm_ref::{Mamba2Config, Mamba2State};

static DEBUG_MEM: LazyLock<bool> =
    LazyLock::new(|| std::env::var("MAMBA_DEBUG_MEM").ok().as_deref() == Some("1"));

static USE_CUDA_GRAPH: LazyLock<bool> =
    LazyLock::new(|| std::env::var("MAMBA_USE_CUDA_GRAPH").ok().as_deref() == Some("1"));

const DT_INIT_FLOOR: f64 = 1e-4;

static MAMBA_CALL_ID: AtomicUsize = AtomicUsize::new(0);

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
            let rms =
                (grouped.pow_tensor_scalar(2).mean_dim(-1, true, Kind::Float) + self.eps).sqrt();
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
    // Pre-allocated empty tensors for Option::None cases
    empty_f32: Tensor,
    empty_i64: Tensor,
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

    fn maybe_to_device_kind(&self, input: &Tensor, device: tch::Device, kind: Kind) -> Tensor {
        let input = if input.device() == device {
            input.shallow_clone()
        } else {
            input.to_device(device)
        };
        if input.kind() == kind {
            input
        } else {
            input.to_kind(kind)
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
        let _nheads = self.nheads;
        let _headdim = self.config.headdim;
        let _d_state = self.config.d_state;
        let _ngroups = self.config.ngroups;

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
            &zxbcdt, batch, seqlen, dt_scale, seq_idx, call_id, debug_mem,
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
                Some(b) => self.maybe_to_device_kind(b, device, kind),
                None => self.empty_tensor(kind, device),
            };
            mamba_fused::rmsnorm_linear(
                u,
                &self.maybe_to_device_kind(norm_weight, device, kind),
                norm_eps,
                &self.maybe_to_device_kind(&self.in_proj.ws, device, kind),
                &bias,
            )
        } else {
            let x_f32 = u.to_kind(Kind::Float);
            let rms =
                (x_f32.pow_tensor_scalar(2).mean_dim(-1, true, Kind::Float) + norm_eps).sqrt();
            let normed = (x_f32 / rms * norm_weight.to_kind(Kind::Float)).to_kind(u.kind());
            normed.apply(&self.in_proj)
        };

        let y_out = self.forward_fused_from_zxbcdt(
            &zxbcdt, batch, seqlen, dt_scale, seq_idx, call_id, debug_mem,
        );

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
            conv_state: Tensor::zeros(
                &[batch_size, conv_dim, d_conv],
                (self.conv1d.ws.kind(), device),
            ),
            ssm_state: Tensor::zeros(
                &[batch_size, nheads, headdim, d_state],
                (Kind::Float, device),
            ),
            has_conv_state: false,
        }
    }

    fn step_impl(&self, u: &Tensor, state: &mut Mamba2State, dt_scale: Option<f64>) -> Tensor {
        let u = if u.dim() == 2 {
            u.unsqueeze(1)
        } else {
            u.shallow_clone()
        };
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
        let dt_raw = zxbcdt.narrow(
            -1,
            z0_dim + x0_dim + z_dim + d_ssm + 2 * ngroups * d_state,
            nheads,
        );

        let xbc = xbc_in;

        let _ = state
            .conv_state
            .narrow(2, 0, self.config.d_conv - 1)
            .copy_(&state.conv_state.narrow(2, 1, self.config.d_conv - 1));
        let _ = state
            .conv_state
            .narrow(2, self.config.d_conv - 1, 1)
            .copy_(&xbc.unsqueeze(-1));
        state.has_conv_state = true;

        let conv_weight = self.conv1d.ws.squeeze_dim(1).to_kind(Kind::Float);
        let xbc_conv = (state.conv_state.to_kind(Kind::Float) * &conv_weight).sum_dim_intlist(
            -1,
            false,
            Kind::Float,
        );
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
            let apply_dt_limit = if self.config.dt_limit == (0.0, f64::INFINITY) {
                0
            } else {
                1
            };
            let (dt_input, dt_bias_input, use_bias) = match dt_scale {
                Some(scale) => {
                    let dt_pre = (&dt_raw.to_kind(Kind::Float)
                        + &self.dt_bias.to_kind(Kind::Float))
                        .softplus()
                        * scale;
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
                    norm.forward(&y_flat.unsqueeze(1), Some(&z_for_gate.unsqueeze(1)))
                        .squeeze_dim(1)
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
            b.view([batch, 1, d_state])
                .expand([batch, nheads, d_state], false)
        } else {
            b.view([batch, ngroups, 1, d_state])
                .expand([batch, ngroups, heads_per_group, d_state], false)
                .reshape([batch, nheads, d_state])
        };

        let c_heads = if ngroups == 1 {
            c.view([batch, 1, d_state])
                .expand([batch, nheads, d_state], false)
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
                norm.forward(&y_flat.unsqueeze(1), Some(&z_for_gate.unsqueeze(1)))
                    .squeeze_dim(1)
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
            let rms =
                (x_f32.pow_tensor_scalar(2).mean_dim(-1, true, Kind::Float) + norm_eps).sqrt();
            (x_f32 / rms * norm_weight.to_kind(Kind::Float)).to_kind(u.kind())
        };

        self.step_impl(&u_normed, state, Some(dt_scale))
    }

    fn forward_fused_from_zxbcdt(
        &self,
        zxbcdt: &Tensor,
        batch: i64,
        _seqlen: i64,
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
            let rms =
                (x_f32.pow_tensor_scalar(2).mean_dim(-1, true, Kind::Float) + norm_eps).sqrt();
            let normed = (x_f32 / rms * norm_weight.to_kind(Kind::Float)).to_kind(u.kind());
            normed.apply(&self.in_proj)
        };

        self.forward_with_state_from_zxbcdt(&zxbcdt, state, dt_scale)
    }

    pub fn forward_with_state_dt_scale(
        &self,
        u: &Tensor,
        state: &mut Mamba2State,
        dt_scale: Option<&Tensor>,
    ) -> Tensor {
        let zxbcdt = u.apply(&self.in_proj);
        self.forward_with_state_from_zxbcdt(&zxbcdt, state, dt_scale)
    }

    fn forward_with_state_from_zxbcdt(
        &self,
        zxbcdt: &Tensor,
        state: &mut Mamba2State,
        dt_scale: Option<&Tensor>,
    ) -> Tensor {
        let _nheads = self.nheads;
        let headdim = self.config.headdim;
        let ngroups = self.config.ngroups;
        let _d_ssm = self.d_ssm;
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

        let initial_state = if state.ssm_state.kind() == kind && state.ssm_state.device() == device
        {
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
            state
                .ssm_state
                .copy_(&new_ssm_state.to_kind(state.ssm_state.kind()));
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
}

/// Create a Mamba2 block with custom config (returns closure for test use)
pub fn mamba_block_cfg(p: &nn::Path, config: Mamba2Config) -> Box<dyn Fn(&Tensor, bool) -> Tensor> {
    let mamba = Mamba2::new(p, config);
    Box::new(move |x: &Tensor, train: bool| mamba.forward(x, train))
}
