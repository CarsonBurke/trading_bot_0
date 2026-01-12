use std::sync::{Mutex, OnceLock};

use libloading::Library;
use tch::{CModule, IValue, Kind, Tensor};

static LIB: OnceLock<Library> = OnceLock::new();
static MODULE: OnceLock<Mutex<CModule>> = OnceLock::new();

fn module_lock() -> &'static Mutex<CModule> {
    let lib_path = env!("MAMBA_FUSED_LIB_PATH");
    let wrapper_path = env!("MAMBA_FUSED_WRAPPER_PATH");
    LIB.get_or_init(|| unsafe { Library::new(lib_path).expect("Failed to load mamba fused ops") });
    MODULE.get_or_init(|| {
        Mutex::new(CModule::load(wrapper_path).expect("Failed to load mamba fused wrapper"))
    })
}


pub fn fused_conv_scan(
    zxbcdt: &Tensor,
    conv_w: &Tensor,
    conv_b: &Tensor,
    dt_bias: &Tensor,
    a_log: &Tensor,
    d_param: &Tensor,
    dt_scale: &Tensor,
    initial_state: &Tensor,
    seq_idx: &Tensor,
    chunk_size: i64,
    ngroups: i64,
    headdim: i64,
    dt_min: f64,
    dt_max: f64,
) -> (Tensor, Tensor) {
    let module = module_lock()
        .lock()
        .expect("Failed to lock mamba fused module");
    let inputs = vec![
        IValue::Tensor(zxbcdt.shallow_clone()),
        IValue::Tensor(conv_w.shallow_clone()),
        IValue::Tensor(conv_b.shallow_clone()),
        IValue::Tensor(dt_bias.shallow_clone()),
        IValue::Tensor(a_log.shallow_clone()),
        IValue::Tensor(d_param.shallow_clone()),
        IValue::Tensor(dt_scale.shallow_clone()),
        IValue::Tensor(initial_state.shallow_clone()),
        IValue::Tensor(seq_idx.shallow_clone()),
        IValue::Int(chunk_size),
        IValue::Int(ngroups),
        IValue::Int(headdim),
        IValue::Double(dt_min),
        IValue::Double(dt_max),
    ];
    let out = module.forward_is(&inputs).expect("mamba fused forward failed");
    let (y, final_state): (Tensor, Tensor) = out.try_into().expect("bad mamba fused output");
    (y, final_state)
}

pub fn fused_conv_scan_infer(
    zxbcdt: &Tensor,
    conv_w: &Tensor,
    conv_b: &Tensor,
    dt_bias: &Tensor,
    a_log: &Tensor,
    d_param: &Tensor,
    dt_scale: &Tensor,
    initial_state: &Tensor,
    seq_idx: &Tensor,
    chunk_size: i64,
    ngroups: i64,
    headdim: i64,
    dt_min: f64,
    dt_max: f64,
    rmsnorm_weight: &Tensor,
    rmsnorm_eps: f64,
    norm_before_gate: bool,
    outproj_w: &Tensor,
    outproj_b: &Tensor,
) -> (Tensor, Tensor) {
    let module = module_lock()
        .lock()
        .expect("Failed to lock mamba fused module");
    let inputs = vec![
        IValue::Tensor(zxbcdt.shallow_clone()),
        IValue::Tensor(conv_w.shallow_clone()),
        IValue::Tensor(conv_b.shallow_clone()),
        IValue::Tensor(dt_bias.shallow_clone()),
        IValue::Tensor(a_log.shallow_clone()),
        IValue::Tensor(d_param.shallow_clone()),
        IValue::Tensor(dt_scale.shallow_clone()),
        IValue::Tensor(initial_state.shallow_clone()),
        IValue::Tensor(seq_idx.shallow_clone()),
        IValue::Int(chunk_size),
        IValue::Int(ngroups),
        IValue::Int(headdim),
        IValue::Double(dt_min),
        IValue::Double(dt_max),
        IValue::Tensor(rmsnorm_weight.shallow_clone()),
        IValue::Double(rmsnorm_eps),
        IValue::Bool(norm_before_gate),
        IValue::Tensor(outproj_w.shallow_clone()),
        IValue::Tensor(outproj_b.shallow_clone()),
    ];
    let out = module
        .method_is("forward_infer", &inputs)
        .expect("mamba fused infer forward failed");
    let (y, final_state): (Tensor, Tensor) = out.try_into().expect("bad mamba fused infer output");
    (y, final_state)
}

pub fn fused_conv_scan_full(
    zxbcdt: &Tensor,
    conv_w: &Tensor,
    conv_b: &Tensor,
    dt_bias: &Tensor,
    a_log: &Tensor,
    d_param: &Tensor,
    dt_scale: &Tensor,
    initial_state: &Tensor,
    seq_idx: &Tensor,
    chunk_size: i64,
    ngroups: i64,
    headdim: i64,
    dt_min: f64,
    dt_max: f64,
    rmsnorm_weight: &Tensor,
    rmsnorm_eps: f64,
    norm_before_gate: bool,
    outproj_w: &Tensor,
    outproj_b: &Tensor,
) -> (Tensor, Tensor) {
    let inputs = vec![
        IValue::Tensor(zxbcdt.shallow_clone()),
        IValue::Tensor(conv_w.shallow_clone()),
        IValue::Tensor(conv_b.shallow_clone()),
        IValue::Tensor(dt_bias.shallow_clone()),
        IValue::Tensor(a_log.shallow_clone()),
        IValue::Tensor(d_param.shallow_clone()),
        IValue::Tensor(dt_scale.shallow_clone()),
        IValue::Tensor(initial_state.shallow_clone()),
        IValue::Tensor(seq_idx.shallow_clone()),
        IValue::Int(chunk_size),
        IValue::Int(ngroups),
        IValue::Int(headdim),
        IValue::Double(dt_min),
        IValue::Double(dt_max),
        IValue::Tensor(rmsnorm_weight.shallow_clone()),
        IValue::Double(rmsnorm_eps),
        IValue::Bool(norm_before_gate),
        IValue::Tensor(outproj_w.shallow_clone()),
        IValue::Tensor(outproj_b.shallow_clone()),
    ];
    let out = {
        let module = module_lock()
            .lock()
            .expect("Failed to lock mamba fused module");
        module.method_is("forward_full", &inputs)
    };
    match out {
        Ok(out) => {
            let (y, final_state): (Tensor, Tensor) =
                out.try_into().expect("bad mamba fused full output");
            (y, final_state)
        }
        Err(err) => {
            let message = err.to_string();
            if message.contains("forward_full") && message.contains("not defined") {
                let (y, final_state) = fused_conv_scan(
                    zxbcdt,
                    conv_w,
                    conv_b,
                    dt_bias,
                    a_log,
                    d_param,
                    dt_scale,
                    initial_state,
                    seq_idx,
                    chunk_size,
                    ngroups,
                    headdim,
                    dt_min,
                    dt_max,
                );
                let y_out = fallback_fused_post(
                    &y,
                    zxbcdt,
                    conv_w,
                    ngroups,
                    headdim,
                    rmsnorm_weight,
                    rmsnorm_eps,
                    norm_before_gate,
                    outproj_w,
                    outproj_b,
                );
                (y_out, final_state)
            } else {
                panic!("mamba fused full forward failed: {}", err);
            }
        }
    }
}

pub fn fused_conv_scan_full_graph(
    zxbcdt: &Tensor,
    conv_w: &Tensor,
    conv_b: &Tensor,
    dt_bias: &Tensor,
    a_log: &Tensor,
    d_param: &Tensor,
    dt_scale: &Tensor,
    initial_state: &Tensor,
    seq_idx: &Tensor,
    chunk_size: i64,
    ngroups: i64,
    headdim: i64,
    dt_min: f64,
    dt_max: f64,
    rmsnorm_weight: &Tensor,
    rmsnorm_eps: f64,
    norm_before_gate: bool,
    outproj_w: &Tensor,
    outproj_b: &Tensor,
) -> (Tensor, Tensor) {
    let inputs = vec![
        IValue::Tensor(zxbcdt.shallow_clone()),
        IValue::Tensor(conv_w.shallow_clone()),
        IValue::Tensor(conv_b.shallow_clone()),
        IValue::Tensor(dt_bias.shallow_clone()),
        IValue::Tensor(a_log.shallow_clone()),
        IValue::Tensor(d_param.shallow_clone()),
        IValue::Tensor(dt_scale.shallow_clone()),
        IValue::Tensor(initial_state.shallow_clone()),
        IValue::Tensor(seq_idx.shallow_clone()),
        IValue::Int(chunk_size),
        IValue::Int(ngroups),
        IValue::Int(headdim),
        IValue::Double(dt_min),
        IValue::Double(dt_max),
        IValue::Tensor(rmsnorm_weight.shallow_clone()),
        IValue::Double(rmsnorm_eps),
        IValue::Bool(norm_before_gate),
        IValue::Tensor(outproj_w.shallow_clone()),
        IValue::Tensor(outproj_b.shallow_clone()),
    ];
    let out = {
        let module = module_lock()
            .lock()
            .expect("Failed to lock mamba fused module");
        module
            .method_is("forward_full_graph", &inputs)
            .expect("mamba fused full graph forward failed")
    };
    let (y, final_state): (Tensor, Tensor) = out.try_into().expect("bad mamba fused graph output");
    (y, final_state)
}

fn fallback_fused_post(
    y: &Tensor,
    zxbcdt: &Tensor,
    conv_w: &Tensor,
    ngroups: i64,
    headdim: i64,
    rmsnorm_weight: &Tensor,
    rmsnorm_eps: f64,
    norm_before_gate: bool,
    outproj_w: &Tensor,
    outproj_b: &Tensor,
) -> Tensor {
    let sizes = y.size();
    let batch = sizes[0];
    let seqlen = sizes[1];
    let nheads = sizes[2];
    let d_ssm = nheads * headdim;
    let d_in_proj = zxbcdt.size()[2];
    let d_state = (conv_w.size()[0] - d_ssm) / (2 * ngroups);
    let d_inner = (d_in_proj - 2 * ngroups * d_state - nheads) / 2;
    let d_mlp = d_inner - d_ssm;

    let y_flat = y.view([batch, seqlen, d_ssm]);
    let y_f = y_flat.to_kind(Kind::Float);
    let z = zxbcdt
        .narrow(2, 2 * d_mlp, d_ssm)
        .to_kind(Kind::Float);

    let y_norm = if rmsnorm_weight.defined() && rmsnorm_weight.numel() > 0 {
        let group_size = d_ssm / ngroups;
        let y_r = y_f.view([batch, seqlen, ngroups, group_size]);
        let z_r = z.view([batch, seqlen, ngroups, group_size]);
        let normed = if norm_before_gate {
            let rms = (y_r
                .pow_tensor_scalar(2.0)
                .mean_dim([3].as_slice(), true, Kind::Float)
                + rmsnorm_eps)
                .sqrt();
            (&y_r / rms) * z_r.silu()
        } else {
            let gated = &y_r * z_r.silu();
            let rms = (gated
                .pow_tensor_scalar(2.0)
                .mean_dim([3].as_slice(), true, Kind::Float)
                + rmsnorm_eps)
                .sqrt();
            gated / rms
        };
        let weight = rmsnorm_weight
            .to_kind(Kind::Float)
            .view([1, 1, ngroups, group_size]);
        (normed * weight).view([batch, seqlen, d_ssm])
    } else {
        y_f * z.silu()
    };

    let y_out = if d_mlp > 0 {
        let z0 = zxbcdt.narrow(2, 0, d_mlp).to_kind(Kind::Float);
        let x0 = zxbcdt.narrow(2, d_mlp, d_mlp).to_kind(Kind::Float);
        let mlp = x0 * z0.silu();
        Tensor::cat(&[mlp, y_norm], -1)
    } else {
        y_norm
    };

    if outproj_w.defined() && outproj_w.numel() > 0 {
        let out_in = y_out.to_kind(outproj_w.kind());
        let out_2d = out_in.view([batch * seqlen, d_inner]);
        let out_w_t = outproj_w.transpose(0, 1).contiguous();
        let out = if outproj_b.defined() && outproj_b.numel() > 0 {
            outproj_b.addmm(&out_2d, &out_w_t)
        } else {
            out_2d.matmul(&out_w_t)
        };
        out.view([batch, seqlen, outproj_w.size()[0]])
    } else {
        y_out
    }
}

pub fn fused_conv_scan_stateful(
    zxbcdt: &Tensor,
    conv_w: &Tensor,
    conv_b: &Tensor,
    dt_bias: &Tensor,
    a_log: &Tensor,
    d_param: &Tensor,
    dt_scale: &Tensor,
    initial_state: &Tensor,
    conv_state: &Tensor,
    seq_idx: &Tensor,
    chunk_size: i64,
    ngroups: i64,
    headdim: i64,
    dt_min: f64,
    dt_max: f64,
    rmsnorm_weight: &Tensor,
    rmsnorm_eps: f64,
    norm_before_gate: bool,
    outproj_w: &Tensor,
    outproj_b: &Tensor,
) -> (Tensor, Tensor, Tensor) {
    let inputs = vec![
        IValue::Tensor(zxbcdt.shallow_clone()),
        IValue::Tensor(conv_w.shallow_clone()),
        IValue::Tensor(conv_b.shallow_clone()),
        IValue::Tensor(dt_bias.shallow_clone()),
        IValue::Tensor(a_log.shallow_clone()),
        IValue::Tensor(d_param.shallow_clone()),
        IValue::Tensor(dt_scale.shallow_clone()),
        IValue::Tensor(initial_state.shallow_clone()),
        IValue::Tensor(conv_state.shallow_clone()),
        IValue::Tensor(seq_idx.shallow_clone()),
        IValue::Int(chunk_size),
        IValue::Int(ngroups),
        IValue::Int(headdim),
        IValue::Double(dt_min),
        IValue::Double(dt_max),
        IValue::Tensor(rmsnorm_weight.shallow_clone()),
        IValue::Double(rmsnorm_eps),
        IValue::Bool(norm_before_gate),
        IValue::Tensor(outproj_w.shallow_clone()),
        IValue::Tensor(outproj_b.shallow_clone()),
    ];
    let out = {
        let module = module_lock()
            .lock()
            .expect("Failed to lock mamba fused module");
        module
            .method_is("forward_stateful", &inputs)
            .expect("mamba fused stateful forward failed")
    };
    let (y, final_state, conv_state_out): (Tensor, Tensor, Tensor) =
        out.try_into().expect("bad mamba fused stateful output");
    (y, final_state, conv_state_out)
}

pub fn selective_state_update(
    state: &Tensor,
    x: &Tensor,
    dt: &Tensor,
    a_log: &Tensor,
    b: &Tensor,
    c: &Tensor,
    d_param: &Tensor,
    z: &Tensor,
    dt_bias: &Tensor,
    dt_softplus: bool,
    dt_min: f64,
    dt_max: f64,
    ngroups: i64,
    headdim: i64,
    apply_dt_limit: i64,
) -> (Tensor, Tensor) {
    let module = module_lock()
        .lock()
        .expect("Failed to lock mamba fused module");
    let inputs = vec![
        IValue::Tensor(state.shallow_clone()),
        IValue::Tensor(x.shallow_clone()),
        IValue::Tensor(dt.shallow_clone()),
        IValue::Tensor(a_log.shallow_clone()),
        IValue::Tensor(b.shallow_clone()),
        IValue::Tensor(c.shallow_clone()),
        IValue::Tensor(d_param.shallow_clone()),
        IValue::Tensor(z.shallow_clone()),
        IValue::Tensor(dt_bias.shallow_clone()),
        IValue::Bool(dt_softplus),
        IValue::Double(dt_min),
        IValue::Double(dt_max),
        IValue::Int(ngroups),
        IValue::Int(headdim),
        IValue::Int(apply_dt_limit),
    ];
    let out = module
        .method_is("selective_state_update", &inputs)
        .expect("mamba selective state update failed");
    let (y, new_state): (Tensor, Tensor) = out.try_into().expect("bad selective update output");
    (y, new_state)
}

/// Empty PyTorch's CUDA caching allocator to release reserved memory back to CUDA.
pub fn cuda_empty_cache() {
    let module = module_lock()
        .lock()
        .expect("Failed to lock mamba fused module");
    let empty: &[IValue] = &[];
    let _ = module.method_is("cuda_empty_cache", empty);
}

/// Get CUDA memory statistics from PyTorch's caching allocator.
/// Returns [allocated_current, reserved_current, active_current, allocated_peak, reserved_peak] in bytes.
pub fn cuda_memory_stats() -> Vec<i64> {
    let module = module_lock()
        .lock()
        .expect("Failed to lock mamba fused module");
    let empty: &[IValue] = &[];
    let out = module
        .method_is("cuda_memory_stats", empty)
        .expect("cuda_memory_stats failed");
    let stats: Vec<i64> = out.try_into().expect("bad memory stats output");
    stats
}

pub fn rmsnorm_forward(x: &Tensor, weight: &Tensor, eps: f64) -> Tensor {
    let module = module_lock()
        .lock()
        .expect("Failed to lock mamba fused module");
    let inputs = vec![
        IValue::Tensor(x.shallow_clone()),
        IValue::Tensor(weight.shallow_clone()),
        IValue::Double(eps),
    ];
    let out = module
        .method_is("rmsnorm_forward", &inputs)
        .expect("mamba rmsnorm forward failed");
    out.try_into().expect("bad rmsnorm output")
}
