use std::sync::OnceLock;

use libloading::Library;
use tch::{CModule, IValue, Tensor};

static LIB: OnceLock<Library> = OnceLock::new();
static MODULE: OnceLock<CModule> = OnceLock::new();

fn load_module() -> &'static CModule {
    let lib_path = env!("MAMBA_FUSED_LIB_PATH");
    let wrapper_path = env!("MAMBA_FUSED_WRAPPER_PATH");
    LIB.get_or_init(|| unsafe { Library::new(lib_path).expect("Failed to load mamba fused ops") });
    MODULE.get_or_init(|| CModule::load(wrapper_path).expect("Failed to load mamba fused wrapper"))
}

pub fn fused_conv_scan(
    zxbcdt: &Tensor,
    conv_w: &Tensor,
    conv_b: &Tensor,
    dt_bias: &Tensor,
    a_log: &Tensor,
    dt_scale: &Tensor,
    initial_state: &Tensor,
    chunk_size: i64,
    ngroups: i64,
    headdim: i64,
    dt_min: f64,
    dt_max: f64,
) -> (Tensor, Tensor) {
    let module = load_module();
    let inputs = vec![
        IValue::Tensor(zxbcdt.shallow_clone()),
        IValue::Tensor(conv_w.shallow_clone()),
        IValue::Tensor(conv_b.shallow_clone()),
        IValue::Tensor(dt_bias.shallow_clone()),
        IValue::Tensor(a_log.shallow_clone()),
        IValue::Tensor(dt_scale.shallow_clone()),
        IValue::Tensor(initial_state.shallow_clone()),
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
