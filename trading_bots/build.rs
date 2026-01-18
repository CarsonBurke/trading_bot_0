use std::path::PathBuf;
use std::process::Command;

fn main() {
    let os = std::env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");

    match os.as_str() {
        "linux" | "windows" => {
            if let Some(lib_path) = std::env::var_os("DEP_TCH_LIBTORCH_LIB") {
                println!("cargo:rustc-link-arg=-Wl,-rpath={}", lib_path.to_string_lossy());
            }
            println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
            println!("cargo:rustc-link-arg=-ltorch");
        }
        _ => {}
    }

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("Missing OUT_DIR"));
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("Missing CARGO_MANIFEST_DIR"));
    let cuda_ops_dir = manifest_dir.join("cuda_ops");
    let cpp_src = cuda_ops_dir.join("mamba_fused_ops.cpp");
    let cu_src = cuda_ops_dir.join("mamba_fused_ops.cu");
    let wrapper_src = cuda_ops_dir.join("gen_wrapper.py");
    let triton_dir = cuda_ops_dir.join("triton");

    println!("cargo:rerun-if-changed={}", cpp_src.display());
    println!("cargo:rerun-if-changed={}", cu_src.display());
    println!("cargo:rerun-if-changed={}", wrapper_src.display());
    for file in [
        "ssd_common.cuh",
        "ssd_state_passing.cu",
        "ssd_bmm.cu",
        "ssd_chunk_state.cu",
        "ssd_chunk_scan.cu",
        "ssd_combined.cu",
        "selective_state_update.cu",
        "rmsnorm.cu",
    ] {
        println!("cargo:rerun-if-changed={}", triton_dir.join(file).display());
    }
    println!("cargo:rerun-if-changed={}", manifest_dir.join("build.rs").display());
    println!(
        "cargo:rerun-if-changed={}",
        manifest_dir.join("src/torch/mamba_fused.rs").display()
    );

    let libtorch_lib = libtorch_lib_dir();
    let libtorch_root = libtorch_lib.parent().expect("Invalid libtorch lib path");
    let include1 = libtorch_root.join("include");
    let include2 = libtorch_root.join("include/torch/csrc/api/include");

    let cxx11_abi = detect_cxx11_abi();
    let cxx11_define = format!("-D_GLIBCXX_USE_CXX11_ABI={}", cxx11_abi);

    let cpp_obj = out_dir.join("mamba_fused_ops.o");
    let cu_obj = out_dir.join("mamba_fused_ops_cuda.o");
    let so_path = out_dir.join("libmamba_fused_ops.so");

    run(Command::new("c++")
        .arg("-O3")
        .arg("-std=c++17")
        .arg("-fPIC")
        .arg(cxx11_define.clone())
        .arg(format!("-I{}", include1.display()))
        .arg(format!("-I{}", include2.display()))
        .arg("-I/opt/cuda/include")
        .arg("-c")
        .arg(&cpp_src)
        .arg("-o")
        .arg(&cpp_obj));

    run(Command::new("nvcc")
        .arg("-O3")
        .arg("-std=c++17")
        .arg("-lineinfo")
        .arg(cxx11_define.clone())
        .arg(format!("-I{}", include1.display()))
        .arg(format!("-I{}", include2.display()))
        .arg("--compiler-options")
        .arg("-fPIC")
        .arg("-c")
        .arg(&cu_src)
        .arg("-o")
        .arg(&cu_obj));

    run(Command::new("nvcc")
        .arg("-shared")
        .arg("-o")
        .arg(&so_path)
        .arg(&cpp_obj)
        .arg(&cu_obj)
        .arg(format!("-L{}", libtorch_lib.display()))
        .arg("-ltorch")
        .arg("-ltorch_cpu")
        .arg("-ltorch_cuda")
        .arg("-lc10")
        .arg("-lcudart"));

    let wrapper_path = out_dir.join("mamba_fused_wrapper.pt");
    // Only regenerate wrapper if it doesn't exist (python3.14 compatibility workaround)
    if !wrapper_path.exists() {
        run(Command::new("python")
            .arg(cuda_ops_dir.join("gen_wrapper.py"))
            .env("MAMBA_FUSED_LIB_PATH", &so_path)
            .env("MAMBA_FUSED_WRAPPER_PATH", &wrapper_path));
    }

    println!("cargo:rustc-env=MAMBA_FUSED_LIB_PATH={}", so_path.display());
    println!("cargo:rustc-env=MAMBA_FUSED_WRAPPER_PATH={}", wrapper_path.display());
}

fn run(cmd: &mut Command) {
    let status = cmd.status().expect("Failed to run command");
    if !status.success() {
        panic!("Command failed: {:?}", cmd);
    }
}

fn libtorch_lib_dir() -> PathBuf {
    if let Ok(path) = std::env::var("DEP_TCH_LIBTORCH_LIB") {
        return PathBuf::from(path);
    }
    if let Ok(path) = std::env::var("LIBTORCH") {
        return PathBuf::from(path).join("lib");
    }
    let output = Command::new("python")
        .arg("-c")
        .arg("import torch; import json; print(json.dumps(torch.utils.cpp_extension.library_paths()))")
        .output()
        .expect("Failed to query torch library paths");
    if output.status.success() {
        let s = String::from_utf8_lossy(&output.stdout);
        if let Ok(paths) = serde_json::from_str::<Vec<String>>(s.trim()) {
            if let Some(path) = paths.first() {
                return PathBuf::from(path);
            }
        }
    }
    panic!("DEP_TCH_LIBTORCH_LIB or LIBTORCH not set, and torch library paths unavailable");
}

fn detect_cxx11_abi() -> i32 {
    let output = Command::new("python")
        .arg("-c")
        .arg("import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
        .output()
        .expect("Failed to query torch ABI");
    if !output.status.success() {
        return 1;
    }
    let s = String::from_utf8_lossy(&output.stdout);
    s.trim().parse().unwrap_or(1)
}
