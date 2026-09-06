// Build glue for the fused CUDA kernels.
//
// WHY A SIBLING CRATE AND NOT `vendor/torch-sys`. Three reasons, in order of weight.
// First, `vendor/torch-sys-0.25.0` is a vendored third-party crate carrying a 767 KB
// generated translation unit; adding a `.cu` file there means every kernel edit
// recompiles all of it, and it means every kernel edit collides with anyone patching
// `torch_api.cpp`. Second, `torch-sys`'s build script has no CUDA compilation path at all
// - it only *includes* CUDA headers - so the nvcc invocation would be new code in a file
// we want to keep close to upstream. Third, this crate owns a `links` key of its own, so
// Cargo enforces a single copy of the kernels per binary. The only thing lost is the
// `torch_last_err` channel, which is why `bridge.cpp` carries its own.
//
// The libtorch discovery below is deliberately the same shape as `torch-sys`'s: ask the
// pinned interpreter, and fail loudly if the harness was bypassed.

use anyhow::{bail, Context, Result};
use std::{env, path::PathBuf, process::Command};

/// Every architecture the project actually runs on: the workstation's GB202 (sm_120), plus
/// the datacenter parts a rented box can be, plus PTX for anything newer. These kernels use
/// nothing beyond bf16 conversion intrinsics, so one fat binary covers all of them.
const ARCHITECTURES: &[&str] = &["90", "100", "120"];

const PYTHON_PROBE: &str = r"
import torch
from torch.utils import cpp_extension
print('CXX11:', int(torch._C._GLIBCXX_USE_CXX11_ABI))
for path in cpp_extension.include_paths():
    print('INCLUDE:', path)
for path in cpp_extension.library_paths():
    print('LIB:', path)
";

fn interpreter() -> PathBuf {
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    env::var_os("PYO3_PYTHON")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("python3"))
}

fn cuda_root() -> Result<PathBuf> {
    for key in ["CUDA_HOME", "CUDA_PATH"] {
        println!("cargo:rerun-if-env-changed={key}");
        if let Some(value) = env::var_os(key) {
            let root = PathBuf::from(value);
            if root.join("bin/nvcc").is_file() {
                return Ok(root);
            }
        }
    }
    for candidate in ["/opt/cuda", "/usr/local/cuda"] {
        let root = PathBuf::from(candidate);
        if root.join("bin/nvcc").is_file() {
            return Ok(root);
        }
    }
    bail!(
        "no nvcc found; the fused kernels need a CUDA toolkit in CUDA_HOME, CUDA_PATH, \
         /opt/cuda or /usr/local/cuda"
    )
}

struct Torch {
    cxx11_abi: String,
    includes: Vec<PathBuf>,
    lib_dir: PathBuf,
}

fn probe_torch() -> Result<Torch> {
    if env::var_os("LIBTORCH_USE_PYTORCH").is_none() {
        bail!("the repository requires its pinned Python PyTorch toolchain; run ./torch-env.sh cargo ...");
    }
    let python = interpreter();
    let output = Command::new(&python)
        .arg("-c")
        .arg(PYTHON_PROBE)
        .output()
        .with_context(|| format!("could not run {}", python.display()))?;
    if !output.status.success() {
        bail!(
            "{} could not describe its torch install: {}",
            python.display(),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let text = String::from_utf8(output.stdout)?;
    let mut cxx11_abi = None;
    let mut includes = Vec::new();
    let mut lib_dir = None;
    for line in text.lines() {
        match line.split_once(": ") {
            Some(("CXX11", value)) => cxx11_abi = Some(value.trim().to_owned()),
            Some(("INCLUDE", value)) => includes.push(PathBuf::from(value.trim())),
            Some(("LIB", value)) if lib_dir.is_none() => {
                lib_dir = Some(PathBuf::from(value.trim()))
            }
            _ => {}
        }
    }
    Ok(Torch {
        cxx11_abi: cxx11_abi.context("torch did not report its C++ ABI")?,
        includes,
        // `torch-sys` publishes the directory it linked against; preferring it keeps this
        // crate's rpath identical to the one the rest of the binary already has.
        lib_dir: env::var_os("DEP_TCH_LIBTORCH_LIB")
            .map(PathBuf::from)
            .or(lib_dir)
            .context("torch did not report a library directory")?,
    })
}

/// Compiles `kernels.cu` with nvcc and archives the object, because `cc`'s CUDA mode
/// forwards user flags verbatim and would need the same explicit `-Xcompiler` handling
/// anyway. Returns the archive's directory and library name.
fn compile_kernels(cuda: &PathBuf) -> Result<()> {
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let object = out_dir.join("fk_kernels.o");
    let mut nvcc = Command::new(cuda.join("bin/nvcc"));
    nvcc.arg("-std=c++17")
        .arg("-O3")
        .arg("-lineinfo")
        .arg("--compiler-options")
        .arg("-fPIC")
        .arg("-c")
        .arg("csrc/kernels.cu")
        .arg("-o")
        .arg(&object);
    for architecture in ARCHITECTURES {
        nvcc.arg(format!(
            "-gencode=arch=compute_{architecture},code=sm_{architecture}"
        ));
    }
    // PTX for the newest target, so a future card runs these without a rebuild.
    let newest = ARCHITECTURES.last().expect("at least one architecture");
    nvcc.arg(format!(
        "-gencode=arch=compute_{newest},code=compute_{newest}"
    ));
    let status = nvcc.status().context("could not run nvcc")?;
    if !status.success() {
        bail!("nvcc failed to compile csrc/kernels.cu: {status}");
    }

    let archive = out_dir.join("libfused_kernels_cuda.a");
    let _ = std::fs::remove_file(&archive);
    let status = cc::Build::new()
        .get_archiver()
        .arg("crs")
        .arg(&archive)
        .arg(&object)
        .status()
        .context("could not run the archiver")?;
    if !status.success() {
        bail!("archiving the fused kernels failed: {status}");
    }
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=fused_kernels_cuda");
    Ok(())
}

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=csrc/kernels.cu");
    println!("cargo:rerun-if-changed=csrc/kernels.h");
    println!("cargo:rerun-if-changed=csrc/bridge.cpp");

    let torch = probe_torch()?;
    let cuda = cuda_root()?;
    compile_kernels(&cuda)?;

    cc::Build::new()
        .cpp(true)
        .pic(true)
        .warnings(false)
        .includes(&torch.includes)
        .include(cuda.join("include"))
        .include("csrc")
        .flag("-std=c++17")
        .flag(format!("-D_GLIBCXX_USE_CXX11_ABI={}", torch.cxx11_abi))
        .flag("-DGLOG_USE_GLOG_EXPORT")
        .flag(format!("-Wl,-rpath={}", torch.lib_dir.display()))
        .file("csrc/bridge.cpp")
        .compile("fused_kernels_bridge");

    // Naming the libraries is not enough. `libtorch.so` is a stub whose only job is to
    // pull in `libtorch_cpu` and `libtorch_cuda`, and nothing in Rust references a symbol
    // from it, so the default `--as-needed` drops it and with it every CUDA dispatch key -
    // the observable symptom being `Cuda::is_available()` returning false in a binary that
    // linked fine. `trading_bots/build.rs` already solves exactly this; these are the same
    // three link arguments, and they apply to this package's own bin and tests.
    println!("cargo:rustc-link-search=native={}", torch.lib_dir.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath={}", torch.lib_dir.display());
    println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
    println!("cargo:rustc-link-arg=-ltorch");
    println!("cargo:rustc-link-arg=-lc10");
    let cudart = cuda.join("lib64");
    if cudart.join("libcudart.so").exists() {
        println!("cargo:rustc-link-search=native={}", cudart.display());
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    Ok(())
}
