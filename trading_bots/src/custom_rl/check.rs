use std::env;
use tch::{Device, Kind, Tensor};

pub fn check() {
    println!("=== tch-rs 0.20.0 ROCm Debug ===");

    // Print environment variables
    println!("Environment:");
    for var in &[
        "DRI_PRIME",
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "HSA_OVERRIDE_GFX_VERSION",
    ] {
        println!("  {}: {:?}", var, env::var(var).ok());
    }

    // Check tch-rs CUDA detection
    println!("\ntch-rs GPU detection:");
    println!("CUDA available: {}", tch::Cuda::is_available());
    println!("Device count: {}", tch::Cuda::device_count());

    if tch::Cuda::device_count() > 0 {
        if tch::Cuda::is_available() {
            println!("Device is CUDA available");
        }
    }

    // Test cuda_if_available()
    let device = Device::cuda_if_available();
    println!("\nDevice::cuda_if_available() selected: {:?}", device);

    // Try to force specific devices
    println!("\nTesting specific devices:");
    for device_id in 0..2 {
        println!("Testing Device::Cuda({}):", device_id);
        match std::panic::catch_unwind(|| {
            let device = Device::Cuda(device_id);
            let tensor = Tensor::ones(&[2, 2], (Kind::Float, device));
            (device, tensor.size())
        }) {
            Ok((device, size)) => {
                println!("  SUCCESS: {:?}, tensor size: {:?}", device, size);
            }
            Err(_) => {
                println!("  FAILED: Device::Cuda({}) not available", device_id);
            }
        }
    }

    // Perform actual computation test
    println!("\n=== Computation Test ===");
    let device = Device::cuda_if_available();

    let a = Tensor::randn(&[1000, 1000], (Kind::Float, device));
    let b = Tensor::randn(&[1000, 1000], (Kind::Float, device));

    println!("Created tensors on device: {:?}", device);

    let start = std::time::Instant::now();
    let c = &a.matmul(&b);
    let duration = start.elapsed();

    println!("Matrix multiplication completed in: {:?}", duration);
    println!("Result tensor device: {:?}", c.device());
    println!("Result tensor shape: {:?}", c.size());

    // Test device transfer
    if tch::Cuda::is_available() && tch::Cuda::device_count() > 0 {
        println!("\n=== Device Transfer Test ===");
        let cpu_tensor = Tensor::ones(&[100, 100], (Kind::Float, Device::Cpu));
        println!("CPU tensor created");

        match std::panic::catch_unwind(|| {
            let gpu_tensor = cpu_tensor.to_device(Device::Cuda(0));
            gpu_tensor.size()
        }) {
            Ok(size) => println!("Successfully transferred to GPU, size: {:?}", size),
            Err(_) => println!("Failed to transfer tensor to GPU"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_available() {
        // This test will pass if GPU is available
        if tch::Cuda::is_available() {
            assert!(tch::Cuda::device_count() > 0);
        }
    }
}
