use burn::{backend::{rocm::RocmDevice, wgpu::{graphics::AutoGraphicsApi, WgpuDevice}, Rocm, Wgpu}, tensor::{backend::{AutodiffBackend, Backend}, Tensor}};
use candle_core::cpu_backend::CpuDevice;

pub fn check() {
    // Would also be nice to check CPU to compare to wgpu
    test_compute::<Wgpu>(&WgpuDevice::default(), "WGPU");
    // test_compute::<Rocm>(&RocmDevice::default(), "ROCm");
}

fn test_compute<B: Backend>(device: &B::Device, test_name: &str)
where
    B::FloatElem: Into<f32>,
{
    println!("Testing {}", test_name);

    let start = std::time::Instant::now();
    
    let a = Tensor::<B, 2>::ones([256, 256], device);
    let b = Tensor::<B, 2>::ones([256, 256], device);
    let c = a.matmul(b);
    let sum: f32 = c.sum().into_scalar().into();
    
    let duration = start.elapsed();
    println!("{} Computation OK, sum = {}, time taken: {:?}", test_name, sum, duration);
}
