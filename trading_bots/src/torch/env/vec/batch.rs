pub struct CpuStepBatch {
    pub actions_f32: Vec<f32>,
    pub(super) actions_f64: Vec<f64>,
    pub reset_indices: Vec<usize>,
    pub reset_price_deltas: Vec<f32>,
}

impl CpuStepBatch {
    pub fn new(nprocs: usize, action_dim: usize, pd_dim: usize) -> Self {
        Self {
            actions_f32: vec![0.0; nprocs * action_dim],
            actions_f64: vec![0.0; nprocs * action_dim],
            reset_indices: Vec::with_capacity(nprocs),
            reset_price_deltas: Vec::with_capacity(nprocs * pd_dim),
        }
    }
}
