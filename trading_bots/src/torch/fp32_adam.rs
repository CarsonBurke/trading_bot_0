use tch::{Kind, Tensor, nn};

struct ParamGroup {
    bf16_param: Tensor,
    fp32_master: Tensor,
    m: Tensor,
    v: Tensor,
}

pub struct Fp32Adam {
    groups: Vec<ParamGroup>,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    step_count: i64,
}

impl Fp32Adam {
    pub fn new(vs: &nn::VarStore, lr: f64) -> Self {
        let groups: Vec<ParamGroup> = vs
            .trainable_variables()
            .into_iter()
            .map(|p| {
                let fp32_master = p.to_kind(Kind::Float).detach();
                let m = Tensor::zeros_like(&fp32_master);
                let v = Tensor::zeros_like(&fp32_master);
                ParamGroup {
                    bf16_param: p,
                    fp32_master,
                    m,
                    v,
                }
            })
            .collect();

        Fp32Adam {
            groups,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-5,
            step_count: 0,
        }
    }

    pub fn step(&mut self) {
        self.step_count += 1;
        let bc1 = 1.0 - self.beta1.powi(self.step_count as i32);
        let bc2 = 1.0 - self.beta2.powi(self.step_count as i32);
        let step_lr = self.lr / bc1;

        for g in &mut self.groups {
            let grad_bf16 = g.bf16_param.grad();
            if !grad_bf16.defined() {
                continue;
            }
            let grad = grad_bf16.to_kind(Kind::Float);

            // m = beta1 * m + (1 - beta1) * grad
            let _ = g.m.g_mul_scalar_(self.beta1);
            let _ = g.m.g_add_(&(&grad * (1.0 - self.beta1)));
            // v = beta2 * v + (1 - beta2) * grad^2
            let _ = g.v.g_mul_scalar_(self.beta2);
            let _ = g.v.g_add_(&(&grad * &grad * (1.0 - self.beta2)));

            // master -= step_lr * m / (sqrt(v / bc2) + eps)
            let denom = (g.v.shallow_clone() / bc2).sqrt_() + self.eps;
            let update = &g.m / &denom * step_lr;
            let _ = g.fp32_master.g_sub_(&update);

            tch::no_grad(|| {
                g.bf16_param
                    .set_data(&g.fp32_master.to_kind(Kind::BFloat16));
            });
        }
    }

    pub fn zero_grad(&self) {
        for g in &self.groups {
            let mut grad = g.bf16_param.grad();
            if grad.defined() {
                let _ = grad.zero_();
            }
        }
    }

    pub fn clip_grad_norm(&self, max_norm: f64) -> f64 {
        let mut total_norm_sq = 0.0f64;
        for g in &self.groups {
            let grad = g.bf16_param.grad();
            if grad.defined() {
                let norm_sq = grad
                    .to_kind(Kind::Float)
                    .pow_tensor_scalar(2)
                    .sum(Kind::Float)
                    .double_value(&[]);
                total_norm_sq += norm_sq;
            }
        }
        let total_norm = total_norm_sq.sqrt();
        if total_norm > max_norm {
            let scale = max_norm / (total_norm + 1e-6);
            for g in &self.groups {
                let mut grad = g.bf16_param.grad();
                if grad.defined() {
                    let _ = grad.g_mul_scalar_(scale);
                }
            }
        }
        total_norm
    }

    pub fn trainable_variables(&self) -> Vec<Tensor> {
        self.groups
            .iter()
            .map(|g| g.bf16_param.shallow_clone())
            .collect()
    }

    pub fn sync_from_params(&mut self) {
        for g in &mut self.groups {
            g.fp32_master = g.bf16_param.to_kind(Kind::Float).detach();
            let _ = g.m.zero_();
            let _ = g.v.zero_();
        }
        self.step_count = 0;
    }
}
