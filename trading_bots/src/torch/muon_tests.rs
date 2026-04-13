#[cfg(test)]
mod tests {
    use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Tensor};

    use crate::torch::muon::{Muon, MuonConfig};

    const HIDDEN: i64 = 128;
    const TRAIN_STEPS: usize = 500;
    const DATASET_SIZE: i64 = 2048;
    const BATCH_SIZE: i64 = 64;
    const INPUT_DIM: i64 = 16;

    fn build_mlp(vs: &nn::Path) -> impl Module {
        nn::seq()
            .add(nn::linear(
                vs / "fc1",
                INPUT_DIM,
                HIDDEN,
                Default::default(),
            ))
            .add_fn(|x| x.gelu("none"))
            .add(nn::linear(vs / "fc2", HIDDEN, HIDDEN, Default::default()))
            .add_fn(|x| x.gelu("none"))
            .add(nn::linear(vs / "fc3", HIDDEN, HIDDEN, Default::default()))
            .add_fn(|x| x.gelu("none"))
            .add(nn::linear(vs / "fc4", HIDDEN, 1, Default::default()))
    }

    /// Fixed dataset; training draws fresh minibatches via random indexing.
    fn make_dataset(device: Device) -> (Tensor, Tensor) {
        let _guard = tch::no_grad_guard();
        let w = Tensor::randn([INPUT_DIM, 4], (Kind::Float, device));
        let x = Tensor::randn([DATASET_SIZE, INPUT_DIM], (Kind::Float, device));
        let h = x.matmul(&w);
        let y = (h.slice(1, 0, 1, 1).sin() * h.slice(1, 1, 2, 1).cos()
            + 0.3 * h.slice(1, 2, 3, 1) * h.slice(1, 3, 4, 1).tanh())
            + 0.05 * Tensor::randn([DATASET_SIZE, 1], (Kind::Float, device));
        (x, y)
    }

    /// Eval loss over full dataset (no grad).
    fn eval_loss(net: &dyn Module, x: &Tensor, y: &Tensor) -> f64 {
        tch::no_grad(|| {
            let pred = net.forward(x);
            (&pred - y).square().mean(Kind::Float).double_value(&[])
        })
    }

    fn train_adamw(device: Device, seed: i64) -> Vec<f64> {
        tch::manual_seed(seed);
        let vs = nn::VarStore::new(device);
        let net = build_mlp(&vs.root());
        let mut opt = nn::AdamW::default().build(&vs, 1e-3).expect("adamw");

        tch::manual_seed(seed + 1000);
        let (x_all, y_all) = make_dataset(device);
        let mut losses = Vec::with_capacity(TRAIN_STEPS);

        for step in 0..TRAIN_STEPS {
            let idx = Tensor::randint(DATASET_SIZE, [BATCH_SIZE], (Kind::Int64, device));
            let xb = x_all.index_select(0, &idx);
            let yb = y_all.index_select(0, &idx);

            let pred = net.forward(&xb);
            let loss = (&pred - &yb).square().mean(Kind::Float);
            opt.backward_step(&loss);

            if step % 50 == 0 || step == TRAIN_STEPS - 1 {
                losses.push(eval_loss(&net, &x_all, &y_all));
            }
        }
        losses
    }

    fn train_muon(device: Device, seed: i64) -> Vec<f64> {
        tch::manual_seed(seed);
        let vs = nn::VarStore::new(device);
        let net = build_mlp(&vs.root());
        let trainable = vs.trainable_variables();
        let mut opt = Muon::new(
            &trainable,
            MuonConfig {
                lr: 0.02,
                adamw_lr: 1e-3,
                ..MuonConfig::default()
            },
        );

        tch::manual_seed(seed + 1000);
        let (x_all, y_all) = make_dataset(device);
        let mut losses = Vec::with_capacity(TRAIN_STEPS);

        for step in 0..TRAIN_STEPS {
            let idx = Tensor::randint(DATASET_SIZE, [BATCH_SIZE], (Kind::Int64, device));
            let xb = x_all.index_select(0, &idx);
            let yb = y_all.index_select(0, &idx);

            let pred = net.forward(&xb);
            let loss = (&pred - &yb).square().mean(Kind::Float);
            loss.backward();
            opt.step();
            opt.zero_grad();

            if step % 50 == 0 || step == TRAIN_STEPS - 1 {
                losses.push(eval_loss(&net, &x_all, &y_all));
            }
        }
        losses
    }

    #[test]
    fn muon_converges_on_synthetic_regression() {
        let device = Device::Cpu;
        let losses = train_muon(device, 42);
        let first = losses[0];
        let last = *losses.last().unwrap();
        println!(
            "Muon: loss {:.6} -> {:.6} ({:.1}x reduction)",
            first,
            last,
            first / last
        );
        assert!(
            last < first * 0.1,
            "Muon failed to converge: {:.6} -> {:.6}",
            first,
            last
        );
    }

    #[test]
    fn adamw_converges_on_synthetic_regression() {
        let device = Device::Cpu;
        let losses = train_adamw(device, 42);
        let first = losses[0];
        let last = *losses.last().unwrap();
        println!(
            "AdamW: loss {:.6} -> {:.6} ({:.1}x reduction)",
            first,
            last,
            first / last
        );
        assert!(
            last < first * 0.1,
            "AdamW failed to converge: {:.6} -> {:.6}",
            first,
            last
        );
    }

    #[test]
    fn muon_vs_adamw_comparison() {
        let device = Device::Cpu;
        let seed = 42;

        let adamw_losses = train_adamw(device, seed);
        let muon_losses = train_muon(device, seed);

        println!(
            "\n{:<8} {:>12} {:>12} {:>10}",
            "Step", "AdamW", "Muon", "Winner"
        );
        println!("{}", "-".repeat(46));
        let steps: Vec<usize> = (0..TRAIN_STEPS)
            .filter(|&s| s % 50 == 0 || s == TRAIN_STEPS - 1)
            .collect();
        for (i, &s) in steps.iter().enumerate() {
            let a = adamw_losses[i];
            let m = muon_losses[i];
            let winner = if m < a { "Muon" } else { "AdamW" };
            println!("{:<8} {:>12.6} {:>12.6} {:>10}", s + 1, a, m, winner);
        }

        let adamw_final = *adamw_losses.last().unwrap();
        let muon_final = *muon_losses.last().unwrap();
        println!(
            "\nFinal ratio (Muon/AdamW): {:.3}x  — {}",
            muon_final / adamw_final,
            if muon_final < adamw_final {
                "Muon wins"
            } else {
                "AdamW wins"
            }
        );

        // Both must converge
        assert!(
            adamw_final < 0.5,
            "AdamW did not converge: {:.6}",
            adamw_final
        );
        assert!(muon_final < 0.5, "Muon did not converge: {:.6}", muon_final);
    }
}
