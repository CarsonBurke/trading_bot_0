use tch::nn::Init;
use tch::{nn, Kind, Tensor};

use crate::torch::constants::{PRICE_DELTAS_PER_TICKER, TICKERS_COUNT};

const NUM_ENSEMBLE: i64 = 5;
const HIDDEN_DIM: i64 = 64;
// Use last N price deltas as context
const CONTEXT_LEN: i64 = 32;
const INPUT_DIM: i64 = CONTEXT_LEN; // last 32 price deltas per ticker
const OUTPUT_DIM: i64 = 1; // predict next price delta

fn trunc_normal(in_f: i64, out_f: i64) -> Init {
    let std = (2.0 / (in_f + out_f) as f64).sqrt() / 0.8796;
    Init::Randn { mean: 0.0, stdev: std }
}

/// Returns (mean_prediction, disagreement, all_predictions for loss)
pub type EnsembleModel = Box<dyn Fn(&Tensor, bool) -> (Tensor, Tensor, Vec<Tensor>)>;

/// Ensemble of forward dynamics models for disagreement-based exploration
/// Uses last CONTEXT_LEN price deltas to predict next delta
/// Disagreement = std across ensemble -> intrinsic exploration reward
pub fn ensemble(p: &nn::Path) -> EnsembleModel {
    let mut fc1_layers = Vec::with_capacity(NUM_ENSEMBLE as usize);
    let mut fc2_layers = Vec::with_capacity(NUM_ENSEMBLE as usize);
    let mut out_layers = Vec::with_capacity(NUM_ENSEMBLE as usize);

    for i in 0..NUM_ENSEMBLE {
        let fc1 = nn::linear(
            p / format!("ens{}_fc1", i),
            INPUT_DIM,
            HIDDEN_DIM,
            nn::LinearConfig { ws_init: trunc_normal(INPUT_DIM, HIDDEN_DIM), ..Default::default() },
        );
        let fc2 = nn::linear(
            p / format!("ens{}_fc2", i),
            HIDDEN_DIM,
            HIDDEN_DIM,
            nn::LinearConfig { ws_init: trunc_normal(HIDDEN_DIM, HIDDEN_DIM), ..Default::default() },
        );
        let out = nn::linear(
            p / format!("ens{}_out", i),
            HIDDEN_DIM,
            OUTPUT_DIM,
            nn::LinearConfig {
                ws_init: Init::Uniform { lo: -0.01, up: 0.01 },
                bs_init: Some(Init::Const(0.0)),
                bias: true,
            },
        );

        fc1_layers.push(fc1);
        fc2_layers.push(fc2);
        out_layers.push(out);
    }

    let device = p.device();
    Box::new(move |price_deltas: &Tensor, train: bool| {
        // price_deltas: [batch, TICKERS * PRICE_DELTAS_PER_TICKER]
        let batch_size = price_deltas.size()[0];

        // Reshape to [batch, TICKERS, PRICE_DELTAS_PER_TICKER]
        let deltas = price_deltas.view([batch_size, TICKERS_COUNT, PRICE_DELTAS_PER_TICKER as i64]);

        // Take last CONTEXT_LEN deltas per ticker: [batch, TICKERS, CONTEXT_LEN]
        let context_start = PRICE_DELTAS_PER_TICKER as i64 - CONTEXT_LEN;
        let context = deltas.narrow(2, context_start, CONTEXT_LEN);

        // Flatten to [batch * TICKERS, CONTEXT_LEN]
        let context_flat = context.reshape([batch_size * TICKERS_COUNT, CONTEXT_LEN]).to_device(device);

        let mut predictions = Vec::with_capacity(NUM_ENSEMBLE as usize);

        for i in 0..NUM_ENSEMBLE as usize {
            let x = context_flat.apply(&fc1_layers[i]).silu();
            let x = x.apply(&fc2_layers[i]).silu();
            let pred = x.apply(&out_layers[i]); // [batch * TICKERS, 1]
            predictions.push(pred.squeeze_dim(-1)); // [batch * TICKERS]
        }

        // Stack: [NUM_ENSEMBLE, batch * TICKERS]
        let stacked = Tensor::stack(&predictions, 0);

        // Mean prediction
        let mean_pred = stacked.mean_dim(0, false, Kind::Float); // [batch * TICKERS]

        // Disagreement = std across ensemble
        let disagreement = stacked.std_dim(0, false, false); // [batch * TICKERS]

        // Average disagreement across tickers for per-sample bonus
        let disagreement_per_step = disagreement
            .view([batch_size, TICKERS_COUNT])
            .mean_dim(1, false, Kind::Float); // [batch]

        // Log disagreement for more stable reward signal
        let disagreement_bonus = if train {
            (disagreement_per_step + 1e-6).log()
        } else {
            disagreement_per_step
        };

        (mean_pred, disagreement_bonus, predictions)
    })
}

/// Compute ensemble loss against actual next price deltas
/// targets: [batch * TICKERS] - the actual next delta for each ticker
pub fn ensemble_loss(predictions: &[Tensor], targets: &Tensor) -> Tensor {
    let mut total_loss = Tensor::zeros(&[], (Kind::Float, predictions[0].device()));

    for pred in predictions {
        let loss = (pred - targets).pow_tensor_scalar(2).mean(Kind::Float);
        total_loss = total_loss + loss;
    }

    total_loss / (predictions.len() as f64)
}
