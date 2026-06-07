use tch::nn::Init;
use tch::{nn, Tensor};

pub(in crate::torch::model) fn linear_with_same_dtype(x: &Tensor, linear: &nn::Linear) -> Tensor {
    let weight = if linear.ws.kind() == x.kind() {
        linear.ws.shallow_clone()
    } else {
        linear.ws.to_kind(x.kind())
    };
    let bias = linear.bs.as_ref().map(|b| {
        if b.kind() == x.kind() {
            b.shallow_clone()
        } else {
            b.to_kind(x.kind())
        }
    });
    x.linear(&weight, bias.as_ref())
}

pub(in crate::torch::model) fn relu_sq_linear(x: &Tensor, out_proj: &nn::Linear) -> Tensor {
    let h = x.relu().square();
    linear_with_same_dtype(&h, out_proj)
}

pub(in crate::torch::model) fn xavier_normal_std(in_features: i64, out_features: i64) -> f64 {
    (2.0 / (in_features + out_features) as f64).sqrt()
}

pub(in crate::torch::model) fn truncated_normal_std(in_features: i64, out_features: i64) -> f64 {
    xavier_normal_std(in_features, out_features) / 0.8796
}

pub(in crate::torch::model) fn truncated_normal_init(in_features: i64, out_features: i64) -> Init {
    Init::Randn {
        mean: 0.0,
        stdev: truncated_normal_std(in_features, out_features),
    }
}

pub(in crate::torch::model) fn linear_truncated(
    p: &nn::Path,
    name: &str,
    in_features: i64,
    out_features: i64,
) -> nn::Linear {
    linear_orthogonal(p, name, in_features, out_features, 1.0)
}

pub(in crate::torch::model) fn linear_orthogonal(
    p: &nn::Path,
    name: &str,
    in_features: i64,
    out_features: i64,
    gain: f64,
) -> nn::Linear {
    nn::linear(
        p / name,
        in_features,
        out_features,
        nn::LinearConfig {
            ws_init: Init::Orthogonal { gain },
            bs_init: None,
            bias: false,
        },
    )
}

pub(in crate::torch::model) fn linear_orthogonal_with_bias(
    p: &nn::Path,
    name: &str,
    in_features: i64,
    out_features: i64,
    gain: f64,
) -> nn::Linear {
    nn::linear(
        p / name,
        in_features,
        out_features,
        nn::LinearConfig {
            ws_init: Init::Orthogonal { gain },
            bs_init: Some(Init::Const(0.0)),
            bias: true,
        },
    )
}

pub(in crate::torch::model) fn linear_residual_out(
    p: &nn::Path,
    name: &str,
    in_features: i64,
    out_features: i64,
) -> nn::Linear {
    nn::linear(
        p / name,
        in_features,
        out_features,
        nn::LinearConfig {
            ws_init: Init::Const(0.0),
            bs_init: None,
            bias: false,
        },
    )
}
