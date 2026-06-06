use crate::torch::model::ModelVariant;

use super::trainer::Trainer;

pub async fn train(
    weights_path: Option<&str>,
    model_variant: ModelVariant,
    run_name: Option<String>,
) {
    let mut trainer = Trainer::new(weights_path, model_variant, run_name);
    trainer.run().await;
}
