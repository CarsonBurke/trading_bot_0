use tch::{Kind, Tensor};

use super::init::linear_with_same_dtype;
use super::trading_model::{ModelOutput, TradingModel};
use crate::torch::action_space::beta_concentration;
use crate::torch::constants::TICKERS_COUNT;

impl TradingModel {
    pub(in crate::torch::model) fn head_from_actor_critic_cls(
        &self,
        actor: &Tensor,
        critic: &Tensor,
        batch_size: i64,
    ) -> ModelOutput {
        let actor_in = actor.view([batch_size, TICKERS_COUNT, self.model_dim]);
        let policy_concentration =
            linear_with_same_dtype(&actor_in, &self.policy_concentration).to_kind(Kind::Float);
        let raw_alpha = policy_concentration.narrow(-1, 0, 1).squeeze_dim(-1);
        let raw_beta = policy_concentration.narrow(-1, 1, 1).squeeze_dim(-1);
        let alpha = beta_concentration(&raw_alpha);
        let beta = beta_concentration(&raw_beta);

        let critic_in = critic.view([batch_size, TICKERS_COUNT * self.model_dim]);
        let value_logits = linear_with_same_dtype(&critic_in, &self.value_proj)
            + self.value_bias.to_kind(critic_in.kind());

        (
            value_logits.to_kind(Kind::Float),
            alpha.to_kind(Kind::Float),
            beta.to_kind(Kind::Float),
        )
    }
}
