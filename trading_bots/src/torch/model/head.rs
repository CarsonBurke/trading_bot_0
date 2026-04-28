use tch::{Kind, Tensor};

use super::{linear_with_same_dtype, ModelOutput, TradingModel};
use crate::torch::action_space::{beta_concentration_from_head, beta_std};
use crate::torch::constants::TICKERS_COUNT;

impl TradingModel {
    pub(super) fn head_from_actor_critic_cls(
        &self,
        actor: &Tensor,
        critic: &Tensor,
        batch_size: i64,
    ) -> ModelOutput {
        let actor_in = actor.view([batch_size, TICKERS_COUNT, self.model_dim]);
        let policy_alpha_beta =
            linear_with_same_dtype(&actor_in, &self.policy_alpha_beta).to_kind(Kind::Float);
        let action_alpha =
            beta_concentration_from_head(&policy_alpha_beta.narrow(-1, 0, 1).squeeze_dim(-1));
        let action_beta =
            beta_concentration_from_head(&policy_alpha_beta.narrow(-1, 1, 1).squeeze_dim(-1));
        let action_std = beta_std(&action_alpha, &action_beta);

        let critic_in = critic.view([batch_size, TICKERS_COUNT * self.model_dim]);
        let value_logits = linear_with_same_dtype(&critic_in, &self.value_proj);

        (
            value_logits.to_kind(Kind::Float),
            action_alpha.to_kind(Kind::Float),
            action_beta.to_kind(Kind::Float),
            action_std.to_kind(Kind::Float),
        )
    }
}
