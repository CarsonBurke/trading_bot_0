use tch::{Kind, Tensor};

use super::init::linear_with_same_dtype;
use super::trading_model::{ModelOutput, TradingModel};
use crate::torch::action_space::gaussian_std_from_log_var;
use crate::torch::constants::TICKERS_COUNT;

impl TradingModel {
    pub(in crate::torch::model) fn head_from_actor_critic_cls(
        &self,
        actor: &Tensor,
        critic: &Tensor,
        batch_size: i64,
    ) -> ModelOutput {
        let actor_in = actor.view([batch_size, TICKERS_COUNT, self.model_dim]);
        let policy_mean_log_var =
            linear_with_same_dtype(&actor_in, &self.policy_mean_log_var).to_kind(Kind::Float);
        let action_mean = policy_mean_log_var.narrow(-1, 0, 1).squeeze_dim(-1);
        let action_log_var = policy_mean_log_var.narrow(-1, 1, 1).squeeze_dim(-1);
        let action_std = gaussian_std_from_log_var(&action_log_var);
        let action_log_std = action_std.log();

        let critic_in = critic.view([batch_size, TICKERS_COUNT * self.model_dim]);
        let value_logits = linear_with_same_dtype(&critic_in, &self.value_proj);

        (
            value_logits.to_kind(Kind::Float),
            action_mean.to_kind(Kind::Float),
            action_log_std.to_kind(Kind::Float),
            action_std.to_kind(Kind::Float),
        )
    }
}
