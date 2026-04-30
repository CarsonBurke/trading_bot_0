use tch::{Device, Tensor};

use crate::torch::cuda::graph::{empty_cuda_cache, synchronize_device, CudaGraph};
use crate::torch::model::TradingModel;
use crate::torch::value::hl_gauss::HlGaussBins;

use super::pmpo::{compute_pmpo_minibatch_outputs, zero_existing_grads, PmpoMinibatchMetrics};

pub(crate) struct PmpoMinibatchCudaGraph {
    graph: CudaGraph,
    captured: bool,
    sample_count: i64,
    windowed: Tensor,
    static_flat: Tensor,
    actions: Tensor,
    old_log_probs: Tensor,
    old_action_alpha: Tensor,
    old_action_beta: Tensor,
    advantages: Tensor,
    returns: Tensor,
    approx_kl: Option<Tensor>,
    action_loss: Option<Tensor>,
    value_loss: Option<Tensor>,
    reverse_kl_loss: Option<Tensor>,
    dist_entropy: Option<Tensor>,
    clipped: Option<Tensor>,
}

impl PmpoMinibatchCudaGraph {
    pub(crate) fn new(
        device: Device,
        windowed: &Tensor,
        static_flat: &Tensor,
        actions: &Tensor,
        old_log_probs: &Tensor,
        old_action_alpha: &Tensor,
        old_action_beta: &Tensor,
        advantages: &Tensor,
        returns: &Tensor,
        sample_count: i64,
    ) -> Self {
        Self {
            graph: CudaGraph::new(device),
            captured: false,
            sample_count,
            windowed: Tensor::zeros_like(windowed),
            static_flat: Tensor::zeros_like(static_flat),
            actions: Tensor::zeros_like(actions),
            old_log_probs: Tensor::zeros_like(old_log_probs),
            old_action_alpha: Tensor::zeros_like(old_action_alpha),
            old_action_beta: Tensor::zeros_like(old_action_beta),
            advantages: Tensor::zeros_like(advantages),
            returns: Tensor::zeros_like(returns),
            approx_kl: None,
            action_loss: None,
            value_loss: None,
            reverse_kl_loss: None,
            dist_entropy: None,
            clipped: None,
        }
    }

    pub(crate) fn matches(
        &self,
        windowed: &Tensor,
        static_flat: &Tensor,
        actions: &Tensor,
        old_log_probs: &Tensor,
        old_action_alpha: &Tensor,
        old_action_beta: &Tensor,
        advantages: &Tensor,
        returns: &Tensor,
        sample_count: i64,
    ) -> bool {
        self.sample_count == sample_count
            && self.windowed.size() == windowed.size()
            && self.static_flat.size() == static_flat.size()
            && self.actions.size() == actions.size()
            && self.old_log_probs.size() == old_log_probs.size()
            && self.old_action_alpha.size() == old_action_alpha.size()
            && self.old_action_beta.size() == old_action_beta.size()
            && self.advantages.size() == advantages.size()
            && self.returns.size() == returns.size()
    }

    fn copy_inputs(
        &mut self,
        windowed: &Tensor,
        static_flat: &Tensor,
        actions: &Tensor,
        old_log_probs: &Tensor,
        old_action_alpha: &Tensor,
        old_action_beta: &Tensor,
        advantages: &Tensor,
        returns: &Tensor,
    ) {
        self.windowed.copy_(windowed);
        self.static_flat.copy_(static_flat);
        self.actions.copy_(actions);
        self.old_log_probs.copy_(old_log_probs);
        self.old_action_alpha.copy_(old_action_alpha);
        self.old_action_beta.copy_(old_action_beta);
        self.advantages.copy_(advantages);
        self.returns.copy_(returns);
    }

    pub(crate) fn capture_or_replay(
        &mut self,
        model: &TradingModel,
        hl_gauss: &HlGaussBins,
        trainable_vars: &[Tensor],
        device: Device,
        windowed: &Tensor,
        static_flat: &Tensor,
        actions: &Tensor,
        old_log_probs: &Tensor,
        old_action_alpha: &Tensor,
        old_action_beta: &Tensor,
        advantages: &Tensor,
        returns: &Tensor,
    ) -> PmpoMinibatchMetrics {
        self.copy_inputs(
            windowed,
            static_flat,
            actions,
            old_log_probs,
            old_action_alpha,
            old_action_beta,
            advantages,
            returns,
        );

        if !self.captured {
            let warmup_outputs = compute_pmpo_minibatch_outputs(
                model,
                hl_gauss,
                &self.windowed,
                &self.static_flat,
                self.sample_count,
                &self.actions,
                &self.old_log_probs,
                &self.old_action_alpha,
                &self.old_action_beta,
                &self.advantages,
                &self.returns,
            );
            warmup_outputs.total_loss.backward();
            drop(warmup_outputs);
            zero_existing_grads(trainable_vars);
            synchronize_device(device);
            empty_cuda_cache();

            self.graph.capture_begin();
            let outputs = compute_pmpo_minibatch_outputs(
                model,
                hl_gauss,
                &self.windowed,
                &self.static_flat,
                self.sample_count,
                &self.actions,
                &self.old_log_probs,
                &self.old_action_alpha,
                &self.old_action_beta,
                &self.advantages,
                &self.returns,
            );
            outputs.total_loss.backward();
            self.graph.capture_end();
            self.approx_kl = Some(outputs.metrics.approx_kl);
            self.action_loss = Some(outputs.metrics.action_loss);
            self.value_loss = Some(outputs.metrics.value_loss);
            self.reverse_kl_loss = Some(outputs.metrics.reverse_kl_loss);
            self.dist_entropy = Some(outputs.metrics.dist_entropy);
            self.clipped = Some(outputs.metrics.clipped);
            self.captured = true;
            self.graph.replay();
        } else {
            self.graph.replay();
        }

        self.metrics()
    }

    fn metrics(&self) -> PmpoMinibatchMetrics {
        PmpoMinibatchMetrics {
            approx_kl: self.approx_kl.as_ref().unwrap().shallow_clone(),
            action_loss: self.action_loss.as_ref().unwrap().shallow_clone(),
            value_loss: self.value_loss.as_ref().unwrap().shallow_clone(),
            reverse_kl_loss: self.reverse_kl_loss.as_ref().unwrap().shallow_clone(),
            dist_entropy: self.dist_entropy.as_ref().unwrap().shallow_clone(),
            clipped: self.clipped.as_ref().unwrap().shallow_clone(),
        }
    }
}
