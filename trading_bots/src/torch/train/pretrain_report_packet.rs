//! Typed device-to-host wire contract for optimizer-step reporting.
//!
//! The packet remains one contiguous tensor and one host transfer. Positional offsets are owned
//! here, while callers consume named fields; adding a segment can no longer silently shift a raw
//! index in the training loop.

use anyhow::{ensure, Result};
use tch::{Kind, Tensor};

use super::growth;
use super::smd_idbd::SMD_METRIC_COUNT;
use crate::torch::bar_dist::BAR_DOF;
use crate::torch::direct_return::DIRECT_RETURN_COUNT;
use crate::torch::optim::muon::ROW_LR_METRIC_COUNT;

const TOTAL: usize = 0;
const NLL: usize = TOTAL + 1;
const NLL_DOF_START: usize = NLL + 1;
const NLL_DOF_END: usize = NLL_DOF_START + BAR_DOF;
const DIRECT_NLL_START: usize = NLL_DOF_END;
const DIRECT_NLL_END: usize = DIRECT_NLL_START + DIRECT_RETURN_COUNT;
const DIRECT_VALID_START: usize = DIRECT_NLL_END;
const DIRECT_VALID_END: usize = DIRECT_VALID_START + DIRECT_RETURN_COUNT;
const DYN: usize = DIRECT_VALID_END;
const KL: usize = DYN + 1;
const GROWTH_DIAGNOSTIC: usize = KL + 1;
const GROWTH_STATS_START: usize = GROWTH_DIAGNOSTIC + 1;
const GROWTH_STATS_END: usize = GROWTH_STATS_START + growth::GROWTH_STAT_COUNT;
const IDENTITY: usize = GROWTH_STATS_END;
const AUTOCORR: usize = IDENTITY + 1;
const GRAD_NORM: usize = AUTOCORR + 1;
const BASE_END: usize = GRAD_NORM + 1;
const ROW_LR_START: usize = BASE_END;
const ROW_LR_END: usize = ROW_LR_START + ROW_LR_METRIC_COUNT;
const SMD_START: usize = ROW_LR_END;
const SMD_END: usize = SMD_START + SMD_METRIC_COUNT;
const DIRECT_OBJECTIVE_NLL: usize = SMD_END;
const JOINT_CATEGORICAL_CE: usize = DIRECT_OBJECTIVE_NLL + 1;
const PACKET_LEN: usize = JOINT_CATEGORICAL_CE + 1;

#[derive(Clone, Debug)]
pub(super) struct PackedStepMetrics([f64; PACKET_LEN]);

impl PackedStepMetrics {
    pub(super) const LEN: usize = PACKET_LEN;

    /// Read the fixed direct-objective packet after the single device-to-host transfer.
    pub(super) fn read(packed: &Tensor) -> Self {
        let values = Vec::<f64>::try_from(packed.to_kind(Kind::Double).reshape([-1]))
            .expect("packed step metrics are convertible");
        assert_eq!(
            values.len(),
            Self::LEN,
            "packed step metrics carry {} entries, expected {}",
            values.len(),
            Self::LEN,
        );
        let mut packet = [f64::NAN; PACKET_LEN];
        packet[..values.len()].copy_from_slice(&values);
        Self(packet)
    }

    pub(super) fn total(&self) -> f64 {
        self.0[TOTAL]
    }

    pub(super) fn canonical_nll(&self) -> f64 {
        self.0[NLL]
    }

    pub(super) fn nll_dof(&self) -> [f64; BAR_DOF] {
        self.0[NLL_DOF_START..NLL_DOF_END]
            .try_into()
            .expect("NLL DOF packet segment has fixed width")
    }

    pub(super) fn direct_nll_horizon(&self) -> [f64; DIRECT_RETURN_COUNT] {
        self.0[DIRECT_NLL_START..DIRECT_NLL_END]
            .try_into()
            .expect("direct-return NLL packet segment has fixed width")
    }

    pub(super) fn direct_valid_horizon(&self) -> [f64; DIRECT_RETURN_COUNT] {
        self.0[DIRECT_VALID_START..DIRECT_VALID_END]
            .try_into()
            .expect("direct-return valid-count packet segment has fixed width")
    }

    pub(super) fn dyn_loss(&self) -> f64 {
        self.0[DYN]
    }

    pub(super) fn kl_loss(&self) -> f64 {
        self.0[KL]
    }

    pub(super) fn growth_diagnostic(&self) -> f64 {
        self.0[GROWTH_DIAGNOSTIC]
    }

    pub(super) fn growth_stats(&self) -> growth::GrowthStats {
        let [mean_abs_f, clamp_bind, min_log_argument] = self.0
            [GROWTH_STATS_START..GROWTH_STATS_END]
            .try_into()
            .expect("growth-stat packet segment has fixed width");
        growth::GrowthStats {
            mean_abs_f,
            clamp_bind,
            min_log_argument,
        }
    }

    pub(super) fn identity(&self) -> f64 {
        self.0[IDENTITY]
    }

    pub(super) fn autocorr(&self) -> f64 {
        self.0[AUTOCORR]
    }

    pub(super) fn grad_norm(&self) -> f64 {
        self.0[GRAD_NORM]
    }

    pub(super) fn row_learned_lr(&self) -> [f64; ROW_LR_METRIC_COUNT] {
        self.0[ROW_LR_START..ROW_LR_END]
            .try_into()
            .expect("controller metric packet segment has fixed width")
    }

    pub(super) fn smd_idbd(&self) -> [f64; SMD_METRIC_COUNT] {
        self.0[SMD_START..SMD_END]
            .try_into()
            .expect("SMD metric packet segment has fixed width")
    }

    pub(super) fn direct_return_nll_mean(&self) -> Option<f64> {
        self.0[DIRECT_OBJECTIVE_NLL]
            .is_finite()
            .then_some(self.0[DIRECT_OBJECTIVE_NLL])
    }

    pub(super) fn joint_categorical_ce(&self) -> f64 {
        self.0[JOINT_CATEGORICAL_CE]
    }

    /// Fail before optimizer mutation if a required segment is non-finite or an optional
    /// segment is only partially populated.
    pub(super) fn ensure_finite(&self, step: usize) -> Result<()> {
        let total = self.total();
        ensure!(
            total.is_finite(),
            "loss is not finite at step {step}: {total}"
        );
        let grad_norm = self.grad_norm();
        ensure!(
            grad_norm.is_finite(),
            "gradient norm is not finite at step {step}: {grad_norm}"
        );
        ensure!(
            self.0[..BASE_END].iter().all(|value| value.is_finite()),
            "packed loss/gradient diagnostics are not finite at step {step}: {:?}",
            self.0
        );
        let controller = &self.0[ROW_LR_START..ROW_LR_END];
        ensure!(
            controller.iter().all(|value| value.is_nan())
                || controller.iter().all(|value| value.is_finite()),
            "row learned-LR diagnostics are partially non-finite at step {step}: {controller:?}"
        );
        let smd = &self.0[SMD_START..SMD_END];
        ensure!(
            smd.iter().all(|value| value.is_nan()) || smd.iter().all(|value| value.is_finite()),
            "SMD-IDBD diagnostics are partially non-finite at step {step}: {smd:?}"
        );
        let direct_objective = self.0[DIRECT_OBJECTIVE_NLL];
        ensure!(
            direct_objective.is_finite(),
            "direct-return likelihood is not finite at step {step}: {direct_objective}"
        );
        let joint_categorical_ce = self.joint_categorical_ce();
        ensure!(
            joint_categorical_ce.is_finite(),
            "joint categorical auxiliary is not finite at step {step}: {joint_categorical_ce}"
        );
        Ok(())
    }
}
