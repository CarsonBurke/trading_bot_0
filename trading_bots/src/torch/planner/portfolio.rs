use crate::torch::constants::{ACTION_THRESHOLD, COMMISSION_RATE};

pub const PLANNER_REWARD_SCALE: f64 = 20.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlannerPortfolio {
    pub cash: f64,
    pub shares: f64,
    pub previous_target_weight: f64,
    pub total_commissions: f64,
    pub previous_turnover: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PortfolioStep {
    pub reward: f64,
    pub assets_before: f64,
    pub assets_after_trade: f64,
    pub assets_after: f64,
    pub commission: f64,
    pub traded_notional: f64,
    pub turnover: f64,
    pub requested_target_weight: f64,
    pub stock_weight_after_trade: f64,
    pub stock_weight_after: f64,
    pub fill_ratio: f64,
}

impl PlannerPortfolio {
    pub fn new(cash: f64) -> Self {
        assert!(
            cash.is_finite() && cash > 0.0,
            "starting cash must be positive and finite"
        );
        Self {
            cash,
            shares: 0.0,
            previous_target_weight: 0.0,
            total_commissions: 0.0,
            previous_turnover: 0.0,
        }
    }

    pub fn from_position(cash: f64, shares: f64, previous_target_weight: f64) -> Self {
        assert!(
            cash.is_finite() && cash >= 0.0,
            "cash must be finite and non-negative"
        );
        assert!(
            shares.is_finite() && shares >= 0.0,
            "shares must be finite and non-negative"
        );
        Self {
            cash,
            shares,
            previous_target_weight: sanitize_target(previous_target_weight),
            total_commissions: 0.0,
            previous_turnover: 0.0,
        }
    }

    pub fn total_assets(&self, price: f64) -> f64 {
        assert_valid_price(price);
        self.cash + self.shares * price
    }

    pub fn stock_weight(&self, price: f64) -> f64 {
        let assets = self.total_assets(price);
        if assets > 0.0 {
            (self.shares * price / assets).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    pub fn planner_state(&self, price: f64) -> [f32; 4] {
        let stock_weight = self.stock_weight(price);
        [
            stock_weight as f32,
            (1.0 - stock_weight) as f32,
            self.previous_target_weight as f32,
            self.previous_turnover as f32,
        ]
    }

    pub fn step(
        &mut self,
        target_weight: f64,
        current_price: f64,
        next_price: f64,
    ) -> PortfolioStep {
        assert_valid_price(current_price);
        assert_valid_price(next_price);

        let target_weight = sanitize_target(target_weight);
        let assets_before = self.total_assets(current_price);
        assert!(assets_before > 0.0, "cannot trade an exhausted portfolio");

        let current_value = self.shares * current_price;
        let target_value = target_weight * assets_before;
        let delta_value = target_value - current_value;
        let min_trade_notional = ACTION_THRESHOLD * target_value.max(current_value);

        let mut commission = 0.0;
        let mut traded_notional = 0.0;
        let mut fill_ratio = 1.0;

        if delta_value.abs() >= min_trade_notional {
            if delta_value < 0.0 {
                let sell_value = (-delta_value).min(current_value);
                if sell_value > 0.0 {
                    let quantity = sell_value / current_price;
                    commission = quantity * COMMISSION_RATE;
                    self.cash += sell_value - commission;
                    self.shares -= quantity;
                    traded_notional = sell_value;
                }
            } else if delta_value > 0.0 {
                let total_buy_demand =
                    delta_value + (delta_value / current_price) * COMMISSION_RATE;
                fill_ratio = (self.cash / total_buy_demand).min(1.0);
                let scaled_amount = delta_value * fill_ratio;
                let quantity = scaled_amount / current_price;
                let buy_commission = quantity * COMMISSION_RATE;
                let total_cost = scaled_amount + buy_commission;
                if total_cost <= self.cash && total_cost > 0.0 {
                    commission = buy_commission;
                    self.cash -= total_cost;
                    self.shares += quantity;
                    traded_notional = scaled_amount;
                }
            }
        }

        self.total_commissions += commission;
        let assets_after_trade = self.total_assets(current_price);
        let assets_after = self.total_assets(next_price);
        let turnover = traded_notional / assets_before;
        let reward = (assets_after / assets_before).ln() * PLANNER_REWARD_SCALE;
        let stock_weight_after_trade = self.stock_weight(current_price);
        let stock_weight_after = self.stock_weight(next_price);
        self.previous_target_weight = target_weight;
        self.previous_turnover = turnover;

        PortfolioStep {
            reward,
            assets_before,
            assets_after_trade,
            assets_after,
            commission,
            traded_notional,
            turnover,
            requested_target_weight: target_weight,
            stock_weight_after_trade,
            stock_weight_after,
            fill_ratio,
        }
    }
}

fn assert_valid_price(price: f64) {
    assert!(
        price.is_finite() && price > 0.0,
        "price must be positive and finite"
    );
}

fn sanitize_target(target_weight: f64) -> f64 {
    if target_weight.is_finite() {
        target_weight.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn full_allocation_matches_env_fill_and_commission_semantics() {
        let mut portfolio = PlannerPortfolio::new(100.0);
        let result = portfolio.step(1.0, 10.0, 11.0);

        let desired_notional = 100.0;
        let fill = 100.0 / (desired_notional + desired_notional / 10.0 * COMMISSION_RATE);
        let bought = desired_notional * fill;
        let shares = bought / 10.0;
        let commission = shares * COMMISSION_RATE;
        let next_assets = shares * 11.0;

        approx_eq(result.fill_ratio, fill, 1e-12);
        approx_eq(portfolio.shares, shares, 1e-12);
        approx_eq(result.commission, commission, 1e-12);
        approx_eq(result.assets_after, next_assets, 1e-10);
        approx_eq(result.reward, 20.0 * (next_assets / 100.0).ln(), 1e-12);
    }

    #[test]
    fn sell_rebalances_before_next_price_and_charges_per_share() {
        let mut portfolio = PlannerPortfolio::from_position(0.0, 10.0, 1.0);
        let result = portfolio.step(0.5, 10.0, 8.0);

        let expected_commission = 5.0 * COMMISSION_RATE;
        approx_eq(portfolio.shares, 5.0, 1e-12);
        approx_eq(portfolio.cash, 50.0 - expected_commission, 1e-12);
        approx_eq(
            result.assets_after_trade,
            100.0 - expected_commission,
            1e-12,
        );
        approx_eq(result.assets_after, 90.0 - expected_commission, 1e-12);
        approx_eq(result.turnover, 0.5, 1e-12);
    }

    #[test]
    fn partial_sell_preserves_sub_epsilon_residual_shares_like_env() {
        let mut portfolio = PlannerPortfolio::from_position(0.0, 1e-8, 1.0);
        portfolio.step(0.5, 1.0, 1.0);
        approx_eq(portfolio.shares, 5e-9, 1e-20);
    }

    #[test]
    fn target_threshold_skips_small_rebalance_but_updates_requested_target() {
        let mut portfolio = PlannerPortfolio::from_position(50.0, 5.0, 0.5);
        let target = 0.5004;
        let result = portfolio.step(target, 10.0, 10.0);

        assert_eq!(result.traded_notional, 0.0);
        assert_eq!(result.commission, 0.0);
        assert_eq!(portfolio.previous_target_weight, target);
        assert_eq!(
            portfolio.planner_state(10.0),
            [0.5, 0.5, target as f32, 0.0]
        );
    }

    #[test]
    fn non_finite_action_maps_to_cash() {
        let mut portfolio = PlannerPortfolio::from_position(0.0, 10.0, 0.8);
        let result = portfolio.step(f64::NAN, 10.0, 10.0);
        assert_eq!(result.requested_target_weight, 0.0);
        assert_eq!(portfolio.shares, 0.0);
    }
}
