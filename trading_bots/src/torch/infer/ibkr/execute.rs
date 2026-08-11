use std::io;
use std::sync::Arc;
use std::time::Duration;

use ibapi::accounts::types::AccountId;
use ibapi::contracts::Contract;
use ibapi::orders::{order_builder, Action, CancelOrder, Orders, PlaceOrder};
use ibapi::Client;

use crate::torch::constants::{ACTION_THRESHOLD, COMMISSION_RATE};
use crate::types::Account;

use super::sync::sync_account_from_ibkr;

const MIN_ORDER_NOTIONAL: f64 = 1.0;
const MIN_ORDER_QUANTITY: f64 = 1e-8;
const ORDER_EVENT_TIMEOUT: Duration = Duration::from_secs(30);
const CANCEL_EVENT_TIMEOUT: Duration = Duration::from_secs(10);
const CANCEL_RETRY_DELAY: Duration = Duration::from_secs(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TradeSide {
    Buy,
    Sell,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OrderProgress {
    Active,
    Filled,
    TerminalFailure,
}

#[derive(Clone, Debug, PartialEq)]
struct PlannedTrade {
    ticker_idx: usize,
    side: TradeSide,
    quantity: f64,
    reference_price: f64,
}

impl PlannedTrade {
    fn notional(&self) -> f64 {
        self.quantity * self.reference_price
    }
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn classify_order_progress(status: &str, filled: f64, quantity: f64) -> OrderProgress {
    if status == "Filled" || filled + MIN_ORDER_QUANTITY >= quantity {
        return OrderProgress::Filled;
    }
    if matches!(
        status,
        "Cancelled" | "Canceled" | "ApiCancelled" | "ApiCanceled" | "Inactive" | "Rejected"
    ) {
        return OrderProgress::TerminalFailure;
    }
    OrderProgress::Active
}

fn record_notice(notices: &mut Vec<String>, notice: String) {
    println!("IBKR order diagnostic: {notice}");
    if !notices.contains(&notice) {
        notices.push(notice);
    }
}

fn notice_suffix(notices: &[String]) -> String {
    if notices.is_empty() {
        String::new()
    } else {
        format!("; notices: {}", notices.join(" | "))
    }
}

fn order_is_open(
    client: &Client,
    account_id: &AccountId,
    order_id: i32,
) -> Result<bool, Box<dyn std::error::Error>> {
    let subscription = client.all_open_orders()?;
    let mut found = false;
    let mut notices = Vec::new();
    for update in &subscription {
        match update {
            Orders::OrderData(data)
                if data.order_id == order_id && data.order.account == account_id.0 =>
            {
                found = true;
            }
            Orders::Notice(notice) => notices.push(notice.to_string()),
            Orders::OrderData(_) | Orders::OrderStatus(_) => {}
        }
    }
    if notices.is_empty() {
        Ok(found)
    } else {
        Err(io::Error::other(format!(
            "open-order reconciliation returned notices: {}",
            notices.join(" | ")
        ))
        .into())
    }
}

fn plan_target_weight_trades(
    target_weights: &[f64],
    current_prices: &[f64],
    account: &Account,
) -> Result<Vec<PlannedTrade>, io::Error> {
    let ticker_count = target_weights.len();
    if current_prices.len() != ticker_count || account.positions.len() != ticker_count {
        return Err(invalid_input(format!(
            "target weights, prices, and positions must have the same length ({} weights, {} prices, {} positions)",
            ticker_count,
            current_prices.len(),
            account.positions.len()
        )));
    }
    if !account.total_assets.is_finite() || account.total_assets <= 0.0 {
        return Err(invalid_input(
            "account total assets must be finite and positive",
        ));
    }
    if !account.cash.is_finite() || account.cash < 0.0 {
        return Err(invalid_input(
            "account cash must be finite and non-negative",
        ));
    }

    let mut weight_sum = 0.0;
    for (ticker_idx, ((&weight, &price), position)) in target_weights
        .iter()
        .zip(current_prices)
        .zip(&account.positions)
        .enumerate()
    {
        if !weight.is_finite() || !(0.0..=1.0).contains(&weight) {
            return Err(invalid_input(format!(
                "target weight for ticker {ticker_idx} must be finite and within [0, 1]"
            )));
        }
        if !price.is_finite() || price <= 0.0 {
            return Err(invalid_input(format!(
                "price for ticker {ticker_idx} must be finite and positive"
            )));
        }
        if !position.quantity.is_finite() || position.quantity < 0.0 {
            return Err(invalid_input(format!(
                "position quantity for ticker {ticker_idx} must be finite and non-negative"
            )));
        }
        weight_sum += weight;
    }
    if weight_sum > 1.0 + f64::EPSILON * ticker_count.max(1) as f64 {
        return Err(invalid_input(format!(
            "target ticker weights sum to {weight_sum:.6}, exceeding the fully-invested limit"
        )));
    }

    let mut sells = Vec::new();
    let mut buy_candidates = Vec::new();
    for (ticker_idx, ((&target_weight, &price), position)) in target_weights
        .iter()
        .zip(current_prices)
        .zip(&account.positions)
        .enumerate()
    {
        let current_value = position.quantity * price;
        let target_value = target_weight * account.total_assets;
        let delta_value = target_value - current_value;
        let rebalance_threshold =
            (ACTION_THRESHOLD * target_value.max(current_value)).max(MIN_ORDER_NOTIONAL);
        if delta_value.abs() < rebalance_threshold {
            continue;
        }

        if delta_value < 0.0 {
            let quantity = (-delta_value / price).min(position.quantity);
            if quantity >= MIN_ORDER_QUANTITY && quantity * price >= MIN_ORDER_NOTIONAL {
                sells.push(PlannedTrade {
                    ticker_idx,
                    side: TradeSide::Sell,
                    quantity,
                    reference_price: price,
                });
            }
        } else {
            let quantity = delta_value / price;
            if quantity >= MIN_ORDER_QUANTITY && quantity * price >= MIN_ORDER_NOTIONAL {
                buy_candidates.push(PlannedTrade {
                    ticker_idx,
                    side: TradeSide::Buy,
                    quantity,
                    reference_price: price,
                });
            }
        }
    }

    let total_buy_cost: f64 = buy_candidates
        .iter()
        .map(|trade| trade.notional() + trade.quantity * COMMISSION_RATE)
        .sum();
    let fill_ratio = if total_buy_cost > account.cash && total_buy_cost > 0.0 {
        account.cash / total_buy_cost
    } else {
        1.0
    };
    let mut available_cash = account.cash;
    let mut buys = Vec::with_capacity(buy_candidates.len());
    for mut trade in buy_candidates {
        let max_affordable_quantity = available_cash / (trade.reference_price + COMMISSION_RATE);
        trade.quantity = (trade.quantity * fill_ratio).min(max_affordable_quantity);
        let cost = trade.notional() + trade.quantity * COMMISSION_RATE;
        if trade.quantity < MIN_ORDER_QUANTITY
            || trade.notional() < MIN_ORDER_NOTIONAL
            || cost > available_cash + f64::EPSILON * available_cash.max(1.0)
        {
            continue;
        }
        available_cash -= cost;
        buys.push(trade);
    }

    sells.extend(buys);
    Ok(sells)
}

fn plan_trade_phase(
    target_weights: &[f64],
    current_prices: &[f64],
    account: &Account,
    side: TradeSide,
) -> Result<Vec<PlannedTrade>, io::Error> {
    Ok(
        plan_target_weight_trades(target_weights, current_prices, account)?
            .into_iter()
            .filter(|trade| trade.side == side)
            .collect(),
    )
}

fn revalue_account(account: &mut Account, current_prices: &[f64]) -> Result<(), io::Error> {
    if current_prices.len() != account.positions.len() {
        return Err(invalid_input(format!(
            "cannot revalue {} positions with {} prices",
            account.positions.len(),
            current_prices.len()
        )));
    }

    let mut total_assets = account.cash;
    for (ticker_idx, (&price, position)) in
        current_prices.iter().zip(&account.positions).enumerate()
    {
        if !price.is_finite() || price <= 0.0 {
            return Err(invalid_input(format!(
                "price for ticker {ticker_idx} must be finite and positive"
            )));
        }
        if !position.quantity.is_finite() || position.quantity < 0.0 {
            return Err(invalid_input(format!(
                "position quantity for ticker {ticker_idx} must be finite and non-negative"
            )));
        }
        total_assets += position.quantity * price;
    }
    if !total_assets.is_finite() || total_assets <= 0.0 {
        return Err(invalid_input(
            "revalued account total assets must be finite and positive",
        ));
    }
    account.total_assets = total_assets;
    Ok(())
}

fn submit_and_wait(
    client: &Client,
    account_id: &AccountId,
    symbol: &str,
    trade: &PlannedTrade,
) -> Result<(), Box<dyn std::error::Error>> {
    let action = match trade.side {
        TradeSide::Buy => Action::Buy,
        TradeSide::Sell => Action::Sell,
    };
    let contract = Contract::stock(symbol).build();
    let order_id = client.next_order_id();
    let mut order = order_builder::market_order(action, trade.quantity);
    order.account = account_id.0.clone();
    let subscription = match client.place_order(order_id, &contract, &order) {
        Ok(subscription) => subscription,
        Err(error) => {
            return cancel_and_confirm_order(
                client,
                account_id,
                order_id,
                symbol,
                trade.quantity,
                format!("place-order transport failed ambiguously: {error}"),
                vec![error.to_string()],
            );
        }
    };
    let mut notices = Vec::new();

    println!(
        "Submitted {action} order {order_id}: {:.6} shares of {symbol} (reference notional ${:.2})",
        trade.quantity,
        trade.notional()
    );

    loop {
        let Some(event) = subscription.next_timeout(ORDER_EVENT_TIMEOUT) else {
            let reason = subscription
                .error()
                .map(|error| format!("order event stream failed: {error}"))
                .unwrap_or_else(|| "timed out waiting for a terminal order event".to_string());
            return cancel_and_confirm_order(
                client,
                account_id,
                order_id,
                symbol,
                trade.quantity,
                reason,
                notices,
            );
        };

        match event {
            PlaceOrder::OrderStatus(status) => {
                match classify_order_progress(&status.status, status.filled, trade.quantity) {
                    OrderProgress::Filled => return Ok(()),
                    OrderProgress::TerminalFailure => {
                        return Err(io::Error::other(format!(
                        "IBKR order {order_id} for {symbol} ended with status {} after filling {:.4} of {:.4} shares{}",
                        status.status, status.filled, trade.quantity, notice_suffix(&notices)
                    ))
                    .into());
                    }
                    OrderProgress::Active => {}
                }
            }
            PlaceOrder::ExecutionData(execution)
                if execution.execution.cumulative_quantity + 1e-8 >= trade.quantity =>
            {
                return Ok(());
            }
            PlaceOrder::Message(notice) => {
                record_notice(&mut notices, notice.to_string());
            }
            PlaceOrder::OpenOrder(_)
            | PlaceOrder::ExecutionData(_)
            | PlaceOrder::CommissionReport(_) => {}
        }
    }
}

fn cancel_and_confirm_order(
    client: &Client,
    account_id: &AccountId,
    order_id: i32,
    symbol: &str,
    quantity: f64,
    abort_reason: String,
    mut notices: Vec<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    loop {
        let cancellation = match client.cancel_order(order_id, "") {
            Ok(cancellation) => cancellation,
            Err(error) => {
                record_notice(
                    &mut notices,
                    format!("could not submit cancellation request: {error}"),
                );
                match order_is_open(client, account_id, order_id) {
                    Ok(false) => {
                        return Err(io::Error::other(format!(
                            "IBKR order {order_id} for {symbol} aborted because {abort_reason}; confirmed absent from open orders{}",
                            notice_suffix(&notices)
                        ))
                        .into());
                    }
                    Ok(true) => std::thread::sleep(CANCEL_RETRY_DELAY),
                    Err(reconcile_error) => {
                        record_notice(
                            &mut notices,
                            format!("open-order reconciliation failed: {reconcile_error}"),
                        );
                        std::thread::sleep(CANCEL_RETRY_DELAY);
                    }
                }
                continue;
            }
        };

        loop {
            let Some(event) = cancellation.next_timeout(CANCEL_EVENT_TIMEOUT) else {
                let detail = cancellation
                    .error()
                    .map(|error| format!("cancellation event stream failed: {error}"))
                    .unwrap_or_else(|| "cancellation confirmation timed out".to_string());
                record_notice(&mut notices, detail);
                match order_is_open(client, account_id, order_id) {
                    Ok(false) => {
                        return Err(io::Error::other(format!(
                            "IBKR order {order_id} for {symbol} aborted because {abort_reason}; confirmed absent from open orders{}",
                            notice_suffix(&notices)
                        ))
                        .into());
                    }
                    Ok(true) => {}
                    Err(reconcile_error) => record_notice(
                        &mut notices,
                        format!("open-order reconciliation failed: {reconcile_error}"),
                    ),
                }
                break;
            };

            match event {
                CancelOrder::OrderStatus(status) => {
                    match classify_order_progress(&status.status, status.filled, quantity) {
                        OrderProgress::Filled => return Ok(()),
                        OrderProgress::TerminalFailure => {
                            return Err(io::Error::other(format!(
                            "IBKR order {order_id} for {symbol} aborted because {abort_reason}; terminal cancellation status {} after filling {:.4} of {quantity:.4} shares{}",
                            status.status,
                            status.filled,
                            notice_suffix(&notices)
                        ))
                        .into());
                        }
                        OrderProgress::Active => {}
                    }
                }
                CancelOrder::Notice(notice) => {
                    record_notice(&mut notices, notice.to_string());
                }
            }
        }
    }
}

pub(super) fn execute_trades(
    client: &Arc<Client>,
    account_id: &AccountId,
    symbols: &[String],
    actions: &[f64],
    current_prices: &[f64],
    account: &mut Account,
) -> Result<(), Box<dyn std::error::Error>> {
    if symbols.len() != actions.len() {
        return Err(invalid_input(format!(
            "received {} actions for {} paper-trading symbols",
            actions.len(),
            symbols.len()
        ))
        .into());
    }

    revalue_account(account, current_prices)?;
    let sells = plan_trade_phase(actions, current_prices, account, TradeSide::Sell)?;
    for trade in &sells {
        submit_and_wait(client, account_id, &symbols[trade.ticker_idx], trade)?;
    }

    if !sells.is_empty() {
        sync_account_from_ibkr(client, account_id, symbols, account)?;
        revalue_account(account, current_prices)?;
    }

    if super::state::MAX_ACCOUNT_VALUE.is_some_and(|max_value| account.total_assets > max_value) {
        println!(
            "Account value ${:.2} exceeds the configured cap; skipping risk-increasing buys",
            account.total_assets
        );
        return Ok(());
    }

    let buys = plan_trade_phase(actions, current_prices, account, TradeSide::Buy)?;
    for trade in &buys {
        submit_and_wait(client, account_id, &symbols[trade.ticker_idx], trade)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        classify_order_progress, plan_target_weight_trades, plan_trade_phase, revalue_account,
        OrderProgress, TradeSide, MIN_ORDER_NOTIONAL,
    };
    use crate::types::{Account, Position};

    fn account(cash: f64, total_assets: f64, quantities: &[f64]) -> Account {
        Account {
            cash,
            total_assets,
            positions: quantities
                .iter()
                .map(|&quantity| Position {
                    quantity,
                    avg_price: 1.0,
                })
                .collect(),
        }
    }

    #[test]
    fn planner_orders_sells_before_cash_limited_buys() {
        let account = account(500.0, 2_000.0, &[10.0, 0.0]);
        let trades = plan_target_weight_trades(&[0.25, 0.25], &[100.0, 100.0], &account)
            .expect("valid rebalance should plan");

        assert_eq!(trades.len(), 2);
        assert_eq!(trades[0].side, TradeSide::Sell);
        assert_eq!(trades[0].ticker_idx, 0);
        assert_eq!(trades[0].quantity, 5.0);
        assert_eq!(trades[1].side, TradeSide::Buy);
        assert_eq!(trades[1].ticker_idx, 1);
        assert!(trades[1].quantity > 4.99 && trades[1].quantity < 5.0);
        let buy_cost =
            trades[1].notional() + trades[1].quantity * crate::torch::constants::COMMISSION_RATE;
        assert!(buy_cost <= account.cash);
    }

    #[test]
    fn planner_rejects_invalid_market_inputs() {
        let account = account(1_000.0, 1_000.0, &[0.0]);
        assert!(plan_target_weight_trades(&[f64::NAN], &[100.0], &account).is_err());
        assert!(plan_target_weight_trades(&[0.5], &[0.0], &account).is_err());
        assert!(plan_target_weight_trades(&[1.1], &[100.0], &account).is_err());
    }

    #[test]
    fn planner_skips_subminimum_orders() {
        let account = account(10_000.0, 10_000.0, &[0.0]);
        let trades = plan_target_weight_trades(
            &[MIN_ORDER_NOTIONAL * 0.5 / account.total_assets],
            &[100.0],
            &account,
        )
        .expect("valid small target");
        assert!(trades.is_empty());
    }

    #[test]
    fn planner_preserves_fractional_sell_quantity() {
        let account = account(750.0, 1_000.0, &[2.5]);
        let trades = plan_target_weight_trades(&[0.2], &[100.0], &account)
            .expect("fractional sell should plan");

        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].side, TradeSide::Sell);
        assert!((trades[0].quantity - 0.5).abs() < 1e-12);
        assert!(trades[0].quantity <= account.positions[0].quantity);
    }

    #[test]
    fn planner_preserves_and_proportionally_scales_fractional_buys() {
        let account = account(75.0, 1_000.0, &[0.0, 0.0]);
        let trades = plan_target_weight_trades(&[0.05, 0.05], &[200.0, 200.0], &account)
            .expect("fractional buys should plan");

        assert_eq!(trades.len(), 2);
        assert!(trades.iter().all(|trade| trade.side == TradeSide::Buy));
        assert!((trades[0].quantity - trades[1].quantity).abs() < 1e-12);
        assert!(trades.iter().all(|trade| trade.quantity > 0.18));
        assert!(trades.iter().all(|trade| trade.quantity < 0.19));
        let total_cost: f64 = trades
            .iter()
            .map(|trade| {
                trade.notional() + trade.quantity * crate::torch::constants::COMMISSION_RATE
            })
            .sum();
        assert!(total_cost <= account.cash + 1e-10);
    }

    #[test]
    fn order_status_classification_requires_terminal_evidence() {
        assert_eq!(
            classify_order_progress("Submitted", 0.0, 1.0),
            OrderProgress::Active
        );
        assert_eq!(
            classify_order_progress("PendingCancel", 0.5, 1.0),
            OrderProgress::Active
        );
        assert_eq!(
            classify_order_progress("Filled", 1.0, 1.0),
            OrderProgress::Filled
        );
        assert_eq!(
            classify_order_progress("Cancelled", 0.5, 1.0),
            OrderProgress::TerminalFailure
        );
    }

    #[test]
    fn refreshed_sale_proceeds_fund_same_cycle_buy_phase() {
        let before_sale = account(0.0, 1_000.0, &[10.0, 0.0]);
        let sells = plan_trade_phase(&[0.0, 1.0], &[100.0, 100.0], &before_sale, TradeSide::Sell)
            .expect("sell phase should plan");
        assert_eq!(sells.len(), 1);
        assert_eq!(sells[0].quantity, 10.0);

        let mut after_sale = account(1_000.0, 1_000.0, &[0.0, 0.0]);
        revalue_account(&mut after_sale, &[100.0, 100.0]).unwrap();
        let buys = plan_trade_phase(&[0.0, 1.0], &[100.0, 100.0], &after_sale, TradeSide::Buy)
            .expect("buy phase should use refreshed proceeds");
        assert_eq!(buys.len(), 1);
        assert_eq!(buys[0].ticker_idx, 1);
        assert!(buys[0].quantity > 9.99);
    }
}
