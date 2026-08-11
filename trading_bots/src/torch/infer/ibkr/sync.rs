use std::io;

use ibapi::{
    accounts::{types::AccountId, AccountUpdate},
    contracts::SecurityType,
    orders::Orders,
    Client,
};

use crate::types::{Account, Position};

use super::state::MAX_ACCOUNT_VALUE;

const POSITION_EPSILON: f64 = 1e-8;
const VALUE_EPSILON: f64 = 1e-6;

#[derive(Debug)]
struct AccountSnapshot {
    account: Account,
    net_liquidation: f64,
    foreign_positions: Vec<String>,
}

fn invalid_account(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn validate_dedicated_snapshot(snapshot: &AccountSnapshot, startup: bool) -> Result<(), io::Error> {
    let account = &snapshot.account;
    if !account.cash.is_finite() || account.cash < 0.0 {
        return Err(invalid_account(
            "dedicated IBKR account cash must be finite and non-negative",
        ));
    }
    if !snapshot.net_liquidation.is_finite() || snapshot.net_liquidation <= 0.0 {
        return Err(invalid_account(
            "dedicated IBKR account net liquidation must be finite and positive",
        ));
    }
    if !snapshot.foreign_positions.is_empty() {
        return Err(invalid_account(format!(
            "dedicated IBKR account contains unmanaged positions: {}",
            snapshot.foreign_positions.join(", ")
        )));
    }
    if account
        .positions
        .iter()
        .any(|position| !position.quantity.is_finite() || position.quantity < -POSITION_EPSILON)
    {
        return Err(invalid_account(
            "dedicated IBKR account contains a short or invalid selected-symbol position",
        ));
    }
    if startup
        && account
            .positions
            .iter()
            .any(|position| position.quantity.abs() > POSITION_EPSILON)
    {
        return Err(invalid_account(
            "dedicated IBKR account must be cash-only at startup; resume/adoption of existing positions is not supported",
        ));
    }
    if startup {
        if let Some(max_value) = MAX_ACCOUNT_VALUE {
            let tolerance = VALUE_EPSILON * max_value.max(1.0);
            if account.cash > max_value + tolerance
                || snapshot.net_liquidation > max_value + tolerance
            {
                return Err(invalid_account(format!(
                "dedicated IBKR account cash and net liquidation must not exceed ${max_value:.2}"
            )));
            }
        }
    }
    if startup {
        let tolerance = VALUE_EPSILON * snapshot.net_liquidation.max(1.0);
        if (account.cash - snapshot.net_liquidation).abs() > tolerance {
            return Err(invalid_account(
                "dedicated IBKR account must be cash-only at startup",
            ));
        }
    }
    Ok(())
}

fn read_account_snapshot(
    client: &Client,
    account_id: &AccountId,
    symbols: &[String],
) -> Result<AccountSnapshot, Box<dyn std::error::Error>> {
    let subscription = client.account_updates(account_id)?;
    let mut cash = None;
    let mut net_liquidation = None;
    let mut positions = vec![Position::default(); symbols.len()];
    let mut seen_selected_contract = vec![false; symbols.len()];
    let mut foreign_positions = Vec::new();
    let mut received_end = false;

    for update in &subscription {
        match update {
            AccountUpdate::AccountValue(value) if value.currency == "BASE" => {
                let target = match value.key.as_str() {
                    "TotalCashValue" => &mut cash,
                    "NetLiquidation" => &mut net_liquidation,
                    _ => continue,
                };
                *target = Some(value.value.parse::<f64>().map_err(|error| {
                    invalid_account(format!(
                        "could not parse IBKR {} value {:?}: {error}",
                        value.key, value.value
                    ))
                })?);
            }
            AccountUpdate::PortfolioValue(portfolio)
                if portfolio.position.abs() > POSITION_EPSILON =>
            {
                if portfolio
                    .account
                    .as_deref()
                    .is_some_and(|id| id != account_id.0)
                {
                    return Err(invalid_account(format!(
                        "received portfolio data for unexpected account {:?}",
                        portfolio.account
                    ))
                    .into());
                }
                let symbol = portfolio.contract.symbol.as_str();
                if portfolio.contract.security_type == SecurityType::Stock {
                    if let Some(ticker_idx) =
                        symbols.iter().position(|candidate| candidate == symbol)
                    {
                        if seen_selected_contract[ticker_idx] {
                            foreign_positions
                                .push(format!("ambiguous duplicate STK contract for {symbol}"));
                            continue;
                        }
                        seen_selected_contract[ticker_idx] = true;
                        positions[ticker_idx].quantity = portfolio.position;
                        positions[ticker_idx].avg_price = portfolio.average_cost;
                        continue;
                    }
                }
                foreign_positions.push(format!(
                    "{} {} ({:.6})",
                    portfolio.contract.security_type, symbol, portfolio.position
                ));
            }
            AccountUpdate::End => {
                received_end = true;
                subscription.cancel();
                break;
            }
            AccountUpdate::AccountValue(_)
            | AccountUpdate::PortfolioValue(_)
            | AccountUpdate::UpdateTime(_) => {}
        }
    }

    if !received_end {
        let detail = subscription
            .error()
            .map(|error| error.to_string())
            .unwrap_or_else(|| "account snapshot ended before AccountUpdate::End".to_string());
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, detail).into());
    }

    let cash =
        cash.ok_or_else(|| invalid_account("IBKR account snapshot omitted BASE TotalCashValue"))?;
    let net_liquidation = net_liquidation
        .ok_or_else(|| invalid_account("IBKR account snapshot omitted BASE NetLiquidation"))?;
    Ok(AccountSnapshot {
        account: Account {
            cash,
            total_assets: net_liquidation,
            positions,
        },
        net_liquidation,
        foreign_positions,
    })
}

pub(super) fn initialize_dedicated_account(
    client: &Client,
    selected_account: &str,
    symbols: &[String],
) -> Result<(AccountId, Account), Box<dyn std::error::Error>> {
    let managed_accounts: Vec<String> = client
        .managed_accounts()?
        .into_iter()
        .filter(|account| !account.trim().is_empty())
        .collect();
    if !managed_accounts
        .iter()
        .any(|account| account == selected_account)
    {
        return Err(invalid_account(format!(
            "selected IBKR account {selected_account:?} is not available to this API client"
        ))
        .into());
    }

    let account_id = AccountId(selected_account.to_string());
    let open_orders = client.all_open_orders()?;
    let mut live_order_ids = Vec::new();
    let mut order_notices = Vec::new();
    for update in &open_orders {
        match update {
            Orders::OrderData(data) if data.order.account == selected_account => {
                live_order_ids.push(data.order_id);
            }
            Orders::Notice(notice) => order_notices.push(notice.to_string()),
            Orders::OrderData(_) | Orders::OrderStatus(_) => {}
        }
    }
    if !order_notices.is_empty() {
        return Err(invalid_account(format!(
            "could not safely reconcile startup open orders: {}",
            order_notices.join(" | ")
        ))
        .into());
    }
    if !live_order_ids.is_empty() {
        return Err(invalid_account(format!(
            "dedicated IBKR account has live orders {:?}; cancel them before starting",
            live_order_ids
        ))
        .into());
    }
    let snapshot = read_account_snapshot(client, &account_id, symbols)?;
    validate_dedicated_snapshot(&snapshot, true)?;
    Ok((account_id, snapshot.account))
}

pub(super) fn sync_account_from_ibkr(
    client: &Client,
    account_id: &AccountId,
    symbols: &[String],
    account: &mut Account,
) -> Result<(), Box<dyn std::error::Error>> {
    let snapshot = read_account_snapshot(client, account_id, symbols)?;
    validate_dedicated_snapshot(&snapshot, false)?;
    *account = snapshot.account;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{validate_dedicated_snapshot, AccountSnapshot};
    use crate::types::{Account, Position};

    fn snapshot(cash: f64, net_liquidation: f64, quantities: &[f64]) -> AccountSnapshot {
        AccountSnapshot {
            account: Account {
                cash,
                total_assets: net_liquidation,
                positions: quantities
                    .iter()
                    .map(|&quantity| Position {
                        quantity,
                        avg_price: 0.0,
                    })
                    .collect(),
            },
            net_liquidation,
            foreign_positions: Vec::new(),
        }
    }

    #[test]
    fn startup_requires_cash_only_account_within_cap() {
        assert!(validate_dedicated_snapshot(&snapshot(10_000.0, 10_000.0, &[0.0]), true).is_ok());
        assert!(validate_dedicated_snapshot(&snapshot(10_000.0, 10_000.0, &[1.0]), true).is_err());
        assert!(validate_dedicated_snapshot(&snapshot(10_001.0, 10_001.0, &[0.0]), true).is_err());
    }

    #[test]
    fn ongoing_sync_rejects_foreign_holdings_and_shorts() {
        let mut foreign = snapshot(5_000.0, 10_000.0, &[50.0]);
        foreign
            .foreign_positions
            .push("STK MANUAL (1.0)".to_string());
        assert!(validate_dedicated_snapshot(&foreign, false).is_err());
        assert!(validate_dedicated_snapshot(&snapshot(5_000.0, 5_000.0, &[-1.0]), false).is_err());
    }
}
