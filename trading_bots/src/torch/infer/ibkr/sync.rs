use ibapi::{
    accounts::{types::AccountGroup, AccountSummaryResult, AccountSummaryTags, PositionUpdate},
    Client,
};

use crate::types::Account;

use super::state::MAX_ACCOUNT_VALUE;

pub(super) fn sync_account_from_ibkr(
    client: &Client,
    symbols: &[String],
    account: &mut Account,
) -> Result<(), Box<dyn std::error::Error>> {
    let account_subscription =
        client.account_summary(&AccountGroup::from("All"), AccountSummaryTags::ALL)?;

    for update in &account_subscription {
        match update {
            AccountSummaryResult::Summary(summary) => {
                if summary.tag == "TotalCashValue" {
                    account.cash = summary.value.parse::<f64>().unwrap_or_else(|_| {
                        println!("Warning: Could not parse cash value");
                        account.cash
                    });

                    if let Some(max_value) = MAX_ACCOUNT_VALUE {
                        account.cash = account.cash.min(max_value);
                    }

                    account_subscription.cancel();
                    break;
                }
            }
            AccountSummaryResult::End => {
                account_subscription.cancel();
                break;
            }
        }
    }

    drop(account_subscription);

    let positions_subscription = client.positions()?;

    for ticker_idx in 0..symbols.len() {
        account.positions[ticker_idx].quantity = 0.0;
        account.positions[ticker_idx].avg_price = 0.0;
    }

    for position in &positions_subscription {
        match position {
            PositionUpdate::Position(pos) => {
                if let Some(ticker_idx) = symbols
                    .iter()
                    .position(|s| s == pos.contract.symbol.as_str())
                {
                    account.positions[ticker_idx].quantity = pos.position;
                    account.positions[ticker_idx].avg_price = pos.average_cost;
                }
            }
            PositionUpdate::PositionEnd => {
                positions_subscription.cancel();
                break;
            }
        }
    }

    Ok(())
}
