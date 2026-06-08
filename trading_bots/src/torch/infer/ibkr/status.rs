use std::time::Instant;

use super::state::LiveMarketState;

pub(super) fn print_status(step: usize, state: &LiveMarketState, start_time: &Instant) {
    let elapsed = Instant::now().duration_since(*start_time).as_secs_f32();
    let current_prices = state.get_current_prices();

    println!("\n--- Step {} (elapsed: {:.1}s) ---", step, elapsed);
    println!("Total Assets: ${:.2}", state.account.total_assets);
    println!("Cash: ${:.2}", state.account.cash);

    for (i, position) in state.account.positions.iter().enumerate() {
        if position.quantity > 0.0 {
            let value = position.value_with_price(current_prices[i]);
            let pnl_pct = position.appreciation(current_prices[i]) * 100.0;
            println!(
                "Position {}: {:.2} shares @ ${:.2} (value: ${:.2}, P&L: {:.2}%)",
                i, position.quantity, current_prices[i], value, pnl_pct
            );
        }
    }
}
