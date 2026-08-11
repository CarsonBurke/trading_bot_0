use std::io;
use std::path::Path;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::thread;
use std::time::{Duration, Instant};

use ibapi::{
    contracts::Contract,
    market_data::{
        realtime::{BarSize, WhatToShow},
        TradingHours,
    },
    Client,
};
use tch::{Device, Kind, Tensor};

use crate::constants::api;
use crate::data::historical::refresh_historical_bars_at;
use crate::torch::constants::{PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS, TICKERS_COUNT};
use crate::torch::infer::offline::{load_model, sample_actions};
use crate::torch::model::ModelVariant;

use super::super::validate_ticker_count;
use super::execute::execute_trades;
use super::state::LiveMarketState;
use super::status::print_status;
use super::sync::{initialize_dedicated_account, sync_account_from_ibkr};

const MAX_FEED_AGE: Duration = Duration::from_secs(30);
const MAX_FEED_POLL_INTERVAL: Duration = Duration::from_secs(5);

fn feed_poll_interval(requested_secs: u64) -> Duration {
    Duration::from_secs(requested_secs).min(MAX_FEED_POLL_INTERVAL)
}

pub fn run_ibkr_paper_trading<P: AsRef<Path>>(
    weight_path: P,
    account: String,
    symbols: Vec<String>,
    update_interval_secs: u64,
    max_steps: usize,
    model_variant: ModelVariant,
) -> Result<(), Box<dyn std::error::Error>> {
    validate_ticker_count(&symbols, "IBKR paper trading")?;
    if update_interval_secs == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "paper update interval must be positive",
        )
        .into());
    }

    println!("=== IBKR Paper Trading (deterministic policy) ===");
    println!("Weight path: {:?}", weight_path.as_ref());
    println!("Account: {account}");
    println!("Symbols: {symbols:?}");
    let poll_interval = feed_poll_interval(update_interval_secs);
    println!("Requested update interval: {update_interval_secs}s");
    println!("Effective feed poll interval: {}s", poll_interval.as_secs());
    println!("Max steps: {max_steps}");

    let client = Client::connect(api::CONNECTION_URL, 100)?;
    println!("Connected to IBKR");

    let (account_id, initial_account) = initialize_dedicated_account(&client, &account, &symbols)?;
    let starting_cash = initial_account.cash;
    println!("Dedicated account cash: ${starting_cash:.2}");

    let device = Device::cuda_if_available();
    let (_vs, model) = load_model(weight_path, device, model_variant)?;

    let history_len = PRICE_DELTAS_PER_TICKER + 1;
    let historical_cutoff = time::OffsetDateTime::now_utc();
    let historical = symbols
        .iter()
        .map(|symbol| refresh_historical_bars_at(&client, symbol, historical_cutoff))
        .collect::<Result<Vec<_>, _>>()?;
    let reference_times = historical
        .first()
        .and_then(|bars| {
            bars.len()
                .checked_sub(history_len)
                .map(|start| &bars[start..])
        })
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("paper trading requires at least {history_len} completed historical bars"),
            )
        })?
        .iter()
        .map(|bar| bar.date)
        .collect::<Vec<_>>();
    let state = Arc::new(Mutex::new(LiveMarketState::new(
        symbols.clone(),
        starting_cash,
    )));
    state.lock().unwrap().account = initial_account;
    {
        let mut state_guard = state.lock().unwrap();
        for (ticker_idx, bars) in historical.iter().enumerate() {
            if bars.len() < history_len {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!(
                        "{} has {} completed historical bars; {history_len} required",
                        symbols[ticker_idx],
                        bars.len()
                    ),
                )
                .into());
            }
            let window = &bars[bars.len() - history_len..];
            let times = window.iter().map(|bar| bar.date).collect::<Vec<_>>();
            if times != reference_times {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "paper historical feeds are not aligned on identical completed bars",
                )
                .into());
            }
            let closes = window.iter().map(|bar| bar.close).collect::<Vec<_>>();
            state_guard.seed_history(ticker_idx, &closes, &times);
        }
    }

    let client = Arc::new(client);
    let stop = Arc::new(AtomicBool::new(false));
    let mut feed_threads = Vec::with_capacity(symbols.len());
    for (ticker_idx, symbol) in symbols.iter().enumerate() {
        let contract = Contract::stock(symbol).build();
        let client = Arc::clone(&client);
        let state = Arc::clone(&state);
        let stop = Arc::clone(&stop);
        let symbol = symbol.clone();
        feed_threads.push(thread::spawn(move || {
            let subscription = match client.realtime_bars(
                &contract,
                BarSize::Sec5,
                WhatToShow::Trades,
                TradingHours::Regular,
            ) {
                Ok(subscription) => subscription,
                Err(error) => {
                    state
                        .lock()
                        .unwrap()
                        .mark_feed_failed(ticker_idx, format!("subscription failed: {error}"));
                    return;
                }
            };

            while !stop.load(Ordering::Relaxed) {
                let Some(bar) = subscription.next_timeout(Duration::from_secs(1)) else {
                    if let Some(error) = subscription.error() {
                        state
                            .lock()
                            .unwrap()
                            .mark_feed_failed(ticker_idx, format!("stream failed: {error}"));
                        return;
                    }
                    continue;
                };
                let mut state = state.lock().unwrap();
                if let Err(error) =
                    state.record_realtime_bar(ticker_idx, bar.date, bar.close, Instant::now())
                {
                    state.mark_feed_failed(ticker_idx, error);
                    return;
                }
            }
            println!("Stopped {symbol} realtime feed");
        }));
    }

    let started_at = Instant::now();
    // Realtime bars arrive every five seconds. Keep feed health independent of
    // the policy cadence so a slow action interval cannot bless old data.
    let mut last_acted_sequence = None;
    let mut static_obs_gpu = Tensor::zeros([1, STATIC_OBSERVATIONS as i64], (Kind::Float, device));
    let mut full_obs_raw_gpu = Tensor::zeros(
        [1, TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64],
        (Kind::Float, device),
    );
    let mut full_obs_gpu = Tensor::zeros([1, model.price_input_dim()], (Kind::Float, device));

    let trading_result = (|| -> Result<(), Box<dyn std::error::Error>> {
        while state.lock().unwrap().step_count < max_steps {
            thread::sleep(poll_interval);

            let action_gate = state
                .lock()
                .unwrap()
                .actionable_prices(
                    Instant::now(),
                    time::OffsetDateTime::now_utc(),
                    MAX_FEED_AGE,
                    last_acted_sequence,
                )
                .map_err(io::Error::other)?;
            let Some((bucket, sequence, _)) = action_gate else {
                println!("Waiting for a fresh aligned completed bar from every feed");
                continue;
            };

            let mut account_snapshot = state.lock().unwrap().account.clone();
            sync_account_from_ibkr(&client, &account_id, &symbols, &mut account_snapshot)?;
            let previous_quantities = account_snapshot
                .positions
                .iter()
                .map(|position| position.quantity)
                .collect::<Vec<_>>();
            let (price_deltas_tensor, static_obs_tensor) = {
                let mut state = state.lock().unwrap();
                state.account = account_snapshot.clone();
                state
                    .prepare_observation(sequence)
                    .map_err(io::Error::other)?;
                let observation_prices = state.get_current_prices();
                state.update_observation_value_with_prices(&observation_prices);
                state.build_observation().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "aligned live frame did not produce a complete observation",
                    )
                })?
            };

            static_obs_gpu.copy_(&static_obs_tensor);
            full_obs_raw_gpu.copy_(&price_deltas_tensor);
            if model.variant() == ModelVariant::UniformStream {
                let layout = model.uniform_stream_layout_from_raw_input(&full_obs_raw_gpu);
                full_obs_gpu.copy_(&layout);
            } else {
                full_obs_gpu.copy_(&full_obs_raw_gpu);
            }
            let mut stream_state = model.init_stream_state();
            let (alpha, beta) = tch::no_grad(|| {
                let (_, alpha, beta) =
                    model.step_on_device(&full_obs_gpu, &static_obs_gpu, &mut stream_state);
                (alpha, beta)
            });
            let actions = sample_actions(&alpha, &beta, true, 1.0);
            let actions = Vec::<f64>::try_from(actions.flatten(0, -1))?;

            // Account sync and inference can block. Revalidate the feed and take
            // the newest executable quotes immediately before order planning.
            let fresh_gate = state
                .lock()
                .unwrap()
                .actionable_prices(
                    Instant::now(),
                    time::OffsetDateTime::now_utc(),
                    MAX_FEED_AGE,
                    last_acted_sequence,
                )
                .map_err(io::Error::other)?;
            let Some((fresh_bucket, fresh_sequence, execution_prices)) = fresh_gate else {
                continue;
            };
            if fresh_bucket != bucket || fresh_sequence != sequence {
                println!("A newer completed frame arrived during inference; recomputing action");
                continue;
            }

            let mut ensure_execution_frame = || -> Result<(), Box<dyn std::error::Error>> {
                let gate = state
                    .lock()
                    .unwrap()
                    .actionable_prices(
                        Instant::now(),
                        time::OffsetDateTime::now_utc(),
                        MAX_FEED_AGE,
                        last_acted_sequence,
                    )
                    .map_err(io::Error::other)?;
                let Some((current_bucket, current_sequence, _)) = gate else {
                    return Err(io::Error::other(
                        "paper feed became stale or incomplete while executing order batch",
                    )
                    .into());
                };
                if current_bucket != bucket || current_sequence != sequence {
                    return Err(io::Error::other(
                        "paper feed advanced while executing order batch",
                    )
                    .into());
                }
                Ok(())
            };
            let outcome = execute_trades(
                &client,
                &account_id,
                &symbols,
                &actions,
                &execution_prices,
                &mut account_snapshot,
                &mut ensure_execution_frame,
            )?;
            {
                let mut state = state.lock().unwrap();
                state.account = account_snapshot;
                state
                    .apply_execution_feedback(&previous_quantities, &outcome)
                    .map_err(io::Error::other)?;
                state.revalue_account_with_prices(&execution_prices);
                if state.step_count % 10 == 0 {
                    print_status(state.step_count, &state, &started_at);
                }
            }
            last_acted_sequence = Some(sequence);
        }
        Ok(())
    })();

    stop.store(true, Ordering::Relaxed);
    let mut feed_panicked = false;
    for handle in feed_threads {
        feed_panicked |= handle.join().is_err();
    }
    trading_result?;
    if feed_panicked {
        return Err(io::Error::other("a paper market-data worker panicked").into());
    }

    let final_state = state.lock().unwrap();
    println!("\n=== Final Summary ===");
    println!("Total steps: {}", final_state.step_count);
    println!("Starting cash: ${starting_cash:.2}");
    println!(
        "Final total assets: ${:.2}",
        final_state.account.total_assets
    );
    println!(
        "Total P&L: ${:.2} ({:.2}%)",
        final_state.account.total_assets - starting_cash,
        (final_state.account.total_assets / starting_cash - 1.0) * 100.0
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{feed_poll_interval, MAX_FEED_POLL_INTERVAL};
    use std::time::Duration;

    #[test]
    fn slow_requested_cadence_cannot_exceed_realtime_feed_poll_bound() {
        assert_eq!(feed_poll_interval(60), MAX_FEED_POLL_INTERVAL);
        assert_eq!(feed_poll_interval(2), Duration::from_secs(2));
    }
}
