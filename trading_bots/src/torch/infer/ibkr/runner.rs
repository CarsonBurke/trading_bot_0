use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;
use tch::{Device, Kind, Tensor};

use ibapi::{
    accounts::{types::AccountGroup, AccountSummaryResult, AccountSummaryTags},
    contracts::Contract,
    market_data::{
        realtime::{BarSize, WhatToShow},
        TradingHours,
    },
    Client,
};

use crate::constants::api;
use crate::data::historical::get_historical_data;
use crate::torch::constants::{PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS, TICKERS_COUNT};
use crate::torch::infer::offline::{load_model, sample_actions};
use crate::torch::model::ModelVariant;

use super::execute::execute_trades;
use super::state::{LiveMarketState, MAX_ACCOUNT_VALUE};
use super::status::print_status;
use super::sync::sync_account_from_ibkr;

pub fn run_ibkr_paper_trading<P: AsRef<Path>>(
    weight_path: P,
    symbols: Vec<String>,
    update_interval_secs: u64,
    max_steps: usize,
    _temperature: f64,
    model_variant: ModelVariant,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== IBKR Paper Trading ===");
    println!("Weight path: {:?}", weight_path.as_ref());
    println!("Symbols: {:?}", symbols);
    println!("Update interval: {}s", update_interval_secs);
    println!("Max steps: {}", max_steps);

    let client = Client::connect(api::CONNECTION_URL, 100)?;
    println!("Connected to IBKR");

    let account_subscription =
        client.account_summary(&AccountGroup::from("All"), AccountSummaryTags::ALL)?;

    let mut starting_cash = 0.0;
    for update in &account_subscription {
        match update {
            AccountSummaryResult::Summary(summary) => {
                if summary.tag == "TotalCashValue" {
                    starting_cash = summary.value.parse::<f64>().unwrap_or_else(|_| {
                        println!(
                            "Warning: Could not parse cash value '{}', using 0.0",
                            summary.value
                        );
                        0.0
                    });

                    if let Some(max_value) = MAX_ACCOUNT_VALUE {
                        starting_cash = starting_cash.min(max_value);
                        println!(
                            "Account cash: ${:.2} (limited to ${:.2})",
                            starting_cash, max_value
                        );
                    } else {
                        println!("Account cash: ${:.2}", starting_cash);
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

    if starting_cash == 0.0 {
        return Err("Could not retrieve account cash from IBKR".into());
    }

    drop(account_subscription);

    let device = Device::cuda_if_available();
    let (_vs, model) = load_model(weight_path, device, model_variant)?;
    let mut stream_state = model.init_stream_state();

    let state = Arc::new(Mutex::new(LiveMarketState::new(
        symbols.clone(),
        starting_cash,
    )));

    // Seed the observation window from historical 5-minute bars (the resolution
    // the model trained on) so inference can start immediately with the correct
    // price/momentum/macro/earnings context rather than waiting weeks for
    // realtime 5-minute bars to accumulate.
    {
        let symbol_refs: Vec<&str> = symbols.iter().map(|s| s.as_str()).collect();
        let historical = get_historical_data(Some(&symbol_refs));
        let mut state_guard = state.lock().unwrap();
        for (ticker_idx, bars) in historical.iter().enumerate() {
            let start = bars.len().saturating_sub(PRICE_DELTAS_PER_TICKER + 1);
            let window = &bars[start..];
            let closes: Vec<f64> = window.iter().map(|b| b.close).collect();
            let dates: Vec<String> = window
                .iter()
                .map(|b| {
                    format!(
                        "{:04}-{:02}-{:02}",
                        b.date.year(),
                        b.date.month() as u8,
                        b.date.day()
                    )
                })
                .collect();
            state_guard.seed_history(ticker_idx, &closes, &dates);
        }
    }

    let client_arc = Arc::new(client);

    for (ticker_idx, symbol) in symbols.iter().enumerate() {
        let contract = Contract::stock(symbol).build();
        let client_clone = Arc::clone(&client_arc);
        let state_clone = Arc::clone(&state);
        let symbol_clone = symbol.clone();

        thread::spawn(move || {
            let subscription = match client_clone.realtime_bars(
                &contract,
                BarSize::Sec5,
                WhatToShow::Trades,
                TradingHours::Regular,
            ) {
                Ok(sub) => sub,
                Err(e) => {
                    println!("Failed to subscribe to {}: {}", symbol_clone, e);
                    return;
                }
            };

            // IBKR realtime bars are 5-second only; training uses 5-minute bars,
            // so aggregate 5s bars into 5-minute buckets to match the trained
            // resolution before feeding the model. A bucket is emitted when the
            // bar timestamp crosses into a new 5-minute window.
            let mut current_bucket: Option<i64> = None;
            let mut bucket_close = 0.0f64;
            let mut bucket_date = String::new();
            for bar in &subscription {
                let bucket = bar.date.unix_timestamp() / 300;
                let date = format!(
                    "{:04}-{:02}-{:02}",
                    bar.date.year(),
                    bar.date.month() as u8,
                    bar.date.day()
                );
                match current_bucket {
                    Some(b) if b == bucket => {
                        bucket_close = bar.close;
                        bucket_date = date;
                    }
                    Some(_) => {
                        let mut state = state_clone.lock().unwrap();
                        state.update_price(ticker_idx, bucket_close);
                        if ticker_idx == 0 {
                            state.push_bar_date(std::mem::take(&mut bucket_date));
                        }
                        state.update_account_total();
                        current_bucket = Some(bucket);
                        bucket_close = bar.close;
                        bucket_date = date;
                    }
                    None => {
                        current_bucket = Some(bucket);
                        bucket_close = bar.close;
                        bucket_date = date;
                    }
                }
            }
        });
    }

    // History is seeded from historical bars; only briefly settle realtime subs.
    thread::sleep(std::time::Duration::from_secs(10));

    let mut step = 0;
    let start_time = Instant::now();
    let mut static_obs_gpu = Tensor::zeros(&[1, STATIC_OBSERVATIONS as i64], (Kind::Float, device));
    let mut full_obs_raw_gpu = Tensor::zeros(
        &[1, TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64],
        (Kind::Float, device),
    );
    let mut full_obs_gpu = Tensor::zeros(&[1, model.price_input_dim()], (Kind::Float, device));
    let mut step_obs_gpu = Tensor::zeros(&[TICKERS_COUNT as i64], (Kind::Float, device));
    loop {
        if step >= max_steps {
            break;
        }

        thread::sleep(std::time::Duration::from_secs(update_interval_secs));

        let mut state_guard = state.lock().unwrap();

        sync_account_from_ibkr(&client_arc, &symbols, &mut state_guard.account)?;
        state_guard.update_account_total();

        if let Some((price_deltas_tensor, static_obs_tensor)) = state_guard.build_observation() {
            static_obs_gpu.copy_(&static_obs_tensor);

            // First step: use full observation to initialize streaming state
            // Subsequent steps: use single delta per ticker for O(1) inference
            let price_deltas_gpu = if !state_guard.model_initialized {
                state_guard.model_initialized = true;
                for deltas in &mut state_guard.pending_deltas {
                    deltas.clear();
                }
                full_obs_raw_gpu.copy_(&price_deltas_tensor);
                if model.variant() == ModelVariant::UniformStream {
                    let layout = model.uniform_stream_layout_from_raw_input(&full_obs_raw_gpu);
                    full_obs_gpu.copy_(&layout);
                } else {
                    full_obs_gpu.copy_(&full_obs_raw_gpu);
                }
                &full_obs_gpu
            } else {
                let Some(step_deltas) = state_guard.take_step_deltas() else {
                    println!("No fresh market delta available yet, skipping inference step");
                    continue;
                };
                step_obs_gpu.copy_(&step_deltas);
                &step_obs_gpu
            };

            let (alpha, beta) = tch::no_grad(|| {
                let (_, alpha, beta) =
                    model.step_on_device(price_deltas_gpu, &static_obs_gpu, &mut stream_state);
                (alpha, beta)
            });
            let actions = sample_actions(
                &alpha, &beta, true, // deterministic
                0.0,  // temperature
            );

            let actions_vec = Vec::<f64>::try_from(actions.flatten(0, -1)).unwrap();

            let current_prices = state_guard.get_current_prices();

            execute_trades(
                &client_arc,
                &symbols,
                &actions_vec,
                &current_prices,
                &state_guard.account,
            )?;

            step += 1;
            state_guard.step_count = step;

            if step % 10 == 0 {
                print_status(step, &state_guard, &start_time);
            }
        } else {
            println!(
                "Waiting for sufficient price data... ({}/{})",
                state_guard
                    .price_deltas
                    .iter()
                    .map(|d| d.len())
                    .min()
                    .unwrap_or(0),
                PRICE_DELTAS_PER_TICKER
            );
        }
    }

    let final_state = state.lock().unwrap();
    println!("\n=== Final Summary ===");
    println!("Total steps: {}", final_state.step_count);
    println!("Starting cash: ${:.2}", starting_cash);
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
