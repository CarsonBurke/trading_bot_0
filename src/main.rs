use clap::{arg, Command};
use constants::{files, TICKER, TICKERS};
use hashbrown::{HashMap, HashSet};
use ibapi::{
    accounts::{AccountSummaries, AccountSummaryTags},
    client::Subscription,
    contracts::Contract,
    market_data::{
        historical::{self, BarSize, HistoricalData, ToDuration, WhatToShow},
        realtime,
    },
    orders::{order_builder, Action, PlaceOrder},
    Client,
};
use std::{fs, os};
use time::{macros::datetime, OffsetDateTime};
use types::{Account, MappedHistorical};
use utils::{candle_chart, chart, convert_historical, get_rsi_values, rsi_chart};

pub mod constants;
pub mod strategies;
mod types;
mod utils;

fn main() {
    println!("Trying to connect to IB API!");
    let connection_url = "127.0.0.1:4001";

    let client = Client::connect(connection_url, 1).expect("connection to TWS failed!");
    println!("Successfully connected to TWS at {connection_url}");

    // account_info(&client);
    let mut account = Account::default();

    // let historical_data = historical_data(&client);
    // market_depth(&client);
    // bars(&client);
    // place_order(&client);

    let mapped_historical = get_historical_data(&client);
    
    let tsla = mapped_historical.get("TSLA").unwrap();
    let data = convert_historical(tsla);
    let rsi_values = get_rsi_values(&data);

    chart(&data).unwrap();
    rsi_chart(&rsi_values).unwrap();

    candle_chart(&tsla).unwrap();

    strategies::basic::basic(&client, &data, &mut account);
}

fn account_info(client: &Client) {
    let subscription = client
        .account_summary("All", AccountSummaryTags::ALL)
        .expect("error requesting account summary");

    for update in &subscription {
        match update {
            AccountSummaries::Summary(summary) => println!("{summary:?}"),
            AccountSummaries::End => subscription.cancel(),
        }
    }
}

fn get_historical_data(client: &Client) -> MappedHistorical {
    if let Some(data) = get_historical_data_from_files() {
        return data;
    };

    get_historical_data_from_ibkr(client)
}

fn get_historical_data_from_files() -> Option<MappedHistorical> {
    let dir = fs::read_dir(files::DATA_PATH).ok()?;

    let mut data = HashMap::new();

    let tickers_set: HashSet<String> =
        HashSet::from_iter(TICKERS.iter().map(|str| str.to_string()));

    for path in dir {
        let Ok(entry) = path else {
            continue;
        };

        let filename_extended = entry.file_name().to_str().unwrap().to_string();
        let (os_filename, _) = filename_extended.split_once('.').unwrap();
        let filename = os_filename.to_string();

        if !tickers_set.contains(&filename) {
            continue;
        }

        let file = fs::read(format!("{}/{}", files::DATA_PATH, filename_extended)).ok()?;
        let bars: Vec<historical::Bar> = postcard::from_bytes(&file).ok()?;

        data.insert(filename, bars);
    }

    Some(data)
}

fn get_historical_data_from_ibkr(client: &Client) -> MappedHistorical {
    let mut data = HashMap::new();

    fs::create_dir(files::DATA_PATH).ok().unwrap();

    for ticker in TICKERS {
        let contract = Contract::stock(ticker);

        let historical_data = client
            .historical_data(
                &contract,
                OffsetDateTime::now_utc(),
                360.days(),
                BarSize::Hour,
                WhatToShow::Trades,
                true,
            )
            .expect("historical data request failed");

        // Write compacted data to a file

        let encoded = postcard::to_allocvec(&historical_data.bars).unwrap();

        fs::write(
            format!("{}/{}.bin", files::DATA_PATH, ticker),
            encoded.as_slice(),
        )
        .ok()
        .unwrap();

        data.insert(ticker.to_string(), historical_data.bars);
    }

    data
}

fn historical_data(client: &Client) -> HistoricalData {
    let contract = Contract::stock(TICKER);

    let historical_data = client
        .historical_data(
            &contract,
            OffsetDateTime::now_utc(),
            360.days(),
            BarSize::Hour,
            WhatToShow::Trades,
            true,
        )
        .expect("historical data request failed");

    /*     println!(
        "start: {:?}, end: {:?}",
        historical_data.start, historical_data.end
    );

    for bar in &historical_data.bars {
        println!("{bar:?}");
    }

    println!("data points: {}", historical_data.bars.len()); */

    historical_data
}

fn market_data(client: &Client) {
    /* for ticker in constants::TICKERS {
        let subscription = client
            .market_data(ticker, vec![], false, false)
            .expect("market data request failed");
    } */
}

fn market_depth(client: &Client) {
    let contract = Contract::stock("AAPL");

    let subscription = client
        .market_depth(&contract, 5, true)
        .expect("error requesting market depth");
    for row in &subscription {
        println!("row: {row:?}")
    }
}

fn place_order(client: &Client) {
    let contract = Contract::stock("AAPL");

    // Creates a market order to purchase 100 shares
    let order_id = client.next_order_id();
    let order = order_builder::market_order(Action::Buy, 100.0);

    let subscription = client
        .place_order(order_id, &contract, &order)
        .expect("place order request failed!");

    for event in &subscription {
        if let PlaceOrder::ExecutionData(data) = event {
            println!(
                "{} {} shares of {}",
                data.execution.side, data.execution.shares, data.contract.symbol
            );
        } else {
            println!("{:?}", event);
        }
    }
}

fn bars(client: &Client) {
    println!("getting bars");

    let contract = Contract::stock("AAPL");

    let subscription = client
        .realtime_bars(
            &contract,
            realtime::BarSize::Sec5,
            realtime::WhatToShow::Trades,
            false,
        )
        .expect("error requesting market depth");
    println!("subscription: {subscription:?}");
    for bar in &subscription {
        println!("bar: {bar:?}")
    }
}

fn find_opportunites(client: &Client) {}
