use std::fs;

use hashbrown::{HashMap, HashSet};
use ibapi::{
    contracts::Contract,
    market_data::historical::{self, BarSize, ToDuration, WhatToShow},
    Client,
};
use time::OffsetDateTime;

use crate::{
    constants::{files, TICKERS},
    types::MappedHistorical,
    utils::create_folder_if_not_exists,
};

pub fn get_historical_data(client: &Client) -> MappedHistorical {
    let mut data = Vec::new();

    for ticker in TICKERS.iter() {
        if let Some(bars) = get_historical_data_from_files(ticker) {
            data.push(bars);
            continue;
        }

        data.push(get_historical_data_from_ibkr(client, ticker));
    }

    data
}

fn get_historical_data_from_files(ticker: &str) -> Option<Vec<historical::Bar>> {
    let path = format!("{}/{}.bin", files::DATA_PATH, ticker);
    let file = fs::read(path).ok()?;

    let bars: Vec<historical::Bar> = postcard::from_bytes(&file).ok()?;
    Some(bars)
}

fn get_historical_data_from_ibkr(client: &Client, ticker: &str) -> Vec<historical::Bar> {
    create_folder_if_not_exists(&files::DATA_PATH.to_string());

    println!("Downloading data for {ticker}");
    let contract = Contract::stock(ticker);

    let historical_data = client
        .historical_data(
            &contract,
            OffsetDateTime::now_utc(),
            365.days(),
            /* 1.years(), */
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

    historical_data.bars
}
