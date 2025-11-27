use std::fs;

use hashbrown::{HashMap, HashSet};
use ibapi::{
    client,
    contracts::Contract,
    market_data::historical::{self, BarSize, ToDuration, WhatToShow},
    Client,
};
use time::OffsetDateTime;

use crate::{
    constants::{TICKERS, api, files::{self, DATA_PATH}},
    types::MappedHistorical,
    utils::create_folder_if_not_exists,
};

pub fn get_historical_data(tickers: Option<&[&str]>) -> MappedHistorical {
    let tickers = tickers.unwrap_or(&TICKERS);

    let mut data = Vec::new();
    let opt_client: Option<Client> = None;

    for ticker in tickers.iter() {
        // Try to get the data from local files
        if let Some(bars) = get_historical_data_from_files(ticker) {
            data.push(bars);
            continue;
        }

        // Otherwise get the data from IBKR

        let client = match &opt_client {
            Some(client) => client,
            None => {
                &Client::connect(api::CONNECTION_URL, 1).expect("connection to TWS failed!")
            }
        };

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
            match DATA_PATH {
                "data" => 356.days(),
                "long_data" => 5.years(),
                "very_long_data" => 10.years(),
                "extra_long_data" => 20.years(),
                _ => panic!("no data path provided"),
            },
            BarSize::Min5,
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
