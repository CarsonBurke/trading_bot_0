use std::fs;

use hashbrown::{HashMap, HashSet};
use ibapi::{contracts::Contract, market_data::historical::{self, BarSize, ToDuration, WhatToShow}, Client};
use time::OffsetDateTime;

use crate::{constants::{files, TICKERS}, types::MappedHistorical};

pub fn get_historical_data(client: &Client) -> MappedHistorical {
    if let Some(data) = get_historical_data_from_files() {
        return data;
    };

    get_historical_data_from_ibkr(client)
}

fn get_historical_data_from_files() -> Option<MappedHistorical> {
    let dir = fs::read_dir(files::DATA_PATH).ok()?;

    let mut data = Vec::new();

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

        data.push(bars);
    }

    Some(data)
}

fn get_historical_data_from_ibkr(client: &Client) -> MappedHistorical {
    let mut data = Vec::new();

    fs::create_dir(files::DATA_PATH).ok().unwrap();

    for ticker in TICKERS {
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

        data.push(historical_data.bars);
    }

    data
}