use ibapi::{client, market_data::historical::{self, HistoricalData}, Client};

use crate::{types::Data, utils::get_rsi_values};

pub fn basic(client: &Client, data: &Data) {

    let rsi_values = get_rsi_values(data);
    
    for price in data {

    }
}