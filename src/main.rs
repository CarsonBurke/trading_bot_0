use agent::train::train_agents;
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

pub mod constants;
pub mod strategies;
pub mod agent;
pub mod data;
mod types;
mod utils;
pub mod charts;

fn main() {
    println!("Trying to connect to IB API!");
    let connection_url = "127.0.0.1:4001";

    let client = Client::connect(connection_url, 1).expect("connection to TWS failed!");
    println!("Successfully connected to TWS at {connection_url}");

    // account_info(&client);
    // let mut account = Account::default();

    // let historical_data = historical_data(&client);
    // market_depth(&client);
    // bars(&client);
    // place_order(&client);

    // let mapped_historical = get_historical_data(&client);
    
    // let stock_data = mapped_historical.get(TICKER).unwrap();
    // let data = convert_historical(stock_data);

    // strategies::basic::basic(&client, &data, &mut account);
    // let rsi_values = get_rsi_values(&data);

    // chart(&data).unwrap();
    // rsi_chart(&rsi_values).unwrap();

    // candle_chart(&stock_data).unwrap();

    train_agents(&client);
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
