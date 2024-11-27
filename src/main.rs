use clap::{arg, Command};
use ibapi::{
    accounts::{AccountSummaries, AccountSummaryTags}, client::Subscription, contracts::Contract, market_data::{historical::{BarSize, HistoricalData, ToDuration, WhatToShow}, realtime}, orders::{order_builder, Action, PlaceOrder}, Client
};
use time::{macros::datetime, OffsetDateTime};
use utils::{chart, convert_historical, get_rsi_values, rsi_chart};

mod constants;
mod utils;
mod strategies;
mod types;

fn main() {
    println!("Trying to connect to IB API!");
    let connection_url = "127.0.0.1:4001";

    let client = Client::connect(connection_url, 1).expect("connection to TWS failed!");
    println!("Successfully connected to TWS at {connection_url}");

    account_info(&client);
    let historical_data = historical_data(&client);
    // market_depth(&client);
    // bars(&client);
    // place_order(&client);

    let data = convert_historical(&historical_data.bars);
    let rsi_values = get_rsi_values(&data);

    chart(&data).unwrap();
    rsi_chart(&rsi_values).unwrap();
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

fn historical_data(client: &Client) -> HistoricalData {
    let ticker = "TSLA";

    let contract = Contract::stock(ticker);

    let historical_data = client
        .historical_data(
            &contract,
            OffsetDateTime::now_utc(),
            30.days(),
            BarSize::Hour,
            WhatToShow::Trades,
            true,
        )
        .expect("historical data request failed");

    println!(
        "start: {:?}, end: {:?}",
        historical_data.start, historical_data.end
    );

    for bar in &historical_data.bars {
        println!("{bar:?}");
    }

    println!("data points: {}", historical_data.bars.len());

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

    let subscription = client.place_order(order_id, &contract, &order).expect("place order request failed!");

    for event in &subscription {
        if let PlaceOrder::ExecutionData(data) = event {
            println!("{} {} shares of {}", data.execution.side, data.execution.shares, data.contract.symbol);
        } else {
            println!("{:?}", event);
        }
    }
}

fn bars(client: &Client) {
    println!("getting bars");

    let contract = Contract::stock("AAPL");

    let subscription = client
        .realtime_bars(&contract, realtime::BarSize::Sec5, realtime::WhatToShow::Trades, false)
        .expect("error requesting market depth");
    println!("subscription: {subscription:?}");
    for bar in &subscription {
        println!("bar: {bar:?}")
    }
}

fn find_opportunites(client: &Client) {}
