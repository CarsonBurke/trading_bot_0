use clap::{arg, Command};
use ibapi::{
    accounts::{AccountSummaries, AccountSummaryTags},
    client::Subscription,
    contracts::Contract,
    market_data::historical::{BarSize, ToDuration, WhatToShow},
    Client,
};
use time::macros::datetime;

fn main() {
    println!("Trying to connect to IB API!");
    let connection_url = "127.0.0.1:4001";

    let client = Client::connect(connection_url, 1).expect("connection to TWS failed!");
    println!("Successfully connected to TWS at {connection_url}");

    account_info(&client);
    historical_data(&client);
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

fn historical_data(client: &Client) {
    let ticker = "TSLA";

    let contract = Contract::stock(ticker);

    let historical_data = client
        .historical_data(
            &contract,
            datetime!(2023-04-11 20:00 UTC),
            1.days(),
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
}

fn find_opportunites(client: &Client) {}
