use ibapi::{accounts::{AccountSummaries, AccountSummaryTags}, client::Subscription, Client};

fn main() {
    println!("Trying to connect to IB API!");
    let connection_url = "127.0.0.1:4001";

    let client = Client::connect(connection_url, 1).expect("connection to TWS failed!");
    println!("Successfully connected to TWS at {connection_url}");

    account_info(&client);
}

fn account_info(client: &Client) {
    let subscription = client.account_summary("All", AccountSummaryTags::ALL).expect("error requesting account summary");

    for update in &subscription {
        match update {
            AccountSummaries::Summary(summary) => println!("{summary:?}"),
            AccountSummaries::End => subscription.cancel(),
        }
    }
}