pub mod ibkr;
pub mod offline;

use std::io;

use crate::torch::constants::TICKERS_COUNT;

pub use ibkr::run_ibkr_paper_trading;
pub use offline::run_inference;

fn validate_ticker_count(tickers: &[String], context: &str) -> Result<(), io::Error> {
    let expected = TICKERS_COUNT as usize;
    if tickers.len() == expected {
        return Ok(());
    }

    Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        format!(
            "{context} requires exactly {expected} ticker(s) for the configured model, received {}",
            tickers.len()
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::validate_ticker_count;
    use crate::torch::constants::TICKERS_COUNT;

    #[test]
    fn ticker_count_validation_matches_model_cardinality() {
        let valid = vec!["TEST".to_string(); TICKERS_COUNT as usize];
        assert!(validate_ticker_count(&valid, "inference").is_ok());

        let invalid = vec!["TEST".to_string(); TICKERS_COUNT as usize + 1];
        let error = validate_ticker_count(&invalid, "inference").unwrap_err();
        assert!(error.to_string().contains("requires exactly"));
    }
}
