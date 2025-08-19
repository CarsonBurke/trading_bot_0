use crate::{data::historical::get_historical_data, types::{Data, MakeCharts, MappedHistorical}};

pub fn price_rl(
    ticker_sets: &[Vec<usize>],
    mapped_data: &MappedHistorical,
    mapped_diffs: &[Data],
    // mapped_indicators: &Vec<Indicators>,
    inputs_count: usize,
    make_charts: Option<MakeCharts>,
) {
    let time = std::time::Instant::now();

    let mapped_historical = get_historical_data(None);
    
    
}
