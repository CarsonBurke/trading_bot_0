use crate::types::{Data, MakeCharts, MappedHistorical};

pub fn price_rl(
    ticker_sets: &[Vec<usize>],
    mapped_data: &MappedHistorical,
    mapped_diffs: &[Data],
    // mapped_indicators: &Vec<Indicators>,
    inputs_count: usize,
    make_charts: Option<MakeCharts>,
) {
    let mut all_min: f64 = f64::MAX;
    
    
}
