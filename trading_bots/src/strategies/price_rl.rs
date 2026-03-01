use crate::{
    data::historical::get_historical_data,
    types::{Data, MakeCharts, MappedHistorical},
};

pub fn price_rl(
    _ticker_sets: &[Vec<usize>],
    _mapped_data: &MappedHistorical,
    _mapped_diffs: &[Data],
    // mapped_indicators: &Vec<Indicators>,
    _inputs_count: usize,
    _make_charts: Option<MakeCharts>,
) {
    let _time = std::time::Instant::now();

    let _mapped_historical = get_historical_data(None);
}
