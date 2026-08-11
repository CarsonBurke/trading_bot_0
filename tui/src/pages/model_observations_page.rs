use anyhow::{bail, Context, Result};
use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};
use shared::{
    constants::{GLOBAL_STATIC_OBS, PER_TICKER_STATIC_OBS},
    report::{read_report, ReportKind},
};
use std::path::Path;

use crate::{components::episode_status, state::generation_browser::GenerationInfo, theme, App};

const GLOBAL_LABELS: [&str; GLOBAL_STATIC_OBS] = [
    "Cash %",
    "PnL",
    "Drawdown",
    "Commissions",
    "Fill ratio",
    "GDP growth",
    "Unemployment",
    "Jobs growth",
    "CPI YoY",
    "Core CPI YoY",
    "Fed funds",
    "Treasury 10Y",
    "Yield spread",
    "Sentiment",
    "Initial claims",
    "Steps to jobs",
    "Steps to CPI",
    "Steps to FOMC",
    "Steps to GDP",
];

const TICKER_LABELS: [&str; PER_TICKER_STATIC_OBS] = [
    "Position %",
    "Appreciation",
    "Trade activity",
    "Trade recency",
    "Position age",
    "Realized weight",
    "Momentum 5",
    "Momentum 20",
    "Momentum 60",
    "Momentum 120",
    "Acceleration",
    "Vol-adjusted",
    "Efficiency",
    "Trend strength",
    "RSI",
    "Range position",
    "Stochastic K",
    "Z-score",
    "MACD",
    "Earnings recency",
    "Revenue growth",
    "Opex growth",
    "Profit growth",
    "EPS",
    "EPS surprise",
];

#[derive(Debug, PartialEq)]
struct ObservationData {
    observation_tickers: Vec<String>,
    action_tickers: Vec<String>,
    static_observations: Vec<Vec<f32>>,
    attention_weights: Vec<Vec<f32>>,
    action_step0: Option<Vec<f32>>,
    action_final: Option<Vec<f32>>,
}

fn load_observations(path: &Path) -> Result<ObservationData> {
    let report = read_report(path)
        .with_context(|| format!("failed reading observations report {}", path.display()))?;
    let ReportKind::Observations {
        observation_tickers,
        action_tickers,
        static_observations,
        attention_weights,
        action_step0,
        action_final,
    } = report.kind
    else {
        bail!("{} is not an observations report", path.display());
    };
    if static_observations.iter().any(|row| {
        row.len() != GLOBAL_STATIC_OBS + observation_tickers.len() * PER_TICKER_STATIC_OBS
    }) {
        bail!("observations report has a row width inconsistent with its ticker labels");
    }
    Ok(ObservationData {
        observation_tickers,
        action_tickers,
        static_observations,
        attention_weights,
        action_step0,
        action_final,
    })
}

pub fn render(f: &mut Frame, app: &mut App) {
    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(0),
            Constraint::Length(4),
        ])
        .split(f.area());
    let content_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(30),
            Constraint::Percentage(45),
            Constraint::Percentage(25),
        ])
        .split(main_chunks[1]);

    let mut title_spans = vec![Span::styled(
        " Model Observations ",
        Style::default()
            .fg(theme::MAUVE)
            .add_modifier(Modifier::BOLD),
    )];
    title_spans.extend(episode_status::episode_status_spans(
        app.is_training_running(),
        app.get_current_episode(),
        app.has_training_progress(),
    ));
    f.render_widget(
        Paragraph::new(Line::from(title_spans)).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(theme::LAVENDER)),
        ),
        main_chunks[0],
    );

    let report = latest_observations_path(&app.generation_browser.generations);
    match report.as_deref().map(load_observations) {
        Some(Ok(data)) if !data.static_observations.is_empty() => {
            render_observations(f, content_chunks.as_ref(), &data)
        }
        Some(Ok(_)) => render_message(f, content_chunks.as_ref(), "Observations report is empty"),
        Some(Err(error)) => render_message(f, content_chunks.as_ref(), &error.to_string()),
        None => render_message(
            f,
            content_chunks.as_ref(),
            "No episodes found. Train a model first.",
        ),
    }

    f.render_widget(
        Paragraph::new(" ESC: Back to Main | R: Refresh ")
            .block(Block::default().borders(Borders::ALL).title(" Controls ")),
        main_chunks[2],
    );
}

fn latest_observations_path(generations: &[GenerationInfo]) -> Option<std::path::PathBuf> {
    generations.iter().rev().find_map(|generation| {
        let report = generation.path.join("observations.report.bin");
        report.is_file().then_some(report)
    })
}

fn render_observations(f: &mut Frame, chunks: &[ratatui::layout::Rect], data: &ObservationData) {
    let step = data.static_observations.len() - 1;
    let row = &data.static_observations[step];
    let mut global_lines = vec![heading(format!(
        "Step {}/{}",
        step + 1,
        data.static_observations.len()
    ))];
    global_lines.extend(
        GLOBAL_LABELS
            .iter()
            .zip(row.iter())
            .map(|(label, value)| metric_line(label, *value)),
    );
    f.render_widget(
        Paragraph::new(global_lines).block(panel("Global Observations")),
        chunks[0],
    );

    let mut ticker_lines = Vec::new();
    for (ticker_index, ticker) in data.observation_tickers.iter().enumerate() {
        ticker_lines.push(heading(ticker.clone()));
        let start = GLOBAL_STATIC_OBS + ticker_index * PER_TICKER_STATIC_OBS;
        ticker_lines.extend(
            TICKER_LABELS
                .iter()
                .zip(row[start..start + PER_TICKER_STATIC_OBS].iter())
                .map(|(label, value)| metric_line(label, *value)),
        );
        ticker_lines.push(Line::from(""));
    }
    f.render_widget(
        Paragraph::new(ticker_lines).block(panel("Per-Ticker Observations")),
        chunks[1],
    );

    let side_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chunks[2]);
    let mut action_lines = vec![heading("Target weights: start / final")];
    let mut action_names = data.action_tickers.clone();
    if data
        .action_step0
        .as_ref()
        .into_iter()
        .chain(data.action_final.as_ref())
        .any(|values| values.len() > data.action_tickers.len())
    {
        action_names.push("CASH".to_owned());
    }
    if data.action_step0.is_none() && data.action_final.is_none() {
        action_lines.push(Line::from("No action snapshots recorded"));
    } else {
        for (index, name) in action_names.iter().enumerate() {
            let start = data
                .action_step0
                .as_ref()
                .and_then(|values| values.get(index));
            let end = data
                .action_final
                .as_ref()
                .and_then(|values| values.get(index));
            action_lines.push(Line::from(format!(
                "{name:>8}  {} / {}",
                format_optional(start),
                format_optional(end)
            )));
        }
    }
    f.render_widget(
        Paragraph::new(action_lines).block(panel("Actions")),
        side_chunks[0],
    );

    let attention_lines = data
        .attention_weights
        .last()
        .filter(|weights| !weights.is_empty())
        .map(|weights| {
            let total = weights
                .iter()
                .copied()
                .filter(|value| value.is_finite())
                .sum::<f32>();
            let (peak_index, peak) = weights
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, value)| value.is_finite())
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .unwrap_or((0, 0.0));
            vec![
                metric_line("Samples", weights.len() as f32),
                metric_line("Total", total),
                metric_line("Peak index", peak_index as f32),
                metric_line("Peak weight", peak),
            ]
        })
        .unwrap_or_else(|| vec![Line::from("No attention data")]);
    f.render_widget(
        Paragraph::new(attention_lines).block(panel("Temporal Attention")),
        side_chunks[1],
    );
}

fn render_message(f: &mut Frame, chunks: &[ratatui::layout::Rect], message: &str) {
    f.render_widget(
        Paragraph::new(message)
            .block(panel("Model Observations"))
            .style(Style::default().fg(theme::SUBTEXT0)),
        chunks[0],
    );
    f.render_widget(Block::default().borders(Borders::ALL), chunks[1]);
    f.render_widget(Block::default().borders(Borders::ALL), chunks[2]);
}

fn panel(title: &str) -> Block<'_> {
    Block::default()
        .borders(Borders::ALL)
        .title(title)
        .border_style(Style::default().fg(theme::SURFACE2))
}

fn heading(text: impl Into<String>) -> Line<'static> {
    Line::from(Span::styled(
        text.into(),
        Style::default()
            .fg(theme::TEXT)
            .add_modifier(Modifier::BOLD),
    ))
}

fn metric_line(label: &str, value: f32) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            format!("{label:>18}: "),
            Style::default().fg(theme::SUBTEXT1),
        ),
        Span::styled(format!("{value:+.5}"), Style::default().fg(theme::SKY)),
    ])
}

fn format_optional(value: Option<&f32>) -> String {
    value
        .map(|value| format!("{value:+.3}"))
        .unwrap_or_else(|| "n/a".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared::report::{write_report, Report, ScaleKind};
    use std::fs;

    #[test]
    fn observations_page_decodes_report_bin_and_validates_schema() {
        let root = std::env::temp_dir().join(format!(
            "tui-observations-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let path = root.join("observations.report.bin");
        let report = Report {
            title: "Observations".to_owned(),
            x_label: None,
            y_label: None,
            scale: ScaleKind::Linear,
            kind: ReportKind::Observations {
                observation_tickers: vec!["BRK.B".to_owned()],
                action_tickers: vec!["AAPL".to_owned()],
                static_observations: vec![vec![0.0; GLOBAL_STATIC_OBS + PER_TICKER_STATIC_OBS]],
                attention_weights: vec![],
                action_step0: Some(vec![0.25]),
                action_final: Some(vec![0.75]),
            },
        };
        write_report(&path, &report).unwrap();
        let decoded = load_observations(&path).unwrap();
        assert_eq!(decoded.observation_tickers, vec!["BRK.B"]);
        assert_eq!(decoded.action_tickers, vec!["AAPL"]);
        assert_eq!(decoded.static_observations.len(), 1);

        let mut invalid = report;
        let ReportKind::Observations {
            static_observations,
            ..
        } = &mut invalid.kind
        else {
            unreachable!()
        };
        static_observations[0].pop();
        write_report(&path, &invalid).unwrap();
        assert!(load_observations(&path).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn observations_page_uses_the_newest_reportable_generation() {
        let root = std::env::temp_dir().join(format!(
            "tui-observation-generation-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let generations = [0usize, 5, 10, 11]
            .into_iter()
            .map(|number| {
                let path = root.join(number.to_string());
                fs::create_dir_all(&path).unwrap();
                GenerationInfo { number, path }
            })
            .collect::<Vec<_>>();
        fs::write(generations[0].path.join("observations.report.bin"), []).unwrap();
        fs::write(generations[2].path.join("observations.report.bin"), []).unwrap();

        assert_eq!(
            latest_observations_path(&generations),
            Some(generations[2].path.join("observations.report.bin"))
        );
        fs::remove_dir_all(root).unwrap();
    }
}
