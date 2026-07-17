use anyhow::Result;
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph},
    Frame,
};
use ratatui_image::{picker::Picker, protocol::StatefulProtocol, StatefulImage};
use shared::report::{Report, ReportKind};
use std::fs;
use std::path::PathBuf;
use walkdir::WalkDir;

use crate::components::episode_status;
use crate::report_renderer::render_report_with_options;
use crate::utils::clipboard;

#[derive(Debug, Clone)]
pub enum ChartSource {
    Path(PathBuf),
    Report(Box<Report>),
}

impl ChartSource {
    fn report(&self) -> Result<Report> {
        match self {
            ChartSource::Path(path) => load_report(path),
            ChartSource::Report(report) => Ok((**report).clone()),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ChartNode {
    Folder {
        name: String,
        path: PathBuf,
        children: Vec<usize>,
    },
    Chart {
        name: String,
        source: ChartSource,
    },
}

pub struct ChartViewer {
    nodes: Vec<ChartNode>,
    root_indices: Vec<usize>,
    list_state: ListState,
    flattened: Vec<(usize, usize)>, // (node_index, depth)
    expanded: Vec<bool>,
    picker: Picker,
    current_image: Option<Box<dyn StatefulProtocol>>,
    viewing_mode: ViewingMode,
    // Row skip for rendering
    row_skip_input: String,
    editing_row_skip: bool,
    row_skip: usize,
    show_legend: bool,
    solo_series: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
enum ViewingMode {
    Generation(usize), // Episode number
    Inference(usize),  // Inference number
    MetaCharts,        // Meta charts from various episodes
}

impl ChartViewer {
    pub fn new() -> Self {
        let mut picker = Picker::from_termios().unwrap_or_else(|_| Picker::new((8, 12)));
        picker.guess_protocol();

        Self {
            nodes: Vec::new(),
            root_indices: Vec::new(),
            list_state: ListState::default(),
            flattened: Vec::new(),
            expanded: Vec::new(),
            picker,
            current_image: None,
            viewing_mode: ViewingMode::MetaCharts,
            row_skip_input: String::new(),
            editing_row_skip: false,
            row_skip: 0,
            show_legend: true,
            solo_series: None,
        }
    }

    pub fn toggle_legend(&mut self) {
        self.show_legend = !self.show_legend;
        self.load_current_image();
    }

    pub fn toggle_solo_series(&mut self, n: usize) {
        if self.current_series_count().is_some_and(|count| n >= count) {
            self.solo_series = None;
            self.load_current_image();
            return;
        }
        self.solo_series = if self.solo_series == Some(n) {
            None
        } else {
            Some(n)
        };
        self.load_current_image();
    }

    fn current_series_count(&self) -> Option<usize> {
        let i = self.list_state.selected()?;
        if i >= self.flattened.len() {
            return None;
        }
        let (node_idx, _) = self.flattened[i];
        let ChartNode::Chart { source, .. } = &self.nodes[node_idx] else {
            return None;
        };
        let report = source.report().ok()?;
        match report.kind {
            ReportKind::Simple { ema_alpha, .. } => Some(if ema_alpha.is_some() { 2 } else { 1 }),
            ReportKind::MultiLine { series } => Some(series.len()),
            ReportKind::Assets {
                positioned,
                benchmark,
                ..
            } => Some(
                2 + positioned
                    .as_ref()
                    .map_or(0, |p| if p.is_empty() { 0 } else { 1 })
                    + benchmark.as_ref().map_or(0, |_| 1),
            ),
            ReportKind::CandleCompare { .. } => Some(2),
            ReportKind::BuySell { .. } | ReportKind::Observations { .. } => None,
        }
    }

    pub fn is_legend_visible(&self) -> bool {
        self.show_legend
    }

    pub fn is_editing_row_skip(&self) -> bool {
        self.editing_row_skip
    }

    pub fn start_editing_row_skip(&mut self) {
        self.editing_row_skip = true;
    }

    pub fn stop_editing_row_skip(&mut self) {
        self.editing_row_skip = false;
    }

    pub fn cancel_editing_row_skip(&mut self) {
        self.editing_row_skip = false;
        self.row_skip_input.clear();
        self.row_skip = 0;
        self.load_current_image();
    }

    pub fn row_skip_input_push(&mut self, c: char) {
        if c.is_ascii_digit() {
            self.row_skip_input.push(c);
            self.apply_row_skip();
        }
    }

    pub fn row_skip_input_pop(&mut self) {
        self.row_skip_input.pop();
        self.apply_row_skip();
    }

    pub fn get_row_skip_input(&self) -> &str {
        &self.row_skip_input
    }

    pub fn get_row_skip(&self) -> usize {
        self.row_skip
    }

    fn apply_row_skip(&mut self) {
        self.row_skip = self.row_skip_input.parse().unwrap_or(0);
        self.load_current_image();
    }

    pub fn load_generation(&mut self, gen_path: &PathBuf) -> Result<()> {
        self.nodes.clear();
        self.root_indices.clear();
        self.expanded.clear();
        self.current_image = None;

        // Extract episode number from path
        let episode_num = gen_path
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        self.viewing_mode = ViewingMode::Generation(episode_num);

        self.build_tree(gen_path)?;
        self.rebuild_flattened();

        if !self.flattened.is_empty() {
            self.list_state.select(Some(0));
            self.load_current_image();
        }

        Ok(())
    }

    pub fn load_inference(&mut self, infer_path: &PathBuf) -> Result<()> {
        self.nodes.clear();
        self.root_indices.clear();
        self.expanded.clear();
        self.current_image = None;

        // Extract inference number from path
        let infer_num = infer_path
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        self.viewing_mode = ViewingMode::Inference(infer_num);

        self.build_tree(infer_path)?;
        self.rebuild_flattened();

        if !self.flattened.is_empty() {
            self.list_state.select(Some(0));
            self.load_current_image();
        }

        Ok(())
    }

    pub fn load_charts(
        &mut self,
        chart_paths: &[PathBuf],
        extra_reports: Vec<(String, Report)>,
    ) -> Result<()> {
        use std::collections::{HashMap, HashSet};
        use std::time::SystemTime;

        let selected_title = self.selected_report_title();
        self.nodes.clear();
        self.root_indices.clear();
        self.expanded.clear();
        self.current_image = None;
        self.viewing_mode = ViewingMode::MetaCharts;

        let artifact_scores = chart_paths
            .iter()
            .filter_map(|path| {
                let report = load_report(path).ok()?;
                Some((report.title.clone(), report_score(&report)))
            })
            .collect::<HashMap<_, _>>();
        let mut suppressed_artifact_titles = HashSet::new();

        for (name, report) in extra_reports {
            let use_fallback = artifact_scores
                .get(&report.title)
                .is_none_or(|artifact| report_score(&report) > *artifact);
            if !use_fallback {
                continue;
            }
            suppressed_artifact_titles.insert(report.title.clone());
            let chart_idx = self.nodes.len();
            self.nodes.push(ChartNode::Chart {
                name,
                source: ChartSource::Report(Box::new(report)),
            });
            self.expanded.push(false);
            self.root_indices.push(chart_idx);
        }

        // Group charts by ticker (None for episode-level charts)
        // Store (path, chart_name, episode_num, modified_time)
        let mut ticker_groups: HashMap<
            Option<String>,
            Vec<(PathBuf, String, Option<usize>, SystemTime)>,
        > = HashMap::new();
        let mut pretrain_sample_groups: HashMap<
            String,
            Vec<(PathBuf, String, Option<usize>, SystemTime)>,
        > = HashMap::new();
        let mut candle_snapshot_charts: Vec<(PathBuf, String, SystemTime)> = Vec::new();

        for path in chart_paths {
            if path.exists() {
                if report_title_from_path(path)
                    .is_some_and(|title| suppressed_artifact_titles.contains(&title))
                {
                    continue;
                }
                // Get modification time
                let modified = path
                    .metadata()
                    .and_then(|m| m.modified())
                    .unwrap_or(SystemTime::UNIX_EPOCH);

                // Extract episode number, ticker, and chart name from report path
                // Expected: gens/123/chart.report.bin or gens/123/TICKER/chart.report.bin
                let parent = path.parent();
                let chart_name =
                    report_title_from_path(path).unwrap_or_else(|| chart_name_from_path(path));
                let chart_name = normalize_title(&chart_name);

                let (episode_num, ticker) = if let Some(parent) = parent {
                    if let Some(parent_name) = parent.file_name().and_then(|n| n.to_str()) {
                        if let Ok(ep) = parent_name.parse::<usize>() {
                            (Some(ep), None)
                        } else if parent_name == "candle_snapshots" {
                            candle_snapshot_charts.push((
                                path.clone(),
                                chart_name.clone(),
                                modified,
                            ));
                            (None, Some("candle snapshots".to_string()))
                        } else if parent_name == "samples" {
                            let chart_parent = parent.parent();
                            if let Some(chart_parent) = chart_parent {
                                if let Some(ep_name) =
                                    chart_parent.file_name().and_then(|n| n.to_str())
                                {
                                    if let Ok(ep) = ep_name.parse::<usize>() {
                                        if let Some((sample_key, chart_name)) =
                                            pretrain_sample_parts(path, Some(ep))
                                        {
                                            pretrain_sample_groups
                                                .entry(sample_key)
                                                .or_insert_with(Vec::new)
                                                .push((
                                                    path.clone(),
                                                    chart_name,
                                                    Some(ep),
                                                    modified,
                                                ));
                                            (Some(ep), Some("pretrain samples".to_string()))
                                        } else {
                                            (Some(ep), Some("pretrain samples".to_string()))
                                        }
                                    } else {
                                        (None, None)
                                    }
                                } else {
                                    (None, None)
                                }
                            } else {
                                (None, None)
                            }
                        } else if is_ticker_name(parent_name) {
                            let chart_parent = parent.parent();
                            if let Some(chart_parent) = chart_parent {
                                if let Some(ep_name) =
                                    chart_parent.file_name().and_then(|n| n.to_str())
                                {
                                    if let Ok(ep) = ep_name.parse::<usize>() {
                                        (Some(ep), Some(parent_name.to_string()))
                                    } else {
                                        (None, None)
                                    }
                                } else {
                                    (None, None)
                                }
                            } else {
                                (None, None)
                            }
                        } else {
                            (None, None)
                        }
                    } else {
                        (None, None)
                    }
                } else {
                    (None, None)
                };

                if episode_num.is_none() && ticker.is_none() {
                    continue;
                }
                if ticker
                    .as_deref()
                    .is_some_and(|name| name != "pretrain samples" && name != "candle snapshots")
                {
                    ticker_groups.entry(ticker).or_insert_with(Vec::new).push((
                        path.clone(),
                        chart_name,
                        episode_num,
                        modified,
                    ));
                } else if ticker.is_none() {
                    ticker_groups.entry(ticker).or_insert_with(Vec::new).push((
                        path.clone(),
                        chart_name,
                        episode_num,
                        modified,
                    ));
                }
            }
        }

        // Add episode-level charts first (no ticker)
        if let Some(mut episode_charts) = ticker_groups.remove(&None) {
            // Sort by modification time (most recent first)
            episode_charts.sort_by(|a, b| b.3.cmp(&a.3));

            for (path, chart_name, episode_num, _) in episode_charts {
                let name = if let Some(ep) = episode_num {
                    format!("{} (ep {})", chart_name, ep)
                } else {
                    chart_name
                };

                let chart_idx = self.nodes.len();
                self.nodes.push(ChartNode::Chart {
                    name,
                    source: ChartSource::Path(path.clone()),
                });
                self.expanded.push(false);
                self.root_indices.push(chart_idx);
            }
        }

        if !pretrain_sample_groups.is_empty() {
            let mut sample_infos: Vec<(String, SystemTime)> = pretrain_sample_groups
                .iter()
                .map(|(name, charts)| {
                    let most_recent = charts
                        .iter()
                        .map(|(_, _, _, modified)| *modified)
                        .max()
                        .unwrap_or(SystemTime::UNIX_EPOCH);
                    (name.clone(), most_recent)
                })
                .collect();
            sample_infos.sort_by(|a, b| b.1.cmp(&a.1));

            let mut sample_chart_indices = Vec::new();
            for (sample_key, _) in sample_infos {
                if let Some(mut charts) = pretrain_sample_groups.remove(&sample_key) {
                    charts.sort_by(|a, b| a.1.cmp(&b.1));
                    let sample_name = sample_key
                        .split_once('|')
                        .map(|(_, display)| display.to_string())
                        .unwrap_or_else(|| sample_key.clone());
                    for (path, chart_name, _, _) in charts {
                        let chart_idx = self.nodes.len();
                        self.nodes.push(ChartNode::Chart {
                            name: format!("{sample_name} - {chart_name}"),
                            source: ChartSource::Path(path),
                        });
                        self.expanded.push(false);
                        sample_chart_indices.push(chart_idx);
                    }
                }
            }

            let root_idx = self.nodes.len();
            self.nodes.push(ChartNode::Folder {
                name: "pretrain samples".to_string(),
                path: PathBuf::new(),
                children: sample_chart_indices,
            });
            self.expanded.push(false);
            self.root_indices.push(root_idx);
        }

        if !candle_snapshot_charts.is_empty() {
            candle_snapshot_charts.sort_by(|a, b| a.1.cmp(&b.1));
            let mut children = Vec::new();
            for (path, chart_name, _) in candle_snapshot_charts {
                let chart_idx = self.nodes.len();
                self.nodes.push(ChartNode::Chart {
                    name: chart_name,
                    source: ChartSource::Path(path),
                });
                self.expanded.push(false);
                children.push(chart_idx);
            }
            let root_idx = self.nodes.len();
            self.nodes.push(ChartNode::Folder {
                name: "candle snapshots".to_string(),
                path: PathBuf::new(),
                children,
            });
            self.expanded.push(false);
            self.root_indices.push(root_idx);
        }

        // Create folders for each ticker, sorted by most recent modification time
        let mut ticker_info: Vec<(String, SystemTime)> = ticker_groups
            .iter()
            .filter_map(|(ticker_opt, charts)| {
                ticker_opt.as_ref().map(|ticker| {
                    // Get the most recent modification time for this ticker
                    let most_recent = charts
                        .iter()
                        .map(|(_, _, _, modified)| *modified)
                        .max()
                        .unwrap_or(SystemTime::UNIX_EPOCH);
                    (ticker.clone(), most_recent)
                })
            })
            .collect();

        // Sort tickers by modification time (most recent first)
        ticker_info.sort_by(|a, b| b.1.cmp(&a.1));

        for (ticker_name, _) in ticker_info {
            if let Some(mut charts) = ticker_groups.remove(&Some(ticker_name.clone())) {
                let mut children = Vec::new();

                // Sort charts within ticker by modification time (most recent first)
                charts.sort_by(|a, b| b.3.cmp(&a.3));

                // Add all charts for this ticker
                for (path, chart_name, episode_num, _) in charts {
                    let name = if let Some(ep) = episode_num {
                        format!("{} (ep {})", chart_name, ep)
                    } else {
                        chart_name
                    };

                    let chart_idx = self.nodes.len();
                    self.nodes.push(ChartNode::Chart {
                        name,
                        source: ChartSource::Path(path.clone()),
                    });
                    self.expanded.push(false);
                    children.push(chart_idx);
                }

                // Create the folder node
                let folder_idx = self.nodes.len();
                self.nodes.push(ChartNode::Folder {
                    name: ticker_name.clone(),
                    path: PathBuf::new(), // Dummy path for folders
                    children,
                });
                self.expanded.push(false);
                self.root_indices.push(folder_idx);
            }
        }

        self.rebuild_flattened();

        if !self.flattened.is_empty() {
            let selected = selected_title
                .as_deref()
                .and_then(|title| self.flattened_report_position(title))
                .unwrap_or(0);
            self.list_state.select(Some(selected));
            self.load_current_image();
        }

        Ok(())
    }

    fn selected_report_title(&self) -> Option<String> {
        let selected = self.list_state.selected()?;
        let (node, _) = *self.flattened.get(selected)?;
        let ChartNode::Chart { source, .. } = self.nodes.get(node)? else {
            return None;
        };
        source.report().ok().map(|report| report.title)
    }

    fn flattened_report_position(&self, title: &str) -> Option<usize> {
        self.flattened.iter().position(|(node, _)| {
            let Some(ChartNode::Chart { source, .. }) = self.nodes.get(*node) else {
                return false;
            };
            source.report().is_ok_and(|report| report.title == title)
        })
    }

    fn build_tree(&mut self, path: &PathBuf) -> Result<()> {
        use std::time::SystemTime;

        let mut folders = Vec::new();
        let mut charts = Vec::new();

        for entry in WalkDir::new(path)
            .min_depth(1)
            .max_depth(1)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let entry_path = entry.path().to_path_buf();
            let name = entry.file_name().to_str().unwrap_or("unknown").to_string();

            if entry.file_type().is_dir() {
                let children = if name == "samples" {
                    self.build_pretrain_sample_tree(&entry_path)
                } else {
                    let mut children = Vec::new();
                    for sub_entry in WalkDir::new(&entry_path)
                        .min_depth(1)
                        .max_depth(1)
                        .into_iter()
                        .filter_map(|e| e.ok())
                    {
                        if !sub_entry.file_type().is_file() {
                            continue;
                        }
                        let file_name = sub_entry.file_name().to_str().unwrap_or("unknown");
                        if !file_name.ends_with(".report.bin") {
                            continue;
                        }
                        let chart_idx = self.nodes.len();
                        self.nodes.push(ChartNode::Chart {
                            name: report_display_name(file_name),
                            source: ChartSource::Path(sub_entry.path().to_path_buf()),
                        });
                        children.push(chart_idx);
                        self.expanded.push(false);
                    }
                    children
                };

                // Get modification time for sorting
                let modified = entry
                    .path()
                    .metadata()
                    .and_then(|m| m.modified())
                    .unwrap_or(SystemTime::UNIX_EPOCH);

                let folder_idx = self.nodes.len();
                self.nodes.push(ChartNode::Folder {
                    name: name.clone(),
                    path: entry_path,
                    children,
                });
                self.expanded.push(false);
                folders.push((folder_idx, modified));
            } else if entry.file_type().is_file() && name.ends_with(".report.bin") {
                let chart_idx = self.nodes.len();
                self.nodes.push(ChartNode::Chart {
                    name: report_display_name(&name),
                    source: ChartSource::Path(entry_path),
                });
                self.expanded.push(false);
                charts.push(chart_idx);
            }
        }

        // Sort folders by modification time (most recent first)
        folders.sort_by(|a, b| b.1.cmp(&a.1));

        self.root_indices.extend(charts);
        self.root_indices
            .extend(folders.into_iter().map(|(idx, _)| idx));

        Ok(())
    }

    fn build_pretrain_sample_tree(&mut self, path: &PathBuf) -> Vec<usize> {
        use std::collections::HashMap;
        use std::time::SystemTime;

        let mut groups: HashMap<String, Vec<(PathBuf, String, SystemTime)>> = HashMap::new();
        for entry in WalkDir::new(path)
            .min_depth(1)
            .max_depth(1)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if !entry.file_type().is_file() {
                continue;
            }
            let entry_path = entry.path().to_path_buf();
            let Some((sample_key, chart_name)) = pretrain_sample_parts(&entry_path, None) else {
                continue;
            };
            let modified = entry
                .path()
                .metadata()
                .and_then(|m| m.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH);
            groups
                .entry(sample_key)
                .or_insert_with(Vec::new)
                .push((entry_path, chart_name, modified));
        }

        let mut sample_infos = groups
            .iter()
            .map(|(key, charts)| {
                let modified = charts
                    .iter()
                    .map(|(_, _, modified)| *modified)
                    .max()
                    .unwrap_or(SystemTime::UNIX_EPOCH);
                (key.clone(), modified)
            })
            .collect::<Vec<_>>();
        sample_infos.sort_by(|a, b| a.0.cmp(&b.0));

        let mut chart_indices = Vec::new();
        for (sample_key, _) in sample_infos {
            let Some(mut charts) = groups.remove(&sample_key) else {
                continue;
            };
            charts.sort_by(|a, b| a.1.cmp(&b.1));
            let sample_name = sample_key
                .split_once('|')
                .map(|(_, display)| display.to_string())
                .unwrap_or_else(|| sample_key.clone());
            for (path, chart_name, _) in charts {
                let chart_idx = self.nodes.len();
                self.nodes.push(ChartNode::Chart {
                    name: format!("{sample_name} - {chart_name}"),
                    source: ChartSource::Path(path),
                });
                self.expanded.push(false);
                chart_indices.push(chart_idx);
            }
        }

        chart_indices
    }

    fn rebuild_flattened(&mut self) {
        self.flattened.clear();

        // Iterate by index to avoid cloning
        for i in 0..self.root_indices.len() {
            let idx = self.root_indices[i];
            self.add_to_flattened(idx, 0);
        }
    }

    fn add_to_flattened(&mut self, idx: usize, depth: usize) {
        self.flattened.push((idx, depth));

        // Check if we should expand children
        let should_expand =
            matches!(&self.nodes[idx], ChartNode::Folder { .. }) && self.expanded[idx];

        if should_expand {
            // Get children count first to avoid borrow issues
            let children_count = if let ChartNode::Folder { children, .. } = &self.nodes[idx] {
                children.len()
            } else {
                0
            };

            // Now iterate using the count
            for i in 0..children_count {
                let child_idx = if let ChartNode::Folder { children, .. } = &self.nodes[idx] {
                    children[i]
                } else {
                    continue;
                };
                self.add_to_flattened(child_idx, depth + 1);
            }
        }
    }

    fn load_current_image(&mut self) {
        self.current_image = None;

        if let Some(i) = self.list_state.selected() {
            if i < self.flattened.len() {
                let (node_idx, _) = self.flattened[i];
                if let ChartNode::Chart { source, .. } = &self.nodes[node_idx] {
                    if let Ok(report) = source.report() {
                        let skip = if self.viewing_mode == ViewingMode::MetaCharts {
                            self.row_skip
                        } else {
                            0
                        };
                        if let Ok(img) = render_report_with_options(
                            &report,
                            skip,
                            self.show_legend,
                            self.solo_series,
                        ) {
                            let protocol = self.picker.new_resize_protocol(img);
                            self.current_image = Some(protocol);
                        }
                    }
                }
            }
        }
    }

    pub fn next(&mut self) {
        if self.flattened.is_empty() {
            return;
        }
        let i = match self.list_state.selected() {
            Some(i) => {
                if i >= self.flattened.len() - 1 {
                    0
                } else {
                    i + 1
                }
            }
            None => 0,
        };
        self.list_state.select(Some(i));
        self.solo_series = None;
        self.load_current_image();
    }

    pub fn previous(&mut self) {
        if self.flattened.is_empty() {
            return;
        }
        let i = match self.list_state.selected() {
            Some(i) => {
                if i == 0 {
                    self.flattened.len() - 1
                } else {
                    i - 1
                }
            }
            None => 0,
        };
        self.list_state.select(Some(i));
        self.solo_series = None;
        self.load_current_image();
    }

    pub fn toggle_expand(&mut self) {
        if let Some(i) = self.list_state.selected() {
            if i < self.flattened.len() {
                let (node_idx, _) = self.flattened[i];
                if matches!(self.nodes[node_idx], ChartNode::Folder { .. }) {
                    self.expanded[node_idx] = !self.expanded[node_idx];
                    self.rebuild_flattened();
                }
            }
        }
    }

    pub fn scroll_up(&mut self, amount: usize) {
        for _ in 0..amount {
            self.previous();
        }
    }

    pub fn scroll_down(&mut self, amount: usize) {
        for _ in 0..amount {
            self.next();
        }
    }

    pub fn copy_current_image(&self) -> Result<()> {
        if let Some(i) = self.list_state.selected() {
            if i < self.flattened.len() {
                let (node_idx, _) = self.flattened[i];
                if let ChartNode::Chart { source, .. } = &self.nodes[node_idx] {
                    if let Ok(report) = source.report() {
                        let skip = if self.viewing_mode == ViewingMode::MetaCharts {
                            self.row_skip
                        } else {
                            0
                        };
                        let temp_path = render_report_to_temp(&report, skip, self.show_legend)?;
                        clipboard::copy_image_to_clipboard(&temp_path)?;
                    } else if let ChartSource::Path(path) = source {
                        clipboard::copy_image_to_clipboard(path)?;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn is_viewing_meta_charts(&self) -> bool {
        self.viewing_mode == ViewingMode::MetaCharts
    }

    pub fn render(
        &mut self,
        f: &mut Frame,
        is_training: bool,
        current_episode: Option<usize>,
        has_progress: bool,
    ) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(25), Constraint::Percentage(75)])
            .split(f.area());

        self.render_list(f, chunks[0], is_training, current_episode, has_progress);
        self.render_preview(f, chunks[1]);
    }

    fn render_list(
        &mut self,
        f: &mut Frame,
        area: Rect,
        is_training: bool,
        current_episode: Option<usize>,
        has_progress: bool,
    ) {
        let show_filter = self.viewing_mode == ViewingMode::MetaCharts;

        let chunks = if show_filter {
            Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3),
                    Constraint::Length(3),
                    Constraint::Min(0),
                    Constraint::Length(4),
                ])
                .split(area)
        } else {
            Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3),
                    Constraint::Min(0),
                    Constraint::Length(4),
                ])
                .split(area)
        };

        let title = match &self.viewing_mode {
            ViewingMode::Generation(ep) => {
                let mut title_spans = vec![Span::styled(
                    format!(" Episode {} Charts ", ep),
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                )];
                title_spans.extend(episode_status::episode_status_spans(
                    is_training,
                    current_episode,
                    has_progress,
                ));
                Paragraph::new(Line::from(title_spans))
            }
            ViewingMode::Inference(num) => {
                let mut title_spans = vec![Span::styled(
                    format!(" Inference {} Charts ", num),
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                )];
                title_spans.extend(episode_status::episode_status_spans(
                    is_training,
                    current_episode,
                    has_progress,
                ));
                Paragraph::new(Line::from(title_spans))
            }
            ViewingMode::MetaCharts => {
                let mut title_spans = vec![Span::styled(
                    " Meta Charts ",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                )];
                title_spans.extend(episode_status::episode_status_spans(
                    is_training,
                    current_episode,
                    has_progress,
                ));
                Paragraph::new(Line::from(title_spans))
            }
        };

        let title_widget = title.block(Block::default().borders(Borders::ALL));
        f.render_widget(title_widget, chunks[0]);

        // Render row skip input for MetaCharts mode
        let (list_chunk, help_chunk) = if show_filter {
            let filter_style = if self.editing_row_skip {
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::DarkGray)
            };

            let filter_text = if self.editing_row_skip {
                format!("Skip: {}_", self.row_skip_input)
            } else if self.row_skip_input.is_empty() {
                "Skip: (press / to set)".to_string()
            } else {
                format!("Skip: {}", self.row_skip_input)
            };

            let filter = Paragraph::new(filter_text)
                .style(filter_style)
                .block(Block::default().borders(Borders::ALL).title("Row Skip"));
            f.render_widget(filter, chunks[1]);

            (chunks[2], chunks[3])
        } else {
            (chunks[1], chunks[2])
        };

        let items: Vec<ListItem> = self
            .flattened
            .iter()
            .map(|(node_idx, depth)| {
                let indent = "  ".repeat(*depth);
                let (text, style) = match &self.nodes[*node_idx] {
                    ChartNode::Folder { name, children, .. } => {
                        let icon = if self.expanded[*node_idx] {
                            "▼"
                        } else {
                            "▶"
                        };
                        let label =
                            format!("{}{} {} ({} items)", indent, icon, name, children.len());
                        (
                            label,
                            Style::default()
                                .fg(Color::Yellow)
                                .add_modifier(Modifier::BOLD),
                        )
                    }
                    ChartNode::Chart { name, .. } => {
                        let label = format!("{}  {}", indent, name);
                        (label, Style::default().fg(Color::White))
                    }
                };
                ListItem::new(text).style(style)
            })
            .collect();

        let list = List::new(items)
            .block(Block::default().borders(Borders::ALL).title("Charts"))
            .highlight_style(
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol(">> ");

        f.render_stateful_widget(list, list_chunk, &mut self.list_state);

        let legend_label = if self.show_legend {
            "Legend"
        } else {
            "Legend (off)"
        };
        let solo_label = match self.solo_series {
            Some(n) => format!("Solo ({})", n + 1),
            None => "Solo".to_string(),
        };
        let (help_line1, help_line2) = if self.viewing_mode == ViewingMode::MetaCharts {
            (
                Line::from(vec![
                    Span::styled("↑/k", Style::default().fg(Color::Cyan)),
                    Span::raw(": Up  "),
                    Span::styled("↓/j", Style::default().fg(Color::Cyan)),
                    Span::raw(": Down  "),
                    Span::styled("Enter", Style::default().fg(Color::Green)),
                    Span::raw(": Expand  "),
                    Span::styled("c", Style::default().fg(Color::Magenta)),
                    Span::raw(": Copy  "),
                    Span::styled("q/Esc", Style::default().fg(Color::Red)),
                    Span::raw(": Back"),
                ]),
                Line::from(vec![
                    Span::styled("/", Style::default().fg(Color::Yellow)),
                    Span::raw(": Filter  "),
                    Span::styled("l", Style::default().fg(Color::Yellow)),
                    Span::raw(format!(": {}  ", legend_label)),
                    Span::styled("1-9", Style::default().fg(Color::Yellow)),
                    Span::raw(format!(": {}  ", solo_label)),
                    Span::styled("r", Style::default().fg(Color::Yellow)),
                    Span::raw(": Refresh"),
                ]),
            )
        } else {
            (
                Line::from(vec![
                    Span::styled("↑/k", Style::default().fg(Color::Cyan)),
                    Span::raw(": Up  "),
                    Span::styled("↓/j", Style::default().fg(Color::Cyan)),
                    Span::raw(": Down  "),
                    Span::styled("Enter", Style::default().fg(Color::Green)),
                    Span::raw(": Expand/Collapse  "),
                    Span::styled("c", Style::default().fg(Color::Magenta)),
                    Span::raw(": Copy  "),
                    Span::styled("q/Esc", Style::default().fg(Color::Red)),
                    Span::raw(": Back"),
                ]),
                Line::from(vec![
                    Span::styled("l", Style::default().fg(Color::Yellow)),
                    Span::raw(format!(": {}  ", legend_label)),
                    Span::styled("1-9", Style::default().fg(Color::Yellow)),
                    Span::raw(format!(": {}", solo_label)),
                ]),
            )
        };

        let help = Paragraph::new(vec![help_line1, help_line2])
            .block(Block::default().borders(Borders::ALL).title("Controls"));
        f.render_widget(help, help_chunk);
    }

    fn render_preview(&mut self, f: &mut Frame, area: Rect) {
        let block = Block::default().borders(Borders::ALL).title("Preview");
        let inner = block.inner(area);
        f.render_widget(block, area);

        if let Some(ref mut protocol) = self.current_image {
            let image = StatefulImage::new(None);
            f.render_stateful_widget(image, inner, protocol);
        } else {
            let selected_is_folder = self
                .list_state
                .selected()
                .and_then(|i| {
                    if i < self.flattened.len() {
                        let (node_idx, _) = self.flattened[i];
                        Some(matches!(self.nodes[node_idx], ChartNode::Folder { .. }))
                    } else {
                        None
                    }
                })
                .unwrap_or(false);

            let msg = if selected_is_folder {
                "Folders cannot be previewed - expand to view charts"
            } else {
                "Select a chart to preview"
            };

            let no_preview = Paragraph::new(msg).style(Style::default().fg(Color::DarkGray));
            f.render_widget(no_preview, inner);
        }
    }
}

fn load_report(path: &PathBuf) -> Result<Report> {
    let bytes = fs::read(path)?;
    let report = postcard::from_bytes(&bytes)?;
    Ok(report)
}

fn report_display_name(name: &str) -> String {
    let trimmed = name.strip_suffix(".report.bin").unwrap_or(name);
    normalize_title(&trimmed.replace('_', " "))
}

fn chart_name_from_path(path: &PathBuf) -> String {
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown");
    report_display_name(file_name)
}

fn normalize_title(name: &str) -> String {
    let mut parts = Vec::new();
    for word in name.split_whitespace() {
        if word.eq_ignore_ascii_case("log") {
            parts.push("(Log)".to_string());
            continue;
        }
        let mut chars = word.chars();
        if let Some(first) = chars.next() {
            let rest = chars.as_str().to_ascii_lowercase();
            let mut word_out = String::new();
            word_out.push(first.to_ascii_uppercase());
            word_out.push_str(&rest);
            parts.push(word_out);
        }
    }
    parts.join(" ")
}

fn is_ticker_name(name: &str) -> bool {
    let mut has_alpha = false;
    for c in name.chars() {
        if c.is_ascii_alphabetic() {
            has_alpha = true;
            if !c.is_ascii_uppercase() {
                return false;
            }
        } else if !c.is_ascii_digit() {
            return false;
        }
    }
    has_alpha
}

fn report_title_from_path(path: &PathBuf) -> Option<String> {
    let bytes = fs::read(path).ok()?;
    let report: Report = postcard::from_bytes(&bytes).ok()?;
    Some(report.title)
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ReportScore {
    latest_finite_index: Option<usize>,
    axis_semantics_current: bool,
    series_count: usize,
}

fn report_score(report: &Report) -> ReportScore {
    let series = match &report.kind {
        ReportKind::Simple { values, .. } => vec![values.as_slice()],
        ReportKind::MultiLine { series } => series
            .iter()
            .map(|series| series.values.as_slice())
            .collect(),
        ReportKind::Assets {
            total,
            cash,
            positioned,
            benchmark,
        } => {
            let mut series = vec![total.as_slice(), cash.as_slice()];
            if let Some(positioned) = positioned {
                series.push(positioned);
            }
            if let Some(benchmark) = benchmark {
                series.push(benchmark);
            }
            series
        }
        _ => Vec::new(),
    };
    let latest_finite_index = series
        .iter()
        .filter_map(|values| values.iter().rposition(|value| value.is_finite()))
        .max();
    let series_count = series
        .iter()
        .filter(|values| values.iter().any(|value| value.is_finite()))
        .count();
    let axis_semantics_current = !(report.title.contains("Outperformance")
        && !report.title.contains("Fraction")
        && matches!(
            &report.kind,
            ReportKind::MultiLine { series }
                if series.iter().any(|series| series.label.to_ascii_lowercase().contains("fraction"))
        ));
    ReportScore {
        latest_finite_index,
        axis_semantics_current,
        series_count,
    }
}

fn pretrain_sample_parts(path: &PathBuf, episode: Option<usize>) -> Option<(String, String)> {
    let file_name = path.file_name()?.to_str()?;
    let base = file_name.strip_suffix(".report.bin")?;
    let parts = base.split('_').collect::<Vec<_>>();
    if parts.len() < 3 {
        return None;
    }
    let kind = parts[0];
    if kind != "sample" && kind != "worst" {
        return None;
    }
    let number = parts[1];
    let suffix = parts[2..].join("_");
    let folder_display = match episode {
        Some(ep) => format!("{} {} (ep {})", kind, number, ep),
        None => format!("{} {}", kind, number),
    };
    let key = match episode {
        Some(ep) => format!("{ep}:{kind}_{number}|{folder_display}"),
        None => format!("{kind}_{number}|{folder_display}"),
    };
    let chart_name = match suffix.as_str() {
        "deltas" => "Returns".to_string(),
        "candles" => "Candles".to_string(),
        _ => normalize_title(&suffix.replace('_', " ")),
    };
    Some((key, chart_name))
}

fn render_report_to_temp(report: &Report, skip: usize, show_legend: bool) -> Result<PathBuf> {
    let image = render_report_with_options(report, skip, show_legend, None)?;
    let mut path = std::env::temp_dir();
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    path.push(format!("report_chart_{stamp}.png"));
    image.save(&path)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared::report::{ReportSeries, ScaleKind};

    fn multiline(title: &str, labels: &[&str]) -> Report {
        Report {
            title: title.to_owned(),
            x_label: Some("update".to_owned()),
            y_label: Some("value".to_owned()),
            scale: ScaleKind::Linear,
            kind: ReportKind::MultiLine {
                series: labels
                    .iter()
                    .map(|label| ReportSeries {
                        label: (*label).to_owned(),
                        values: vec![1.0],
                    })
                    .collect(),
            },
        }
    }

    #[test]
    fn meta_refresh_preserves_selected_report_title() {
        let mut viewer = ChartViewer::new();
        viewer
            .load_charts(
                &[],
                vec![
                    ("A".to_owned(), multiline("A", &["a"])),
                    ("B".to_owned(), multiline("B", &["b"])),
                ],
            )
            .unwrap();
        viewer.next();
        assert_eq!(viewer.selected_report_title().as_deref(), Some("B"));

        viewer
            .load_charts(
                &[],
                vec![
                    ("A".to_owned(), multiline("A", &["a"])),
                    ("B".to_owned(), multiline("B", &["b", "new"])),
                ],
            )
            .unwrap();

        assert_eq!(viewer.selected_report_title().as_deref(), Some("B"));
        assert_eq!(viewer.current_series_count(), Some(2));
    }

    #[test]
    fn canonical_artifact_dedup_prefers_equally_current_artifact() {
        let root = std::env::temp_dir().join(format!(
            "chart-viewer-pretrain-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let generation = root.join("gens/1");
        fs::create_dir_all(&generation).unwrap();
        let artifact_path = generation.join("outperformance.report.bin");
        fs::write(
            &artifact_path,
            postcard::to_stdvec(&multiline("Pretrain Probe", &["train", "validation"])).unwrap(),
        )
        .unwrap();
        let csv_fallback = multiline("Pretrain Probe", &["train", "validation"]);
        let mut viewer = ChartViewer::new();

        viewer
            .load_charts(
                &[artifact_path],
                vec![("Pretrain Probe".to_owned(), csv_fallback)],
            )
            .unwrap();

        assert_eq!(viewer.current_series_count(), Some(2));
        let selected = viewer.list_state.selected().unwrap();
        let (node, _) = viewer.flattened[selected];
        let ChartNode::Chart { source, .. } = &viewer.nodes[node] else {
            panic!("expected chart");
        };
        assert!(matches!(source, ChartSource::Path(_)));
        assert_eq!(
            viewer
                .nodes
                .iter()
                .filter(|node| matches!(node, ChartNode::Chart { .. }))
                .count(),
            1
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn newer_pretrain_fallback_replaces_stale_artifact() {
        let root = std::env::temp_dir().join(format!(
            "chart-viewer-stale-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let generation = root.join("gens/1");
        fs::create_dir_all(&generation).unwrap();
        let artifact_path = generation.join("reward.report.bin");
        fs::write(
            &artifact_path,
            postcard::to_stdvec(&multiline("Pretrain Loss", &["train"])).unwrap(),
        )
        .unwrap();
        let mut fallback = multiline("Pretrain Loss", &["train"]);
        let ReportKind::MultiLine { series } = &mut fallback.kind else {
            unreachable!();
        };
        series[0].values = vec![1.0, 2.0, 3.0];
        let mut viewer = ChartViewer::new();

        viewer
            .load_charts(
                &[artifact_path],
                vec![("Pretrain Loss".to_owned(), fallback)],
            )
            .unwrap();

        let selected = viewer.list_state.selected().unwrap();
        let (node, _) = viewer.flattened[selected];
        let ChartNode::Chart { source, .. } = &viewer.nodes[node] else {
            panic!("expected chart");
        };
        assert!(matches!(source, ChartSource::Report(_)));
        assert_eq!(source.report().unwrap().kind.to_lines().len(), 3);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn equally_fresh_richer_fallback_outranks_incomplete_artifact() {
        let incomplete = multiline("Pretrain Loss", &["train"]);
        let richer = multiline("Pretrain Loss", &["train", "validation"]);
        assert!(report_score(&richer) > report_score(&incomplete));
    }
}
