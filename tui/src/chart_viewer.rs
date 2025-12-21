use anyhow::Result;
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph},
    Frame,
};
use ratatui_image::{picker::Picker, protocol::StatefulProtocol, StatefulImage};
use shared::report::Report;
use std::fs;
use std::path::PathBuf;
use walkdir::WalkDir;

use crate::components::episode_status;
use crate::report_renderer::render_report;
use crate::utils::clipboard;

#[derive(Debug, Clone)]
pub enum ChartNode {
    Folder { name: String, path: PathBuf, children: Vec<usize> },
    Chart { name: String, path: PathBuf },
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
}

#[derive(Debug, Clone, PartialEq)]
enum ViewingMode {
    Generation(usize),  // Episode number
    Inference(usize),   // Inference number
    MetaCharts,         // Meta charts from various episodes
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
        }
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

    pub fn load_charts(&mut self, chart_paths: &[PathBuf]) -> Result<()> {
        use std::collections::HashMap;
        use std::time::SystemTime;

        self.nodes.clear();
        self.root_indices.clear();
        self.expanded.clear();
        self.current_image = None;
        self.viewing_mode = ViewingMode::MetaCharts;

        // Group charts by ticker (None for episode-level charts)
        // Store (path, chart_name, episode_num, modified_time)
        let mut ticker_groups: HashMap<Option<String>, Vec<(PathBuf, String, Option<usize>, SystemTime)>> = HashMap::new();

        for path in chart_paths {
            if path.exists() {
                // Get modification time
                let modified = path
                    .metadata()
                    .and_then(|m| m.modified())
                    .unwrap_or(SystemTime::UNIX_EPOCH);

                // Extract episode number, ticker, and chart name from report path
                // Expected: gens/123/chart.report.bin or gens/123/TICKER/chart.report.bin
                let parent = path.parent();
                let chart_name = report_title_from_path(path)
                    .unwrap_or_else(|| chart_name_from_path(path));
                let chart_name = normalize_title(&chart_name);

                let (episode_num, ticker) = if let Some(parent) = parent {
                    if let Some(parent_name) = parent.file_name().and_then(|n| n.to_str()) {
                        if let Ok(ep) = parent_name.parse::<usize>() {
                            (Some(ep), None)
                        } else if is_ticker_name(parent_name) {
                            let chart_parent = parent.parent();
                            if let Some(chart_parent) = chart_parent {
                                if let Some(ep_name) = chart_parent.file_name().and_then(|n| n.to_str())
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

                ticker_groups.entry(ticker)
                    .or_insert_with(Vec::new)
                    .push((path.clone(), chart_name, episode_num, modified));
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
                    path: path.clone(),
                });
                self.expanded.push(false);
                self.root_indices.push(chart_idx);
            }
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
                        path: path.clone(),
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
            self.list_state.select(Some(0));
            self.load_current_image();
        }

        Ok(())
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
            let name = entry
                .file_name()
                .to_str()
                .unwrap_or("unknown")
                .to_string();

            if entry.file_type().is_dir() {
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
                        path: sub_entry.path().to_path_buf(),
                    });
                    children.push(chart_idx);
                    self.expanded.push(false);
                }

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
                    path: entry_path,
                });
                self.expanded.push(false);
                charts.push(chart_idx);
            }
        }

        // Sort folders by modification time (most recent first)
        folders.sort_by(|a, b| b.1.cmp(&a.1));

        self.root_indices.extend(charts);
        self.root_indices.extend(folders.into_iter().map(|(idx, _)| idx));

        Ok(())
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
        let should_expand = matches!(&self.nodes[idx], ChartNode::Folder { .. }) && self.expanded[idx];

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
                if let ChartNode::Chart { path, .. } = &self.nodes[node_idx] {
                    if let Ok(report) = load_report(path) {
                        if let Ok(img) = render_report(&report) {
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
                if let ChartNode::Chart { path, .. } = &self.nodes[node_idx] {
                    if let Ok(report) = load_report(path) {
                        let temp_path = render_report_to_temp(&report)?;
                        clipboard::copy_image_to_clipboard(&temp_path)?;
                    } else {
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

    pub fn render(&mut self, f: &mut Frame, is_training: bool, current_episode: Option<usize>) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(25), Constraint::Percentage(75)])
            .split(f.area());

        self.render_list(f, chunks[0], is_training, current_episode);
        self.render_preview(f, chunks[1]);
    }

    fn render_list(&mut self, f: &mut Frame, area: Rect, is_training: bool, current_episode: Option<usize>) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(0), Constraint::Length(3)])
            .split(area);

        let title = match &self.viewing_mode {
            ViewingMode::Generation(ep) => {
                let mut title_spans = vec![
                    Span::styled(format!(" Episode {} Charts ", ep), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                ];
                title_spans.extend(episode_status::episode_status_spans(is_training, current_episode));
                Paragraph::new(Line::from(title_spans))
            }
            ViewingMode::Inference(num) => {
                let mut title_spans = vec![
                    Span::styled(format!(" Inference {} Charts ", num), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                ];
                title_spans.extend(episode_status::episode_status_spans(is_training, current_episode));
                Paragraph::new(Line::from(title_spans))
            }
            ViewingMode::MetaCharts => {
                let mut title_spans = vec![
                    Span::styled(" Meta Charts ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                ];
                title_spans.extend(episode_status::episode_status_spans(is_training, current_episode));
                Paragraph::new(Line::from(title_spans))
            }
        };

        let title_widget = title.block(Block::default().borders(Borders::ALL));
        f.render_widget(title_widget, chunks[0]);

        let items: Vec<ListItem> = self
            .flattened
            .iter()
            .map(|(node_idx, depth)| {
                let indent = "  ".repeat(*depth);
                let (text, style) = match &self.nodes[*node_idx] {
                    ChartNode::Folder { name, children, .. } => {
                        let icon = if self.expanded[*node_idx] { "▼" } else { "▶" };
                        let label = format!("{}{} {} ({} items)", indent, icon, name, children.len());
                        (label, Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
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

        f.render_stateful_widget(list, chunks[1], &mut self.list_state);

        let help_line = if self.viewing_mode == ViewingMode::MetaCharts {
            Line::from(vec![
                Span::styled("↑/k", Style::default().fg(Color::Cyan)),
                Span::raw(": Up  "),
                Span::styled("↓/j", Style::default().fg(Color::Cyan)),
                Span::raw(": Down  "),
                Span::styled("Enter", Style::default().fg(Color::Green)),
                Span::raw(": Expand/Collapse  "),
                Span::styled("c", Style::default().fg(Color::Magenta)),
                Span::raw(": Copy  "),
                Span::styled("r", Style::default().fg(Color::Yellow)),
                Span::raw(": Refresh  "),
                Span::styled("q/Esc", Style::default().fg(Color::Red)),
                Span::raw(": Back"),
            ])
        } else {
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
            ])
        };

        let help = Paragraph::new(vec![help_line])
            .block(Block::default().borders(Borders::ALL).title("Controls"));
        f.render_widget(help, chunks[2]);
    }

    fn render_preview(&mut self, f: &mut Frame, area: Rect) {
        let block = Block::default().borders(Borders::ALL).title("Preview");
        let inner = block.inner(area);
        f.render_widget(block, area);

        if let Some(ref mut protocol) = self.current_image {
            let image = StatefulImage::new(None);
            f.render_stateful_widget(image, inner, protocol);
        } else {
            let selected_is_folder = self.list_state.selected().and_then(|i| {
                if i < self.flattened.len() {
                    let (node_idx, _) = self.flattened[i];
                    Some(matches!(self.nodes[node_idx], ChartNode::Folder { .. }))
                } else {
                    None
                }
            }).unwrap_or(false);

            let msg = if selected_is_folder {
                "Folders cannot be previewed - expand to view charts"
            } else {
                "Select a chart to preview"
            };

            let no_preview = Paragraph::new(msg)
                .style(Style::default().fg(Color::DarkGray));
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
    let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("unknown");
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

fn render_report_to_temp(report: &Report) -> Result<PathBuf> {
    let image = render_report(report)?;
    let mut path = std::env::temp_dir();
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    path.push(format!("report_chart_{stamp}.png"));
    image.save(&path)?;
    Ok(path)
}
