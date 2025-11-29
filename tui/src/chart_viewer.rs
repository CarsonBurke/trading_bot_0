use anyhow::Result;
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph},
    Frame,
};
use ratatui_image::{picker::Picker, protocol::StatefulProtocol, StatefulImage};
use std::path::PathBuf;
use walkdir::WalkDir;

use crate::components::episode_status;

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
        // Store (path, episode_num, modified_time)
        let mut ticker_groups: HashMap<Option<String>, Vec<(PathBuf, Option<usize>, SystemTime)>> = HashMap::new();

        for path in chart_paths {
            if path.exists() {
                // Get modification time
                let modified = path
                    .metadata()
                    .and_then(|m| m.modified())
                    .unwrap_or(SystemTime::UNIX_EPOCH);

                // Extract episode number and ticker from path
                let parent = path.parent();
                let grandparent = parent.and_then(|p| p.parent());

                // Check if this is a ticker-specific chart (path like: gens/123/NVDA/assets.png)
                let (episode_num, ticker) = if let Some(parent) = parent {
                    if let Some(ticker_str) = parent.file_name().and_then(|n| n.to_str()) {
                        // Check if parent is a ticker folder (not numeric)
                        if ticker_str.parse::<usize>().is_err() {
                            // This is a ticker folder, get episode from grandparent
                            let ep = grandparent
                                .and_then(|gp| gp.file_name())
                                .and_then(|n| n.to_str())
                                .and_then(|s| s.parse::<usize>().ok());
                            (ep, Some(ticker_str.to_string()))
                        } else {
                            // This is an episode folder (meta chart)
                            let ep_num = ticker_str.parse::<usize>().ok();
                            (ep_num, None)
                        }
                    } else {
                        (None, None)
                    }
                } else {
                    (None, None)
                };

                ticker_groups.entry(ticker)
                    .or_insert_with(Vec::new)
                    .push((path.clone(), episode_num, modified));
            }
        }

        // Add episode-level charts first (no ticker)
        if let Some(mut episode_charts) = ticker_groups.remove(&None) {
            // Sort by modification time (most recent first)
            episode_charts.sort_by(|a, b| b.2.cmp(&a.2));

            for (path, episode_num, _) in episode_charts {
                let base_name = path
                    .file_stem()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown");

                let name = if let Some(ep) = episode_num {
                    format!("{} (ep {})", base_name, ep)
                } else {
                    base_name.to_string()
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
                        .map(|(_, _, modified)| *modified)
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
                charts.sort_by(|a, b| b.2.cmp(&a.2));

                // Add all charts for this ticker
                for (path, episode_num, _) in charts {
                    let base_name = path
                        .file_stem()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown");

                    let name = if let Some(ep) = episode_num {
                        format!("{} (ep {})", base_name, ep)
                    } else {
                        base_name.to_string()
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
                    let ext = sub_entry.path().extension().and_then(|s| s.to_str());
                    if sub_entry.file_type().is_file()
                        && (ext == Some("png") || ext == Some("webp"))
                    {
                        let chart_name = sub_entry
                            .file_name()
                            .to_str()
                            .unwrap_or("unknown")
                            .to_string();
                        let chart_idx = self.nodes.len();
                        self.nodes.push(ChartNode::Chart {
                            name: chart_name,
                            path: sub_entry.path().to_path_buf(),
                        });
                        children.push(chart_idx);
                        self.expanded.push(false);
                    }
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
            } else {
                let ext = entry.path().extension().and_then(|s| s.to_str());
                if entry.file_type().is_file() && (ext == Some("png") || ext == Some("webp")) {
                    let chart_idx = self.nodes.len();
                    self.nodes.push(ChartNode::Chart {
                        name,
                        path: entry_path,
                    });
                    self.expanded.push(false);
                    charts.push(chart_idx);
                }
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
                    if let Ok(img) = image::open(path) {
                        // Use original image size, picker will handle fitting to terminal
                        let protocol = self.picker.new_resize_protocol(img);
                        self.current_image = Some(protocol);
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
