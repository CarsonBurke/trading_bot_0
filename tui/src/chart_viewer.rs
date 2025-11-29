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
        }
    }

    pub fn load_generation(&mut self, gen_path: &PathBuf) -> Result<()> {
        self.nodes.clear();
        self.root_indices.clear();
        self.expanded.clear();
        self.current_image = None;

        self.build_tree(gen_path)?;
        self.rebuild_flattened();

        if !self.flattened.is_empty() {
            self.list_state.select(Some(0));
            self.load_current_image();
        }

        Ok(())
    }

    pub fn load_inference(&mut self, infer_path: &PathBuf) -> Result<()> {
        // Inferences use the same structure as generations
        self.load_generation(infer_path)
    }

    pub fn load_charts(&mut self, chart_paths: &[PathBuf]) -> Result<()> {
        self.nodes.clear();
        self.root_indices.clear();
        self.expanded.clear();
        self.current_image = None;

        for path in chart_paths {
            if path.exists() {
                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string();

                let chart_idx = self.nodes.len();
                self.nodes.push(ChartNode::Chart {
                    name,
                    path: path.clone(),
                });
                self.expanded.push(false);
                self.root_indices.push(chart_idx);
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

        let root_indices = self.root_indices.clone();
        for &idx in &root_indices {
            self.add_to_flattened(idx, 0);
        }
    }

    fn add_to_flattened(&mut self, idx: usize, depth: usize) {
        self.flattened.push((idx, depth));

        if let ChartNode::Folder { children, .. } = &self.nodes[idx] {
            if self.expanded[idx] {
                let children_clone = children.clone();
                for &child_idx in &children_clone {
                    self.add_to_flattened(child_idx, depth + 1);
                }
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

    pub fn render(&mut self, f: &mut Frame) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(25), Constraint::Percentage(75)])
            .split(f.area());

        self.render_list(f, chunks[0]);
        self.render_preview(f, chunks[1]);
    }

    fn render_list(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(0), Constraint::Length(3)])
            .split(area);

        let title = Paragraph::new(" Meta Charts ")
            .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
            .block(Block::default().borders(Borders::ALL));
        f.render_widget(title, chunks[0]);

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
                        let label = format!("{}  📊 {}", indent, name);
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

        let mut list_state = self.list_state.clone();
        f.render_stateful_widget(list, chunks[1], &mut list_state);

        let help = Paragraph::new(vec![
            Line::from(vec![
                Span::styled("↑/k", Style::default().fg(Color::Cyan)),
                Span::raw(": Up  "),
                Span::styled("↓/j", Style::default().fg(Color::Cyan)),
                Span::raw(": Down  "),
                Span::styled("Enter", Style::default().fg(Color::Green)),
                Span::raw(": Expand/Collapse  "),
                Span::styled("q/Esc", Style::default().fg(Color::Red)),
                Span::raw(": Back"),
            ]),
        ])
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
