use anyhow::Result;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, MouseButton, MouseEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap},
    Frame, Terminal,
};
use shared::paths::{TRAINING_PATH, WEIGHTS_PATH};
use std::{
    io,
    path::PathBuf,
    process::{Child, Command},
    time::{Duration, SystemTime},
};
use walkdir::WalkDir;

mod chart_viewer;
use chart_viewer::ChartViewer;

// Catppuccin Mocha theme colors
mod theme {
    use ratatui::style::Color;

    pub const BASE: Color = Color::Rgb(30, 30, 46);           // #1e1e2e
    pub const MANTLE: Color = Color::Rgb(24, 24, 37);        // #181825
    pub const CRUST: Color = Color::Rgb(17, 17, 27);         // #11111b
    pub const TEXT: Color = Color::Rgb(205, 214, 244);       // #cdd6f4
    pub const SUBTEXT0: Color = Color::Rgb(166, 173, 200);   // #a6adc8
    pub const SUBTEXT1: Color = Color::Rgb(186, 194, 222);   // #bac2de
    pub const SURFACE0: Color = Color::Rgb(49, 50, 68);      // #313244
    pub const SURFACE1: Color = Color::Rgb(69, 71, 90);      // #45475a
    pub const SURFACE2: Color = Color::Rgb(88, 91, 112);     // #585b70
    pub const OVERLAY0: Color = Color::Rgb(108, 112, 134);   // #6c7086
    pub const OVERLAY1: Color = Color::Rgb(127, 132, 156);   // #7f849c
    pub const OVERLAY2: Color = Color::Rgb(147, 153, 178);   // #9399b2
    pub const BLUE: Color = Color::Rgb(137, 180, 250);       // #89b4fa
    pub const LAVENDER: Color = Color::Rgb(180, 190, 254);   // #b4befe
    pub const SAPPHIRE: Color = Color::Rgb(116, 199, 236);   // #74c7ec
    pub const SKY: Color = Color::Rgb(137, 220, 235);        // #89dceb
    pub const TEAL: Color = Color::Rgb(148, 226, 213);       // #94e2d5
    pub const GREEN: Color = Color::Rgb(166, 227, 161);      // #a6e3a1
    pub const YELLOW: Color = Color::Rgb(249, 226, 175);     // #f9e2af
    pub const PEACH: Color = Color::Rgb(250, 179, 135);      // #fab387
    pub const MAROON: Color = Color::Rgb(235, 160, 172);     // #eba0ac
    pub const RED: Color = Color::Rgb(243, 139, 168);        // #f38ba8
    pub const MAUVE: Color = Color::Rgb(203, 166, 247);      // #cba6f7
    pub const PINK: Color = Color::Rgb(245, 194, 231);       // #f5c2e7
    pub const FLAMINGO: Color = Color::Rgb(242, 205, 205);   // #f2cdcd
    pub const ROSEWATER: Color = Color::Rgb(245, 224, 220);  // #f5e0dc
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum AppMode {
    Main,
    GenerationBrowser,
    ChartViewer,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum DialogMode {
    None,
    WeightsInput { for_training: bool, for_inference: bool },
    ConfirmQuit,
    ConfirmStopTraining,
}

struct App {
    mode: AppMode,
    dialog_mode: DialogMode,
    training_process: Option<Child>,
    inference_process: Option<Child>,
    generations: Vec<GenerationInfo>,
    filtered_generations: Vec<GenerationInfo>,
    selected_generation: Option<usize>,
    list_state: ListState,
    chart_viewer: ChartViewer,
    input: String,
    search_input: String,
    weights_path: Option<String>,
    searching: bool,
    list_area: Rect,
    latest_meta_charts: Vec<PathBuf>,
}

#[derive(Debug, Clone)]
struct GenerationInfo {
    number: usize,
    path: PathBuf,
}

impl App {
    fn new() -> Result<Self> {
        let mut app = App {
            mode: AppMode::Main,
            dialog_mode: DialogMode::None,
            training_process: None,
            inference_process: None,
            generations: Vec::new(),
            filtered_generations: Vec::new(),
            selected_generation: None,
            list_state: ListState::default(),
            chart_viewer: ChartViewer::new(),
            input: String::new(),
            search_input: String::new(),
            weights_path: None,
            searching: false,
            list_area: Rect::default(),
            latest_meta_charts: Vec::new(),
        };
        app.load_generations()?;
        app.load_latest_meta_charts()?;
        Ok(app)
    }

    fn load_latest_meta_charts(&mut self) -> Result<()> {
        self.latest_meta_charts.clear();

        let data_path = PathBuf::from(TRAINING_PATH).join("data");
        if !data_path.exists() {
            return Ok(());
        }

        let mut meta_charts: Vec<(PathBuf, SystemTime)> = Vec::new();

        for entry in WalkDir::new(&data_path)
            .max_depth(1)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if entry.file_type().is_file() {
                if let Some(name) = entry.file_name().to_str() {
                    if name.contains("meta_history") && name.ends_with(".png") {
                        if let Ok(metadata) = entry.metadata() {
                            if let Ok(modified) = metadata.modified() {
                                meta_charts.push((entry.path().to_path_buf(), modified));
                            }
                        }
                    }
                }
            }
        }

        meta_charts.sort_by(|a, b| b.1.cmp(&a.1));
        self.latest_meta_charts = meta_charts.into_iter().take(4).map(|(path, _)| path).collect();

        Ok(())
    }

    fn is_anything_running(&self) -> bool {
        self.is_training_running() || self.inference_process.is_some()
    }

    fn is_training_running(&self) -> bool {
        if self.training_process.is_some() {
            return true;
        }

        // Check if there's a cargo process running training in trading_bots directory
        // This matches: "cargo run ... train" in the trading_bots context
        if let Ok(output) = Command::new("pgrep")
            .args(["-f", "cargo.*trading_bots.*train"])
            .output()
        {
            if !output.stdout.is_empty() {
                return true;
            }
        }

        // Also check for the actual binary running with train argument
        if let Ok(output) = Command::new("pgrep")
            .args(["-f", "trading_bot_0.*train"])
            .output()
        {
            if !output.stdout.is_empty() {
                // Make sure it's not the TUI itself
                let stdout = String::from_utf8_lossy(&output.stdout);
                for pid in stdout.lines() {
                    if let Ok(pid_num) = pid.trim().parse::<u32>() {
                        // Check if this PID is actually a training process, not the TUI
                        if let Ok(cmdline_output) = Command::new("ps")
                            .args(["-p", &pid_num.to_string(), "-o", "args="])
                            .output()
                        {
                            let cmdline = String::from_utf8_lossy(&cmdline_output.stdout);
                            if cmdline.contains("train") && !cmdline.contains("tui") {
                                return true;
                            }
                        }
                    }
                }
            }
        }

        false
    }

    fn load_generations(&mut self) -> Result<()> {
        self.generations.clear();
        let training_path = PathBuf::from("../training/gens");

        if !training_path.exists() {
            return Ok(());
        }

        for entry in WalkDir::new(&training_path)
            .min_depth(1)
            .max_depth(1)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if entry.file_type().is_dir() {
                if let Some(name) = entry.file_name().to_str() {
                    if let Ok(num) = name.parse::<usize>() {
                        self.generations.push(GenerationInfo {
                            number: num,
                            path: entry.path().to_path_buf(),
                        });
                    }
                }
            }
        }

        self.generations.sort_by_key(|g| g.number);
        self.filter_generations();
        Ok(())
    }

    fn filter_generations(&mut self) {
        if self.search_input.is_empty() {
            self.filtered_generations = self.generations.clone();
        } else {
            self.filtered_generations = self
                .generations
                .iter()
                .filter(|g| g.number.to_string().contains(&self.search_input))
                .cloned()
                .collect();
        }

        // Reset selection if filtered list is shorter
        if let Some(selected) = self.list_state.selected() {
            if selected >= self.filtered_generations.len() && !self.filtered_generations.is_empty() {
                self.list_state.select(Some(0));
                self.center_list(0);
            } else if !self.filtered_generations.is_empty() {
                self.center_list(selected);
            }
        } else if !self.filtered_generations.is_empty() {
            self.list_state.select(Some(0));
            self.center_list(0);
        }
    }

    fn scroll_up(&mut self, amount: usize) {
        for _ in 0..amount {
            self.previous_generation();
        }
    }

    fn scroll_down(&mut self, amount: usize) {
        for _ in 0..amount {
            self.next_generation();
        }
    }

    fn start_training(&mut self, weights_file: Option<String>) -> Result<()> {
        if self.training_process.is_some() {
            return Ok(());
        }

        let mut cmd = Command::new("cargo");
        cmd.arg("run")
            .arg("--release")
            .arg("--")
            .arg("train")
            .current_dir("../trading_bots");

        if let Some(weights) = weights_file {
            let weights_path = if weights.starts_with('/') || weights.starts_with("..") {
                weights
            } else {
                format!("{}/{}", WEIGHTS_PATH, weights)
            };
            cmd.arg("-w").arg(weights_path);
        }

        let child = cmd.spawn()?;
        self.training_process = Some(child);
        Ok(())
    }

    fn start_inference(&mut self, weights_file: Option<String>) -> Result<()> {
        if self.inference_process.is_some() {
            return Ok(());
        }

        let mut cmd = Command::new("cargo");
        cmd.arg("run")
            .arg("--release")
            .arg("--")
            .arg("infer")
            .current_dir("../trading_bots");

        let weights = weights_file.unwrap_or_else(|| "infer.ot".to_string());
        let weights_path = if weights.starts_with('/') || weights.starts_with("..") {
            weights
        } else {
            format!("{}/{}", WEIGHTS_PATH, weights)
        };
        cmd.arg("-w").arg(weights_path);

        let child = cmd.spawn()?;
        self.inference_process = Some(child);
        Ok(())
    }

    fn stop_training(&mut self) -> Result<()> {
        if let Some(mut child) = self.training_process.take() {
            child.kill()?;
            child.wait()?;
        }
        Ok(())
    }

    fn center_list(&mut self, selected: usize) {
        // Calculate visible height (subtract borders and title)
        let visible_height = self.list_area.height.saturating_sub(2) as usize;
        let center = visible_height / 2;

        let offset = if selected >= center {
            selected.saturating_sub(center)
        } else {
            0
        };

        self.list_state = ListState::default().with_selected(Some(selected)).with_offset(offset);
    }

    fn next_generation(&mut self) {
        if self.filtered_generations.is_empty() {
            return;
        }
        let i = match self.list_state.selected() {
            Some(i) => {
                if i >= self.filtered_generations.len() - 1 {
                    0
                } else {
                    i + 1
                }
            }
            None => 0,
        };
        self.list_state.select(Some(i));
        self.center_list(i);
    }

    fn previous_generation(&mut self) {
        if self.filtered_generations.is_empty() {
            return;
        }
        let i = match self.list_state.selected() {
            Some(i) => {
                if i == 0 {
                    self.filtered_generations.len() - 1
                } else {
                    i - 1
                }
            }
            None => 0,
        };
        self.list_state.select(Some(i));
        self.center_list(i);
    }

    fn select_generation(&mut self) -> Result<()> {
        if let Some(idx) = self.list_state.selected() {
            if idx < self.filtered_generations.len() {
                self.selected_generation = Some(idx);
                self.chart_viewer.load_generation(&self.filtered_generations[idx].path)?;
                self.mode = AppMode::ChartViewer;
            }
        }
        Ok(())
    }

    fn handle_generation_click(&mut self, row: u16) -> Result<()> {
        // Adjust for border (1) and title (1)
        let adjusted_row = row.saturating_sub(2);

        // Account for current offset
        let current_offset = self.list_state.offset();
        let actual_index = current_offset + adjusted_row as usize;

        if actual_index < self.filtered_generations.len() {
            self.center_list(actual_index);
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new()?;
    let res = run_app(&mut terminal, &mut app);

    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        println!("{err:?}");
    }

    Ok(())
}

fn run_app<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    app: &mut App,
) -> Result<()> {
    loop {
        terminal.draw(|f| ui(f, app))?;

        if event::poll(Duration::from_millis(250))? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => {
                    // Handle dialogs first (they take priority)
                    match app.dialog_mode {
                        DialogMode::WeightsInput { for_training, for_inference } => match key.code {
                            KeyCode::Esc => {
                                app.dialog_mode = DialogMode::None;
                                app.input.clear();
                            }
                            KeyCode::Enter => {
                                let weights = if app.input.is_empty() {
                                    None
                                } else {
                                    Some(app.input.clone())
                                };
                                app.input.clear();
                                app.dialog_mode = DialogMode::None;

                                if for_training {
                                    app.start_training(weights)?;
                                } else if for_inference {
                                    app.start_inference(weights)?;
                                }
                            }
                            KeyCode::Char(c) => {
                                app.input.push(c);
                            }
                            KeyCode::Backspace => {
                                app.input.pop();
                            }
                            _ => {}
                        }
                        DialogMode::ConfirmQuit => match key.code {
                            KeyCode::Char('y') | KeyCode::Enter => {
                                app.stop_training()?;
                                return Ok(());
                            }
                            KeyCode::Char('n') | KeyCode::Esc => {
                                app.dialog_mode = DialogMode::None;
                            }
                            _ => {}
                        }
                        DialogMode::ConfirmStopTraining => match key.code {
                            KeyCode::Char('y') | KeyCode::Enter => {
                                app.stop_training()?;
                                app.dialog_mode = DialogMode::None;
                            }
                            KeyCode::Char('n') | KeyCode::Esc => {
                                app.dialog_mode = DialogMode::None;
                            }
                            _ => {}
                        }
                        DialogMode::None => {
                            // Global keybindings (work in all modes when no dialog is open)
                            match key.code {
                                KeyCode::Char('c') if key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) => {
                                    app.dialog_mode = DialogMode::ConfirmQuit;
                                }
                                KeyCode::Char('d') if key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) => {
                                    app.dialog_mode = DialogMode::ConfirmQuit;
                                }
                                _ => {}
                            }

                            match app.mode {
                            AppMode::Main => match key.code {
                                KeyCode::Char('q') => {
                                    app.dialog_mode = DialogMode::ConfirmQuit;
                                }
                                KeyCode::Char('s') => {
                                    app.dialog_mode = DialogMode::WeightsInput { for_training: true, for_inference: false };
                                }
                                KeyCode::Char('f') => {
                                    if !app.is_anything_running() {
                                        app.dialog_mode = DialogMode::WeightsInput { for_training: false, for_inference: true };
                                    }
                                }
                                KeyCode::Char('x') => {
                                    if app.is_training_running() {
                                        app.dialog_mode = DialogMode::ConfirmStopTraining;
                                    }
                                }
                                KeyCode::Char('g') => {
                                    app.load_generations()?;
                                    app.mode = AppMode::GenerationBrowser;
                                }
                                _ => {}
                            },
                    AppMode::GenerationBrowser => {
                        if app.searching {
                            match key.code {
                                KeyCode::Esc => {
                                    app.searching = false;
                                    app.search_input.clear();
                                    app.filter_generations();
                                }
                                KeyCode::Enter => {
                                    app.searching = false;
                                }
                                KeyCode::Char(c) => {
                                    app.search_input.push(c);
                                    app.filter_generations();
                                }
                                KeyCode::Backspace => {
                                    app.search_input.pop();
                                    app.filter_generations();
                                }
                                _ => {}
                            }
                        } else {
                            match key.code {
                                KeyCode::Esc | KeyCode::Char('q') => {
                                    app.mode = AppMode::Main;
                                }
                                KeyCode::Char('/') => {
                                    app.searching = true;
                                }
                                KeyCode::Down | KeyCode::Char('j') => {
                                    app.next_generation();
                                }
                                KeyCode::Up | KeyCode::Char('k') => {
                                    app.previous_generation();
                                }
                                KeyCode::Enter => {
                                    app.select_generation()?;
                                }
                                KeyCode::Char('r') => {
                                    app.load_generations()?;
                                }
                                _ => {}
                            }
                        }
                    }
                    AppMode::ChartViewer => match key.code {
                        KeyCode::Esc | KeyCode::Char('q') => {
                            app.mode = AppMode::GenerationBrowser;
                        }
                        KeyCode::Down | KeyCode::Char('j') => {
                            app.chart_viewer.next();
                        }
                        KeyCode::Up | KeyCode::Char('k') => {
                            app.chart_viewer.previous();
                        }
                        KeyCode::Enter => {
                            app.chart_viewer.toggle_expand();
                        }
                        _ => {}
                    },
                            }
                        }
                    }
                }
                Event::Mouse(mouse) => match mouse.kind {
                    MouseEventKind::ScrollUp => match app.mode {
                        AppMode::GenerationBrowser => app.scroll_up(3),
                        AppMode::ChartViewer => app.chart_viewer.scroll_up(3),
                        _ => {}
                    },
                    MouseEventKind::ScrollDown => match app.mode {
                        AppMode::GenerationBrowser => app.scroll_down(3),
                        AppMode::ChartViewer => app.chart_viewer.scroll_down(3),
                        _ => {}
                    },
                    MouseEventKind::Down(MouseButton::Left) => {
                        if app.mode == AppMode::GenerationBrowser {
                            if mouse.column >= app.list_area.x
                                && mouse.column < app.list_area.x + app.list_area.width
                                && mouse.row >= app.list_area.y
                                && mouse.row < app.list_area.y + app.list_area.height
                            {
                                let _ = app.handle_generation_click(mouse.row - app.list_area.y);
                            }
                        }
                    }
                    _ => {}
                },
                _ => {}
            }
        }

        // Check if training process finished
        if let Some(child) = &mut app.training_process {
            match child.try_wait() {
                Ok(Some(_)) => {
                    app.training_process = None;
                    app.load_generations()?;
                    app.load_latest_meta_charts()?;
                }
                Ok(None) => {}
                Err(_) => {
                    app.training_process = None;
                }
            }
        }
    }
}

fn ui(f: &mut Frame, app: &mut App) {
    match app.mode {
        AppMode::Main => render_main(f, app),
        AppMode::GenerationBrowser => render_generation_browser(f, app),
        AppMode::ChartViewer => render_chart_viewer(f, app),
    }

    // Render dialog on top if active
    match app.dialog_mode {
        DialogMode::WeightsInput { for_training, for_inference } => {
            render_weights_dialog(f, app, for_training, for_inference);
        }
        DialogMode::ConfirmQuit => {
            render_confirm_dialog(f, "Quit?", "This will stop any running processes.");
        }
        DialogMode::ConfirmStopTraining => {
            render_confirm_dialog(f, "Stop Training?", "This will terminate the training process.");
        }
        DialogMode::None => {}
    }
}

fn render_main(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(7),
            Constraint::Min(0),
            Constraint::Length(6),
        ])
        .split(f.area());

    let title = Paragraph::new("Trading Bot TUI")
        .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(title, chunks[0]);

    let is_training = app.is_training_running();
    let status = if is_training {
        "Training: RUNNING"
    } else {
        "Training: STOPPED"
    };

    let status_color = if is_training {
        Color::Green
    } else {
        Color::Red
    };

    let weights_info = if let Some(weights) = &app.weights_path {
        format!("Weights: {}", weights)
    } else {
        format!("Weights: {} (default)", WEIGHTS_PATH)
    };

    let info_text = vec![
        Line::from(vec![
            Span::styled("Status: ", Style::default().fg(Color::Yellow)),
            Span::styled(status, Style::default().fg(status_color)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Weights File: ", Style::default().fg(Color::Yellow)),
            Span::raw(&app.input),
        ]),
        Line::from(weights_info),
    ];

    let info = Paragraph::new(info_text)
        .block(Block::default().borders(Borders::ALL).title("Info"))
        .wrap(Wrap { trim: false });
    f.render_widget(info, chunks[1]);

    let meta_chart_items: Vec<Line> = if app.latest_meta_charts.is_empty() {
        vec![Line::from(Span::styled(
            "No meta-history charts found yet",
            Style::default().fg(Color::DarkGray),
        ))]
    } else {
        app.latest_meta_charts
            .iter()
            .map(|path| {
                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown");
                Line::from(vec![
                    Span::styled("• ", Style::default().fg(Color::Cyan)),
                    Span::raw(name),
                ])
            })
            .collect()
    };

    let meta_charts = Paragraph::new(meta_chart_items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Latest Meta-History Charts"),
        )
        .wrap(Wrap { trim: false });
    f.render_widget(meta_charts, chunks[2]);

    let help_text = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("s", Style::default().fg(Color::Green)),
            Span::raw(": Start Training  "),
            Span::styled("f", Style::default().fg(Color::Blue)),
            Span::raw(": Run Inference  "),
            Span::styled("x", Style::default().fg(Color::Red)),
            Span::raw(": Stop Training"),
        ]),
        Line::from(vec![
            Span::styled("g", Style::default().fg(Color::Cyan)),
            Span::raw(": View Generations  "),
            Span::styled("q", Style::default().fg(Color::Red)),
            Span::raw(": Quit  "),
            Span::styled("Ctrl+C/D", Style::default().fg(Color::DarkGray)),
            Span::raw(": Quit"),
        ]),
    ];

    let help = Paragraph::new(help_text)
        .block(Block::default().borders(Borders::ALL).title("Controls"));
    f.render_widget(help, chunks[3]);
}

fn render_generation_browser(f: &mut Frame, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Min(0),
            Constraint::Length(4),
        ])
        .split(f.area());

    let title = Paragraph::new("Generation Browser")
        .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(title, chunks[0]);

    let search_style = if app.searching {
        Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let search_text = if app.searching {
        format!("Search: {}_", app.search_input)
    } else {
        format!("Search: {} (press / to search)", app.search_input)
    };

    let search = Paragraph::new(search_text)
        .style(search_style)
        .block(Block::default().borders(Borders::ALL).title("Filter"));
    f.render_widget(search, chunks[1]);

    let items: Vec<ListItem> = app
        .filtered_generations
        .iter()
        .map(|gen| {
            ListItem::new(format!("Generation {}", gen.number))
                .style(Style::default().fg(Color::White))
        })
        .collect();

    let list_title = format!(
        "Generations ({}/{})",
        app.filtered_generations.len(),
        app.generations.len()
    );

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(list_title))
        .highlight_style(
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol(">> ");

    app.list_area = chunks[2];
    let mut list_state = app.list_state.clone();
    f.render_stateful_widget(list, chunks[2], &mut list_state);

    let help = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("↑/k", Style::default().fg(Color::Cyan)),
            Span::raw(": Up  "),
            Span::styled("↓/j", Style::default().fg(Color::Cyan)),
            Span::raw(": Down  "),
            Span::styled("Wheel", Style::default().fg(Color::Cyan)),
            Span::raw(": Scroll  "),
            Span::styled("Click", Style::default().fg(Color::Cyan)),
            Span::raw(": Select"),
        ]),
        Line::from(vec![
            Span::styled("/", Style::default().fg(Color::Yellow)),
            Span::raw(": Search  "),
            Span::styled("Enter", Style::default().fg(Color::Green)),
            Span::raw(": View  "),
            Span::styled("r", Style::default().fg(Color::Yellow)),
            Span::raw(": Refresh  "),
            Span::styled("q/Esc", Style::default().fg(Color::Red)),
            Span::raw(": Back"),
        ]),
    ])
    .block(Block::default().borders(Borders::ALL).title("Controls"));
    f.render_widget(help, chunks[3]);
}

fn render_chart_viewer(f: &mut Frame, app: &mut App) {
    app.chart_viewer.render(f);
}

fn render_weights_dialog(f: &mut Frame, app: &App, for_training: bool, for_inference: bool) {
    let area = centered_rect(60, 30, f.area());

    let title = if for_training {
        "Start Training"
    } else if for_inference {
        "Run Inference"
    } else {
        "Weights Input"
    };

    let default_text = if for_inference {
        "infer.ot"
    } else {
        "none (train from scratch)"
    };

    let dialog = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .style(Style::default().bg(Color::Black));

    let inner = dialog.inner(area);
    f.render_widget(dialog, area);
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),
            Constraint::Length(3),
            Constraint::Length(2),
            Constraint::Min(0),
        ])
        .split(inner);

    let prompt_text = vec![
        Line::from(vec![
            Span::raw("Enter weights filename from "),
            Span::styled(WEIGHTS_PATH, Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled("Default: ", Style::default().fg(Color::Yellow)),
            Span::raw(default_text),
        ]),
    ];
    let prompt = Paragraph::new(prompt_text);
    f.render_widget(prompt, chunks[0]);

    let input_display = format!("> {}_", app.input);
    let input_widget = Paragraph::new(input_display)
        .style(Style::default().fg(Color::Green))
        .block(Block::default().borders(Borders::ALL).title("Weights File"));
    f.render_widget(input_widget, chunks[1]);

    let help = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("Enter", Style::default().fg(Color::Green)),
            Span::raw(": Confirm  "),
            Span::styled("Esc", Style::default().fg(Color::Red)),
            Span::raw(": Cancel"),
        ]),
    ]);
    f.render_widget(help, chunks[2]);
}

fn render_confirm_dialog(f: &mut Frame, title: &str, message: &str) {
    let area = centered_rect(50, 25, f.area());

    let dialog = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .style(Style::default().bg(Color::Black));

    let inner = dialog.inner(area);
    f.render_widget(dialog, area);
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(0),
            Constraint::Length(2),
        ])
        .split(inner);

    let message_widget = Paragraph::new(message)
        .style(Style::default().fg(Color::Yellow))
        .wrap(Wrap { trim: false });
    f.render_widget(message_widget, chunks[0]);

    let help = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("y/Enter", Style::default().fg(Color::Green)),
            Span::raw(": Yes  "),
            Span::styled("n/Esc", Style::default().fg(Color::Red)),
            Span::raw(": No"),
        ]),
    ]);
    f.render_widget(help, chunks[2]);
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}
