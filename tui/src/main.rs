use anyhow::Result;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, MouseButton, MouseEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::Rect,
    widgets::ListState,
    Frame, Terminal,
};
use shared::paths::WEIGHTS_PATH;
use std::{
    io,
    path::PathBuf,
    process::{Child, Command},
    time::{Duration, Instant},
};
use walkdir::WalkDir;

mod chart_viewer;
mod components;
mod pages;
mod theme;
mod utils;

use chart_viewer::ChartViewer;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AppMode {
    Main,
    GenerationBrowser,
    InferenceBrowser,
    ChartViewer,
    Logs,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DialogMode {
    None,
    WeightsInput { for_training: bool, for_inference: bool },
    InferenceInput { focused_field: InferenceField },
    ConfirmQuit,
    ConfirmStopTraining,
    PageJump { selected: usize },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InferenceField {
    Weights,
    Ticker,
    Episodes,
}

pub struct App {
    pub mode: AppMode,
    pub previous_mode: AppMode,
    pub dialog_mode: DialogMode,
    pub inference_process: Option<Child>,
    pub training_process: Option<Child>,
    pub training_output: Vec<String>,
    pub generations: Vec<GenerationInfo>,
    pub filtered_generations: Vec<GenerationInfo>,
    pub selected_generation: Option<usize>,
    pub list_state: ListState,
    pub inferences: Vec<InferenceInfo>,
    pub filtered_inferences: Vec<InferenceInfo>,
    pub selected_inference: Option<usize>,
    pub inference_list_state: ListState,
    pub chart_viewer: ChartViewer,
    pub input: String,
    pub ticker_input: String,
    pub episodes_input: String,
    pub search_input: String,
    pub inference_search_input: String,
    pub weights_path: Option<String>,
    pub searching: bool,
    pub inference_searching: bool,
    pub list_area: Rect,
    pub inference_list_area: Rect,
    pub latest_meta_charts: Vec<PathBuf>,
    pub logs_scroll: usize,
    last_refresh: Instant,
}

#[derive(Debug, Clone)]
pub struct GenerationInfo {
    pub number: usize,
    pub path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct InferenceInfo {
    pub number: usize,
    pub path: PathBuf,
}

impl App {
    fn coerce_weights_filename(input: &str) -> String {
        let trimmed = input.trim();

        // If it's just a number, expand to ppo_ep{N}.ot
        if trimmed.parse::<u32>().is_ok() {
            return format!("ppo_ep{}.ot", trimmed);
        }

        // If it already has the pattern, use as-is
        trimmed.to_string()
    }

    fn new() -> Result<Self> {
        let mut app = App {
            mode: AppMode::Main,
            previous_mode: AppMode::Main,
            dialog_mode: DialogMode::None,
            inference_process: None,
            training_process: None,
            training_output: Vec::new(),
            generations: Vec::new(),
            filtered_generations: Vec::new(),
            selected_generation: None,
            list_state: ListState::default(),
            inferences: Vec::new(),
            filtered_inferences: Vec::new(),
            selected_inference: None,
            inference_list_state: ListState::default(),
            chart_viewer: ChartViewer::new(),
            input: String::new(),
            ticker_input: String::new(),
            episodes_input: String::new(),
            search_input: String::new(),
            inference_search_input: String::new(),
            weights_path: None,
            searching: false,
            inference_searching: false,
            list_area: Rect::default(),
            inference_list_area: Rect::default(),
            latest_meta_charts: Vec::new(),
            logs_scroll: 0,
            last_refresh: Instant::now(),
        };
        app.load_generations()?;
        app.load_inferences()?;
        app.load_latest_meta_charts()?;
        Ok(app)
    }

    fn load_latest_meta_charts(&mut self) -> Result<()> {
        use std::collections::HashMap;
        use std::fs;
        use std::time::SystemTime;

        self.latest_meta_charts.clear();

        let gens_path = PathBuf::from("../training/gens");
        if !gens_path.exists() {
            return Ok(());
        }

        // Meta chart base names (without extension) - these are the "namespaces"
        let meta_chart_bases = vec![
            "final_assets", "cum_reward", "outperformance", "loss",
            "assets", "reward", "total_commissions"
        ];

        // Track the latest file for each chart type: base_name -> (modified_time, path)
        let mut latest_per_type: HashMap<String, (SystemTime, PathBuf)> = HashMap::new();

        // Scan all generation directories for meta charts
        if let Ok(entries) = fs::read_dir(&gens_path) {
            for entry in entries.filter_map(|e| e.ok()) {
                if !entry.file_type().ok().map(|ft| ft.is_dir()).unwrap_or(false) {
                    continue;
                }
                // Only process numeric directories (generation folders)
                if let Some(name) = entry.file_name().to_str() {
                    if name.parse::<usize>().is_err() {
                        continue;
                    }
                }

                let gen_path = entry.path();
                if let Ok(files) = fs::read_dir(&gen_path) {
                    for file in files.filter_map(|e| e.ok()) {
                        let file_path = file.path();
                        let ext = file_path.extension().and_then(|s| s.to_str());
                        if ext != Some("png") && ext != Some("webp") {
                            continue;
                        }

                        let file_name = file_path.file_stem().and_then(|s| s.to_str()).unwrap_or("");

                        // Check if this file matches any of our meta chart types
                        for base in &meta_chart_bases {
                            if file_name == *base {
                                if let Ok(metadata) = file.metadata() {
                                    if let Ok(modified) = metadata.modified() {
                                        let key = base.to_string();
                                        let should_update = latest_per_type
                                            .get(&key)
                                            .map(|(t, _)| modified > *t)
                                            .unwrap_or(true);

                                        if should_update {
                                            latest_per_type.insert(key, (modified, file_path.clone()));
                                        }
                                    }
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }

        // Collect paths from all found chart types
        for (_, (_, path)) in latest_per_type {
            self.latest_meta_charts.push(path);
        }

        // Sort by filename for consistent ordering
        self.latest_meta_charts.sort();

        Ok(())
    }

    fn is_anything_running(&self) -> bool {
        self.is_training_running() || self.inference_process.is_some()
    }

    pub fn is_training_running(&self) -> bool {
        // First check our own child process
        if self.training_process.is_some() {
            return true;
        }

        // Check if training was started externally
        if let Ok(output) = Command::new("pgrep")
            .args(["-f", "trading.*train"])
            .output()
        {
            if !output.stdout.is_empty() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for pid in stdout.lines() {
                    if let Ok(pid_num) = pid.trim().parse::<u32>() {
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

    fn load_inferences(&mut self) -> Result<()> {
        self.inferences.clear();
        let inference_path = PathBuf::from("../training/inferences");

        if !inference_path.exists() {
            return Ok(());
        }

        for entry in WalkDir::new(&inference_path)
            .min_depth(1)
            .max_depth(1)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if entry.file_type().is_dir() {
                if let Some(name) = entry.file_name().to_str() {
                    if let Ok(num) = name.parse::<usize>() {
                        self.inferences.push(InferenceInfo {
                            number: num,
                            path: entry.path().to_path_buf(),
                        });
                    }
                }
            }
        }

        self.inferences.sort_by_key(|i| i.number);
        self.filter_inferences();
        Ok(())
    }

    fn filter_inferences(&mut self) {
        if self.inference_search_input.is_empty() {
            self.filtered_inferences = self.inferences.clone();
        } else {
            self.filtered_inferences = self
                .inferences
                .iter()
                .filter(|i| i.number.to_string().contains(&self.inference_search_input))
                .cloned()
                .collect();
        }

        if let Some(selected) = self.inference_list_state.selected() {
            if selected >= self.filtered_inferences.len() && !self.filtered_inferences.is_empty() {
                self.inference_list_state.select(Some(0));
                self.center_inference_list(0);
            } else if !self.filtered_inferences.is_empty() {
                self.center_inference_list(selected);
            }
        } else if !self.filtered_inferences.is_empty() {
            self.inference_list_state.select(Some(0));
            self.center_inference_list(0);
        }
    }

    fn center_inference_list(&mut self, selected: usize) {
        let visible_height = self.inference_list_area.height.saturating_sub(2) as usize;
        let center = visible_height / 2;

        let offset = if selected >= center {
            selected.saturating_sub(center)
        } else {
            0
        };

        self.inference_list_state = ListState::default().with_selected(Some(selected)).with_offset(offset);
    }

    fn next_inference(&mut self) {
        if self.filtered_inferences.is_empty() {
            return;
        }
        let i = match self.inference_list_state.selected() {
            Some(i) => {
                if i >= self.filtered_inferences.len() - 1 {
                    0
                } else {
                    i + 1
                }
            }
            None => 0,
        };
        self.inference_list_state.select(Some(i));
        self.center_inference_list(i);
    }

    fn previous_inference(&mut self) {
        if self.filtered_inferences.is_empty() {
            return;
        }
        let i = match self.inference_list_state.selected() {
            Some(i) => {
                if i == 0 {
                    self.filtered_inferences.len() - 1
                } else {
                    i - 1
                }
            }
            None => 0,
        };
        self.inference_list_state.select(Some(i));
        self.center_inference_list(i);
    }

    fn scroll_inference_up(&mut self, amount: usize) {
        for _ in 0..amount {
            self.previous_inference();
        }
    }

    fn scroll_inference_down(&mut self, amount: usize) {
        for _ in 0..amount {
            self.next_inference();
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

    fn maybe_refresh(&mut self) -> Result<()> {
        let now = Instant::now();
        if now.duration_since(self.last_refresh) >= Duration::from_secs(1) {
            self.load_generations()?;
            self.load_inferences()?;
            self.load_latest_meta_charts()?;
            self.last_refresh = now;
        }
        Ok(())
    }

    fn poll_training_output(&mut self) {
        use std::fs;

        // Check if our child process has exited
        if let Some(ref mut child) = self.training_process {
            if let Ok(Some(_status)) = child.try_wait() {
                self.training_process = None;
            }
        }

        // Read from log file (works whether training was started by TUI or externally)
        let log_path = "../training/training.log";
        if let Ok(content) = fs::read_to_string(log_path) {
            let new_lines: Vec<String> = content.lines().map(|s| s.to_string()).collect();

            // Only update if there are new lines
            if new_lines.len() > self.training_output.len() || new_lines != self.training_output {
                self.training_output = new_lines;

                // Keep only last 1000 lines
                if self.training_output.len() > 1000 {
                    self.training_output.drain(0..self.training_output.len() - 1000);
                }
            }
        }
    }

    fn start_training(&mut self, weights_file: Option<String>) -> Result<()> {
        if self.is_training_running() {
            return Ok(());
        }

        use std::fs::OpenOptions;
        use std::process::Stdio;

        let log_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("../training/training.log")?;

        let log_file_stderr = log_file.try_clone()?;

        let mut cmd = Command::new("cargo");
        cmd.arg("run")
            .arg("--release")
            .arg("--")
            .arg("train")
            .current_dir("../trading_bots")
            .stdin(Stdio::null())
            .stdout(Stdio::from(log_file))
            .stderr(Stdio::from(log_file_stderr));

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

    fn start_inference(&mut self, weights_file: Option<String>, ticker: Option<String>, episodes: Option<usize>) -> Result<()> {
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

        if let Some(ticker_override) = ticker {
            cmd.env("TICKER_OVERRIDE", ticker_override);
        }

        if let Some(ep) = episodes {
            cmd.arg("-e").arg(ep.to_string());
        }

        let child = cmd.spawn()?;
        self.inference_process = Some(child);
        Ok(())
    }

    fn stop_training(&mut self) -> Result<()> {
        if let Some(mut child) = self.training_process.take() {
            let _ = child.kill();
            let _ = child.wait();
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
                self.previous_mode = self.mode;
                self.mode = AppMode::ChartViewer;
            }
        }
        Ok(())
    }

    fn view_meta_charts(&mut self) -> Result<()> {
        if !self.latest_meta_charts.is_empty() {
            self.chart_viewer.load_charts(&self.latest_meta_charts)?;
            self.previous_mode = self.mode;
            self.mode = AppMode::ChartViewer;
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

        app.maybe_refresh()?;
        app.poll_training_output();

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
                                    Some(App::coerce_weights_filename(&app.input))
                                };
                                app.input.clear();
                                app.dialog_mode = DialogMode::None;

                                if for_training {
                                    app.start_training(weights)?;
                                } else if for_inference {
                                    app.start_inference(weights, None, None)?;
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
                        DialogMode::InferenceInput { focused_field } => match key.code {
                            KeyCode::Esc => {
                                app.dialog_mode = DialogMode::None;
                                app.input.clear();
                                app.ticker_input.clear();
                                app.episodes_input.clear();
                            }
                            KeyCode::Tab => {
                                app.dialog_mode = DialogMode::InferenceInput {
                                    focused_field: match focused_field {
                                        InferenceField::Weights => InferenceField::Ticker,
                                        InferenceField::Ticker => InferenceField::Episodes,
                                        InferenceField::Episodes => InferenceField::Weights,
                                    }
                                };
                            }
                            KeyCode::BackTab => {
                                app.dialog_mode = DialogMode::InferenceInput {
                                    focused_field: match focused_field {
                                        InferenceField::Weights => InferenceField::Episodes,
                                        InferenceField::Ticker => InferenceField::Weights,
                                        InferenceField::Episodes => InferenceField::Ticker,
                                    }
                                };
                            }
                            KeyCode::Enter => {
                                let weights = if app.input.is_empty() {
                                    None
                                } else {
                                    Some(App::coerce_weights_filename(&app.input))
                                };
                                let ticker = if app.ticker_input.is_empty() {
                                    None
                                } else {
                                    Some(app.ticker_input.clone())
                                };
                                let episodes = if app.episodes_input.is_empty() {
                                    None
                                } else {
                                    app.episodes_input.parse::<usize>().ok()
                                };

                                app.input.clear();
                                app.ticker_input.clear();
                                app.episodes_input.clear();
                                app.dialog_mode = DialogMode::None;

                                app.start_inference(weights, ticker, episodes)?;
                            }
                            KeyCode::Char(c) => {
                                match focused_field {
                                    InferenceField::Weights => app.input.push(c),
                                    InferenceField::Ticker => app.ticker_input.push(c),
                                    InferenceField::Episodes => {
                                        if c.is_numeric() {
                                            app.episodes_input.push(c);
                                        }
                                    }
                                }
                            }
                            KeyCode::Backspace => {
                                match focused_field {
                                    InferenceField::Weights => { app.input.pop(); }
                                    InferenceField::Ticker => { app.ticker_input.pop(); }
                                    InferenceField::Episodes => { app.episodes_input.pop(); }
                                }
                            }
                            _ => {}
                        }
                        DialogMode::ConfirmQuit => match key.code {
                            KeyCode::Char('y') | KeyCode::Enter => {
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
                        DialogMode::PageJump { selected } => {
                            const PAGE_COUNT: usize = 5;
                            match key.code {
                                KeyCode::Esc => {
                                    app.dialog_mode = DialogMode::None;
                                }
                                KeyCode::Char('j') | KeyCode::Down => {
                                    app.dialog_mode = DialogMode::PageJump {
                                        selected: (selected + 1) % PAGE_COUNT,
                                    };
                                }
                                KeyCode::Char('k') | KeyCode::Up => {
                                    app.dialog_mode = DialogMode::PageJump {
                                        selected: if selected == 0 { PAGE_COUNT - 1 } else { selected - 1 },
                                    };
                                }
                                KeyCode::Enter => {
                                    app.dialog_mode = DialogMode::None;
                                    match selected {
                                        0 => app.mode = AppMode::Main,
                                        1 => {
                                            app.load_generations()?;
                                            if !app.filtered_generations.is_empty() {
                                                app.list_state.select(Some(0));
                                                app.center_list(0);
                                            }
                                            app.mode = AppMode::GenerationBrowser;
                                        }
                                        2 => {
                                            app.load_inferences()?;
                                            app.mode = AppMode::InferenceBrowser;
                                        }
                                        3 => {
                                            app.view_meta_charts()?;
                                        }
                                        4 => app.mode = AppMode::Logs,
                                        _ => {}
                                    }
                                }
                                KeyCode::Char('1') => {
                                    app.dialog_mode = DialogMode::None;
                                    app.mode = AppMode::Main;
                                }
                                KeyCode::Char('2') => {
                                    app.dialog_mode = DialogMode::None;
                                    app.load_generations()?;
                                    if !app.filtered_generations.is_empty() {
                                        app.list_state.select(Some(0));
                                        app.center_list(0);
                                    }
                                    app.mode = AppMode::GenerationBrowser;
                                }
                                KeyCode::Char('3') => {
                                    app.dialog_mode = DialogMode::None;
                                    app.load_inferences()?;
                                    app.mode = AppMode::InferenceBrowser;
                                }
                                KeyCode::Char('4') => {
                                    app.dialog_mode = DialogMode::None;
                                    app.view_meta_charts()?;
                                }
                                KeyCode::Char('5') => {
                                    app.dialog_mode = DialogMode::None;
                                    app.mode = AppMode::Logs;
                                }
                                _ => {}
                            }
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
                                KeyCode::Char('o') => {
                                    // Don't open page jump when searching
                                    if !app.searching && !app.inference_searching {
                                        app.dialog_mode = DialogMode::PageJump { selected: 0 };
                                    }
                                }
                                _ => {}
                            }

                            match app.mode {
                            AppMode::Main => match key.code {
                                KeyCode::Char('q') => {
                                    app.dialog_mode = DialogMode::ConfirmQuit;
                                }
                                KeyCode::Char('s') => {
                                    if !app.is_training_running() {
                                        app.dialog_mode = DialogMode::WeightsInput { for_training: true, for_inference: false };
                                    }
                                }
                                KeyCode::Char('f') => {
                                    if !app.is_anything_running() {
                                        app.dialog_mode = DialogMode::InferenceInput { focused_field: InferenceField::Weights };
                                    }
                                }
                                KeyCode::Char('x') => {
                                    if app.is_training_running() {
                                        app.dialog_mode = DialogMode::ConfirmStopTraining;
                                    }
                                }
                                KeyCode::Char('e') => {
                                    app.load_generations()?;
                                    // Auto-select latest generation (first in list)
                                    if !app.filtered_generations.is_empty() {
                                        app.list_state.select(Some(0));
                                        app.center_list(0);
                                    }
                                    app.mode = AppMode::GenerationBrowser;
                                }
                                KeyCode::Char('i') => {
                                    app.load_inferences()?;
                                    app.mode = AppMode::InferenceBrowser;
                                }
                                KeyCode::Char('m') => {
                                    app.view_meta_charts()?;
                                }
                                KeyCode::Char('l') => {
                                    app.mode = AppMode::Logs;
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
                    AppMode::InferenceBrowser => {
                        if app.inference_searching {
                            match key.code {
                                KeyCode::Esc => {
                                    app.inference_searching = false;
                                    app.inference_search_input.clear();
                                    app.filter_inferences();
                                }
                                KeyCode::Enter => {
                                    app.inference_searching = false;
                                }
                                KeyCode::Char(c) => {
                                    app.inference_search_input.push(c);
                                    app.filter_inferences();
                                }
                                KeyCode::Backspace => {
                                    app.inference_search_input.pop();
                                    app.filter_inferences();
                                }
                                _ => {}
                            }
                        } else {
                            match key.code {
                                KeyCode::Esc | KeyCode::Char('q') => {
                                    app.mode = AppMode::Main;
                                }
                                KeyCode::Char('/') => {
                                    app.inference_searching = true;
                                }
                                KeyCode::Down | KeyCode::Char('j') => {
                                    app.next_inference();
                                }
                                KeyCode::Up | KeyCode::Char('k') => {
                                    app.previous_inference();
                                }
                                KeyCode::Char('r') => {
                                    app.load_inferences()?;
                                }
                                _ => {}
                            }
                        }
                    }
                    AppMode::ChartViewer => match key.code {
                        KeyCode::Esc | KeyCode::Char('q') => {
                            app.mode = app.previous_mode;
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
                    AppMode::Logs => match key.code {
                        KeyCode::Esc | KeyCode::Char('q') => {
                            app.mode = AppMode::Main;
                        }
                        KeyCode::Char('c') => {
                            app.training_output.clear();
                            app.logs_scroll = 0;
                        }
                        KeyCode::Up | KeyCode::Char('k') => {
                            app.logs_scroll = app.logs_scroll.saturating_sub(1);
                        }
                        KeyCode::Down | KeyCode::Char('j') => {
                            if !app.training_output.is_empty() {
                                app.logs_scroll = (app.logs_scroll + 1).min(app.training_output.len().saturating_sub(1));
                            }
                        }
                        KeyCode::PageUp => {
                            app.logs_scroll = app.logs_scroll.saturating_sub(10);
                        }
                        KeyCode::PageDown => {
                            if !app.training_output.is_empty() {
                                app.logs_scroll = (app.logs_scroll + 10).min(app.training_output.len().saturating_sub(1));
                            }
                        }
                        KeyCode::Home => {
                            app.logs_scroll = 0;
                        }
                        KeyCode::End => {
                            app.logs_scroll = app.training_output.len().saturating_sub(1);
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
                        AppMode::InferenceBrowser => app.scroll_inference_up(3),
                        AppMode::ChartViewer => app.chart_viewer.scroll_up(3),
                        AppMode::Logs => {
                            app.logs_scroll = app.logs_scroll.saturating_sub(3);
                        }
                        _ => {}
                    },
                    MouseEventKind::ScrollDown => match app.mode {
                        AppMode::GenerationBrowser => app.scroll_down(3),
                        AppMode::InferenceBrowser => app.scroll_inference_down(3),
                        AppMode::ChartViewer => app.chart_viewer.scroll_down(3),
                        AppMode::Logs => {
                            if !app.training_output.is_empty() {
                                app.logs_scroll = (app.logs_scroll + 3).min(app.training_output.len().saturating_sub(1));
                            }
                        }
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
    }
}

fn ui(f: &mut Frame, app: &mut App) {
    match app.mode {
        AppMode::Main => pages::main_page::render(f, app),
        AppMode::GenerationBrowser => pages::generation_browser::render(f, app),
        AppMode::InferenceBrowser => pages::inference_browser::render(f, app),
        AppMode::ChartViewer => app.chart_viewer.render(f),
        AppMode::Logs => pages::logs_page::render(f, app),
    }

    // Render dialog on top if active
    match app.dialog_mode {
        DialogMode::WeightsInput { for_training, for_inference } => {
            components::dialogs::weights::render(f, app, for_training, for_inference);
        }
        DialogMode::InferenceInput { focused_field } => {
            components::dialogs::inference::render(f, app, focused_field);
        }
        DialogMode::ConfirmQuit => {
            components::dialogs::confirm::render(f, "Quit?", "Training processes will continue running in background.");
        }
        DialogMode::ConfirmStopTraining => {
            components::dialogs::confirm::render(f, "Stop Training?", "This will terminate the training process.");
        }
        DialogMode::PageJump { selected } => {
            components::dialogs::page_jump::render(f, selected, app.mode);
        }
        DialogMode::None => {}
    }
}

