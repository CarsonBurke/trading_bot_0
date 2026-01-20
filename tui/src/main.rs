use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, MouseButton, MouseEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Frame, Terminal};
use shared::paths::WEIGHTS_PATH;
use std::{
    io,
    path::PathBuf,
    time::{Duration, Instant},
};

mod chart_viewer;
mod components;
mod pages;
mod report_renderer;
mod state;
mod theme;
mod utils;

use chart_viewer::ChartViewer;
use state::{GenerationBrowserState, InferenceBrowserState, LogsPageState, ProcessManagerState};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AppMode {
    Main,
    GenerationBrowser,
    InferenceBrowser,
    ChartViewer,
    Logs,
    ModelObservations,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DialogMode {
    None,
    WeightsInput {
        for_training: bool,
        for_inference: bool,
    },
    InferenceInput {
        focused_field: InferenceField,
    },
    ConfirmQuit,
    ConfirmStopTraining,
    PageJump {
        selected: usize,
    },
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
    pub chart_viewer: ChartViewer,
    pub input: String,
    pub ticker_input: String,
    pub episodes_input: String,
    pub weights_path: Option<String>,
    pub latest_meta_charts: Vec<PathBuf>,
    last_refresh: Instant,
    pub generation_browser: GenerationBrowserState,
    pub inference_browser: InferenceBrowserState,
    pub logs_page: LogsPageState,
    pub process_manager: ProcessManagerState,
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
        let mut generation_browser = GenerationBrowserState::new();
        generation_browser.load_generations()?;

        let mut inference_browser = InferenceBrowserState::new();
        inference_browser.load_inferences()?;

        let mut app = App {
            mode: AppMode::Main,
            previous_mode: AppMode::Main,
            dialog_mode: DialogMode::None,
            chart_viewer: ChartViewer::new(),
            input: String::new(),
            ticker_input: String::new(),
            episodes_input: String::new(),
            weights_path: None,
            latest_meta_charts: Vec::new(),
            last_refresh: Instant::now(),
            generation_browser,
            inference_browser,
            logs_page: LogsPageState::new(),
            process_manager: ProcessManagerState::new(),
        };

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

        // Meta chart base names (episode-level charts without ticker)
        let meta_chart_bases = vec![
            "assets",
            "reward",
            "final_assets",
            "cumulative_reward",
            "outperformance",
            "loss_log",
            "advantage_stats_log",
            "total_commissions",
            "logit_noise",
            "grad_norm_log",
            "target_weights",
            "clip_fraction",
            "value_mae"
        ];

        // Ticker-specific chart base names
        let ticker_chart_bases = vec!["assets", "buy_sell", "raw_action"];

        // Track the latest file for each chart type: base_name -> (modified_time, path)
        let mut latest_per_type: HashMap<String, (SystemTime, PathBuf)> = HashMap::new();

        // Scan all generation directories
        if let Ok(entries) = fs::read_dir(&gens_path) {
            for entry in entries.filter_map(|e| e.ok()) {
                if !entry
                    .file_type()
                    .ok()
                    .map(|ft| ft.is_dir())
                    .unwrap_or(false)
                {
                    continue;
                }
                // Only process numeric directories (generation folders)
                if let Some(name) = entry.file_name().to_str() {
                    if name.parse::<usize>().is_err() {
                        continue;
                    }
                }

                let gen_path = entry.path();

                // Process episode-level meta charts
                for base in &meta_chart_bases {
                    let report_path = gen_path.join(format!("{base}.report.bin"));
                    if !report_path.exists() {
                        continue;
                    }
                    if let Ok(metadata) = fs::metadata(&report_path) {
                        if let Ok(modified) = metadata.modified() {
                            let key = format!("meta_{}", base);
                            if latest_per_type
                                .get(&key)
                                .map(|(t, _)| modified > *t)
                                .unwrap_or(true)
                            {
                                latest_per_type.insert(key, (modified, report_path));
                            }
                        }
                    }
                }

                // Process ticker-specific charts in subdirectories
                if let Ok(items) = fs::read_dir(&gen_path) {
                    for item in items.filter_map(|e| e.ok()) {
                        let item_path = item.path();
                        if !item.file_type().ok().map(|ft| ft.is_dir()).unwrap_or(false) {
                            continue;
                        }
                        let ticker_name = item.file_name();
                        let ticker_str = ticker_name.to_str().unwrap_or("");

                        for base in &ticker_chart_bases {
                            let report_path = item_path.join(format!("{base}.report.bin"));
                            if !report_path.exists() {
                                continue;
                            }
                            if let Ok(metadata) = fs::metadata(&report_path) {
                                if let Ok(modified) = metadata.modified() {
                                    let key = format!("{}_{}", ticker_str, base);
                                    if latest_per_type
                                        .get(&key)
                                        .map(|(t, _)| modified > *t)
                                        .unwrap_or(true)
                                    {
                                        latest_per_type.insert(key, (modified, report_path));
                                    }
                                }
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

    pub fn is_training_running(&mut self) -> bool {
        self.process_manager.is_training_running()
    }

    fn is_anything_running(&mut self) -> bool {
        self.process_manager.is_anything_running()
    }

    pub fn get_current_episode(&self) -> Option<usize> {
        for line in self.logs_page.training_output.iter().rev() {
            // Look for actual episode completion logs: "Episode N - Total Assets..."
            // Skip PPO progress logs: "[Ep N] Episodes: ..."
            if line.contains("Episode") && line.contains("Total Assets") && !line.starts_with("[Ep")
            {
                if let Some(ep_str) = line.split("Episode").nth(1) {
                    if let Some(num_str) = ep_str.trim().split_whitespace().next() {
                        // Strip ANSI codes if present
                        let clean_num = num_str
                            .chars()
                            .filter(|c| c.is_ascii_digit())
                            .collect::<String>();
                        if let Ok(ep) = clean_num.parse::<usize>() {
                            return Some(ep);
                        }
                    }
                }
            }
        }
        None
    }

    fn maybe_refresh(&mut self) -> Result<()> {
        let now = Instant::now();
        if now.duration_since(self.last_refresh) >= Duration::from_secs(1) {
            self.generation_browser.load_generations()?;
            self.inference_browser.load_inferences()?;
            self.load_latest_meta_charts()?;
            self.logs_page.poll_training_output();
            self.last_refresh = now;
        }
        Ok(())
    }

    fn start_training(&mut self, weights_file: Option<String>) -> Result<()> {
        let weights = weights_file.map(|w| {
            if w.starts_with('/') || w.starts_with("..") {
                w
            } else {
                format!("{}/{}", WEIGHTS_PATH, w)
            }
        });
        self.process_manager.start_training(weights)
    }

    fn start_inference(
        &mut self,
        weights_file: Option<String>,
        ticker: Option<String>,
        episodes: Option<usize>,
    ) -> Result<()> {
        let weights = weights_file.unwrap_or_else(|| "infer.ot".to_string());
        let weights_path = if weights.starts_with('/') || weights.starts_with("..") {
            weights
        } else {
            format!("{}/{}", WEIGHTS_PATH, weights)
        };
        self.process_manager
            .start_inference(weights_path, ticker, episodes.unwrap_or(10))
    }

    fn stop_training(&mut self) -> Result<()> {
        self.process_manager.stop_training()
    }

    fn select_generation(&mut self) -> Result<()> {
        if let Some(gen) = self.generation_browser.get_selected() {
            let path = gen.path.clone();
            self.generation_browser.selected_generation =
                self.generation_browser.list_state.selected();
            self.chart_viewer.load_generation(&path)?;
            self.previous_mode = self.mode;
            self.mode = AppMode::ChartViewer;
        }
        Ok(())
    }

    fn select_inference(&mut self) -> Result<()> {
        if let Some(inf) = self.inference_browser.get_selected() {
            let path = inf.path.clone();
            self.inference_browser.selected_inference =
                self.inference_browser.list_state.selected();
            self.chart_viewer.load_inference(&path)?;
            self.previous_mode = self.mode;
            self.mode = AppMode::ChartViewer;
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
        let adjusted_row = row.saturating_sub(2);
        let current_offset = self.generation_browser.list_state.offset();
        let actual_index = current_offset + adjusted_row as usize;

        if actual_index < self.generation_browser.filtered_generations.len() {
            self.generation_browser.center_list(actual_index);
        }
        Ok(())
    }

    fn next_log_line(&mut self) {
        self.logs_page.next();
    }

    fn previous_log_line(&mut self) {
        self.logs_page.previous();
    }
}

fn main() -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new()?;
    let res = run_app(&mut terminal, &mut app);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        println!("{err:?}");
    }

    Ok(())
}

fn run_app<B: ratatui::backend::Backend>(terminal: &mut Terminal<B>, app: &mut App) -> Result<()> {
    loop {
        terminal.draw(|f| ui(f, app))?;

        app.maybe_refresh()?;
        app.logs_page.poll_training_output();

        // Wait for event, then drain all pending events before redrawing
        if event::poll(Duration::from_millis(16))? {
            // ~60fps
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => {
                    // Handle dialogs first (they take priority)
                    match app.dialog_mode {
                        DialogMode::WeightsInput {
                            for_training,
                            for_inference,
                        } => match key.code {
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
                        },
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
                                    },
                                };
                            }
                            KeyCode::BackTab => {
                                app.dialog_mode = DialogMode::InferenceInput {
                                    focused_field: match focused_field {
                                        InferenceField::Weights => InferenceField::Episodes,
                                        InferenceField::Ticker => InferenceField::Weights,
                                        InferenceField::Episodes => InferenceField::Ticker,
                                    },
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
                            KeyCode::Char(c) => match focused_field {
                                InferenceField::Weights => app.input.push(c),
                                InferenceField::Ticker => app.ticker_input.push(c),
                                InferenceField::Episodes => {
                                    if c.is_numeric() {
                                        app.episodes_input.push(c);
                                    }
                                }
                            },
                            KeyCode::Backspace => match focused_field {
                                InferenceField::Weights => {
                                    app.input.pop();
                                }
                                InferenceField::Ticker => {
                                    app.ticker_input.pop();
                                }
                                InferenceField::Episodes => {
                                    app.episodes_input.pop();
                                }
                            },
                            _ => {}
                        },
                        DialogMode::ConfirmQuit => match key.code {
                            KeyCode::Char('y') | KeyCode::Char('Y') | KeyCode::Enter => {
                                return Ok(());
                            }
                            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                                app.dialog_mode = DialogMode::None;
                            }
                            _ => {}
                        },
                        DialogMode::ConfirmStopTraining => match key.code {
                            KeyCode::Char('y') | KeyCode::Char('Y') | KeyCode::Enter => {
                                app.stop_training()?;
                                app.dialog_mode = DialogMode::None;
                            }
                            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                                app.dialog_mode = DialogMode::None;
                            }
                            _ => {}
                        },
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
                                        selected: if selected == 0 {
                                            PAGE_COUNT - 1
                                        } else {
                                            selected - 1
                                        },
                                    };
                                }
                                KeyCode::Enter => {
                                    app.dialog_mode = DialogMode::None;
                                    match selected {
                                        0 => app.mode = AppMode::Main,
                                        1 => {
                                            app.generation_browser.load_generations()?;
                                            if !app
                                                .generation_browser
                                                .filtered_generations
                                                .is_empty()
                                                && app
                                                    .generation_browser
                                                    .list_state
                                                    .selected()
                                                    .is_none()
                                            {
                                                app.generation_browser.list_state.select(Some(0));
                                                app.generation_browser.center_list(0);
                                            }
                                            app.mode = AppMode::GenerationBrowser;
                                        }
                                        2 => {
                                            app.inference_browser.load_inferences()?;
                                            if !app.inference_browser.filtered_inferences.is_empty()
                                                && app
                                                    .inference_browser
                                                    .list_state
                                                    .selected()
                                                    .is_none()
                                            {
                                                app.inference_browser.list_state.select(Some(0));
                                                app.inference_browser.center_list(0);
                                            }
                                            app.mode = AppMode::InferenceBrowser;
                                        }
                                        3 => {
                                            app.view_meta_charts()?;
                                        }
                                        4 => {
                                            app.logs_page.enter();
                                            app.mode = AppMode::Logs;
                                        }
                                        _ => {}
                                    }
                                }
                                KeyCode::Char('1') => {
                                    app.dialog_mode = DialogMode::None;
                                    app.mode = AppMode::Main;
                                }
                                KeyCode::Char('2') => {
                                    app.dialog_mode = DialogMode::None;
                                    app.generation_browser.load_generations()?;
                                    if !app.generation_browser.filtered_generations.is_empty()
                                        && app.generation_browser.list_state.selected().is_none()
                                    {
                                        app.generation_browser.list_state.select(Some(0));
                                        app.generation_browser.center_list(0);
                                    }
                                    app.mode = AppMode::GenerationBrowser;
                                }
                                KeyCode::Char('3') => {
                                    app.dialog_mode = DialogMode::None;
                                    app.inference_browser.load_inferences()?;
                                    if !app.inference_browser.filtered_inferences.is_empty()
                                        && app.inference_browser.list_state.selected().is_none()
                                    {
                                        app.inference_browser.list_state.select(Some(0));
                                        app.inference_browser.center_list(0);
                                    }
                                    app.mode = AppMode::InferenceBrowser;
                                }
                                KeyCode::Char('4') => {
                                    app.dialog_mode = DialogMode::None;
                                    app.view_meta_charts()?;
                                }
                                KeyCode::Char('5') => {
                                    app.dialog_mode = DialogMode::None;
                                    app.logs_page.enter();
                                    app.mode = AppMode::Logs;
                                }
                                _ => {}
                            }
                        }
                        DialogMode::None => {
                            // Global keybindings (work in all modes when no dialog is open)
                            match key.code {
                                KeyCode::Char('c')
                                    if key
                                        .modifiers
                                        .contains(crossterm::event::KeyModifiers::CONTROL) =>
                                {
                                    app.dialog_mode = DialogMode::ConfirmQuit;
                                }
                                KeyCode::Char('d')
                                    if key
                                        .modifiers
                                        .contains(crossterm::event::KeyModifiers::CONTROL) =>
                                {
                                    app.dialog_mode = DialogMode::ConfirmQuit;
                                }
                                KeyCode::Char('o') => {
                                    // Don't open page jump when searching
                                    if !app.generation_browser.searching
                                        && !app.inference_browser.searching
                                    {
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
                                            app.dialog_mode = DialogMode::WeightsInput {
                                                for_training: true,
                                                for_inference: false,
                                            };
                                        }
                                    }
                                    KeyCode::Char('f') => {
                                        if !app.is_anything_running() {
                                            app.dialog_mode = DialogMode::InferenceInput {
                                                focused_field: InferenceField::Weights,
                                            };
                                        }
                                    }
                                    KeyCode::Char('x') => {
                                        if app.is_training_running() {
                                            app.dialog_mode = DialogMode::ConfirmStopTraining;
                                        }
                                    }
                                    KeyCode::Char('e') => {
                                        app.generation_browser.load_generations()?;
                                        // Auto-select latest generation (first in list) only if no selection exists
                                        if !app.generation_browser.filtered_generations.is_empty()
                                            && app
                                                .generation_browser
                                                .list_state
                                                .selected()
                                                .is_none()
                                        {
                                            app.generation_browser.list_state.select(Some(0));
                                            app.generation_browser.center_list(0);
                                        }
                                        app.mode = AppMode::GenerationBrowser;
                                    }
                                    KeyCode::Char('i') => {
                                        app.inference_browser.load_inferences()?;
                                        // Auto-select latest inference (first in list) only if no selection exists
                                        if !app.inference_browser.filtered_inferences.is_empty()
                                            && app.inference_browser.list_state.selected().is_none()
                                        {
                                            app.inference_browser.list_state.select(Some(0));
                                            app.inference_browser.center_list(0);
                                        }
                                        app.mode = AppMode::InferenceBrowser;
                                    }
                                    KeyCode::Char('m') => {
                                        app.view_meta_charts()?;
                                    }
                                    KeyCode::Char('l') => {
                                        app.logs_page.enter();
                                        app.mode = AppMode::Logs;
                                    }
                                    KeyCode::Char('v') => {
                                        app.mode = AppMode::ModelObservations;
                                    }
                                    _ => {}
                                },
                                AppMode::GenerationBrowser => {
                                    if app.generation_browser.searching {
                                        match key.code {
                                            KeyCode::Esc => {
                                                app.generation_browser.searching = false;
                                                app.generation_browser.search_input.clear();
                                                app.generation_browser.filter_generations();
                                            }
                                            KeyCode::Enter => {
                                                app.generation_browser.searching = false;
                                            }
                                            KeyCode::Char(c) => {
                                                app.generation_browser.search_input.push(c);
                                                app.generation_browser.filter_generations();
                                            }
                                            KeyCode::Backspace => {
                                                app.generation_browser.search_input.pop();
                                                app.generation_browser.filter_generations();
                                            }
                                            _ => {}
                                        }
                                    } else {
                                        match key.code {
                                            KeyCode::Esc | KeyCode::Char('q') => {
                                                app.mode = AppMode::Main;
                                            }
                                            KeyCode::Char('/') => {
                                                app.generation_browser.searching = true;
                                            }
                                            KeyCode::Down
                                                if key.modifiers.contains(
                                                    crossterm::event::KeyModifiers::CONTROL,
                                                ) =>
                                            {
                                                app.generation_browser.scroll_down(5);
                                            }
                                            KeyCode::Up
                                                if key.modifiers.contains(
                                                    crossterm::event::KeyModifiers::CONTROL,
                                                ) =>
                                            {
                                                app.generation_browser.scroll_up(5);
                                            }
                                            KeyCode::Down | KeyCode::Char('j') => {
                                                app.generation_browser.next();
                                            }
                                            KeyCode::Up | KeyCode::Char('k') => {
                                                app.generation_browser.previous();
                                            }
                                            KeyCode::Enter => {
                                                app.select_generation()?;
                                            }
                                            KeyCode::Char('r') => {
                                                app.generation_browser.load_generations()?;
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                                AppMode::InferenceBrowser => {
                                    if app.inference_browser.searching {
                                        match key.code {
                                            KeyCode::Esc => {
                                                app.inference_browser.searching = false;
                                                app.inference_browser.search_input.clear();
                                                app.inference_browser.filter_inferences();
                                            }
                                            KeyCode::Enter => {
                                                app.inference_browser.searching = false;
                                            }
                                            KeyCode::Char(c) => {
                                                app.inference_browser.search_input.push(c);
                                                app.inference_browser.filter_inferences();
                                            }
                                            KeyCode::Backspace => {
                                                app.inference_browser.search_input.pop();
                                                app.inference_browser.filter_inferences();
                                            }
                                            _ => {}
                                        }
                                    } else {
                                        match key.code {
                                            KeyCode::Esc | KeyCode::Char('q') => {
                                                app.mode = AppMode::Main;
                                            }
                                            KeyCode::Enter => {
                                                app.select_inference()?;
                                            }
                                            KeyCode::Char('/') => {
                                                app.inference_browser.searching = true;
                                            }
                                            KeyCode::Down
                                                if key.modifiers.contains(
                                                    crossterm::event::KeyModifiers::CONTROL,
                                                ) =>
                                            {
                                                app.inference_browser.scroll_down(5);
                                            }
                                            KeyCode::Up
                                                if key.modifiers.contains(
                                                    crossterm::event::KeyModifiers::CONTROL,
                                                ) =>
                                            {
                                                app.inference_browser.scroll_up(5);
                                            }
                                            KeyCode::Down | KeyCode::Char('j') => {
                                                app.inference_browser.next();
                                            }
                                            KeyCode::Up | KeyCode::Char('k') => {
                                                app.inference_browser.previous();
                                            }
                                            KeyCode::Char('r') => {
                                                app.inference_browser.load_inferences()?;
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
                                    KeyCode::Char('r') => {
                                        if app.chart_viewer.is_viewing_meta_charts() {
                                            app.load_latest_meta_charts()?;
                                            app.chart_viewer
                                                .load_charts(&app.latest_meta_charts)?;
                                        }
                                    }
                                    KeyCode::Char('c') => {
                                        let _ = app.chart_viewer.copy_current_image();
                                    }
                                    _ => {}
                                },
                                AppMode::Logs => match key.code {
                                    KeyCode::Esc | KeyCode::Char('q') => {
                                        app.mode = AppMode::Main;
                                    }
                                    KeyCode::Char('c') => {
                                        app.logs_page.clear_logs();
                                    }
                                    KeyCode::Down | KeyCode::Char('j') => {
                                        app.logs_page.next();
                                    }
                                    KeyCode::Up | KeyCode::Char('k') => {
                                        app.logs_page.previous();
                                    }
                                    KeyCode::PageDown => {
                                        app.logs_page.page_down();
                                    }
                                    KeyCode::PageUp => {
                                        app.logs_page.page_up();
                                    }
                                    KeyCode::Home => {
                                        app.logs_page.jump_to_top();
                                    }
                                    KeyCode::End => {
                                        app.logs_page.jump_to_bottom();
                                    }
                                    _ => {}
                                },
                                AppMode::ModelObservations => match key.code {
                                    KeyCode::Esc | KeyCode::Char('q') => {
                                        app.mode = AppMode::Main;
                                    }
                                    KeyCode::Char('r') => {}
                                    _ => {}
                                },
                            }
                        }
                    }
                }
                Event::Mouse(mouse) => match mouse.kind {
                    MouseEventKind::ScrollUp => match app.mode {
                        AppMode::GenerationBrowser => app.generation_browser.scroll_up(3),
                        AppMode::InferenceBrowser => app.inference_browser.scroll_up(3),
                        AppMode::ChartViewer => app.chart_viewer.scroll_up(3),
                        AppMode::Logs => {
                            for _ in 0..3 {
                                app.logs_page.previous();
                            }
                        }
                        _ => {}
                    },
                    MouseEventKind::ScrollDown => match app.mode {
                        AppMode::GenerationBrowser => app.generation_browser.scroll_down(3),
                        AppMode::InferenceBrowser => app.inference_browser.scroll_down(3),
                        AppMode::ChartViewer => app.chart_viewer.scroll_down(3),
                        AppMode::Logs => {
                            for _ in 0..3 {
                                app.logs_page.next();
                            }
                        }
                        _ => {}
                    },
                    MouseEventKind::Down(MouseButton::Left) => {
                        if app.mode == AppMode::GenerationBrowser {
                            let list_area = app.generation_browser.list_area;
                            if mouse.column >= list_area.x
                                && mouse.column < list_area.x + list_area.width
                                && mouse.row >= list_area.y
                                && mouse.row < list_area.y + list_area.height
                            {
                                let _ = app.handle_generation_click(mouse.row - list_area.y);
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
        AppMode::ChartViewer => {
            let is_training = app.is_training_running();
            let current_episode = app.get_current_episode();
            app.chart_viewer.render(f, is_training, current_episode);
        }
        AppMode::Logs => pages::logs_page::render(f, app),
        AppMode::ModelObservations => pages::model_observations_page::render(f, app),
    }

    // Render dialog on top if active
    match app.dialog_mode {
        DialogMode::WeightsInput {
            for_training,
            for_inference,
        } => {
            components::dialogs::weights::render(f, app, for_training, for_inference);
        }
        DialogMode::InferenceInput { focused_field } => {
            components::dialogs::inference::render(f, app, focused_field);
        }
        DialogMode::ConfirmQuit => {
            components::dialogs::confirm::render(
                f,
                "Quit?",
                "Training processes will continue running in background.",
            );
        }
        DialogMode::ConfirmStopTraining => {
            components::dialogs::confirm::render(
                f,
                "Stop Training?",
                "This will terminate the training process.",
            );
        }
        DialogMode::PageJump { selected } => {
            components::dialogs::page_jump::render(f, selected, app.mode);
        }
        DialogMode::None => {}
    }
}
