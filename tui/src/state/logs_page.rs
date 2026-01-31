use ratatui::{layout::Rect, widgets::ListState};
use std::fs;

const MAX_LOGS: usize = 1000;

pub struct LogsPageState {
    pub training_output: Vec<String>,
    pub logs_list_state: ListState,
    pub logs_list_area: Rect,
    follow: bool,
}

impl LogsPageState {
    pub fn new() -> Self {
        Self {
            training_output: Vec::new(),
            logs_list_state: ListState::default(),
            logs_list_area: Rect::default(),
            follow: true,
        }
    }

    pub fn poll_training_output(&mut self) {
        let log_path = "../training/training.log";
        if let Ok(content) = fs::read_to_string(log_path) {
            let new_lines: Vec<String> = content.lines().collect::<Vec<_>>()
                .into_iter().rev().take(MAX_LOGS).rev().map(|s| s.to_string()).collect();

            if new_lines != self.training_output {
                self.training_output = new_lines;
                if self.follow {
                    self.jump_to_bottom();
                } else if let Some(selected) = self.logs_list_state.selected() {
                    self.set_logs_position(selected);
                }
            }
        }
    }

    pub fn clear_logs(&mut self) {
        let log_path = "../training/training.log";
        let _ = fs::write(log_path, "");
        self.training_output.clear();
        self.logs_list_state.select(None);
    }

    pub fn set_logs_position(&mut self, selected: usize) {
        let visible_height = self.logs_list_area.height.saturating_sub(2) as usize;
        let offset = if selected >= visible_height.saturating_sub(1) {
            selected.saturating_sub(visible_height.saturating_sub(1))
        } else {
            0
        };
        self.logs_list_state = ListState::default().with_selected(Some(selected)).with_offset(offset);
    }

    pub fn next(&mut self) {
        self.follow = false;
        if self.training_output.is_empty() {
            return;
        }
        let i = match self.logs_list_state.selected() {
            Some(i) => {
                if i >= self.training_output.len() - 1 {
                    self.training_output.len() - 1
                } else {
                    i + 1
                }
            }
            None => 0,
        };
        self.logs_list_state.select(Some(i));
        self.set_logs_position(i);
    }

    pub fn previous(&mut self) {
        self.follow = false;
        if self.training_output.is_empty() {
            return;
        }
        let i = match self.logs_list_state.selected() {
            Some(i) => {
                if i == 0 {
                    0
                } else {
                    i - 1
                }
            }
            None => 0,
        };
        self.logs_list_state.select(Some(i));
        self.set_logs_position(i);
    }

    pub fn jump_to_top(&mut self) {
        self.follow = false;
        if !self.training_output.is_empty() {
            self.logs_list_state.select(Some(0));
            self.set_logs_position(0);
        }
    }

    pub fn jump_to_bottom(&mut self) {
        self.follow = true;
        if !self.training_output.is_empty() {
            let last = self.training_output.len() - 1;
            self.logs_list_state.select(Some(last));
            self.set_logs_position(last);
        }
    }

    pub fn enter(&mut self) {
        self.follow = true;
    }

    pub fn page_up(&mut self) {
        let page_size = self.logs_list_area.height.saturating_sub(2) as usize;
        for _ in 0..page_size {
            self.previous();
        }
    }

    pub fn page_down(&mut self) {
        let page_size = self.logs_list_area.height.saturating_sub(2) as usize;
        for _ in 0..page_size {
            self.next();
        }
    }
}
