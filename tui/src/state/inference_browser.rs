use anyhow::Result;
use ratatui::{layout::Rect, widgets::ListState};
use std::path::PathBuf;
use walkdir::WalkDir;

#[derive(Debug, Clone)]
pub struct InferenceInfo {
    pub number: usize,
    pub path: PathBuf,
}

pub struct InferenceBrowserState {
    pub inferences: Vec<InferenceInfo>,
    pub filtered_inferences: Vec<InferenceInfo>,
    pub selected_inference: Option<usize>,
    pub list_state: ListState,
    pub search_input: String,
    pub searching: bool,
    pub list_area: Rect,
}

impl InferenceBrowserState {
    pub fn new() -> Self {
        Self {
            inferences: Vec::new(),
            filtered_inferences: Vec::new(),
            selected_inference: None,
            list_state: ListState::default(),
            search_input: String::new(),
            searching: false,
            list_area: Rect::default(),
        }
    }

    pub fn load_inferences(&mut self) -> Result<()> {
        self.inferences.clear();
        let inference_path = PathBuf::from("../infer");

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

        self.inferences.sort_by(|a, b| a.number.cmp(&b.number));
        self.filter_inferences();
        Ok(())
    }

    pub fn filter_inferences(&mut self) {
        if self.search_input.is_empty() {
            if self.filtered_inferences.len() != self.inferences.len() {
                self.filtered_inferences = self.inferences.clone();
            }
        } else {
            if let Ok(search_num) = self.search_input.parse::<usize>() {
                self.filtered_inferences = self
                    .inferences
                    .iter()
                    .filter(|i| i.number == search_num)
                    .cloned()
                    .collect();
            } else {
                self.filtered_inferences = self
                    .inferences
                    .iter()
                    .filter(|i| i.number.to_string().contains(&self.search_input))
                    .cloned()
                    .collect();
            }
        }

        if let Some(selected) = self.list_state.selected() {
            if selected >= self.filtered_inferences.len() && !self.filtered_inferences.is_empty() {
                self.list_state.select(Some(0));
                self.center_list(0);
            } else if !self.filtered_inferences.is_empty() {
                self.center_list(selected);
            }
        } else if !self.filtered_inferences.is_empty() {
            self.list_state.select(Some(0));
            self.center_list(0);
        }
    }

    pub fn center_list(&mut self, selected: usize) {
        let visible_height = self.list_area.height.saturating_sub(2) as usize;
        let center = visible_height / 2;
        let offset = selected.saturating_sub(center);
        self.list_state = ListState::default().with_selected(Some(selected)).with_offset(offset);
    }

    pub fn next(&mut self) {
        if self.filtered_inferences.is_empty() {
            return;
        }
        let i = match self.list_state.selected() {
            Some(i) => {
                if i >= self.filtered_inferences.len() - 1 {
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

    pub fn previous(&mut self) {
        if self.filtered_inferences.is_empty() {
            return;
        }
        let i = match self.list_state.selected() {
            Some(i) => {
                if i == 0 {
                    self.filtered_inferences.len() - 1
                } else {
                    i - 1
                }
            }
            None => 0,
        };
        self.list_state.select(Some(i));
        self.center_list(i);
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

    pub fn jump_to_latest(&mut self) {
        if !self.filtered_inferences.is_empty() {
            self.list_state.select(Some(0));
            self.center_list(0);
        }
    }

    pub fn get_selected(&self) -> Option<&InferenceInfo> {
        self.list_state
            .selected()
            .and_then(|i| self.filtered_inferences.get(i))
    }
}
