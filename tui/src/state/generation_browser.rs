use anyhow::Result;
use ratatui::{layout::Rect, widgets::ListState};
use std::path::PathBuf;
use walkdir::WalkDir;

#[derive(Debug, Clone)]
pub struct GenerationInfo {
    pub number: usize,
    pub path: PathBuf,
}

pub struct GenerationBrowserState {
    pub generations: Vec<GenerationInfo>,
    pub filtered_generations: Vec<GenerationInfo>,
    pub selected_generation: Option<usize>,
    pub list_state: ListState,
    pub search_input: String,
    pub searching: bool,
    pub list_area: Rect,
}

impl GenerationBrowserState {
    pub fn new() -> Self {
        Self {
            generations: Vec::new(),
            filtered_generations: Vec::new(),
            selected_generation: None,
            list_state: ListState::default(),
            search_input: String::new(),
            searching: false,
            list_area: Rect::default(),
        }
    }

    pub fn load_generations(&mut self) -> Result<()> {
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

        self.generations.sort_by(|a, b| b.number.cmp(&a.number));
        self.filter_generations();
        Ok(())
    }

    pub fn filter_generations(&mut self) {
        if self.search_input.is_empty() {
            if self.filtered_generations.len() != self.generations.len() {
                self.filtered_generations = self.generations.clone();
            }
        } else {
            if let Ok(search_num) = self.search_input.parse::<usize>() {
                self.filtered_generations = self
                    .generations
                    .iter()
                    .filter(|g| g.number == search_num)
                    .cloned()
                    .collect();
            } else {
                self.filtered_generations = self
                    .generations
                    .iter()
                    .filter(|g| g.number.to_string().contains(&self.search_input))
                    .cloned()
                    .collect();
            }
        }

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

    pub fn center_list(&mut self, selected: usize) {
        let visible_height = self.list_area.height.saturating_sub(2) as usize;
        let center = visible_height / 2;
        let offset = selected.saturating_sub(center);
        self.list_state = ListState::default().with_selected(Some(selected)).with_offset(offset);
    }

    pub fn next(&mut self) {
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

    pub fn previous(&mut self) {
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
        if !self.filtered_generations.is_empty() {
            self.list_state.select(Some(0));
            self.center_list(0);
        }
    }

    pub fn get_selected(&self) -> Option<&GenerationInfo> {
        self.list_state
            .selected()
            .and_then(|i| self.filtered_generations.get(i))
    }
}
