# Trading Bot Monorepo

This repository has been reorganized into a monorepo structure with multiple components.

## Repository Structure

```
.
├── trading_bots/     # Main trading bot implementation (Rust)
│   ├── src/
│   ├── Cargo.toml
│   └── ...
├── tui/              # Terminal User Interface for managing training
│   ├── src/
│   └── Cargo.toml
├── weights/          # Model weights (shared)
├── training/         # Training outputs and charts (shared)
├── data/             # Historical price data (shared)
├── long_data/        # Extended historical data (shared)
└── infer/            # Inference results (shared)
```

## Components

### Trading Bots (`trading_bots/`)

The main trading bot implementation using PPO (Proximal Policy Optimization) with PyTorch.

**Run training:**
```bash
cd trading_bots
cargo run --release -- train
```

**Run training with existing weights:**
```bash
cd trading_bots
cargo run --release -- train -w ../weights/ppo_ep300.ot
```

**Run inference:**
```bash
cd trading_bots
cargo run --release -- infer -w ../weights/ppo_ep300.ot
```

### TUI (`tui/`)

Terminal User Interface for managing training sessions and viewing results.

**Features:**
- Start/stop training with or without loading weights
- Browse training generations
- View charts organized by folder/file structure
- Real-time training status monitoring

**Run the TUI:**
```bash
cd tui
cargo run --release
```

**Controls:**

*Main Screen:*
- `s` - Start training from scratch
- `w` - Start training with weights (set weights path first)
- `x` - Stop training
- `g` - Open generation browser
- `i` - Set weights path (type path and press `i` to confirm)
- `q` - Quit

*Generation Browser:*
- `↑/k` - Move up
- `↓/j` - Move down
- `Enter` - View generation charts
- `r` - Refresh generation list
- `q/Esc` - Back to main screen

*Chart Viewer:*
- `↑/k` - Move up
- `↓/j` - Move down
- `Enter` - Expand/collapse folders
- `q/Esc` - Back to generation browser

## Shared Folders

These folders are accessible by both the trading bot and the TUI:

- **`weights/`** - Stores trained model weights (`.ot` files)
- **`training/`** - Contains training outputs, generation charts, and data analysis
  - `training/gens/` - Per-generation charts and metrics
  - `training/data/` - Data analysis charts (volatility, returns distribution, etc.)
- **`data/`** - Historical price data
- **`long_data/`** - Extended historical price data for training
- **`infer/`** - Inference results and charts

## Development

### Building

Each component can be built independently:

```bash
# Build trading bots
cd trading_bots && cargo build --release

# Build TUI
cd tui && cargo build --release
```

### Dependencies

The trading bots component requires:
- CUDA-enabled GPU (for training/inference)
- PyTorch C++ libraries (libtorch)

The TUI component requires:
- Standard Rust toolchain
- No GPU required

## Workflow

1. **Start the TUI:**
   ```bash
   cd tui && cargo run --release
   ```

2. **Configure weights path** (optional):
   - Type the path to weights file (e.g., `../weights/ppo_ep300.ot`)
   - Press `i` to confirm

3. **Start training:**
   - Press `s` for training from scratch
   - Press `w` for training with loaded weights

4. **Monitor progress:**
   - Training runs in the background
   - Status updates show "Training: RUNNING"

5. **View results:**
   - Press `g` to open generation browser
   - Navigate with `↑/↓` or `j/k`
   - Press `Enter` to view charts for a generation

6. **Stop training:**
   - Press `x` to stop training
   - Training process will be killed gracefully

## Notes

- All paths in the code use relative paths (`../`) to reference shared folders
- The TUI runs training as a child process and monitors its status
- Generation charts are automatically loaded when training completes
- Press `r` in the generation browser to refresh the list after training
