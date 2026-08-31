# Trading bot 0

## Monorepo Members

- `trading_bots/` the various trading bots, most notably the `torch/` RL model
- `tui/` control and data viewing for training and inference, designed for the torch RL model
- `shared/` shared code between the members
- `report_cli/` CLI for debugging training and inference results

## Usage

The supported ML toolchain is declared in `.pytorch-version`. Install and verify it,
then run ML builds through the validating launcher:

```bash
./setup-fa4.sh
./torch-env.sh check
./torch-env.sh cargo check -p trading_bot_0
./torch-env.sh cargo run -p benchmarks --release
./trading_bots/run-release-cuda.sh train
./torch-env.sh ldd target/release/trading_bot_0
```

The categorical world-model pretrainer remains the `pretrain` command. The maintained
c277 MSE-JEPA/LeJEPA model is an isolated path with its own checkpoints and report family:

```bash
./trading_bots/run-release-cuda.sh pretrain
./trading_bots/run-release-cuda.sh pretrain-mse-jepa --run mse_jepa_v1
./trading_bots/run-release-cuda.sh pretrain-mse-jepa \
  --weights training/runs/mse_jepa_v1/weights/mse_jepa_best.ot \
  --run mse_jepa_v1_warmstart
```

`pretrain-mse-jepa` uses the mmap bar corpus, a fixed 6,000-bar context, and the
LeWM future-latent SIGReg default `--lambda-sigreg 0.09`. It accepts authenticated
`mse_jepa*.ot` bundles and authenticated historical c277 `pretrain_heads*.ot` artifacts;
a historical `pretrain_model*.ot` argument resolves only to its sibling heads artifact.
Categorical and other LeJEPA checkpoint families are rejected.

The launcher rejects mismatched PyTorch/CUDA builds before Cargo starts. ML binaries
embed the matching wheel's library directory so normal execution does not depend on
`LD_LIBRARY_PATH`; the `ldd` check deliberately clears that variable while validating
the binary's actual resolution.

It's recommended to use the `tui/` to start training or inference, or to view training or inference data. It's recommended to begin training/inference from inside the tui using the controls it provides so that it can show logs and track episodes correctly.

```rust
cd tui && cargo run --release
```

## Project Structure

- `src/torch/` high-performance high-results torch RL model
- `src/agents/` and `src/strategies/` programmatic strategies with genetic algorithm training
- `src/burn/` burn model, abandoned due to poor performance
- `training/` training episode reports (`gens/`) and metadata (`data/`)
- `weights/` trained or partially trained model schemas
- `infer/` inference reports and metadata

## PPO Trading Agent

Multi-asset trading bot using deep RL (PPO) and historical price data.

- `src/torch/` using tch-rs, model derives from their RL example / SB3 PPO implementation with heavy modifications mentioned below
- drastically outperforms the programmatic strategies with comparable training time
- Performs exceptionally well when trained sufficiently (thousands of epochs)

### Notable Architecture Implementations

- 1.8M parameters
- GQA-based temporal attention stack with streamed prefix/suffix cache for `uniform-stream`
- Continuous action space for position sizing [-1, 1] implicit sell/buy/hold and direction (short term memory/goal setting) for each ticker
- Timesnet-inspired Conv layers for price delta % observations
- self-attention layers for weighting time, static inputs, and cross-ticker
- Separate FC paths after shared conv features for actor and critic independent policy/value optimization
- Inputs price delta %s from current and previous time steps, intention to support more inputs such as news and social sentiment scores per-ticker and for economy

### Features

- Can trade a single ticker or multiple tickers simultaneously, balancing a portfolio
- Can train and infer on some consumer hardware, needs ~12GB VRAM (I use an RTX 5090 with good success)
- Built to use the IBKR API to download historical data for training
- Intention to do live trading / paper trading with IBKR API

### Single ticker (NVDA) training results

|                             Training Visualizations in Custom TUI                              |
| :--------------------------------------------------------------------------------------------: |
|                      ![](assets/6_ticker_ep51_assets_benchmarked_tui.png)                      |
| 6 ticker assets (red) cash (green) combined total assets (blue) benchmarked by "index" of tickers being traded (yellow); shown here outperforming the index by 11.79% |
|                             ![](assets/msft_ep51_buy_sell_tui.png)                             |
|      buy (red) and sell (yellow) locations on randomly selected active region of trading       |
|                        ![](assets/msft_ep51_assets_benchmarked_tui.png)                        |
|                    assets and benchmark as before, but MSFT specific                     |
|                             ![](assets/intc_ep51_buy_sell_tui.png)                             |
|                                buy and sell locations for intel                                |
|                        ![](assets/intc_ep51_assets_benchmarked_tui.png)                        |
|                       assets and benchmark as before, but INTC specific                       |

## Programmatic Strategies with Genetic Algorithm Optimization

This was the original project, but was effectively superseded by the RL models. They're here if you want to see some interesting use of genetic algorithms for parameter refinement.

- `src/agents/` and `src/strategies/` a few different strategies with parameters optimized using genetic algorithm, sees significant improvement over training
- Interesting research project but not designed for or expecting significant results

### Single ticker (NVDA) training results

|      ![Screenshot From 2024-11-27 21-38-01](https://github.com/user-attachments/assets/b7d867be-14d1-4f08-9c2f-ca6bc66d830a)      |
| :-------------------------------------------------------------------------------------------------------------------------------: |
| _Total assets doubling from $10,000 -> $20,000 over the course of a year using one of the optimized strategic trading algorithms_ |

Thanks to [rust-ibapi](https://github.com/wboayue/rust-ibapi) and its contributors for making this possible
