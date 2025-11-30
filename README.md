# Trading bot 0

## Monorepo Members

- `trading_bots/` the various trading bots, most notably the `torch/` RL model
- `tui/` control and data viewing for training and inference, designed for the torch RL model
- `shared/` shared code between the members

## Usage

It's recommended to use the `tui/` to start training or inference, or to view training or inference data. It's recommended to begin training/inference from inside the tui using the controls it provides so that it can show logs and track episodes correctly.

```rust
cd tui && cargo run --release
```

## Project Structure

- `src/torch/` high-performance high-results torch RL model
- `src/agents/` and `src/strategies/` programmatic strategies with genetic algorithm training
- `src/burn/` burn model, abandoned due to poor performance
- `training/` training episodes (`gens/`) and metadata (`data/`)
- `weights/` trained or partially trained model schemas
- `infer/` inference results and metadata

## PPO Trading Agent

Multi-asset trading bot using deep RL (PPO) and historical price data.

- `src/torch/` using tch-rs, model derives from their RL example / SB3 PPO implementation with heavy modifications mentioned below
- drastically outperforms the programmatic strategies with comparable training time
- Performs exceptionally well when trained sufficiently (thousands of epochs)

### Architecture

- Continuous action space for position sizing [-1, 1] implicit sell/buy/hold and direction (short term memory/goal setting) for each ticker
- ConvNeXt-inspired Conv layers, depthwise-separable blocks for price delta % inputs
- 4-head self-attention learn inter-asset correlations
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
| 6 ticker assets (red) cash (green) combined total assets (blue) benchmarked over time (yellow) |
|                             ![](assets/msft_ep51_buy_sell_tui.png)                             |
|      buy (red) and sell (yellow) locations on randomly selected active region of trading       |
|                        ![](assets/msft_ep51_assets_benchmarked_tui.png)                        |
|                    assets and benchmark as before, but Microosoft specific                     |
|                             ![](assets/intc_ep51_buy_sell_tui.png)                             |
|                                buy and sell locations for intel                                |
|                        ![](assets/intc_ep51_assets_benchmarked_tui.png)                        |
|                       assets and benchmark as before, but Intel specific                       |

## Programmatic Strategies with Genetic Algorithm Optimization

This was the original project, but was effectively superseded by the RL models. They're here if you want to see some interesting use of genetic algorithms for parameter refinement.

- `src/agents/` and `src/strategies/` a few different strategies with parameters optimized using genetic algorithm, sees significant improvement over training
- Interesting research project but not designed for or expecting significant results

### Single ticker (NVDA) training results

|      ![Screenshot From 2024-11-27 21-38-01](https://github.com/user-attachments/assets/b7d867be-14d1-4f08-9c2f-ca6bc66d830a)      |
| :-------------------------------------------------------------------------------------------------------------------------------: |
| _Total assets doubling from $10,000 -> $20,000 over the course of a year using one of the optimized strategic trading algorithms_ |

Thanks to [rust-ibapi](https://github.com/wboayue/rust-ibapi) and its contributors for making this possible
