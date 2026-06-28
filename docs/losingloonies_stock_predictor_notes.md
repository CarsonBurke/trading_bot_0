# LosingLoonies Stock Predictor Notes

Source scope: transcripts were pulled from the 10 videos surfaced by YouTube's "Machine Learning Stock Predictions" playlist on the LosingLoonies channel. The adjacent channel page also has related one-off tools, but the stock-predictor playlist is the clean scope for the community AI/LSTM predictor series.

Raw captions were used for analysis but are not checked in. YouTube links and timestamps below point to the relevant source material.

## Video Set

| Date | Video | Notes |
| --- | --- | --- |
| 2025-09-18 | [Can AI Predict Tomorrow's Stock Price?](https://www.youtube.com/watch?v=aiFpAl3mgGk) | Initial Python LSTM stock predictor. |
| 2025-10-10 | [Letting My Viewers Fix My Stock Prediction Program](https://www.youtube.com/watch?v=q24MVGCUr4w) | Moves from binary up/down to log-return regression. |
| 2025-11-08 | [My AI Stock Predictor Just Got Smarter](https://www.youtube.com/watch?v=V2l7cZxUpQs) | Multi-horizon predicted returns plus uncertainty penalty. |
| 2025-12-11 | [My LSTM Stock Predictor Just Got A Reality Check](https://www.youtube.com/watch?v=kRa3PUxNBTM) | Identifies leakage/overlap and criticizes MSE regression on noisy returns. |
| 2026-02-14 | [MASSIVE Improvements to My AI Stock Predictor](https://www.youtube.com/watch?v=1F0gYkk7YYw) | More realistic no-trade behavior, adjusted probability, purged walk-forward split. |
| 2026-03-21 | [My AI Stock Predictor Learned to Be Afraid](https://www.youtube.com/watch?v=XZwnrpSOUvU) | Adds volatility/fear/greed features. |
| 2026-03-28 | [Can Politician Trading Beat the Market? I Tested It With AI](https://www.youtube.com/watch?v=fS86gf6E4jg) | Adds politician-trade features; results appear worse in that run. |
| 2026-05-02 | [My AI Stock Predictor vs 10,000 Monkeys](https://www.youtube.com/watch?v=Lh1vrIcpJN4) | Compares live picks against random-stock baselines. |
| 2026-05-16 | [My AI Stock Predictor Can Now See Hidden Signals](https://www.youtube.com/watch?v=AZ-H7Fp2cDk) | Adds Fourier-derived frequency features; emphasizes repeated trials. |
| 2026-06-20 | [I Gave My AI the World's Biggest Money Printer](https://www.youtube.com/watch?v=t2f0vyfABdM) | Adds Federal Reserve/rate-regime features; evaluates mean AI portfolios over many runs. |

## What The Predictor Is

It is described as an LSTM-based stock-prediction system that trains on historical market windows and auxiliary features, then ranks stocks for possible buying. The repeated framing is not "forecast one ticker's exact price" as an end in itself; it evolves into "select tradable opportunities with expected outsized returns and acceptable uncertainty."

Core implementation concepts mentioned across the transcripts:

- LSTM/long short-term memory network over rolling historical windows, initially around one month of market data.
- Monte Carlo dropout to produce repeated predictions, then estimate a mean prediction and standard deviation/uncertainty.
- Train/validation/test splits intended to avoid future leakage, later revised to walk-forward/purged splits.
- Backtests that compare the model against buy-and-hold SPY, random-stock portfolios, and later repeated AI runs.
- A trading layer that can choose not to buy when confidence or adjusted score is too low.

Feature families mentioned:

- Price/market data: close/open/high/low style price history, intraday range, returns/log returns, rolling volatility, momentum.
- Volume-based or engineered features are discussed as possible/viewer-suggested improvements.
- News and sentiment: news sentiment scores, article counts.
- Insider/political signals: insider trading activity and later politician trades.
- Market regime/emotion: VIX, CNN Fear and Greed Index, rolling volatility averages, and per-stock fear/greed sensitivity or correlation.
- Signal-processing features: Fourier-transform peak/frequency features from price time series.
- Macro/Fed features: Fed balance sheet/QE-style data, current rate level, recent rate changes, and a binary "hiking mode" indicator.
- The videos also make broad claims that fundamentals and earnings-report-derived data can be inputs, but the transcripts do not enumerate the exact schema.

## Target Evolution

The target changes materially over the series.

1. Initial next-day direction classification.
   - The first video says the model creates a binary target for whether next-day return is positive or negative, then forecasts the probability a stock will go up.
   - It also describes buy/hold signals based on a probability threshold, and a top-3 strategy selecting stocks with the highest probability of going up the next day.
   - Timestamp anchors: `aiFpAl3mgGk` around 1:12, 1:27, 3:26, 4:42, 7:11.

2. Next-day log-return regression.
   - The second video accepts a viewer suggestion to predict tomorrow's log return rather than only "up or down."
   - The rationale is that direction ignores magnitude: a small up move and a large up move are not equivalent trading opportunities.
   - The backtest then ranks stocks by predicted log return and avoids buying when all predicted log returns are below zero.
   - Timestamp anchors: `q24MVGCUr4w` around 2:27, 2:45, 2:59, 4:02.

3. Multi-horizon expected return.
   - The third video expands predictions across horizons like 1 day, 1 week, 1 month, and 6 months.
   - Selection is normalized across horizons, roughly by comparing predicted return per unit time rather than raw return alone.
   - It uses Monte Carlo dropout to estimate mean predicted log return and uncertainty, then penalizes predictions by subtracting an uncertainty multiple.
   - The trading decision uses adjusted expected return and probability of positive return, with portfolio diversification among top candidates. In the transcript, one gate is that probability of positive return should exceed about 70%.
   - Timestamp anchors: `V2l7cZxUpQs` around 0:58, 1:21, 3:19, 3:46, 4:06, 4:23, 5:00.

4. Tradable-threshold classification.
   - The V4/reality-check video argues that exact daily log-return regression is too noisy and pushes the model toward trivial mean predictions.
   - It changes the question to whether return exceeds a tradable threshold, for example whether a one-day return is above 2%.
   - It further suggests deriving thresholds from average returns plus a sigma multiple per forecast horizon, so labels mean "unusually high return" rather than merely "positive return."
   - Timestamp anchors: `kRa3PUxNBTM` around 5:12, 6:03, 6:28, 6:38, 7:03, 7:25.

5. Uncertainty-adjusted buy/no-buy classification.
   - The February 2026 video frames the predictor as a classifier asking whether returns exceed a meaningful minimum threshold over horizons such as weeks to months.
   - The trading rule becomes: buy only if adjusted probability clears a minimum threshold, where adjusted probability is predicted probability minus uncertainty/std of that probability.
   - This is the clearest "learn to do nothing" version: the model should avoid forced buys.
   - Timestamp anchors: `1F0gYkk7YYw` around 0:55, 1:29, 1:40, 2:01, 2:58, 3:07.

6. Live-trading shorthand.
   - The "10,000 Monkeys" video explains the system more simply as using about one month of inputs to predict tomorrow/market-close value, then buying at open and selling near close.
   - This is less precise than the target evolution above and may be presentation shorthand. The transcript does not prove the live system returned to literal exact-price regression.
   - Timestamp anchors: `Lh1vrIcpJN4` around 0:52, 1:40, 2:47.

## Loss / Objective

This is the key part to treat carefully.

- Explicitly stated loss: the V4/reality-check video says the noisy log-return regression target was paired with mean squared error, MSE. It criticizes MSE because it rewards predicting the mean on tiny noisy daily returns, producing a low loss while learning little of trading value. Timestamp: `kRa3PUxNBTM` around 6:03-6:19.
- Initial binary up/down classification loss: not named in the transcript. The video discusses accuracy and probability thresholds, but does not explicitly say binary cross entropy, cross entropy, focal loss, etc.
- Later tradable-threshold classification loss: also not named in the transcript. The videos discuss classification targets, predicted probabilities, AUC, validation AUC, accuracy, and confidence thresholds. It is reasonable to infer a classification objective is used, but the exact training loss is not transcript-supported.
- Practical objective after V4: optimize useful trading selection rather than minimize raw forecast error. The videos increasingly evaluate whether predictions beat random portfolios and whether confidence-adjusted signals produce better equity curves.

In short: the only named loss from transcripts is MSE for the log-return regression phase, and it is presented as a mistake. The later classifier's precise loss remains unknown from these public videos.

## Evaluation And Backtest Behavior

Notable evaluation details and caveats:

- Early sanity checks intentionally include target leakage to confirm the model can latch onto an obvious feature, then remove leaked fields such as next-day close/target columns. `aiFpAl3mgGk` around 4:15-6:08.
- Initial training often shows training accuracy improving while validation accuracy does not, suggesting memorization/overfit rather than learnable signal. `aiFpAl3mgGk` around 6:26-6:55.
- The first backtest compares top-probability stock selection against buy-and-hold SPY and random-stock distributions; the creator concludes apparent outperformance could be random. `aiFpAl3mgGk` around 6:51-7:43.
- A major V4 critique is overlapping-window leakage: training and validation windows can differ by only one day, letting the model memorize nearby prices/noise rather than generalize. `kRa3PUxNBTM` around 3:25-4:22.
- The February 2026 video moves to walk-forward splits and purging: train on past, test on future, and drop training windows whose label reaches into validation. Validation/test remain in time order. `1F0gYkk7YYw` around 5:31-5:56.
- Survivorship bias is explicitly acknowledged: the ticker universe consists of stocks that exist today. `1F0gYkk7YYw` around 3:30.
- Simple accuracy is repeatedly treated as insufficient. One example: 87 of 147 buys positive, around 59%, is not meaningful if roughly 60% of days in the validation window are positive. `V2l7cZxUpQs` around 5:13-5:22.
- AUC is used later, but the reported behavior is poor: training AUC climbs while validation AUC drifts from about 0.6 toward 0.5, again suggesting no robust signal. `1F0gYkk7YYw` around 7:06.
- Random baselines become increasingly important: random-stock runs, 10,000 "monkeys," repeated AI portfolios, mean AI equity, and one-sigma bands are used to distinguish luck from signal. `Lh1vrIcpJN4` around 5:39-9:27 and `t2f0vyfABdM` around 2:30-4:46.
- Single live trades are treated as anecdotes, not proof. The "monkeys" video includes examples where random selections beat the AI on a given day.
- Live execution assumptions are fragile: the "monkeys" video assumes buy-at-open/sell-near-close behavior, while the creator admits missing some intended entries/exits. This means simulated open-close returns can diverge from real fills. `Lh1vrIcpJN4` around 2:47 and 6:38.
- Transaction costs, slippage, liquidity constraints, taxes, borrow constraints, and delisting handling were not meaningfully evidenced in the transcripts reviewed.

## Feature Experiments

Fear and greed:

- Adds VIX-derived volatility features, then considers CNN Fear and Greed data.
- Adds a per-stock fear/greed sensitivity factor based on backward-looking correlation over about 60 days, allowing stocks to be positively or negatively correlated with market fear.
- The video reports backtest improvement but explicitly says this does not prove the model works.
- Timestamp anchors: `XZwnrpSOUvU` around 1:51, 2:40, 5:10, 6:25, 7:17.

Politician trades:

- Adds Senate/politician trades as model inputs, including trade counts, buys/sells, options/stocks, and amount estimates; transforms large values so they do not dominate.
- In the shown run, compound annual growth with politician trades is reported around 22%, lower than a prior 36% figure, so it appears not to help in that experiment.
- The transcript also notes STOCK Act filing delays, which weakens the case that the signal would remain unpriced by the time it is public.
- Timestamp anchors: `fS86gf6E4jg` around 0:52, 2:42, 4:50.

Fourier features:

- Runs Fourier transforms over stock price series and initially considers feeding frequencies into the model.
- Then simplifies to finding the largest peak in frequency data and using that as an input feature.
- The important methodological point is the warning against one backtest: the video argues for many repeated runs to estimate true accuracy.
- Timestamp anchors: `AZ-H7Fp2cDk` around 3:39, 4:22, 5:07, 5:46, 6:03.

Federal Reserve / rates:

- Adds Fed/rate-regime information: Fed balance sheet/QE-style data, rate level, changes over time, and a binary hiking-mode signal.
- Evaluation uses 100 Monte Carlo AI runs, with mean AI portfolio slightly above a baseline in the shown result.
- The creator is skeptical because Fed data is watched by everyone and likely priced in; he argues future useful features probably need unusual, less-priced-in information.
- Timestamp anchors: `t2f0vyfABdM` around 0:35, 2:53, 7:06, 8:11.

## Takeaways For Our RL Trading Work

- Do not optimize an easy loss that is disconnected from trading value. The explicit MSE example is a warning: low regression error on noisy tiny returns can be meaningless.
- Classification targets need to be economically meaningful. "Positive return" is weak; "exceeds costs/slippage/risk-adjusted threshold" is closer to a useful decision label.
- A no-trade action matters. Forced daily buying causes the model to express weak/noisy scores as trades.
- Evaluation needs random-policy distributions, repeated seeds, and confidence bands, not one equity curve.
- Time-series validation must purge overlapping label windows and preserve chronological test order.
- Accuracy is usually the wrong headline metric. Compare against base rates, random portfolios, drawdowns, risk-adjusted returns, turnover, and transaction costs.
- Uncertainty estimates are useful only if calibrated. The series uses Monte Carlo dropout std as a penalty, but the videos do not demonstrate probability calibration.
- Feature additions should be judged by out-of-sample incremental value, not intuitive appeal. Several "obvious" features are acknowledged as likely priced in.
