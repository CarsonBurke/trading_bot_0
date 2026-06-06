# Recommended Readings

Collection of readings I found helpful when constructing the torch trading bot. Much of these are time series prediction focused and not as much trading focused, but I think those are more similar than one might originally think. In the end, they are time and state analysis models, which is why the same families of ideas show up across forecasting, RL, and sequence modeling.

## Transformer Models

The active architecture is attention-based, so these are the most directly relevant references.

- [iTransformer code+paper](https://github.com/thuml/iTransformer)
- [EiFormer](https://arxiv.org/html/2503.10858)
  - Efficiency optimizations to iTransformer
- [TimeXer](https://arxiv.org/abs/2402.19072)
  - extension of iTransformer by the original authors - improved efficiency, ability to digest exogenous variables (momentum, volatility indicators; ema, etc.) as opposed to just endogenous variables (time series data)
  
## Conv and LSTM

From what I've read, pure CNNs and LSTM are generally overshadowed by more modern transformer architectures for this kind of long-context sequence modeling, though conv layers are still useful inside the feature stack.

- [TimesNet](https://openreview.net/pdf?id=ju_Uqw384Oq)

## Trading models

- https://arxiv.org/pdf/2501.06832
  - comparison against models like PatchTST and FinRL
- [MetaTrader](https://arxiv.org/pdf/2210.01774)
- I had more but I've lost them in my notes.
