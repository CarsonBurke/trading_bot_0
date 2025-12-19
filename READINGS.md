# Recommended Readings

Collection of readings I found helpful when constructing the torch trading bot. Much of these are time series prediction focused and not as much trading focused, but I think those are more similar than one might originally think. In the end, they are *time and state analysis* models, hence why you see transformers and mamba also used in LLMs and image models.

## Mamba and SSM (State Space Models)

Current architecture is based on mamba2/ssd.

I've found Mamba very compelling because of its performance with long sequences while having attention mechanisms like transformers.

- [original mamba paper](https://arxiv.org/pdf/2312.00752)
- [mamba2/ssd paper](https://arxiv.org/pdf/2405.21060)
  - transformer-like attention with better performance and suitability for time series analysis
- [tridao blog](https://tridao.me/blog/2024/mamba2-part1-model/) (1/2 co-author of the model)
  - more accessible than the paper, details architecture
- [nvidia paper mamba2 paper](https://research.nvidia.com/publication/2024-06_empirical-study-mamba-based-language-models)
  - comparison of mamba2 with more conventional transformer architectures
- [Autoregressive Image Generation with Mamba](https://arxiv.org/pdf/2408.12245)
  
## Transformer Models

My focus has been more on mamba, but these architectures seem interesting and capable. I've yet to find good benchmarks to compare SoTA implementations of mamba2 versus TimeXer/EiFormer.

- [iTransformer code+paper](https://github.com/thuml/iTransformer)
- [EiFormer](https://arxiv.org/html/2503.10858)
  - Efficiency optimizations to iTransformer
- [TimeXer](https://arxiv.org/abs/2402.19072)
  - extension of iTransformer by the original authors - improved efficiency, ability to digest exogenous variables (momentum, volatility indicators; ema, etc.) as opposed to just endogenous variables (time series data)
  
## Conv and LSTM

From what I've read, pure CNNs and LSTM are generally overshadowed by more modern Transformer or SSM architectures - though conv layers are used *inside* mamba2, for example.

- [TimesNet](https://openreview.net/pdf?id=ju_Uqw384Oq)

## Trading models

- https://arxiv.org/pdf/2501.06832
  - comparison against models like PatchTST and FinRL
- [MetaTrader](https://arxiv.org/pdf/2210.01774)
- I had more but I've lost them in my notes.
