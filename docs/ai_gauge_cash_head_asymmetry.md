# Gauge-Fixed Log-Prob: Cash Head Asymmetry Problem

## Summary

The gauge-fixed log-prob `diff = diff_ticker - diff_cash` divides the
relative logit shift by `action_noise_std` (ticker std only). When the cash
and ticker heads have different gradient magnitudes per step, the cash shift
dominates `diff` and inflates KL.

## Mechanism

Current architecture:
- `actor_proj`: `[seq_len * model_dim] -> 1` (per-ticker)
- `cash_proj`: `[TICKERS_COUNT * seq_len * model_dim] -> 1` (all-ticker)

The fan-in ratio is `TICKERS_COUNT:1`. Different fan-in → different gradient
magnitudes → cash logit drifts at a different rate than ticker logits.

The gauge-fixed log-prob measures:
```
diff_i = (u_i - mu_new_i) - (u_cash - mu_new_cash)
```
normalized by `action_noise_std` (ticker-only). If cash drifts faster than
tickers, the subtracted `diff_cash` term inflates the residual and KL.

In 5b, ticker and cash share most of the head path (`head_proj` → `head_ln`
→ `actor_out`), so their logit drifts are correlated and the gauge
subtraction stays small.

## Impact

Likely contributes to noisy KL signal independent of the separate
"high initial KL" problem (which reproduces in cleanrl without this
architecture).

## Potential Fixes

- Match fan-in: derive cash logit from the same per-ticker projections
  (e.g., mean-pool ticker logits or use a shared projection)
- Separate cash std: normalize `diff_cash` by its own learned std rather
  than reusing ticker std
- Scale cash_proj learning rate inversely with fan-in ratio
