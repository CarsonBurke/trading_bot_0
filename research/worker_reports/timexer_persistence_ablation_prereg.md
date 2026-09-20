# Pre-registration: contemporaneous vs persistent cross-section channels

Written before the arm exists. Thresholds fixed here are NOT adjusted after seeing results.

## The observation being explained

Step-matched `held-out sample` market-neutral four-channel MSE ratio (below 1.0 beats persistence),
first family (`dispersion` + `cross-section-z`, 14 aux channels) vs second family (those plus
`cross-section-rank` + `relative-volume` + `range-z`, 22 aux channels):

| h | 1st @1000 | 2nd @1000 | 1st @2000 | 2nd @2000 | 2nd @2500 |
|---|-----------|-----------|-----------|-----------|-----------|
| 1 | 0.9542 | 0.94674 | 0.9489 | 0.95399 | 0.95971 |
| 8 | 0.9677 | 0.94248 | 0.9347 | 0.95237 | 0.96734 |
| 16 | 0.9736 | 0.95542 | 0.9375 | 0.95805 | 0.96987 |
| 32 | 0.9876 | 0.96669 | 0.9564 | 0.97232 | 0.98075 |

The second family is better at EVERY horizon at step 1000 and worse at EVERY horizon at step 2000,
and its own trajectory runs backwards from step 1000 onward. The first family improved over the
same interval.

## Hypothesis

The three added channels are not one kind of quantity:

- `cross-section-rank` is CONTEMPORANEOUS. It is the normal score of the own return's rank inside
  the slot. Return ranks are not persistent, so the channel carries no stable per-ticker label.
- `relative-volume` and `range-z` are PERSISTENT CHARACTERISTICS. A large-cap's cross-sectional
  `ln(volume)` z-score is nearly constant across years; a volatile name stays volatile. Both are
  therefore close to a per-ticker identity/size label that happens to be expressed in
  cross-sectional units.

A stable per-ticker label lets the model memorize that ticker's mean return in the training span.
That buys early fit - a per-ticker intercept is genuinely predictive in-sample - and then degrades
as memorization displaces generalization. This predicts exactly the observed shape: an early gain
at every horizon followed by a monotone reversal.

The first family is immune by construction: `dispersion` is universe-level and identical for every
row in a slot, and `cross-section-z` is standardized against the slot, so neither carries a
per-ticker constant.

## The arm

`--features time-of-day,day-of-week,session-gap,volume,market,spy,dispersion,cross-section-z,cross-section-rank`
with everything else identical to the two arms above, `--max-steps 2500`.

Aux channels 22 -> 18. This is the second family MINUS the two persistent channels.

## Predictions, in falsifiable form

CONFIRMED (persistence hypothesis) requires BOTH:
1. Step 1000 keeps most of the second family's early gain: h=1 ratio <= 0.9500 and h=8 <= 0.9500.
   The second family reached 0.94674 / 0.94248; the first family was 0.9542 / 0.9677. A rank-only
   arm at or below 0.9500 at both horizons places it on the second family's side of that gap.
2. The reversal is ABSENT: h=1 and h=8 ratios at step 2000 are each <= their own step-1000 value.
   Direction only, because the step-to-step improvement size is not predicted.

REFUTED if the reversal persists with the two persistent channels removed, i.e. either h=1 or h=8
at step 2000 exceeds its step-1000 value by more than 0.0020 (twice the largest step-matched
reproducibility perturbation observed, 8.5e-17 relative being far below this, so 0.0020 is set by
the arm-to-arm trajectory scatter rather than by numerical noise). That outcome would mean the
degradation is caused by added capacity or by the 27% host-batch growth rather than by persistence,
and the next test is a channel-count control: three contemporaneous channels of pure noise.

UNRESOLVED otherwise, and must not be written up as a verdict.

## Secondary reading, labelled and conditional

Anchored `held-out full` student IC at h=1 against the first family's 0.19362 (single-arm SE
0.009976 over 245 paired cross-sections). Reported, not decisive: this arm's purpose is the
trajectory shape, and one IC point cannot separate the two hypotheses. A rank-only IC at or above
0.19362 alongside CONFIRMED trajectory behaviour would make the rank channel the recipe.

## What this pre-registration does not claim

- Nothing about calibrated ratios. The amplitude gain was identity in every arm measured so far
  (the fit's intercept gate refused it), so no gained number from any existing arm is quotable.
- Nothing about which of `relative-volume` or `range-z` is responsible. If CONFIRMED, separating
  them needs a further arm and is not attempted here.
- No claim that persistence is the ONLY mechanism; the channel-count control named under REFUTED is
  the test that would distinguish capacity from persistence.
