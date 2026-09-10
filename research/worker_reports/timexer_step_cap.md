# `--max-steps`: a stated-step-count run length for CausalPatch arms

## The flag

`train-timexer-segment --max-steps <N>`, default `0` = no cap (run to `--epochs` /
`--preview-patience` exactly as before). Non-zero `N` stops the run at exactly `N` optimizer
steps with a clean shutdown, not a truncation.

`TrainArgs::validate` (runner.rs) is now the FIRST statement of `train`, before
`RunDir::ensure_creatable`, `cuda_device` and `Corpus::load`. It refuses `1..=CAPTURE_AFTER_STEPS`
(i.e. 1–5): an arm capped at or below the CUDA-graph capture warmup never runs a single captured
step, so it does not measure the execution path the baselines ran. `0` and `>= 6` are accepted.
DEMONSTRATED by `an_invalid_step_cap_is_refused_before_the_corpus_loads`, which calls `train`
with a nonexistent `--data-dir` and gets the cap's error rather than a missing-data one — only
possible if the check precedes the corpus load.

## Exit-path semantics

`StopRules` (runner.rs) owns every run-length rule; the step loop holds no stopping policy of
its own, which is what makes the policy testable without a device.

- `StopRules::evaluates(step, epoch_complete)` = `step % eval_every == 0 || epoch_complete ||
  step >= cap`. The cap therefore forces a report interval of its own, so a cap that no
  `--eval-every` boundary lands on (2,500) still evaluates, checkpoints and writes every report
  base AT 2,500. Nothing about a capped run's last interval is lost.
- `StopRules::termination(Progress)` returns the exit, in precedence order:
  `preview-patience` > `step-cap` > `epoch-patience` > `epoch-limit`. Preview patience outranks
  the cap because "the held-out NLL stopped improving" is the stronger statement about the arm;
  the cap outranks both epoch exits, so an arm whose cap lands on the last step of its last
  epoch is still reported as capped.
- Four reasons, not three: `--patience` (complete epochs without improved held-out full NLL) and
  `--epochs` are different exits and a run with `--epochs > 1` can take either. Collapsing them
  would reintroduce exactly the ambiguity this field exists to remove.

The manifest gains `max_steps: Option<usize>` and `termination: Option<Termination>`
(`"step-cap" | "preview-patience" | "epoch-patience" | "epoch-limit"`), both
`skip_serializing_if = "Option::is_none"`. `termination` is `None` in every `preview-latest`
written while the run was still going, and `Some` in the last one — a capped arm can never be
read as one that early-stopped on patience or ran its epoch out. Because both fields are skipped
when absent, an UNCAPPED run's manifest JSON (and the digest authenticating it) is byte-identical
to a pre-cap manifest, so every checkpoint already on disk still authenticates; DEMONSTRATED in
the manifest test. FORMAT and OBJECTIVE are unchanged: the numerical contract did not move.

Nothing was added to the report registry and no base was added or renamed.

## `--max-steps` does not touch the learning-rate schedule

`schedule_budget(&args, steps_per_epoch)` is the schedule's only input: `--schedule-budget`
where stated, `steps_per_epoch * epochs` where 0. `max_steps` is not an input.
`a_step_cap_leaves_the_learning_rate_schedule_untouched` pins it bit-for-bit: at
`--schedule-budget 9590`, `scale(step).to_bits()` and `muon_momentum(step).to_bits()` are equal
for every step in `0..4000` between a capped and an uncapped arm, `cooldown_start()` is 3,836 in
both (the base rate holds through step 3,835 and the capped arm spends only its last 164 steps at
the very start of the warmdown, ending at 0.976 of the base rate), and a schedule that HAD leaked
the cap in would have cooled from step 1,600 and ended at the 0.15 floor. `--schedule-budget 0`
with a cap still means the whole planned epoch (9,590), not the cap.

## The mid-epoch exit and the final full-split validation: YES, it fires

A cap at 4,000 lands mid-epoch, where the interval scores the held-out SAMPLE (2,048 windows);
only an epoch-complete interval scores `held-out full`. So without an extra pass a capped arm
would carry no full split at all — the state `--preview-patience` stops have always been in.

Implemented: any run that ends MID-EPOCH (step cap or preview patience) runs one final
full-split pass over the 433,303 held-out origins before exit. It is deliberately NOT the
interval's own evaluation — selection, `best_step`, `preview-best` and every step-matched sample
curve stay exactly what an uncapped run wrote at that step, which is the entire point of a
capped arm — and it lands as the `held-out full` series at the same step, plus `full_curve` for
the horizon and trading families.

Cost: **103.4 s**, one pass, once per run (DEMONSTRATED: 103,368 ms for 433,303 windows / 1,693
batches at 61 ms per batch, `research/worker_reports/timexer_eval_speed.md`). Against a 4,000-step
arm that is +15.3 % of its 674.4 s of stepping (4,000 x 168.59 ms) and +13.7 % of its ~756 s wall
clock (71.8 s corpus load + stepping + four report intervals). Skipping it would save 103 s and
lose the split; that trade is not worth taking, and preview-patience stops now get it too so
every arm in a batch carries the same two splits whichever way it ended.

`write_metrics` was narrowed rather than relaxed to carry the pair: the step axis is now one row
per STEP (not per point), so the terminal step's sample point and full point are two SERIES on
one x instead of two rows sharing an x. The monotonicity guard now states the positive condition
— a step may carry its `held-out sample` point followed by exactly one `held-out full` point —
and still refuses a repeated split at one step, a sample point after a full one, and a backwards
step. DEMONSTRATED by
`reports::tests::a_capped_terminal_step_carries_both_splits_as_series_on_one_step`, which reads
the `.report.bin` back: `steps == [1000, 4000]`, `held-out sample NLL = 2.11` and
`held-out full NLL = 2.02` both at 4,000, the full series NaN at 1,000, the scope naming 433,303
origins, and all four illegal shapes refused. The on-disk report format is unchanged, so
`report_cli` and the TUI need no edit.

## Verification

- `./torch-env.sh cargo check -p trading_bot_0 --tests` — 0 errors.
- `./torch-env.sh cargo test -p trading_bot_0 timexer_segment` — **79 passed**, 0 failed (74
  before + 5 new: exact stop step, LR bit-identity, the four termination reasons, invalid caps
  refused pre-corpus, and the two-split terminal step end to end through the file format).
- `./torch-env.sh cargo check -p trading-bot-tui --tests` — 0 errors;
  `cargo test -p trading-bot-tui` — 36 passed.
- `train-timexer-segment --help` shows `--max-steps <MAX_STEPS> [default: 0]`.
- Not exercised on device: no GPU/mlq job was run or submitted (the queue is Main's and
  contended). The device-side final pass reuses the same `score(&corpus, &model,
  &corpus.validation_refs, batch_size, device)` call the epoch-end validation makes, so its
  behaviour on device is [INFERENCE] from that call site, not measured here.

## The 4,000-step capped arm

```bash
mlq submit --name timexer-cap4k --max-parallel-runs 1 --time-limit 1h \
  --cwd "$PWD" -- ./trading_bots/run-release-cuda.sh train-timexer-segment \
  --run timexer-cap4k-20260907 \
  --features all --layers 8 --d-model 512 --heads 8 --ffn 2048 --min-history 256 \
  --batch-size 256 --optimizer polar-express --preview-patience 3 --x0-lambdas disabled \
  --max-steps 4000
```

Differing flags versus the shared set: `--max-steps 4000` only (and `--time-limit 1h` instead of
12h, since the arm is ~14 min of wall clock including the final full-split pass). `--schedule-budget`
is deliberately left at its default, so the warmdown is still shaped against the 9,590-step epoch
and these are the first 4,000 steps of exactly the trajectory the long baselines took. To cap an
arm whose schedule is also short, pass `--schedule-budget` explicitly: the two knobs are
independent by construction and by test. Not submitted: the mlq queue is Main's.
