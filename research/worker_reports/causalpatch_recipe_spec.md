# CausalPatch recipe port spec (BarTrunk/modded-nanogpt residual recipe, record-11 UNet skips, value residual)

Reference checkout: `/home/marvin/Documents/repositories/modded-nanogpt` (branch master). Record numbering from its `README.md:118-131` (record 5 = ReLU²/zero-init/QK-norm, record 9 = value + embedding shortcuts, record 11 = U-net skips). In-repo BarTrunk (`trading_bots/src/torch/world_model.rs`) is cited only as evidence that the recipe already runs in this codebase. All `model.rs`/`compute.rs` lines are `trading_bots/src/torch/timexer_segment/`.

## 0. Corrections to the task text

1. **CausalPatch is already PRE-norm, not post-norm.** `Block::forward` normalizes the input before QKV (`model.rs:319`), adds `input + linear(attended, output)` (`:343`), normalizes `state` before the FFN (`:345`) and adds `state + ff` (`:351`); `final_norm` closes the stack (`:676`). `docs/timexer_segment.md:16` says "pre-norm transformer". The reference is also pre-norm (record-11 `:207-216`, current `train_gpt.py:1598,:1643,:1665`; BarTrunk `world_model.rs:1717,1419`). So the lambda scheme drops in without a norm-placement change; only LayerNorm(affine, eps 1e-5) → gainless RMSNorm(eps 1e-6) changes.
2. **No autocast in CausalPatch** (`docs/timexer_segment.md:16`): activations are bf16, every fp32 parameter is cast at point of use (`model.rs:199-203, 216-222`). Every new `[1]`-shaped fp32 scalar MUST be `.to_kind(activation.kind())` before touching an activation — a dimensioned fp32 `[1]` tensor times a bf16 tensor promotes the RESULT to fp32 (this is not the zero-dim exception), silently doubling traffic and breaking the fused-kernel dtype invariant. The reference casts its lambdas to bf16 explicitly (`train_gpt.py:1509-1512`).
3. The value residual is NOT in the current `train_gpt.py` (replaced by value embeddings, record 14, and MUDD `v_mudd`/`aux_v`, `:1541-1543, :1607-1618`). Its authoritative form is the record-9..13 era code.

## 1. Authoritative formula / init table

| # | Component | Formula | Learnable? shape, param, init | Reference (primary) | In-repo (secondary) |
|---|---|---|---|---|---|
| 1a | Norm | `rms_norm(x, [d], weight=None)`; no gain, no bias | none | `train_gpt.py:952-953` (`F.rms_norm(x,(x.size(-1),))`, default eps 1e-6 in torch) | `world_model.rs:109-110, 2949-2952` (eps 1e-6, "no learnable gain anywhere") |
| 1b | QK-norm | `q = rms_norm(q, [head_dim])`, `k = rms_norm(k, [head_dim])`, THEN rotary | none | `train_gpt.py:1103-1109` (`q,k = norm(q),norm(k)` at :1106 then `yarn.rotary` at :1109); record-11 `:180-181` same order | `world_model.rs:2955-2957` applied at `:1391-1399` before `pope_expand_qk` `:1722-1724` |
| 1c | Activation | `relu(h).square()` | none | `train_gpt.py:46-48`; record-11 `:199-202` | `world_model.rs:1419-1420` |
| 1d | Zero-init | attention output projection and MLP down projection are zero at init; QKV and MLP up are `uniform(±√3·0.5·d_model^-½)` (std `0.5·d^-½`) | — | `train_gpt.py:1288-1294` (attn), `:1303-1308` (mlp, `c_proj` zero); record-11 `:168-169, :195-196` | `world_model.rs:1357-1361, 1381, 2904-2909` |
| 1e | Attn residual | `x = λ_ra·x + λ_pa·attn_out + λ_x0·x0` | per layer, raw scalars: `λ_ra` init `√1.1 = 1.0488088481701516`, `λ_pa` init `1.0`, `λ_x0` init `0.0` | `train_gpt.py:1333-1338` (post ones, resid √1.1), applied `:1638`; x0 lambda init 0: `train_gpt.py:1389`, `train_gpt_medium.py:1078,:1181` | `world_model.rs:112-115, 1369-1375, 1415` |
| 1f | MLP residual | `x = λ_rm·x + λ_pm·mlp_out` | per layer raw scalars, `λ_rm` init `√1.1`, `λ_pm` init `1.0` | `train_gpt.py:1665` | `world_model.rs:1382-1383, 1421` |
| 1g | x0 | `x0` = the residual stream entering layer 0 (embedding after norm) | — | record-11 `:254-256` (`x0 = x` after `rms_norm`), current `:1561` | `world_model.rs:1687-1693, 1734` (x0 = normed token embedding) |
| 2 | UNet skip | encoder layers `0..L/2-1` push `x`; decoder layer `j` (0-based within decoder) does `x = x + w_j · pop()` BEFORE its block; pairing 3→4, 2→5, 1→6, 0→7 | record 11: `w_j` raw, `ones(decoder_layers)`; current: `σ(skip_lambda)`, raw init `-1.5` ⇒ 0.182 | record-11 `:239-244` (`skip_weights = ones`), loop `:257-270`; current `train_gpt.py:1346` (`-1.5`), `:1508`, `:1592-1595` (`x + sigmoid(skip_lambda)*gate*cache[3]` at layer 6 only); medium `train_gpt_medium.py:1091, :1174-1177` (3 skips, σ(-1.5)·2σ(gate(x0))) | none in-repo |
| 3 | Value residual | `v = (1-λ)·v + λ·v₁`, `v₁` = layer 0's raw V; layer 0 uses `v₁ = v` (identity) | per layer raw scalar `λ` init `0.5` | record-9 `dd7304a6…txt:168, 175-178`; record-11 `:174-181` identical; record-12 variant uses two scalars `lambdas[0]*v + lambdas[1]*v1` (`2024-11-09_Replicateleloykun/1621af10…txt:179`) | none in-repo |
| opt | Adam groups | resid_lambdas lr×5, betas (.9,.95), wd 0; post_lambdas lr×1, (.9,.95), wd 0; `scalars` (sa/skip/smear) lr×5, (.9,.99), wd 0; x0_lambdas lr×5, wd 0 | — | `train_gpt.py:2030, 2035-2036`, `train_gpt_medium.py:1078-1081`; Adam base lr 0.008, eps 1e-10, wd 0.005 `:2065-2070`; record-11 skip_weights in Adam lr 0.04 betas (.9,.95) `:432-434` | `pretrain.rs:240-258, 4895-4935, 4989-4995` |

**Difference between record 11 and current, and which to prefer:** record 11 initializes skip weights at 1.0 for 6 skips over 12 layers. Current uses ONE skip (3→6) with `σ(-1.5)≈0.18` and a MUDD gate; medium uses three (2→11, 4→10, 6→9) at `σ(-1.5)`. Prefer the current parameterization (sigmoid of a raw scalar initialized at -1.5) with the record-11 TOPOLOGY (all four pairs) as the task demands; skip the MUDD/x0 gates (they are LM-specific, sec. 6). Record 11 uses `lambdas=[1,0]` on `x` and `x0` with plain `x + branch`; current uses the `√1.1 / 1.0 / 0` triple — prefer current (it is also what BarTrunk runs).

**Identity-at-init check.** With zero-init out/down projections every branch is 0, so after `2L=16` sublayers `x = 1.1^8·x0 ≈ 2.14·x0`. That is identity UP TO A SCALAR, and every consumer of `x` is a gainless RMSNorm (pre-norms, final norm), so the head input at init is bit-for-bit the normalized `x0` — the same as today up to LN→RMS. Skips keep `x ∝ x0` at init (`x0·(1.1^k + 0.18·1.1^m)`). The value residual at λ=0.5 mixes V₁ into V from step 0; it does not change the forward at init (W_out = 0) but does change gradients. λ=0 would make it exactly inert at init (decision (b)).

## 2. Edit plan against current `model.rs` (implementation order)

### 2.1 Change 1 — residual recipe

1. **Norm** (`model.rs:205-224`): replace `LayerNorm(nn::LayerNorm)` with a parameter-free `RmsNorm { width }` whose `forward` is `input.rms_norm([width], None::<&Tensor>, Some(1e-6))` (exact form: `world_model.rs:2949-2952`). Remove `LayerNorm::new`'s `nn::layer_norm` registration (no vars). `Block.norms: [LayerNorm;2]` (`:247`) becomes `[RmsNorm;2]` (or drop the field and call a free fn); `final_norm` (`:441`, `:531`) likewise. Kernel class "LayerNorm" (`:727-734`) → "RMSNorm", `parameters: Vec::new()`, `forward_bytes: 2·state + fp32(tokens)` (rstd only; no mean, no gain cast).
2. **Projection init and bias** (`:184-200`, `:262-265`): add a `residual_out(path, in, out)` = `nn::linear` with `ws_init: Const(0.0)`, `bias: false`, used for `output` and `second`. For `qkv` and `first` use `Init::Uniform{±√3·0.5/√fan_in}` (`world_model.rs:2904-2909`), `bias: false`. Decision (a): the reference has NO biases in blocks (record-11 `:165-168`, current `:1103`, medium `:944-945`); recommended: drop all four block biases (kernel_classes' `projected()` `:715-719` already handles `bs == None`). If biases are kept instead, `output.bias` and `second.bias` MUST be zero-init or the block is not identity at init.
3. **QK-norm** (`:319-321`): after `packed = linear(norm(input), qkv).split_with_sizes([2w, w])`, insert `let qk = packed[0].reshape([batch, length, 2*heads, head_dim]).rms_norm([head_dim], None, Some(1e-6)).reshape([batch, length, 2*width]);` and pass `qk` (not `packed[0]`) to `self.rotate`. Column layout is unchanged (`t·width + h·head_dim + s·half + r`, `:281-284`), so `rotate` is untouched. Norm BEFORE rotation is mandatory: rotation is a norm-preserving rotation of each (r, r+half) pair, so normalizing after would be equivalent only in exact arithmetic, and the reference order is norm→rotary (`train_gpt.py:1106-1109`). Add a kernel class "QK norm" (inputs `activation(2·width)`, `forward_bytes: 4·state + fp32(2·heads·tokens)`).
4. **ReLU²** (`:345-349`): `linear(norm(state), first).relu()` then `let h2 = &h * &h` (or `.square()`), then `linear(h2, second)`. Do NOT keep `dropout` calls around it if `dropout == 0` matters for graph count — they are no-ops at 0 but are still recorded; leave as is unless the perf sibling removed them.
5. **Lambdas** (new `Block` fields, all `[1]` fp32 vars under the block path): `attn_resid_lambda` (√1.1), `attn_post_lambda` (1.0), `x0_lambda` (0.0), `ff_resid_lambda` (√1.1), `ff_post_lambda` (1.0). Names MUST contain `_lambda` and MUST NOT end in `.weight` (sec. 3). Forward (`:343`, `:351`), with `x0` threaded from `backbone` (`:672-675`: `let x0 = state.shallow_clone()` right after the patch embedding, before the loop; pass `&x0` into `block.forward`):
   - fold post lambdas into the weights the way BarTrunk (`world_model.rs:1414`) and the reference (`train_gpt.py:1161`, "pre-multiplied to O") do: `attended.linear(&(output.ws.to_kind(k) * attn_post_lambda.to_kind(k)), None)` — scaling a 512×512 weight costs nothing; scaling a `[tokens,512]` activation costs 3 traffic units. Same for `second.ws * ff_post_lambda`.
   - attention residual: `let state = out.addcmul(input, &attn_resid_lambda.to_kind(k)).addcmul(x0, &x0_lambda.to_kind(k));` (two kernels; `addcmul(self,t1,t2) = self + t1*t2`, `[1]` broadcasts). Replaces `input + out`.
   - FFN residual: `ff.addcmul(&state, &ff_resid_lambda.to_kind(k))`. Replaces `state + ff`.
   - The `dropout(...)` on the branch outputs stays where it is (applied to `out`/`ff` before the addcmul).
6. **`step_cost`** (`:139-183`): per-layer width units go from `2+3+8+6` to `2 (norms) + 3 (qkv) + 2 (qk-norm) + 8 (rotation) + 6 (attn out, proj, residuals) + 1 (extra addcmul) + 2·ffn/width·2 (relu AND square each materialize `ffn`)`; i.e. `19·width + 2·ffn` → `22·width + 4·ffn` (+`1·width` more with the x0 addcmul counted above). Update the comment and the kernel-class list so the profile attribution stays honest.
7. **Accessors for the shared report base** (agreed with `ValueResidual`): expose `CausalPatchModel::residual_lambdas() -> Vec<f64>` (16, attn/ff per layer), `post_lambdas()`, `x0_lambdas()` returning EFFECTIVE values (raw here). The integrator owns the single `timexer_segment_recipe_scalars` report base; do not touch `reports.rs`/`runner.rs` from the model change.

### 2.2 Change 2 — record-11 UNet skips

1. Add to `CausalPatchModel`: `skip_lambdas: Vec<Tensor>` of length `layers/2` (4), each `[1]` fp32 under `path / "skip_{j}"` named e.g. `skip_3_to_4_lambda`… simplest: `path / format!("skip_{j}") / "lambda"`, `Init::Const(-1.5)` (current reference `train_gpt.py:1346`). Effective weight `w_j = sigmoid(skip_lambdas[j])` computed in fp32 on the `[1]` tensor, then `.to_kind(k)`.
2. `backbone` (`:672-675`) becomes:
```
let encoder = self.config.layers / 2;
let mut skips: Vec<Tensor> = Vec::with_capacity(encoder);
for (i, block) in self.blocks.iter().enumerate() {
    if i >= encoder {
        let skip = skips.pop().expect("one push per encoder layer");
        state = state.addcmul(&skip, &self.skip_lambdas[i - encoder].sigmoid().to_kind(state.kind()));
    }
    state = block.forward(&state, &x0, rotation, train);
    if i < encoder { skips.push(state.shallow_clone()); }
}
```
   giving exactly 3→4, 2→5, 1→6, 0→7 (record-11 `:257-270`: push after each encoder block, pop before each decoder block). `validate()` (`:88-118`) must `ensure!(self.layers % 2 == 0)` or define the odd case (record 11: `decoder = n - n//2`, extra decoder layer gets no skip; simplest is to require even).
3. Why NOT 1.0: with `√1.1` residual gains, encoder output `i` at init is `1.1^(i+1)·x0`; adding it with `w=1` at decoder layer 4 makes the stream `(1.1^4 + 1.1^4)·x0` = 2× and compounds through 5, 6, 7 (the task's own observation). At 0.18 the stream stays within ~20% of its no-skip norm. Since every consumer is gainless RMSNorm the scale itself is harmless at init, but the RELATIVE weight of the skip vs. the stream is what 0.18 vs 1.0 sets, and NorMuon's update size is scale-free so the skip's gradient share is what matters.
4. Accessor: `skip_weights() -> Vec<f64>` = `sigmoid(raw)` (effective), for the shared report base.

### 2.3 Change 3 — value residual

1. `Block` gets `value_lambda: Option<Tensor>` (`None` for block 0, `Some([1], Init::Const(0.5))` for 1..7; reference `record-9 :168` holds one per layer including layer 0 where it is inert — don't allocate a dead parameter here).
2. `Block::forward` takes `v1: Option<&Tensor>` and returns `(Tensor, Tensor)` = (state, this layer's raw V) — only block 0's return value is used; cheaper: `backbone` reads block 0's V through a dedicated return, other blocks return `state` only. Concretely in `:326-330` the V operand becomes:
```
let v = packed[1].reshape([batch, length, heads, head_dim]);          // strided view, as today
let v = match (v1, &self.value_lambda) {
    (Some(v1), Some(l)) => v1.lerp(&v, &(1.0 - l).to_kind(k)),        // = λ·v1 + (1-λ)·v, ONE kernel
    _ => v,
};
sdpa(q, k, v.transpose(1,2), ...)
```
   `lerp(start=v1, end=v, weight=1-λ) = v1 + (1-λ)(v - v1) = λ v1 + (1-λ) v` ✓. `v1` is block 0's `packed[1].reshape([batch,length,heads,head_dim])` view kept by `backbone` for the loop's lifetime.
3. Accessor: `value_lambdas() -> Vec<f64>` (7 entries, raw) — `ValueResidual` is implementing exactly this.

## 3. Parameter routing (compute.rs:93-155, muon.rs:1307-1316)

Rule today: NorMuon iff `dim()==2 && name contains "block_" && no force_adamw fragment (patch., covariates., norm, .bias, head.)`; the assert at `compute.rs:141-152` additionally demands every NorMuon tensor be `block_*…*.weight` and every non-NorMuon tensor NOT be (2-D, `.weight`, `block_`). AdamW WD 0.005 applies unless the name contains `norm` or `.bias` (`:109`).

| Tensor | shape | route | WD | Action needed |
|---|---|---|---|---|
| `block_i.qkv.weight`, `first.weight` | 2-D | NorMuon (unchanged) | 1.2 quadratic | none |
| `block_i.output.weight`, `second.weight` (now zero-init) | 2-D | NorMuon (unchanged; zero init is fine for NorMuon — BarTrunk does it, `world_model.rs:151,1357-1361`) | 1.2 | none |
| `block_i.*.bias` | removed (decision a) | — | — | if kept: AdamW, no WD (`.bias`) as today |
| norm gains/biases | NONE (gainless RMSNorm) | — | — | `norm_*.weight/bias`, `final_norm.*` disappear from AdamW; if a gain were kept it is 1-D → AdamW, no WD via `norm` |
| `block_i.attn_resid_lambda`, `ff_resid_lambda` | `[1]` | AdamW (dim 1 fails the 2-D test; assert passes because the name does not end in `.weight`) | MUST be 0 → add `"_lambda"` to `adamw_no_weight_decay_name_substrings` | + `adamw_beta_overrides: ("resid_lambda", (0.9,0.95))`, `set_named_lr_scale(["resid_lambda"], 5.0)` per `train_gpt.py:2036`, `pretrain.rs:246,256-258` |
| `block_i.attn_post_lambda`, `ff_post_lambda` | `[1]` | AdamW | 0 (same fragment) | betas (0.9,0.95), lr×1 (`train_gpt.py:2035`) |
| `block_i.x0_lambda` | `[1]` | AdamW | 0 | lr×5, betas: medium says "no beta smoothing" special group (`train_gpt_medium.py:1077-1081`); BarTrunk uses (0.9,0.99) lr×1 (`pretrain.rs:242,254,12241`). Pick (0.9,0.99), lr×5 to match the reference `x0_lambdas.lr_mul=5.0`. |
| `skip_{j}.lambda` | `[1]` | AdamW | 0 → the fragment must be covered: name it `skip_lambda` so `"_lambda"` catches it | current reference puts it in `scalars`: betas (0.9,0.99), lr×5 (`train_gpt.py:2030`); record 11: betas (0.9,0.95) (`:434`). Use (0.9,0.99), lr×5. |
| `block_i.value_lambda` | `[1]` | AdamW | 0 (same fragment) | record-era: plain scalar Adam lr 0.04 betas (0.9,0.95) (`record-11:432-434`). Use (0.9,0.95), lr×1 (it has no modern reference group). |

Silent-bug guards: (i) never name a scalar `*.weight`; (ii) never make a lambda 2-D (e.g. `[1,1]`) — it would be NorMuon-routed and the assert would only fire if it also ended in `.weight`; (iii) a `[layers,2]` packed lambda bank as in the reference (`:1333-1338`) is 2-D and WOULD be caught by NorMuon → keep per-layer `[1]` vars as BarTrunk does; (iv) `adamw_wd 0.005` with quadratic-lr decay on a lambda whose target is ~1.05 is a bias toward 0 — reference wd_mul is 0 for all of them; (v) `capture_step_graphs: true` (`compute.rs:128`) — new parameters change the captured optimizer graph; the `benchmark` capture audit (`benchmark.rs:23-25`) covers it, run it.

## 4. Interaction warnings

1. **QK-norm order**: the reference and BarTrunk both do norm THEN rotary (`train_gpt.py:1106-1109`; `world_model.rs:1396-1397` → `:1722-1724`). CausalPatch currently applies rotary to the raw packed q‖k (`:321`) — insert the per-head RMS norm on `packed[0]` before `rotate`. SDPA scale stays the default `1/√64 = 0.125` (record 11 used the default with head_dim 128; current uses 0.1 with hd 128, `:1035` — not ported). With unit-RMS q,k the max logit is `√hd·√hd/√hd = 8`; fine.
2. **ReLU² vs GELU output scale at fixed ffn 2048**: no correction needed. The down projection is zero-init and NorMuon's update is scale-free (orthogonalized), and the post-lambda absorbs any residual scale; the reference changed GELU→ReLU² with no width or scale change (record 5, README:118). Note ReLU² second moment for unit-Gaussian input is `E[relu(z)^4] = 1.5` vs GELU ≈ 0.4, i.e. ~2× RMS on the hidden, harmless for the reasons above. Cost is the real issue (item 4).
3. **Zero-init out projections vs the folded μP head scale**: orthogonal. `HEAD_OUTPUT_SCALE` multiplies `head.output.weight` (`:966-970`), which is outside the blocks and stays zero-init; the head sees `final_norm(x)` which at init is `rmsnorm(1.1^8·x0)` = `rmsnorm(x0)` — the same unit-RMS input LayerNorm produced (gain 1, bias 0 at init). The only head-input change is LN→RMS (no mean subtraction, no affine), which shifts the optimum slightly but not the scale. Zero-init block outputs make the initial forward a 1-layer model (patch embed → head), which is exactly BarTrunk's regime.
4. **Traffic / memory profile** (batch 256, 96 000 tokens; one bf16 `[tokens,512]` tensor `U` = 98.3 MB; the step is ~144 GB/step memory-bound, `step_cost` counts each materialized activation ×2 (r/w) ×3 (fwd + 2 bwd)):
   - RMSNorm: −fp32 mean, −gain cast: negligible saving.
   - QK-norm: +2U materialized per layer → +2·6·8 = 96 U ≈ **+9.4 GB (+6.5%)**.
   - ReLU² unfused (`relu` then `square`, two kernels each saving one tensor; autograd keeps a single copy since square's input is relu's output): the activation pass traffic doubles relative to one GELU kernel: +4U materialized per layer → 4·6·8 = 192 U ≈ **+18.9 GB (+13%)**. The reference avoids this with a fused Triton kernel (`train_gpt.py:46-48`); tch has no fused `relu²`. This is the single biggest cost of change 1 and is a candidate for the perf sibling (custom kernel or `KernelTraffic`'s fusion work).
   - Lambda residuals with post lambdas folded into weights and `addcmul` chains: attn residual becomes 2 kernels instead of 1 (+1U materialized fwd; each addcmul backward adds `grad·λ` (2 moves) + `Σ grad·t1` reduction (2 reads)), FFN residual same kernel count but +4 backward moves → ≈ +15 U-moves/layer ≈ 120 U ≈ **+11.8 GB (+8%)**. Without folding (`λ·out` on the activation) or with naïve `a*x + b*out + c*x0` (5 kernels) it is roughly 3× that (~+30 GB); the reference's `x0` term is only cheap because it is fused into one expression by torch.compile. Memory: `x0` is already saved by block 0's norm; the addcmul saves `input`/`x0` which are already saved → ~0 extra bytes.
   - Change-1 total unfused ≈ **+40 GB ≈ +28% step time** on a bandwidth-bound step. Decision (c).
   - UNet skips: 4 × (`addcmul` fwd 3 moves + bwd `grad·w` 2 + `Σ grad·skip` 2 + gradient accumulation into the encoder output's grad 3) ≈ 40 U ≈ **+3.9 GB (+2.7%)**. Memory: encoder outputs 0..3 are already retained by the next block's norm backward (norm saves its input) — pushing them retains nothing extra. Autograd accumulation of the two gradient contributions into each encoder output is the one genuinely new backward buffer (`[tokens,512]` bf16, transient).
   - Value residual with `lerp` (one kernel): fwd 3 moves (reads strided V and V₁, writes a contiguous `v_mix`), bwd ≈ 7 moves (`grad·(1-w)`, `grad·w`, `Σ grad·(end−start)`) + accumulation of `grad_V₁` across 7 layers (3 moves each) ≈ 13 U/layer × 7 ≈ 91 U ≈ **+8.9 GB (+6%)**. Memory: `v_mix` is a new `[tokens,512]` bf16 per layer saved by SDPA (+98 MB × 7 ≈ 0.7 GB peak); V₁ itself is a view of block 0's packed projection, which flash SDPA already saves for its backward, so keeping it alive for all 8 layers costs 0 extra bytes. SDPA flash compatibility: flash only needs unit stride on `head_dim` (`:322-330`); `lerp` writes a contiguous `[B,L,H,hd]` whose `.transpose(1,2)` is accepted, and V₁ is consumed by elementwise ops, not by SDPA, so nothing forces a materialized V beyond `v_mix` itself. Layer 0 must feed its RAW V (before any mixing) forward — do not accidentally pass `v_mix`.
5. **bf16 lambdas**: `addcmul`/`lerp` with an fp32 `[1]` weight against a bf16 activation promotes to fp32 — cast the `[1]` to bf16 first (item 0.2). bf16 has 8 mantissa bits: `√1.1 = 1.0488` rounds to 1.046875 (0.2% off), `σ(-1.5) = 0.1824` rounds to 0.18261 — acceptable, and it is what the reference does (`:1509-1512`).
6. **Kernel classes and the capture audit**: `kernel_classes` (`:692+`) closes over `block.norms`, `block.qkv`, `block.output` and builds the SDPA class from `activation(3·width)` (`:756-778`); the new RMSNorm/QK-norm/ReLU²/addcmul/lerp kernels must be added or the "composed layer minus parts" attribution silently grows. `GraphCapture`/`KernelTraffic` are editing these regions concurrently — coordinate before touching `model.rs`.
7. **Dropout**: the reference has none; CausalPatch defaults to 0 (`:47`). If anyone runs dropout > 0, the addcmul form applies it to the branch before scaling — same as today.

## 5. Expected effect and measurement (report bases `timexer_segment_skill`, `_loss`, `_calibration`, `_horizon*`, `reports.rs:410-500, 764-790`)

- **Change 1**: expected faster early loss descent (identity-at-init + NorMuon on zero matrices) and a lower final validation NLL (`timexer_segment_loss`, nats vs persistence prior). ReLU²/QK-norm are the reference's "~1-2% better than GELU" and stability items; the honest expectation for this task is a small NLL gain and no calibration change (`_calibration` bands should stay where they are — a drift there means the RMS head input changed the σ head's operating point). Falsified if `_loss` at matched wall-clock is worse than the LN/GELU baseline by more than seed noise (use ≥2 seeds), given the +28% traffic — the comparison MUST be at equal wall-clock, not equal steps.
- **Change 2**: expected effect is on `_horizon` at long horizons and `_skill` (encoder-level features reachable by decoder layers without 4 layers of residual mixing). Falsified if `skip_weights()` decay toward 0 during training (the model rejects the path) or `_skill` is unchanged at equal wall-clock; +2.7% traffic is the price.
- **Change 3**: expected to help causal attention over 375 tokens (value information from layer 0 available everywhere; the record-9 gain was mostly on deep-layer attention). Watch `value_lambdas()`: if they drift to ~0 the path is unused; if to ~1 the later V projections are dead weight. Falsified by no `_loss` improvement at equal wall-clock with +6% traffic.
- Per-change ablation: one flag each (`--recipe`, `--unet-skips`, `--value-residual`) so each can be turned off in the SAME binary; the integrator's single `timexer_segment_recipe_scalars` base charts all learned scalars over training.

## 6. Not clearly applicable — do NOT port blindly

1. **sa_lambdas** (QKV scalar init 0.5, out scalar 1.0, `train_gpt.py:1344, :1103, :1161`; BarTrunk `qkv_lambda`/`attn_out_lambda` init 1.0, `world_model.rs:1355,1362`) — in the reference, not in the three requested changes. Skip.
2. **MUDD / dynamic per-token lambdas, attention gates, XSA, paired heads, value embeddings, bigram hash embeddings, smear, backout** (`train_gpt.py:1252-1265, 1577-1665`) — token-vocabulary LM machinery; value embeddings need a vocabulary (`:1261`). Not applicable to continuous patch tokens.
3. **Skip gates conditioned on `x0[..., :16]`** (`train_gpt_medium.py:1175`, current `:1594` via MUDD) — depends on a 16-dim embedding slice carrying token identity; port only the scalar `σ(skip_lambda)`.
4. **Record-11's `skip_weights = ones`** — the task's own reasoning (8 layers, `√1.1` gains) and the current reference both say ~0.18.
5. **Logit softcap, untied embed/head, FP8 MLP, YaRN windows, sliding-window schedules** — LM/long-context items; the 375-token dense causal SDPA has no window.
6. **Reference 2-D init `uniform(±√3·0.5·d^-½)`** vs current `±1/√fan_in` (`:184-190`): std 0.5 vs 0.577 · d^-½ — near-identical; adopt the reference form for fidelity but it is not load-bearing.
7. **Value residual on layer 0's V with only 8 layers**: the record-era models had 12 layers; with 8 the benefit is smaller and the +6% traffic may not pay. Treat as the most falsifiable of the three.
8. **x0 shortcut** — x0 here is a linear patch embedding of σ-normalized prices, not a token-identity embedding; BarTrunk's argument (`world_model.rs:1402-1406`: the current bar's own bins are the strongest next-bar predictor) transfers, but expect `x0_lambda` to stay small; it is free at init (0) and cheap with the addcmul form.

## 7. Decisions needed from Main before implementation

(a) Drop the four block projection biases (reference-faithful, recommended) or keep with zero-init on `output`/`second`.
(b) Value-residual λ init: 0.5 (reference, not identity-at-init in gradient terms) or 0.0 (inert at init).
(c) Accept ≈+28% unfused traffic for change 1 (dominated by unfused ReLU² +13% and the addcmul residuals +8%), or gate the merge on a fused `relu²` kernel from the perf track.
(d) Confirm sa_lambdas stay out of scope.

## 8. Delivery note

This agent has no file-write tool (read/grep/glob/web_search/hub/yield only); the assignment's single permitted write could not be performed. Main must save this report verbatim as `research/worker_reports/causalpatch_recipe_spec.md`.