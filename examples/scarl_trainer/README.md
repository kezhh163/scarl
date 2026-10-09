# SCARL Trainer Examples

SCARL (Sequence Confidence-Aligned Reinforcement Learning) reshapes,
but never re-signs, each response's advantage with its **alignment
gap**: the distance between the response's group-relative sequence
confidence and the outcome its advantage certifies, normalized into
shares of each outcome class's gradient budget. Misaligned responses
(fragile successes, confident errors) absorb the update; aligned ones
(mastered successes, exploratory errors) are left nearly untouched.

## Mechanism (matches the paper's Section 3)

1. **Sequence confidence**: `c_i = sum_t log pi_old(y_t | ...)`, the
   old-policy sequence log-likelihood already materialized for the
   importance ratios (`prob` mode, used for all paper runs; the
   alternative `logit` mode uses `sum_t logsigmoid(logit_t)` over the
   selected tokens via a dedicated single-forward path).
2. **Group-relative confidence**: standardize `c_i` within the group
   (adaptive tau), then softmax: `p_i = softmax(c_i / (tau * sigma_G))`.
3. **Alignment gap**: `t_i = 1{A_i >= 0}`;
   `w_raw_i = |t_i - p_i|` (= `1 - p_i` for positives, `p_i` for
   negatives).
4. **Class-wise normalization**: `w_i = w_raw_i / mean_class(w_raw)`,
   so each outcome class keeps its gradient budget
   (`E[w | class] = 1`), and `A_i <- w_i * A_i`.

The factor is strictly positive (never flips an update) and reduces
exactly to the base estimator when confidence carries no signal.

## Key parameters

| Parameter | Paper value | Description |
|---|---|---|
| `algorithm.adv_estimator` | `scarl` | Enable SCARL |
| `algorithm.scarl_tau` | `5` | Temperature; the single substantive hyperparameter (sweep: 0.2, 0.5, 2, 5, 10) |
| `algorithm.scarl_confidence_source` | `prob` | `prob` (old-policy log-probs, Eq. 5; all paper runs) or `logit` (raw selected-token logits) |
| `algorithm.scarl_adaptive_tau` | `True` | Scale logits by the group's std before the temperature |
| `algorithm.scarl_sign_aware_weighting` | `True` | `False` = outcome-blind ablation (gap without the binarized target) |
| `algorithm.scarl_class_wise_normalization` | `True` | `False` = remove the conservation step (catastrophic in the paper's ablation) |
| `algorithm.scarl_lam`, `algorithm.scarl_power` | `1`, `1` | Identity defaults; no-ops at these values |

## Scripts

| Script | Base estimator | Backbone |
|---|---|---|
| `run_scarl_grpo_qwen3_{1.7b,4b}_base.sh` | GRPO | Qwen3-1.7B/4B-Base |
| `run_scarl_dapo_qwen3_{1.7b,4b}_base.sh` | DAPO | Qwen3-1.7B/4B-Base |
| `run_scarl_gspo_qwen3_{1.7b,4b}_base.sh` | GSPO | Qwen3-1.7B/4B-Base |
| `run_scarl_sapo_qwen3_{1.7b,4b}_base.sh` | SAPO | Qwen3-1.7B/4B-Base |
| `run_scarl_dapo_qwen3-{1.7b,4b}_base_single_node.sh` | DAPO | single-node convenience variants |

Other backbones reuse the same scripts with `MODEL_PATH` overridden.
SCARL leaves rewards, ratios, and clipping untouched, so each script
inherits its base estimator's published recipe and changes only the
advantages.

## Ablation examples

The scripts define their flags in a variable block at the top; edit
these before launching:

```bash
# w/o class-wise normalization (variable at the top of the script)
scarl_class_wise_normalization=False

# temperature sweep
scarl_tau=0.5

# outcome-blind shaping (w/o alignment target): change the literal
# in the submit command from
#   +algorithm.scarl_sign_aware_weighting=True
# to
#   +algorithm.scarl_sign_aware_weighting=False
```
