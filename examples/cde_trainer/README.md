# CDE Baseline

Reimplementation of the actor-side perplexity (PPL) bonus of CDE
("CDE: Curiosity-Driven Exploration for Efficient Reinforcement
Learning in Large Language Models", Dai et al., ICLR 2026,
arXiv:2509.09675). No official code was released for the paper; this
port follows the ICLR camera-ready text. CDE's second component, the
multi-head critic bonus, is PPO-specific and is not ported.

Mechanism (paper's Eqs. 1-2):

1. **Curiosity signal**: `B(q, o) = -(1/T) sum_t log pi(o_t | o_<t, q)`,
   the mean negative token log-probability of the response under the
   behavior policy (log-perplexity), computed from the `old_log_probs`
   already materialized for the importance ratios.
2. **Shaped reward**: `r~ = r + omega_t * min(|r| / kappa, alpha * B)`.
   The cap `|r| / kappa` keeps the bonus a fraction of the outcome reward
   and prevents it from reversing a negative reward (`r = -1` for
   incorrect responses in this repo's reward manager).
3. **Base estimator**: the shaped reward goes through the unmodified
   GRPO group normalization (`compute_grpo_outcome_advantage`).

Hyperparameters are the paper's PPL-bonus configuration (Table 4b):
`cde_kappa=3`, `cde_alpha=1`, and a **staircase** bonus-weight schedule
(Figure 7: `omega_t = 1` for the first half of training, `0` afterwards;
`cde_omega=1.0`, `cde_omega_decay_fraction=0.5`). The paper's other
schedules are available as `cde_omega_schedule=linear|cosine|none`.
The weight is scheduled by the fraction of training completed,
`(global_step - 1) / total_training_steps`, so the staircase switches
off after step 75 of the shared 150-step recipe.

Dedicated code paths (shared functions untouched; dispatch seams only):

- `verl/trainer/ppo/core_algos.py`: `AdvantageEstimator.CDE`,
  `cde_bonus_weight`, `compute_cde_ppl_bonus`,
  `compute_cde_outcome_advantage` (calls the unmodified GRPO estimator)
- `verl/trainer/ppo/ray_trainer.py`: `CDE` branch in
  `compute_advantage`; critic-free list; step-fraction and metrics seams
  in `fit`
- `recipe/dapo/dapo_ray_trainer.py`: step-fraction and metrics seams in
  `fit`

Logged metrics: `cde/omega_t`, `cde/bonus_mean{,_pos,_neg}`,
`cde/ppl_mean{,_pos,_neg}`, `cde/cap_active_frac` (fraction of
responses whose bonus is capped at `|r| / kappa`).

## Scripts

- `run_cde_grpo_qwen3_4b_base.sh`: CDE's native configuration (GRPO
  base), all other settings identical to
  `examples/scarl_trainer/run_scarl_grpo_qwen3_4b_base.sh`.
- `run_cde_dapo_qwen3_4b_base.sh`: same-base counterpart of
  `examples/scarl_trainer/run_scarl_dapo_qwen3_4b_base.sh` (DAPO base),
  all other settings identical to that script.
