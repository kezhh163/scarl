# UCAS Baseline

Reimplementation of UCAS (Uncertainty-aware Advantage Shaping;
"Unlocking Exploration in RLVR: Uncertainty-aware Advantage Shaping
for Deeper Reasoning") used as a compute-matched baseline in the
paper. Two-stage shaping: (1) response level, the group-normalized
advantage is modulated by an asymmetric exponential of the z-scored
KL(U||p) self-certainty; (2) token level, a min-max-normalized raw
logit penalty is subtracted, with a clamp protecting positive
trajectories from sign flips.

Hyperparameters in the shipped scripts are the official-repo defaults
(`ucas_beta_positive=0.2`, `ucas_beta_negative=0.2`,
`ucas_logit_penalty_weight=0.01`), which performed better than the
paper-recommended settings in our runs and are what the paper's
tables report.

Dedicated code paths (shared functions untouched):

- `verl/trainer/ppo/core_algos.py`: `compute_ucas_advantage`,
  `_normalize_kl_certainty`
- `verl/utils/torch_functional.py`: `self_certainty_from_logits`
- `verl/workers/actor/dp_actor.py`: `_forward_micro_batch_for_ucas`,
  `compute_log_prob_for_ucas` (single forward returning log-probs,
  entropy, self-certainty, and pre-temperature selected-token logits)
- `verl/workers/fsdp_workers.py`: `compute_log_prob_for_ucas` worker
  endpoint

## Scripts

- `run_ucas_dapo_qwen3_4b_base.sh` (main comparison)
- `run_ucas_dapo_qwen3_1.7b_base.sh`
