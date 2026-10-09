# SCARL: Sequence Confidence-Aligned Reinforcement Learning

Code for the paper *SCARL: Sequence Confidence-Aligned Reinforcement Learning*.

SCARL repairs the sequence-confidence misalignment of group-relative
RLVR: it reshapes, but never re-signs, each response's advantage with
its **alignment gap** -- the distance between the response's
group-relative sequence confidence and the outcome its advantage
certifies -- normalized into shares of each outcome class's gradient
budget. It is a drop-in advantage transformation for any
group-relative estimator (GRPO, DAPO, GSPO, SAPO, ...), adds no extra
forward passes, and has a single substantive hyperparameter, the
temperature `tau`.

This repository is a fork of **verl v0.4.1** (HybridFlow). Framework
architecture, installation details, and general usage are documented
in [`README_verl.md`](README_verl.md) and the
[verl documentation](https://verl.readthedocs.io/); this README covers
only what the paper adds.

## Repository map (paper-specific code)

| Path | Content |
|---|---|
| `verl/trainer/ppo/core_algos.py` | SCARL estimator (`compute_scarl_outcome_advantage` and the weight computation) plus all baseline estimators, including the ported UCAS baseline (`compute_ucas_advantage`) |
| `verl/workers/actor/dp_actor.py` | Single-forward log-prob + selected-logits path (SCARL's optional `logit` mode); dedicated UCAS forward (adds KL(U&#124;&#124;p) self-certainty) |
| `verl/trainer/ppo/ray_trainer.py`, `recipe/dapo/dapo_ray_trainer.py` | Advantage dispatch and training-loop wiring |
| `examples/scarl_trainer/` | SCARL run scripts for all four base estimators (see its README) |
| `examples/ucas_trainer/`, `examples/{grpo,gspo,sapo,dr_grpo,seed_grpo,entropy_adv_shaping}_trainer/`, `recipe/dapo/` | Run scripts for the nine compute-matched methods |
| `recipe/dapo/prepare_dapo_data.sh` | Downloads the pinned public training and evaluation parquets |

## Installation

The paper's runs use verl v0.4.1's official Docker image
(vLLM + FSDP variant):

```bash
docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN \
  -v .:/workspace/scarl --name scarl \
  verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.2-te2.2 sleep infinity
docker start scarl && docker exec -it scarl bash
pip3 install --no-deps -e .
```

The image ships vLLM 0.8.5, PyTorch 2.6.0, and CUDA 12.4; training
uses the FSDP backend with vLLM rollout. Alternative installation
routes are described in `README_verl.md` and the
[verl documentation](https://verl.readthedocs.io/). All runs in the
paper use a single node with 8x NVIDIA A100 40GB GPUs.

## Data

All data are public Hugging Face releases with pre-rendered,
chat-format prompts (a single user turn; templates are reproduced in
the paper's technical supplement):

- **Training**: `dapo-math-17k-processed-all.parquet` from
  `open-r1/DAPO-Math-17k-Processed` (config `all`), a deduplicated
  release of DAPO-Math-17K; the runs consume the original DAPO
  answer-line prompt (the release's `source_prompt` column) under
  the `prompt` key expected by the dataloader.
- **Evaluation**: the public AIME24 (`BytedTsinghua-SIA/AIME-2024`),
  AIME25, AMC23, MATH-500, and Minerva Math parquets, placed under
  `${RAY_DATA_HOME}/data/` as referenced at the top of each run
  script.

`recipe/dapo/prepare_dapo_data.sh` is upstream's helper for the raw
DAPO-Math-17k and AIME-2024 files.

## Reproducing the paper

Every script reads `MODEL_PATH` (backbone), `RAY_DATA_HOME` (data and
checkpoint root), and standard Ray environment variables; defaults
match the paper's configuration (150 steps, group size 8, prompt
batch 512, evaluation of 32 samples per problem every 5 steps).

- **Main comparison** (Qwen3-4B-Base, DAPO base):
  `examples/scarl_trainer/run_scarl_dapo_qwen3_4b_base.sh` for
  DAPO+SCARL; the DAPO baseline and the other seven methods run from
  `recipe/dapo/run_dapo_qwen3_4b_base_custom4.sh` and the
  corresponding `examples/*_trainer/` scripts.
- **Generality grid** (four estimators x four backbones):
  `examples/scarl_trainer/run_scarl_{grpo,dapo,gspo,sapo}_qwen3_{1.7b,4b}_base.sh`;
  the remaining backbones use the same scripts with `MODEL_PATH`
  overridden.
- **Ablations**: edit the flag block at the top of a SCARL script --
  set `scarl_class_wise_normalization=False` (no class-wise
  normalization), change the
  `+algorithm.scarl_sign_aware_weighting=True` line to `False`
  (outcome-blind shaping), or set `scarl_tau` to one of
  {0.2, 0.5, 2, 5, 10} for the temperature sweep.
- **Metrics**: per-benchmark Avg@32 and Pass@32 are logged at each
  evaluation; tables report each run at its best checkpoint by the
  five-benchmark average, under the same rule for every method.

## Key configuration

| Flag | Paper value | Meaning |
|---|---|---|
| `algorithm.adv_estimator` | `scarl` | Enables SCARL |
| `algorithm.scarl_tau` | `5` | Softmax temperature; the single substantive hyperparameter |
| `algorithm.scarl_confidence_source` | `prob` | Sequence confidence from the old-policy log-probs already materialized for the importance ratios (Eq. 5 of the paper; zero extra cost). `logit` mode instead uses raw selected-token logits via a dedicated single-forward path |
| `algorithm.scarl_adaptive_tau` | `True` | Divide by the group's confidence std before the temperature (calibrated sharpness) |
| `algorithm.scarl_sign_aware_weighting` | `True` | Gap taken against the outcome the advantage certifies; `False` is the outcome-blind ablation |
| `algorithm.scarl_class_wise_normalization` | `True` | Normalize gaps into shares within each outcome class; `False` is the catastrophic ablation |
| `algorithm.scarl_lam`, `algorithm.scarl_power` | `1`, `1` | Identity defaults retained for config compatibility; at these values both are no-ops, so the executed computation matches the paper's single-hyperparameter description |

Baseline scripts follow each method's published recipe; per-estimator
settings are tabulated in the paper's technical supplement.

## License

Apache 2.0, inherited from verl (see `LICENSE`).
