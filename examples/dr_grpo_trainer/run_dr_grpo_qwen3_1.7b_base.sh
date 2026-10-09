# Dr.GRPO (Liu et al., "Understanding R1-Zero-Like Training: A Critical
# Perspective", arXiv:2503.20783) on Qwen3-1.7B-Base.
#
# Dr.GRPO is enabled by three overrides on top of the standard GRPO recipe;
# verl supports it natively, no source code change is required:
#   1. algorithm.norm_adv_by_std_in_grpo=False
#        => drop the per-group std normalisation (removes difficulty bias).
#   2. actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm
#        => aggregate token-level losses by a global constant instead of
#           per-sequence mean (removes length bias).
#   3. actor_rollout_ref.actor.use_kl_loss=False
#        => Dr.GRPO drops the KL loss term.
#
# Uses recipe.dapo.main_dapo as dispatcher (same convention as seed_grpo /
# eas examples). DAPO-specific tricks (filter_groups, overlong buffer,
# asymmetric clip) are left at their defaults so only Dr.GRPO is exercised.

# Ray runtime env conflict resolution
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1

# NCCL config: lower comm pressure and raise timeouts
export NCCL_NVLS_ENABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=7200
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_IB_TIMEOUT=23
export NCCL_DEBUG=INFO

project_name="scarl"
adv_estimator_name="DR_GRPO"
current_date_time=$(date +'%m%d-%H%M')

adv_estimator=grpo
# Dr.GRPO: no std norm in advantage.
norm_adv_by_std_in_grpo=False
# Dr.GRPO: global-constant token-loss aggregation (no length bias).
loss_agg_mode=seq-mean-token-sum-norm

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 4))

train_prompt_bsz=512
n_resp_per_prompt=8
train_prompt_mini_bsz=32

# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/examples/dr_grpo_trainer/runtime_env.yaml"}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
echo "${NGPUS_PER_NODE} NGPUS_PER_NODE"

# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3-1.7B-Base"}

model_name=$(basename "${MODEL_PATH:-Qwen3-1.7B-Base}")
exp_name="${adv_estimator_name}-${model_name}-n${n_resp_per_prompt}-bsz${train_prompt_bsz}-minibsz${train_prompt_mini_bsz}-${current_date_time}"

CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k-processed-all.parquet"}
AIME24="${RAY_DATA_HOME}/data/aime_2024/aime_2024.parquet"
AIME25="${RAY_DATA_HOME}/data/aime_2025/aime_2025.parquet"
MATH_500="${RAY_DATA_HOME}/data/math_500/math_500.parquet"
AMC23="${RAY_DATA_HOME}/data/amc23/amc23.parquet"
OlympiadBench="${RAY_DATA_HOME}/data/olympiadbench.parquet"
MINERVA="${RAY_DATA_HOME}/data/simplerl-minerva-math/minerva.parquet"

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.95
val_temperature=0.6
val_top_k=20

# Performance Related Parameter
sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
offload=True
gen_tp=1
fsdp_size=-1

ray job submit --runtime-env="${RUNTIME_ENV}" \
    -- python3 -m recipe.dapo.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="[${AIME24},${AIME25},${MATH_500},${AMC23},${MINERVA}]" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.norm_adv_by_std_in_grpo=${norm_adv_by_std_in_grpo} \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.95 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    actor_rollout_ref.rollout.name=vllm \
    reward_model.reward_manager=dapo \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.val_before_train=False \
    trainer.total_training_steps=150 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    data.shuffle=False \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.ref.use_torch_compile=True \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.ref.entropy_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=True \
    trainer.total_epochs=7 $@
