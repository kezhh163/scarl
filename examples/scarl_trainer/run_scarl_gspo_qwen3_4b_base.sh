# run Qwen3-30B GSPO with new model engine
set -x

HDFS_ROOT=${HDFS_ROOT:-$PWD}
DATA_ROOT=${DATA_ROOT:-$PWD}

# project_name="GRPO"
project_name="scarl"
adv_estimator_name="scarl"
adv_base_estimator_name="GSPO"

# wandb
# experiment_name=qwen3-4B-base-grpo-$backend
default_local_dir=$DATA_ROOT/checkpoint/$project_name/$experiment_name
current_date_time=$(date +'%m%d-%H%M')

# ===================================== Algorithm =====================================
adv_estimator=scarl
scarl_base_estimator=grpo
scarl_confidence_source=prob

scarl_tau=5
scarl_lam=1
scarl_class_wise_normalization=True
scarl_power=1

loss_mode=gspo

# reference policy
use_kl_in_reward=False
kl_coef=0.001
use_kl_loss=False
kl_loss_coef=0.001

clip_ratio_low=3e-4
clip_ratio_high=4e-4

actor_lr=1e-6
critic_lr=2e-6
gae_gamma=1.0
gae_lam=0.95
critic_warmup=0

# ===================================== Data/Model =====================================
# train_files=$DATA_ROOT/dataset/BytedTsinghua-SIA/DAPO-Math-17k/data/dapo-math-17k.parquet
# test_files=$DATA_ROOT/dataset/aime-2024.parquet

# actor_model_path=$HDFS_ROOT/model/Qwen3-4B-Base
# critic_model_path=$actor_model_path

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 4))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

train_prompt_bsz=512
train_prompt_mini_bsz=32
n_resp_per_prompt=8
n_resp_per_prompt_val=32


# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/examples/scarl_trainer/runtime_env.yaml"}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
echo "${NGPUS_PER_NODE} NGPUS_PER_NODE"
# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3-4B-Base"}

model_name=$(basename "${MODEL_PATH:-Qwen3-4B-Base}")
exp_name="${adv_estimator_name}-adv${adv_base_estimator_name}-${model_name}-Custom-power${scarl_power}-tau${scarl_tau}-lam${scarl_lam}-n${n_resp_per_prompt}-bsz${train_prompt_bsz}-minibsz${train_prompt_mini_bsz}-${current_date_time}"

CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k-processed-all.parquet"}
# TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime_2024/aime_2024.parquet"}
AIME24="${RAY_DATA_HOME}/data/aime_2024/aime_2024.parquet"
AIME25="${RAY_DATA_HOME}/data/aime_2025/aime_2025.parquet"
MATH_500="${RAY_DATA_HOME}/data/math_500/math_500.parquet"
# GSM8K="${RAY_DATA_HOME}/data/gsm8k/test.parquet"
AMC23="${RAY_DATA_HOME}/data/amc23/amc23.parquet"
OlympiadBench="${RAY_DATA_HOME}/data/olympiadbench.parquet"
MINERVA="${RAY_DATA_HOME}/data/simplerl-minerva-math/minerva.parquet"
# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.95
val_temperature=0.6
val_top_k=20

sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=38192
critic_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=True
gen_tp=1
fsdp_size=-1

# # ===================================== Training =====================================
# actor_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) * 3))
# critic_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) * 4))

# # FSDP parallelism config
USP_SIZE=4
ACTOR_FSDP_CONFIG="
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1"

# # Megatron parallelism config
# TP_SIZE=2
# CP_SIZE=1
# PP_SIZE=1
# VPP_SIZE=null
# EP_SIZE=8
# ETP_SIZE=1
ACTOR_MEGATRON_CONFIG="
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.actor.megatron.context_parallel_size=$CP_SIZE \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP_SIZE \
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=$VPP_SIZE \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP_SIZE \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$ETP_SIZE \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    actor_rollout_ref.actor.megatron.use_mbridge=True"

# Actor model config
ACTOR_CONFIG="
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.grad_clip=1.0"

# Ref Model config
REF_CONFIG="
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}"

# Critic model config
CIRITC_CONFIG="
    critic.optim.lr=$critic_lr \
    critic.model.path=$critic_model_path \
    critic.model.use_remove_padding=True \
    critic.ppo_max_token_len_per_gpu=${critic_ppo_max_token_len} \
    critic.ulysses_sequence_parallel_size=$USP_SIZE"

CRITIC_FSDP_CONFIG="${ACTOR_FSDP_CONFIG//actor_rollout_ref.actor/critic.model}"
CRITIC_MEGATRON_CONFIG="${ACTOR_MEGATRON_CONFIG//actor_rollout_ref.actor/critic}"

if [[ $backend == "megatron" ]]; then
    CONFIG_NAME=ppo_megatron_trainer
    ACTOR_CONFIG="$ACTOR_CONFIG $ACTOR_MEGATRON_CONFIG"
    if [[ $adv_estimator == "gae" ]]; then
        CIRITC_CONFIG="$CIRITC_CONFIG $CRITIC_MEGATRON_CONFIG"
    else
        CIRITC_CONFIG=""
    fi
else # fsdp, fsdp2
    CONFIG_NAME=ppo_trainer
    ACTOR_CONFIG="$ACTOR_CONFIG $ACTOR_FSDP_CONFIG"
    if [[ $adv_estimator == "gae" ]]; then
        CIRITC_CONFIG="$CIRITC_CONFIG $CRITIC_FSDP_CONFIG"
    else
        CIRITC_CONFIG=""
    fi
fi

# ===================================== Inference =====================================
rollout_name=vllm
# if [ "$rollout_name" = "vllm" ]; then
#     export VLLM_USE_V1=1
# fi
infer_tp=4
infer_dp=1
infer_ep=1
gpu_memory_utilization=0.8

ROLLOUT_CONFIG="
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    actor_rollout_ref.rollout.name=$rollout_name"


# ===================================== Reward =====================================
# REWARD_CONFIG="
#     reward_model.reward_manager=dapo \
#     reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
#     reward_model.overlong_buffer.len=${overlong_buffer_len} \
#     reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
#     +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \ #
#     +reward_model.reward_kwargs.max_resp_len=${max_response_length}" #

# python3 -m verl.trainer.main_ppo \
ray job submit --runtime-env="${RUNTIME_ENV}" \
    -- python3 -m recipe.dapo.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="[${AIME24},${AIME25},${MATH_500},${AMC23},${MINERVA}]" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    algorithm.scarl_tau=${scarl_tau} \
    algorithm.scarl_lam=${scarl_lam} \
    algorithm.scarl_base_estimator=${scarl_base_estimator} \
    algorithm.scarl_confidence_source=${scarl_confidence_source} \
    +algorithm.scarl_sign_aware_weighting=True \
    +algorithm.scarl_class_wise_normalization=${scarl_class_wise_normalization} \
    +algorithm.scarl_power=${scarl_power} \
    +algorithm.scarl_adaptive_tau=True \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.total_epochs=5 \
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
    $ACTOR_CONFIG \
    $CIRITC_CONFIG \
    $ROLLOUT_CONFIG \
