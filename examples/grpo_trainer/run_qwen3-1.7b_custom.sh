# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.

set -x

# project_name="GRPO"
project_name="scarl"
adv_estimator_name="GRPO"
# 0128-1430
current_date_time=$(date +'%m%d-%H%M')

# exp_name="GRPO-Qwen3-4B-Base-Custom-0125"
# exp_name="GRPO-Qwen3-4B-Base-Custom-0125-n8"
# exp_name="GRPO-Qwen3-4B-Base-Custom-0125-n8-2"
# exp_name="GRPO-Qwen3-4B-Base-Custom-0127"
# exp_name="GRPO-Qwen3-4B-Base-Custom-0127-2epoch"
# exp_name="GRPO-Qwen3-4B-Base-Custom-0127-n16"
# exp_name="GRPO-Qwen3-4B-Base-Custom-0128-n8-a800"

    # actor_rollout_ref.model.trust_remote_code=True \
    # actor_rollout_ref.model.custom_chat_template="{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}" \

adv_estimator=grpo

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 4))

train_prompt_bsz=512  # 1024
# n_resp_per_prompt=16  # 5
n_resp_per_prompt=8
train_prompt_mini_bsz=32  # 256

# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3-1.7B-Base"}
# MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/MiniCPM-2B-128k"}
# MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/OLMo-2-0425-1B"}
# MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/DeepSeek-R1-Distill-Qwen-1.5B"}
# MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Hunyuan-1.8B-Pretrain"}

model_name=$(basename "${MODEL_PATH:-Qwen3-1.7B-Base}")
# model_name=$(basename "${MODEL_PATH:-MiniCPM-2B-128k}")
# model_name=$(basename "${MODEL_PATH:-OLMo-2-0425-1B}")
# model_name=$(basename "${MODEL_PATH:-DeepSeek-R1-Distill-Qwen-1.5B}")
# model_name=$(basename "${MODEL_PATH:-Hunyuan-1.8B-Pretrain}")

exp_name="${adv_estimator_name}-a${adv_base_estimator_name}-${model_name}-Custom-n${n_resp_per_prompt}-bsz${train_prompt_bsz}-minibsz${train_prompt_mini_bsz}-${current_date_time}"

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

micro_batch_size_per_gpu=$((train_prompt_mini_bsz / NGPUS_PER_NODE))

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.95
val_temperature=0.6
val_top_k=20


# Performance Related Parameter
sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=38192
offload=True
gen_tp=1
fsdp_size=-1

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
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
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
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.95 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
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