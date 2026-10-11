# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""

__all__ = ["register_adv_est", "get_adv_estimator_fn", "AdvantageEstimator"]

from collections import defaultdict
from enum import Enum
from typing import Any, Optional

import numpy as np
import torch

import verl.utils.torch_functional as verl_F

POLICY_LOSS_REGISTRY = {}


def register_policy_loss(name):
    def decorator(func):
        POLICY_LOSS_REGISTRY[name] = func
        return func

    return decorator


def get_policy_loss_fn(name):
    """Get the policy loss with a given name.

    Args:
        name: `(str)`
            The name of the policy loss.

    Returns:
        `(callable)`: The policy loss function.
    """
    loss_name = name
    if loss_name not in POLICY_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(POLICY_LOSS_REGISTRY.keys())}"
        )
    return POLICY_LOSS_REGISTRY[loss_name]


ADV_ESTIMATOR_REGISTRY = {}


def register_adv_est(name_or_enum):
    """Decorator to register a advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    """

    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in ADV_ESTIMATOR_REGISTRY and ADV_ESTIMATOR_REGISTRY[name] != fn:
            raise ValueError(
                f"Adv estimator {name} has already been registered: {ADV_ESTIMATOR_REGISTRY[name]} vs {fn}"
            )
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn

    return decorator


def get_adv_estimator_fn(name_or_enum):
    """Get the advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    Returns:
        `(callable)`: The advantage estimator function.
    """
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in ADV_ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown advantage estimator simply: {name}")
    return ADV_ESTIMATOR_REGISTRY[name]


class AdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estimator.

    Note(haibin.lin): this enum class is immutable after creation. Extending this
    enum for new estimators may not be necessary since users can always just call
    `verl.trainer.ppo.core_algos.register` with string name for a custom advantage
    estimator instead.
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"
    SCARL = "scarl"
    SEED_GRPO = "seed_grpo"
    UCAS = "ucas"
    CDE = "cde"


def _normalize_kl_certainty(
    certainty: torch.Tensor, mask: torch.Tensor, epsilon: float = 1e-6
) -> torch.Tensor:
    """Helper function for Min-Max normalization of KL-Divergence certainties."""
    # We operate on a copy
    masked_certainty = certainty.clone()

    # To correctly find the min and max only on valid tokens, we mask out the padding tokens.
    # By setting padding tokens to infinity, they won't be chosen as the minimum.
    masked_certainty[~mask.bool()] = float("inf")
    batch_min = masked_certainty.min()

    # By setting padding tokens to negative infinity, they won't be chosen as the maximum.
    masked_certainty[~mask.bool()] = float("-inf")
    batch_max = masked_certainty.max()

    # Handle the edge case where all values in the batch are identical.
    if torch.isclose(batch_max, batch_min):
        # All tokens have the same confidence, so we can treat them as neutral (0.5).
        return torch.full_like(certainty, 0.5) * mask.float()

    # Perform min-max scaling to map values to the [0, 1] range.
    normalized = (certainty - batch_min) / (batch_max - batch_min + epsilon)

    # Re-apply the mask to ensure padding tokens are zero.
    return normalized * mask.float()


@register_adv_est(AdvantageEstimator.UCAS)
def compute_ucas_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    token_level_certaintys: torch.Tensor = None,
    token_level_logits: torch.Tensor = None,
    beta_positive: float = 0.2,
    beta_negative: float = 0.2,
    logit_penalty_weight: float = 0.01,
    **kwargs,
):
    """
    Compute the UCAS (Uncertainty-aware Advantage Shaping) advantage.

    Implements two-stage advantage shaping:
      Stage 1 (Response-Level): Modulates the group-normalized GRPO advantage by the
        response's z-scored self-confidence, using an asymmetric exponential weight so
        that novel correct responses are amplified and confident wrong ones are penalized.
      Stage 2 (Token-Level): Subtracts a Min-Max normalized logit penalty from each
        token to discourage local overconfidence, with an asymmetric clamp that protects
        positive trajectories from flipping to negative.

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length). Sparse reward tensor (non-zero at EOS only).
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length).
        index: `(np.ndarray)` prompt group indices for group-wise normalization.
        epsilon: small constant for numerical stability.
        norm_adv_by_std_in_grpo: if True, normalize advantages by group std (standard GRPO);
            if False, subtract only the mean (Dr. GRPO variant).
        token_level_certaintys: `(torch.Tensor)`
            shape: (bs, response_length). Per-token KL(U||p) self-confidence values.
        token_level_logits: `(torch.Tensor)`
            shape: (bs, response_length). Raw logit of the generated token at each step.

    Returns:
        advantages: `(torch.Tensor)` shape: (bs, response_length)
        returns:    `(torch.Tensor)` shape: (bs, response_length)  [same as advantages]
    """

    scores = token_level_rewards.sum(dim=-1)
    certaintys = token_level_certaintys.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    id2certainty = defaultdict(list)
    id2mean_certainty = {}
    id2std_certainty = {}

    with torch.no_grad():
        normalized_logits = _normalize_kl_certainty(token_level_logits, response_mask)
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2certainty[index[i]].append(certaintys[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")

        for idx in id2certainty:
            if len(id2certainty[idx]) == 1:
                id2mean_certainty[idx] = torch.tensor(0.0)
                id2std_certainty[idx] = torch.tensor(1.0)
            elif len(id2certainty[idx]) > 1:
                id2mean_certainty[idx] = torch.mean(torch.tensor(id2certainty[idx]))
                id2std_certainty[idx] = torch.std(torch.tensor([id2certainty[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")

        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
                certaintys[i] = (certaintys[i] - id2mean_certainty[index[i]]) / (
                    id2std_certainty[index[i]] + epsilon
                )
            else:
                scores[i] = scores[i] - id2mean[index[i]]
                certaintys[i] = certaintys[i] - id2mean_certainty[index[i]]

        scores = scores.unsqueeze(-1) * response_mask
        certaintys = certaintys.unsqueeze(-1) * response_mask

        positive_modifier = torch.exp(-beta_positive * certaintys)
        # When advantage <= 0 (wrong/worse answer), penalize high confidence.
        # exp(beta * certainty) -> larger certainty gives larger value -> stronger penalty on |negative advantage|
        negative_modifier = torch.exp(beta_negative * certaintys)
        is_positive_advantage = scores > 0
        certainty_modifier = torch.where(is_positive_advantage, positive_modifier, negative_modifier)

        raw = scores * certainty_modifier - logit_penalty_weight * normalized_logits
        mask = (scores * certainty_modifier) > 0
        final_scores = torch.where(mask, torch.maximum(raw, torch.tensor(0.0)), raw)

    return final_scores, final_scores


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(
            init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon
        )
    else:
        raise NotImplementedError


@register_adv_est(AdvantageEstimator.GAE)  # or simply: @register_adv_est("gae")
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@register_adv_est(AdvantageEstimator.GRPO)  # or simply: @register_adv_est("grpo")
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: str = True,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.
            If True, the advantage is scaled by the std, as in the original GRPO.
            If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.GRPO_PASSK)  # or simply: @register_adv_est("grpo_passk")
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
    **kwargs,
):
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        config: (dict) algorithm settings, which contains "norm_adv_by_std_in_grpo"

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    assert config is not None
    # if True, normalize advantage by std within group
    norm_adv_by_std_in_grpo = config.get("norm_adv_by_std_in_grpo", True)
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)

        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(
                    f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}."
                )
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


@register_adv_est(
    AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE
)  # or simply: @register_adv_est("reinforce_plus_plus_baseline")
def compute_reinforce_plus_plus_baseline_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
    config=None,
    **kwargs,
):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO)  # or simply: @register_adv_est("rloo")
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config=None,
    **kwargs,
):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[
                    index[i]
                ] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.OPO)  # or simply: @register_adv_est("opo")
def compute_opo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config=None,
    **kwargs,
):
    """
    Compute advantage for OPO based on https://arxiv.org/pdf/2505.23585

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.sum(dim=-1)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2len = defaultdict(list)
    id2bsl = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2len[index[i]].append(response_length[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2bsl[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                score_tensor = torch.tensor(id2score[idx])
                len_tensor = torch.tensor(id2len[idx])
                id2bsl[idx] = (len_tensor * score_tensor).sum() / len_tensor.sum()
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2bsl[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(
    AdvantageEstimator.REINFORCE_PLUS_PLUS
)  # or simply: @register_adv_est("reinforce_plus_plus")
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, config=None, **kwargs
):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    assert config is not None
    gamma = config.gamma
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.REMAX)  # or simply: @register_adv_est("remax")
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor,
    reward_baselines: torch.Tensor,
    response_mask: torch.Tensor,
    config=None,
    **kwargs,
):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


def compute_scarl_entropy_weights_from_logits(
    selected_logits: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    tau: float = 5.0,
    lam: float = 1.0,
    adv_per_sequence: torch.Tensor = None,
    sign_aware_weighting: bool = True,
    class_wise_normalization: bool = True,
    power: float = 1.0,
    adaptive_tau: bool = True,
):
    """
    Compute SCARL entropy-based weights from raw logits for advantage modulation.

    This version uses raw logits of selected tokens as absolute confidence values
    that can be compared across different sequences.

    Args:
        selected_logits: `(torch.Tensor)`
            shape: (bs, response_length). Raw logits of selected (generated) tokens.
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length). Mask for valid response tokens.
        index: `(np.ndarray)`
            shape: (bs,). Group indices for each sample.
        tau: `(float)` Temperature coefficient.
        lam: `(float)` Scaling coefficient.
        adv_per_sequence: `(torch.Tensor)`
            shape: (bs,). Sequence-level advantage for sign determination.
            If provided, used for sign-aware weighting logic.
        class_wise_normalization: `(bool)` Whether to normalize weights within positive/negative classes.
        power: `(float)` Power factor to apply to the confidence weights before normalization. Default: 1.0.
        adaptive_tau: `(bool)` Whether to divide by group std before temperature scaling. Default: True.

    Returns:
        e_s: `(torch.Tensor)` shape: (bs,). Entropy advantage weights per sequence.
    """
    with torch.no_grad():
        # Use logsigmoid to map raw logits to confidence scores
        # c_t = logsigmoid(logits_t), value in (-inf, 0)
        # Higher logits -> closer to 0 (more confident)
        c_t = torch.nn.functional.logsigmoid(selected_logits)
        masked_c_t = c_t * response_mask
        c_s = masked_c_t.sum(dim=-1)  # (bs,)

        e_s, conf_metrics = _compute_scarl_weights_from_confidence(
            c_s,
            index,
            tau,
            lam,
            adv_per_sequence,
            sign_aware_weighting,
            class_wise_normalization,
            power,
            adaptive_tau,
        )
        return e_s, conf_metrics


def compute_scarl_entropy_weights_from_probs(
    old_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    tau: float = 5.0,
    lam: float = 1.0,
    adv_per_sequence: torch.Tensor = None,
    sign_aware_weighting: bool = True,
    class_wise_normalization: bool = True,
    power: float = 1.0,
    adaptive_tau: bool = True,
):
    """
    Compute SCARL entropy-based weights from log probabilities for advantage modulation.

    This version uses log probabilities as confidence proxy. Note that log probabilities
    are normalized within vocabulary, so they only provide relative confidence within
    each token position, not absolute confidence across sequences.

    Args:
        old_log_probs: `(torch.Tensor)`
            shape: (bs, response_length). Log probabilities of selected tokens.
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length). Mask for valid response tokens.
        index: `(np.ndarray)`
            shape: (bs,). Group indices for each sample.
        tau: `(float)` Temperature coefficient.
        lam: `(float)` Scaling coefficient.
        adv_per_sequence: `(torch.Tensor)`
            shape: (bs,). Sequence-level advantage for sign determination.
            If provided, used for sign-aware weighting logic.
        class_wise_normalization: `(bool)` Whether to normalize weights within positive/negative classes.
        power: `(float)` Power factor to apply to the confidence weights before normalization. Default: 1.0.
        adaptive_tau: `(bool)` Whether to divide by group std before temperature scaling. Default: True.

    Returns:
        e_s: `(torch.Tensor)` shape: (bs,). Entropy advantage weights per sequence.
    """
    with torch.no_grad():
        # old_log_probs are already log probabilities, serving as confidence proxy
        # More negative = less confident; closer to 0 = more confident
        masked_log_probs = old_log_probs * response_mask
        c_s = masked_log_probs.sum(dim=-1)  # (bs,)

        e_s, conf_metrics = _compute_scarl_weights_from_confidence(
            c_s,
            index,
            tau,
            lam,
            adv_per_sequence,
            sign_aware_weighting,
            class_wise_normalization,
            power,
            adaptive_tau,
        )
        return e_s, conf_metrics


def _compute_scarl_weights_from_confidence(
    c_s: torch.Tensor,
    index: np.ndarray,
    tau: float = 5.0,
    lam: float = 1.0,
    adv_per_sequence: torch.Tensor = None,
    sign_aware_weighting: bool = True,
    class_wise_normalization: bool = True,
    power: float = 1.0,
    adaptive_tau: bool = True,
):
    """
    Compute SCARL entropy weights from sequence-level confidence scores.

    This is the shared implementation for both logit-based and prob-based confidence.

    Args:
        c_s: `(torch.Tensor)` shape: (bs,). Sequence-level confidence scores.
        index: `(np.ndarray)` shape: (bs,). Group indices for each sample.
        tau: `(float)` Temperature coefficient.
        lam: `(float)` Scaling coefficient.
        adv_per_sequence: `(torch.Tensor)` shape: (bs,). Sequence-level advantage for sign determination.
        sign_aware_weighting: `(bool)` Whether to enable sign-aware weighting. Default: True.
            When enabled: positive adv uses (1 - softmax), negative adv uses softmax.
            When disabled: always use softmax for all samples.
        class_wise_normalization: `(bool)` Whether to normalize weights within positive/negative classes.
            Default: True. When enabled, normalizes so that each class has mean weight of 1.
        power: `(float)` Power factor to apply to the confidence weights before normalization. Default: 1.0.
        adaptive_tau: `(bool)` Whether to divide by group std before temperature scaling. Default: True.
            When enabled, tau_effective = std * tau, making softmax sharpness adaptive per group.

    Returns:
        e_s: `(torch.Tensor)` shape: (bs,). Entropy advantage weights per sequence.
    """
    # Compute entropy advantage weights within each group
    # Group by prompt index to compute softmax within groups
    bsz = c_s.shape[0]
    e_s = torch.zeros(bsz, device=c_s.device, dtype=c_s.dtype)

    # Track per-group confidence (c_s) stats
    pos_conf_vars = []
    neg_conf_vars = []
    group_mean_diffs = []
    # Track per-group softmax_probs stats
    pos_sp_vars = []
    neg_sp_vars = []
    group_sp_mean_diffs = []
    all_pos_sp = []
    all_neg_sp = []
    # Track per-group weight (group_e_s) stats
    pos_wt_vars = []
    neg_wt_vars = []
    group_wt_mean_diffs = []
    all_pos_wt = []
    all_neg_wt = []

    # Group samples by their prompt index
    id2indices = defaultdict(list)
    for i in range(bsz):
        id2indices[index[i]].append(i)

    for idx, sample_indices in id2indices.items():
        if len(sample_indices) == 1:
            # Single sample in group: assign neutral weight
            e_s[sample_indices[0]] = 1.0
        else:
            # Multiple samples: compute softmax-based weights
            group_indices = torch.tensor(sample_indices, device=c_s.device)
            group_c_s = c_s[group_indices]  # (group_size,)

            # Apply temperature scaling
            # When adaptive_tau is enabled, divide by group std first
            # so that tau_effective = std * tau, making softmax sharpness adaptive per group
            if adaptive_tau:
                group_std = group_c_s.std()
                scaled_c_s = group_c_s / (group_std.clamp(min=1e-8) * tau)
            else:
                scaled_c_s = group_c_s / tau

            # Compute softmax probabilities
            softmax_probs = torch.softmax(scaled_c_s, dim=0)

            # # Default weighting logic (original SCARL): 1 - softmax
            # # Encourages exploration by upweighting less confident samples
            # base_weights = 1.0 - softmax_probs

            # Sign-aware weighting logic
            if sign_aware_weighting and adv_per_sequence is not None:
                group_adv = adv_per_sequence[group_indices]
                # Identify positive advantage samples
                # For positive advantage: weight = 1 - softmax (encourage low confidence success)
                # For negative advantage: weight = softmax (penalize high confidence failure)
                pos_mask = group_adv >= 0

                base_weights = 1.0 - softmax_probs

                # Apply conditional weighting
                raw_weights = torch.where(pos_mask, base_weights, softmax_probs)

                # Apply power factor
                if power != 1.0:
                    raw_weights = torch.pow(raw_weights, power)

                # Class-wise normalization: normalize positive and negative samples separately
                # so that each class has mean weight of 1
                if class_wise_normalization:
                    eps = 1e-8
                    neg_mask = ~pos_mask
                    # Use new_tensor to keep device and dtype consistent
                    pos_mean = raw_weights[pos_mask].mean() if pos_mask.any() else raw_weights.new_tensor(1.0)
                    neg_mean = raw_weights[neg_mask].mean() if neg_mask.any() else raw_weights.new_tensor(1.0)

                    # Build per-sample scale factors with numerical stability
                    scale = torch.where(
                        pos_mask, 1.0 / pos_mean.clamp(min=eps), 1.0 / neg_mean.clamp(min=eps)
                    )
                    weights = raw_weights * scale
                else:
                    weights = raw_weights
            else:
                # When sign-aware weighting is disabled, use 1 - softmax for all samples
                raw_weights = 1.0 - softmax_probs

                # Apply power factor
                if power != 1.0:
                    raw_weights = torch.pow(raw_weights, power)

                # Normalize so that mean weight is 1
                if class_wise_normalization:
                    eps = 1e-8
                    weights = raw_weights / raw_weights.mean().clamp(min=eps)
                else:
                    weights = raw_weights

            # Collect per-group stats for c_s and softmax_probs
            if adv_per_sequence is not None:
                group_adv_for_var = adv_per_sequence[group_indices]
                pos_mask_for_var = group_adv_for_var >= 0
                neg_mask_for_var = ~pos_mask_for_var
                # --- c_s stats ---
                pos_c = group_c_s[pos_mask_for_var]
                neg_c = group_c_s[neg_mask_for_var]
                if pos_c.numel() >= 2:
                    pos_conf_vars.append(pos_c.var())
                if neg_c.numel() >= 2:
                    neg_conf_vars.append(neg_c.var())
                if pos_c.numel() >= 1 and neg_c.numel() >= 1:
                    group_mean_diffs.append(pos_c.mean() - neg_c.mean())
                # --- softmax_probs stats ---
                pos_sp = softmax_probs[pos_mask_for_var]
                neg_sp = softmax_probs[neg_mask_for_var]
                if pos_sp.numel() >= 1:
                    all_pos_sp.append(pos_sp)
                if neg_sp.numel() >= 1:
                    all_neg_sp.append(neg_sp)
                if pos_sp.numel() >= 2:
                    pos_sp_vars.append(pos_sp.var())
                if neg_sp.numel() >= 2:
                    neg_sp_vars.append(neg_sp.var())
                if pos_sp.numel() >= 1 and neg_sp.numel() >= 1:
                    group_sp_mean_diffs.append(pos_sp.mean() - neg_sp.mean())

            # Apply scaling factor
            group_e_s = lam * weights

            # --- weight (group_e_s) stats ---
            if adv_per_sequence is not None:
                pos_wt = group_e_s[pos_mask_for_var]
                neg_wt = group_e_s[neg_mask_for_var]
                if pos_wt.numel() >= 1:
                    all_pos_wt.append(pos_wt)
                if neg_wt.numel() >= 1:
                    all_neg_wt.append(neg_wt)
                if pos_wt.numel() >= 2:
                    pos_wt_vars.append(pos_wt.var())
                if neg_wt.numel() >= 2:
                    neg_wt_vars.append(neg_wt.var())
                if pos_wt.numel() >= 1 and neg_wt.numel() >= 1:
                    group_wt_mean_diffs.append(pos_wt.mean() - neg_wt.mean())

            # Assign back to batch positions
            for local_idx, global_idx in enumerate(sample_indices):
                e_s[global_idx] = group_e_s[local_idx]

    # ========== Build all SCARL metrics ==========
    scarl_conf_metrics = {}

    # Helper: extract (mean, var-avg, group-diff-avg, max, min) for a given signal.
    # - global_vals: concatenated tensor of all pos/neg values across groups (or None)
    # - per_group_vars: list of per-group variance tensors
    # - group_diffs: list of per-group mean-diff tensors
    def _fill_metrics(prefix, global_pos_vals, global_neg_vals, pg_vars_pos, pg_vars_neg, pg_diffs):
        pos_t = torch.cat(global_pos_vals) if global_pos_vals else None
        neg_t = torch.cat(global_neg_vals) if global_neg_vals else None
        scarl_conf_metrics[f"scarl/{prefix}_pos_mean"] = pos_t.mean().item() if pos_t is not None else 0.0
        scarl_conf_metrics[f"scarl/{prefix}_neg_mean"] = neg_t.mean().item() if neg_t is not None else 0.0
        scarl_conf_metrics[f"scarl/{prefix}_pos_var"] = (
            torch.stack(pg_vars_pos).mean().item() if pg_vars_pos else 0.0
        )
        scarl_conf_metrics[f"scarl/{prefix}_neg_var"] = (
            torch.stack(pg_vars_neg).mean().item() if pg_vars_neg else 0.0
        )
        scarl_conf_metrics[f"scarl/{prefix}_group_mean_diff"] = (
            torch.stack(pg_diffs).mean().item() if pg_diffs else 0.0
        )
        scarl_conf_metrics[f"scarl/{prefix}_pos_max"] = pos_t.max().item() if pos_t is not None else 0.0
        scarl_conf_metrics[f"scarl/{prefix}_pos_min"] = pos_t.min().item() if pos_t is not None else 0.0
        scarl_conf_metrics[f"scarl/{prefix}_neg_max"] = neg_t.max().item() if neg_t is not None else 0.0
        scarl_conf_metrics[f"scarl/{prefix}_neg_min"] = neg_t.min().item() if neg_t is not None else 0.0

    # --- confidence (c_s): global tensor available, split by adv sign ---
    if adv_per_sequence is not None:
        global_pos_mask = adv_per_sequence >= 0
        global_neg_mask = ~global_pos_mask
        pos_c_all = [c_s[global_pos_mask]] if global_pos_mask.any() else []
        neg_c_all = [c_s[global_neg_mask]] if global_neg_mask.any() else []
    else:
        pos_c_all, neg_c_all = [], []
    _fill_metrics("confidence", pos_c_all, neg_c_all, pos_conf_vars, neg_conf_vars, group_mean_diffs)

    # --- softmax_probs: per-group tensors collected in loop ---
    _fill_metrics("softmax", all_pos_sp, all_neg_sp, pos_sp_vars, neg_sp_vars, group_sp_mean_diffs)

    # --- weight (group_e_s): per-group tensors collected in loop ---
    _fill_metrics("weight", all_pos_wt, all_neg_wt, pos_wt_vars, neg_wt_vars, group_wt_mean_diffs)

    return e_s, scarl_conf_metrics


def compute_confidence_diagnostic_metrics(
    old_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    advantages: torch.Tensor,
    index: np.ndarray,
    tau: float = 5.0,
    lam: float = 1.0,
    sign_aware_weighting: bool = True,
    class_wise_normalization: bool = True,
    power: float = 1.0,
):
    """
    Compute confidence diagnostic metrics for any advantage estimator (non-SCARL).

    Uses old_log_probs (prob mode) as the confidence source. Computes hypothetical
    SCARL-style softmax_probs and weights for diagnostic comparison, without modifying
    the actual advantages.

    Args:
        old_log_probs: (bs, response_length) log probabilities of selected tokens.
        response_mask: (bs, response_length) mask for valid response tokens.
        advantages: (bs, response_length) computed advantages from the base estimator.
        index: (bs,) group indices for each sample.
        tau, lam, sign_aware_weighting, class_wise_normalization, power:
            SCARL hyperparams with sensible defaults for diagnostic purposes.

    Returns:
        dict: Diagnostic metrics with keys prefixed by "scarl/".
    """
    with torch.no_grad():
        # Compute sequence-level confidence from log probs (same as SCARL prob mode)
        masked_log_probs = old_log_probs * response_mask
        c_s = masked_log_probs.sum(dim=-1)  # (bs,)

        # Compute sequence-level advantage for sign determination
        adv_per_sequence = advantages.mean(dim=-1)  # (bs,)

        # Reuse the internal function to compute hypothetical weights AND metrics
        _, metrics = _compute_scarl_weights_from_confidence(
            c_s=c_s,
            index=index,
            tau=tau,
            lam=lam,
            adv_per_sequence=adv_per_sequence,
            sign_aware_weighting=sign_aware_weighting,
            class_wise_normalization=class_wise_normalization,
            power=power,
        )
        return metrics


@register_adv_est(AdvantageEstimator.SCARL)
def compute_scarl_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    selected_logits: torch.Tensor = None,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
    **kwargs,
):
    """
    Compute advantage for SCARL (Gradient-weighted Entropy Penalized Optimization).

    SCARL modulates the GRPO advantage estimates with entropy-based weights that
    emphasize learning from uncertain (high entropy) samples while dampening
    learning from overly confident samples.

    The core idea:
    - Compute base advantages using GRPO (or other base estimator)
    - Calculate sequence-level confidence from raw logits (absolute values)
    - Apply entropy-based reweighting: uncertain sequences get higher weights

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.ndarray)`
            shape: (bs,). Group indices for each sample.
        selected_logits: `(torch.Tensor)`
            shape: (bs, response_length). Raw logits of selected (generated) tokens.
            These absolute values enable cross-sequence confidence comparison.
        epsilon: `(float)`
            Small constant for numerical stability.
        norm_adv_by_std_in_grpo: `(bool)`
            Whether to normalize advantages by std (from base GRPO).
        config: `(dict)`
            Algorithm configuration containing scarl parameters:
            - scarl_tau: temperature coefficient (default: 0.1)
            - scarl_lam: scaling coefficient (default: 2.0)
            - scarl_base_estimator: base advantage estimator (default: "grpo")
            - scarl_power: power factor applied to confidence weights before normalization (default: 1.0)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length). SCARL-weighted advantages.
        returns: `(torch.Tensor)`
            shape: (bs, response_length). Same as advantages for outcome-based methods.
    """
    # Get SCARL hyperparameters from config
    if config is not None:
        tau = config.get("scarl_tau", 5.0)
        lam = config.get("scarl_lam", 1.0)
        base_estimator = config.get("scarl_base_estimator", "grpo")
        norm_adv_by_std_in_grpo = config.get("norm_adv_by_std_in_grpo", True)
    else:
        tau = 5.0
        lam = 1.0
        base_estimator = "grpo"

    with torch.no_grad():
        # Step 1: Compute base advantages using the specified base estimator
        if base_estimator == "grpo":
            base_advantages, base_returns = compute_grpo_outcome_advantage(
                token_level_rewards=token_level_rewards,
                response_mask=response_mask,
                index=index,
                epsilon=epsilon,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            )
        elif base_estimator == "rloo":
            base_advantages, base_returns = compute_rloo_outcome_advantage(
                token_level_rewards=token_level_rewards,
                response_mask=response_mask,
                index=index,
                epsilon=epsilon,
                config=config,
            )
        elif base_estimator == "reinforce_plus_plus_baseline":
            base_advantages, base_returns = compute_reinforce_plus_plus_baseline_outcome_advantage(
                token_level_rewards=token_level_rewards,
                response_mask=response_mask,
                index=index,
                epsilon=epsilon,
                config=config,
            )
        elif base_estimator == "opo":
            base_advantages, base_returns = compute_opo_outcome_advantage(
                token_level_rewards=token_level_rewards,
                response_mask=response_mask,
                index=index,
                epsilon=epsilon,
                config=config,
            )
        else:
            # Default to GRPO
            base_advantages, base_returns = compute_grpo_outcome_advantage(
                token_level_rewards=token_level_rewards,
                response_mask=response_mask,
                index=index,
                epsilon=epsilon,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            )

        # Step 2: Compute entropy-based weights based on confidence source
        confidence_source = config.get("scarl_confidence_source", "logit") if config else "logit"
        sign_aware_weighting = config.get("scarl_sign_aware_weighting", True) if config else True
        class_wise_normalization = config.get("scarl_class_wise_normalization", True) if config else True
        scarl_power = config.get("scarl_power", 1.0) if config else 1.0
        adaptive_tau = config.get("scarl_adaptive_tau", True) if config else True

        # Calculate sequence-level advantage for sign determination
        # adv_per_sequence: shape (bs,)
        # Use mean over sequence length as a robust aggregator
        adv_per_sequence = base_advantages.mean(dim=-1)

        e_s = None
        scarl_metrics = {}
        if confidence_source == "logit" and selected_logits is not None:
            # Use raw logits for absolute confidence (cross-sequence comparable)
            e_s, scarl_metrics = compute_scarl_entropy_weights_from_logits(
                selected_logits=selected_logits,
                response_mask=response_mask,
                index=index,
                tau=tau,
                lam=lam,
                adv_per_sequence=adv_per_sequence,
                sign_aware_weighting=sign_aware_weighting,
                class_wise_normalization=class_wise_normalization,
                power=scarl_power,
                adaptive_tau=adaptive_tau,
            )
        elif confidence_source == "prob" and "old_log_probs" in kwargs:
            # Use log probabilities for relative confidence (within-vocabulary normalized)
            e_s, scarl_metrics = compute_scarl_entropy_weights_from_probs(
                old_log_probs=kwargs["old_log_probs"],
                response_mask=response_mask,
                index=index,
                tau=tau,
                lam=lam,
                adv_per_sequence=adv_per_sequence,
                sign_aware_weighting=sign_aware_weighting,
                class_wise_normalization=class_wise_normalization,
                power=scarl_power,
                adaptive_tau=adaptive_tau,
            )

        if e_s is not None:
            # Step 3: Apply entropy weights to advantages
            # Expand sequence-level weights to token-level
            token_weights = e_s.unsqueeze(-1).expand_as(base_advantages)  # (bs, response_length)

            # Modulate advantages with entropy weights
            weighted_advantages = token_weights * base_advantages
        else:
            # Fallback: use base advantages without entropy weighting
            weighted_advantages = base_advantages

        return weighted_advantages, weighted_advantages, scarl_metrics


# =============================================================================
# SEED-GRPO: Semantic Entropy Enhanced GRPO
# =============================================================================
# Faithful port of https://github.com/AndyLiao404/SEED-GRPO (seed_grpo.py).
# Original author Mingjia Wang et al., 2025.
#
# Algorithm:
#   1. For each prompt group (uid), cluster the K rollout responses by their
#      extracted final answer (string equality on \boxed{} content).
#   2. Aggregate per-response log-likelihood (mean over response tokens) into
#      cluster-level log-likelihoods via logsumexp normalised by the total
#      logsumexp ("sum_normalized" mode in the source).
#   3. Compute Rao-style predictive entropy `pe = -sum(exp(lp) * lp)` over the
#      cluster log-probabilities. This is the prompt-level Semantic Entropy.
#   4. Modulate the (Dr.GRPO) advantage:
#         A_modulated = A * (1 / (1 + alpha * SE))
#      with `alpha = 0.0417 / log(num_samples)`.
# =============================================================================


def _seed_grpo_get_semantic_ids_by_answers(answers_list):
    """Group responses by string equality of extracted answers.

    Faithful port of `get_semantic_ids_by_answers` in seed_grpo.py:286-304.
    Responses whose answer is None / "NO_ANSWER_FOUND" each get a unique
    cluster id so that they maximise prompt-level entropy.
    """
    NO_ANSWER = "NO_ANSWER_FOUND"
    semantic_set_ids = {}
    semantic_ids = [-1] * len(answers_list)
    next_id = 0
    for idx, ans in enumerate(answers_list):
        if ans is None or ans == NO_ANSWER:
            semantic_ids[idx] = next_id
            next_id += 1
            continue
        if ans not in semantic_set_ids:
            semantic_set_ids[ans] = next_id
            next_id += 1
        semantic_ids[idx] = semantic_set_ids[ans]
    assert all(i != -1 for i in semantic_ids)
    return semantic_ids


def _seed_grpo_predictive_entropy_rao(log_probs):
    """`-sum(exp(lp) * lp)` over a cluster log-prob distribution.

    Faithful port of `predictive_entropy_rao` in seed_grpo.py:232-234.
    """
    log_probs = np.asarray(log_probs, dtype=np.float64)
    return float(-np.sum(np.exp(log_probs) * log_probs))


def _seed_grpo_logsumexp_by_id_sum_normalized(semantic_ids, log_likelihoods):
    """Aggregate log-likelihoods per cluster using `sum_normalized` mode.

    Faithful port of `logsumexp_by_id` (agg='sum_normalized') in
    seed_grpo.py:259-281: for each cluster, compute logsumexp of member
    log-likelihoods, then subtract the global logsumexp so the cluster
    log-probabilities sum (in log-space exp) to 1.
    """
    from scipy.special import logsumexp

    unique_ids = sorted(set(semantic_ids))
    log_likelihood_per_id = []
    for uid_ in unique_ids:
        members = [log_likelihoods[i] for i, sid in enumerate(semantic_ids) if sid == uid_]
        log_likelihood_per_id.append(logsumexp(members))
    log_likelihood_per_id = np.asarray(log_likelihood_per_id, dtype=np.float64)
    log_likelihood_per_id = log_likelihood_per_id - logsumexp(log_likelihood_per_id)
    return log_likelihood_per_id.tolist()


def compute_seed_grpo_semantic_entropies(
    index: np.ndarray,
    answers,
    log_lik_agg: torch.Tensor,
):
    """Compute prompt-level semantic entropy and broadcast to per-sample.

    Args:
        index: (bs,) group ids (uid).
        answers: list/array length bs; per-response extracted answer string
            (or None / "NO_ANSWER_FOUND" when extraction failed).
        log_lik_agg: (bs,) per-response aggregated log-likelihood (mean over
            response tokens of `old_log_probs` in the SEED-GRPO source).

    Returns:
        torch.Tensor of shape (bs,) — the prompt's SE replicated across its
        group members (matches the source which uses one SE per prompt
        broadcast across the K rollouts).
    """
    bs = len(index)
    log_lik_agg_np = log_lik_agg.detach().cpu().to(torch.float64).numpy()

    groups = defaultdict(list)
    for i in range(bs):
        groups[index[i]].append(i)

    se_per_sample = np.zeros(bs, dtype=np.float64)
    for group_idx in groups.values():
        group_answers = [answers[i] for i in group_idx]
        group_log_liks = [log_lik_agg_np[i] for i in group_idx]
        semantic_ids = _seed_grpo_get_semantic_ids_by_answers(group_answers)
        cluster_log_probs = _seed_grpo_logsumexp_by_id_sum_normalized(semantic_ids, group_log_liks)
        se = _seed_grpo_predictive_entropy_rao(cluster_log_probs)
        for i in group_idx:
            se_per_sample[i] = se

    return torch.tensor(se_per_sample, dtype=torch.float32)


@register_adv_est(AdvantageEstimator.SEED_GRPO)
def compute_seed_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    semantic_entropies: torch.Tensor = None,
    num_samples: int = None,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = False,
    config=None,
    **kwargs,
):
    """SEED-GRPO outcome advantage.

    Computes a Dr.GRPO-style baseline advantage and modulates it by a
    weighting function f(SE) of the prompt-level semantic entropy.

    Two layers of options are exposed:

    1. **Alpha schedule** (`seed_grpo_alpha_mode`):
       - "source"  (default): faithful to the released source
         (seed_grpo.py:793-799). `alpha = alpha_numerator / log(K)` with
         `alpha_numerator = 0.0417`. At K=8 this yields alpha ~ 0.02006,
         matching the paper's reported best alpha=0.02.
       - "direct": use `seed_grpo_alpha_direct` as alpha verbatim,
         independent of K. Use this to faithfully reproduce the paper's
         ablation (Table 4(b)) at any K.

    2. **Weighting function** (`seed_grpo_weight_function`):
       Let s_tilde = SE / log(K) in [0, 1] (paper Appendix D, Eq. 9).
       - "inverse"     (default, source-faithful):
                       f(SE) = 1 / (1 + alpha * SE), uses RAW SE.
                       With alpha_mode=source this equals
                       1 / (1 + 0.0417 * s_tilde), so the source implicitly
                       normalises SE through its alpha schedule.
       - "linear"      (paper Appendix D Eq. 10, reported best in Table 4(c)):
                       f(s_tilde) = 1 - alpha * s_tilde.
       - "exponential" (paper Appendix D Eq. 11):
                       f(s_tilde) = exp(-alpha * s_tilde).
       - "focal"       (paper Appendix D Eq. 12):
                       f(s_tilde) = (1 - s_tilde) ** gamma.
                       NOTE: alpha is NOT used for focal; the paper fixes
                       gamma = 2.0 (configurable via seed_grpo_focal_gamma).

    Defaults reproduce the released source code exactly. Switch
    weight_function="linear" + alpha_mode="direct" + alpha_direct=0.02 to
    follow the paper's reported best configuration (OpenReview v).

    Args:
        token_level_rewards: (bs, response_length).
        response_mask: (bs, response_length).
        index: (bs,) group ids.
        semantic_entropies: (bs,) per-sample semantic entropy (broadcasted
            from the prompt-level SE).
        num_samples: K, the number of rollouts per prompt. Used in the
            alpha schedule and SE normalisation. Falls back to group sizes
            inferred from `index`.
        config: optional algorithm config, may carry:
            - seed_grpo_alpha_mode: "source" | "direct".
            - seed_grpo_alpha_numerator: override 0.0417 (source mode).
            - seed_grpo_alpha_direct: alpha for direct mode (default 0.02).
            - seed_grpo_weight_function: "inverse"|"linear"|"exponential"|"focal".
            - seed_grpo_focal_gamma: gamma for focal (default 2.0, paper).
            - norm_adv_by_std_in_grpo: override Dr.GRPO default (False).

    Returns:
        (advantages, returns, metrics)
    """
    if config is not None:
        norm_adv_by_std_in_grpo = config.get("norm_adv_by_std_in_grpo", norm_adv_by_std_in_grpo)
        alpha_mode = config.get("seed_grpo_alpha_mode", "source")
        alpha_numerator = config.get("seed_grpo_alpha_numerator", 0.0417)
        alpha_direct = config.get("seed_grpo_alpha_direct", 0.02)
        weight_function = config.get("seed_grpo_weight_function", "inverse")
        focal_gamma = config.get("seed_grpo_focal_gamma", 2.0)
    else:
        alpha_mode = "source"
        alpha_numerator = 0.0417
        alpha_direct = 0.02
        weight_function = "inverse"
        focal_gamma = 2.0

    base_advantages, base_returns = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        epsilon=epsilon,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
    )

    metrics = {}
    if semantic_entropies is None:
        # No SE available — degrade to base GRPO/Dr.GRPO.
        return base_advantages, base_returns, metrics

    if num_samples is None or num_samples <= 1:
        # Infer K as the (max) group size; matches how the source picks K.
        from collections import Counter

        counts = Counter(index.tolist() if hasattr(index, "tolist") else list(index))
        num_samples = max(counts.values()) if counts else 1
    num_samples = max(int(num_samples), 2)  # log(1)=0; SEED-GRPO requires K>=2

    max_possible_entropy = float(np.log(num_samples))
    if alpha_mode == "direct":
        alpha = float(alpha_direct)
    elif alpha_mode == "source":
        alpha = alpha_numerator / max_possible_entropy
    else:
        raise ValueError(f"Unknown seed_grpo_alpha_mode: {alpha_mode}. Use 'source' or 'direct'.")

    se = semantic_entropies.to(base_advantages.device, dtype=base_advantages.dtype)
    se_b = se.unsqueeze(-1)  # (bs, 1) → broadcast to (bs, response_length)
    # Normalised semantic entropy s_tilde = SE / log(K), clamped to [0, 1].
    # Used by the paper's Linear/Exponential/Focal forms (Appendix D, Eq. 9).
    se_norm = torch.clamp(se_b / max_possible_entropy, min=0.0, max=1.0)

    if weight_function == "inverse":
        # Source-faithful: f(SE) = 1 / (1 + alpha * SE), uses RAW SE.
        # Note: with alpha_mode=source, alpha=0.0417/log(K), so this equals
        # 1 / (1 + 0.0417 * s_tilde), i.e. the source implicitly normalises.
        modulator = 1.0 / (1.0 + alpha * se_b)
    elif weight_function == "linear":
        # Paper Appendix D Eq. 10: f(s_tilde) = 1 - alpha * s_tilde.
        # Clamp to >= 0 to avoid sign flips at large alpha * s_tilde.
        modulator = torch.clamp(1.0 - alpha * se_norm, min=0.0)
    elif weight_function == "exponential":
        # Paper Appendix D Eq. 11: f(s_tilde) = exp(-alpha * s_tilde).
        modulator = torch.exp(-alpha * se_norm)
    elif weight_function == "focal":
        # Paper Appendix D Eq. 12: f(s_tilde) = (1 - s_tilde) ** gamma.
        # Note: alpha is NOT used here; the paper fixes gamma = 2.0.
        modulator = torch.pow(1.0 - se_norm, float(focal_gamma))
    else:
        raise ValueError(
            f"Unknown seed_grpo_weight_function: {weight_function}. "
            "Use 'inverse', 'linear', 'exponential', or 'focal'."
        )

    advantages = base_advantages * modulator

    metrics["seed_grpo/alpha"] = float(alpha)
    metrics["seed_grpo/max_possible_entropy"] = max_possible_entropy
    metrics["seed_grpo/semantic_entropy_mean"] = float(se.mean().item())
    metrics["seed_grpo/semantic_entropy_max"] = float(se.max().item())
    metrics["seed_grpo/se_norm_mean"] = float(se_norm.mean().item())
    metrics["seed_grpo/modulator_mean"] = float(modulator.mean().item())

    return advantages, advantages, metrics


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def apply_entropy_advantage_shaping(
    advantages: torch.Tensor,
    entropy: torch.Tensor,
    alpha: float,
    kappa: float,
) -> torch.Tensor:
    """Entropy-Based Advantage Shaping (Cheng et al., AAAI 2026, arXiv:2506.14758).

    A_shaped_t = A_t + psi(H_t),  where  psi(H_t) = min(alpha * H_t.detach(), |A_t| / kappa).

    The detached entropy term acts as a fixed offset, and the |A|/kappa clip prevents the
    shaping term from dominating or flipping the sign of the original advantage. Because
    the operation only modifies the advantage tensor and is independent of the policy loss
    formulation, it is orthogonal to PPO/GRPO/GSPO/SAPO/etc. (paper Listing 1).

    Args:
        advantages: shape (bsz, response_length).
        entropy: token-level policy entropy from current policy logits, shape
            (bsz, response_length). Must be the entropy H_t = -sum_v pi(v|.) log pi(v|.).
        alpha: scaling coefficient (paper: 0.4 for GRPO, 0.1 for PPO).
        kappa: clipping threshold, kappa > 1 (paper: 2).

    Returns:
        shaped advantages, same shape as input.
    """
    if alpha == 0.0:
        return advantages
    psi = torch.min(alpha * entropy.detach(), advantages.abs() / kappa)
    return advantages + psi


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str, **kwargs):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
        **kwargs: Additional arguments for distributed training optimization (e.g., dp_size,
            batch_num_tokens, global_batch_size, loss_scale_factor). These are accepted for
            compatibility with v0.7.0+ but not used in this simplified implementation.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


@register_policy_loss("gspo")
def compute_policy_loss_gspo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config=None,
    rollout_is_weights: torch.Tensor | None = None,
):
    """
    Compute the clipped policy objective and related metrics for GSPO.

    See https://arxiv.org/pdf/2507.18071 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For GSPO, it is recommended to use "seq-mean-token-mean".
        config: Actor configuration (DictConfig or dataclass with .get() support).

    Returns:
        pg_loss: Policy gradient loss.
        pg_clipfrac: Fraction of clipped policy gradients.
        ppo_kl: KL divergence between old and new policy.
        pg_clipfrac_lower: Lower clip fraction (always 0.0 for GSPO).
    """

    assert config is not None
    # Compatible with both DictConfig and dataclass config
    clip_ratio = (
        config.get("clip_ratio", 0.2) if hasattr(config, "get") else getattr(config, "clip_ratio", 0.2)
    )
    clip_ratio_low = (
        config.get("clip_ratio_low", None)
        if hasattr(config, "get")
        else getattr(config, "clip_ratio_low", None)
    )
    clip_ratio_high = (
        config.get("clip_ratio_high", None)
        if hasattr(config, "get")
        else getattr(config, "clip_ratio_high", None)
    )
    clip_ratio_low = clip_ratio_low if clip_ratio_low is not None else clip_ratio
    clip_ratio_high = clip_ratio_high if clip_ratio_high is not None else clip_ratio

    negative_approx_kl = log_prob - old_log_prob

    # compute sequence-level importance ratio:
    # si(θ) = (π_θ(yi|x)/π_θold(yi|x))^(1/|yi|) =
    # exp [(1/|y_i|) * Σ_t log(π_θ(y_i,t|x,y_i,<t)/π_θold(y_i,t|x,y_i,<t))]
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

    # Combined ratio at token level:
    # s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
    # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_prob - sg[log_prob]
    log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    log_seq_importance_ratio = torch.clamp(
        log_seq_importance_ratio, max=10.0
    )  # clamp for numerical stability

    # finaly exp() to remove log
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # for GSPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode="seq-mean-token-mean",
    )

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard GSPO)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Return 4 values to match scarl's dp_actor.py calling convention
    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


@register_policy_loss("sapo")
def compute_policy_loss_sapo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config=None,
    rollout_is_weights: torch.Tensor | None = None,
):
    """
    Compute the smoothed policy objective and related metrics for SAPO.

    See https://arxiv.org/pdf/2511.20347 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For SAPO, it is recommended to use "seq-mean-token-mean".
        config: Actor configuration (DictConfig or dataclass with .get() support).
            - tau_pos: temperature for positive advantage tokens (default: 1.0)
            - tau_neg: temperature for negative advantage tokens (default: 1.05)

    Returns:
        pg_loss: Policy gradient loss.
        pg_clipfrac: Fraction of clipped policy gradients (always 0.0 for SAPO).
        ppo_kl: KL divergence between old and new policy.
        pg_clipfrac_lower: Lower clip fraction (always 0.0 for SAPO).
    """

    assert config is not None
    # Compatible with both DictConfig and dataclass config
    # Get SAPO-specific parameters with defaults matching the paper
    tau_pos_val = config.get("tau_pos", 1.0) if hasattr(config, "get") else getattr(config, "tau_pos", 1.0)
    tau_neg_val = config.get("tau_neg", 1.05) if hasattr(config, "get") else getattr(config, "tau_neg", 1.05)

    # temperature for positive and negative token updates
    tau_pos = torch.as_tensor(tau_pos_val, dtype=advantages.dtype, device=advantages.device)
    tau_neg = torch.as_tensor(tau_neg_val, dtype=advantages.dtype, device=advantages.device)

    def gate_function(x, tau):
        """The gating function used in SAPO"""
        return torch.sigmoid(tau * (x - 1.0)) * (4.0 / tau)

    # compute IS at token level:
    # r_{i,t}(θ) = π_θ(y_{i,t}|x, y_{i,<t}) / π_θold(y_{i,t}|x, y_{i,<t})]
    # In log space: log(r_{i,t}(θ)) = log_prob - ol_log_prob
    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    # finally exp() to remove log and get r_{i,t}(θ)
    ratio = torch.exp(negative_approx_kl)

    # tau_{i,t} is tau_pos if adv > 0 else tau_neg
    taus = torch.where(
        condition=advantages > 0,
        input=tau_pos,  # if A_{i,t} > 0 we set to tau_pos
        other=tau_neg,  # if A_{i,t} <= 0 we set to tau_neg
    )

    # compute the gates f_{i,t}(r_{i,t}(θ)) at token level
    gates = gate_function(ratio, taus)

    # compute policy gradient loss
    pg_losses = -gates * advantages

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # for SAPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode="seq-mean-token-mean",
    )

    # For compatibility, return zero for both pg_clipfrac and pg_clipfrac_lower (not used in SAPO)
    pg_clipfrac = torch.tensor(0.0, device=pg_loss.device)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)
    # compute KL for metrics tracking
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Return 4 values to match scarl's dp_actor.py calling convention
    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


@register_policy_loss("clip_cov")
def compute_policy_loss_clip_cov(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    loss_agg_mode="token-mean",
    config=None,
):
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        clip_cvo_ratio (float, optional):
            Ratio for clipping the covariance. Defaults to 0.0002.
        clip_cov_lb (float, optional):
            Lower bound for clipping covariance. Defaults to 1.0.
        clip_cov_ub (float, optional):
            Upper bound for clipping covariance. Defaults to 5.0.
    """
    clip_cov_ratio = (
        config.policy_loss.clip_cov_ratio if config.policy_loss.clip_cov_ratio is not None else 0.0002
    )
    cliprange = config.clip_ratio
    cliprange_low = config.clip_ratio_low if config.clip_ratio_low is not None else cliprange
    cliprange_high = config.clip_ratio_high if config.clip_ratio_high is not None else cliprange
    clip_cov_ub = config.policy_loss.clip_cov_ub if config.policy_loss.clip_cov_ub is not None else 5.0
    clip_cov_lb = config.policy_loss.clip_cov_lb if config.policy_loss.clip_cov_lb is not None else 1.0

    assert clip_cov_ratio > 0, "clip_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio

    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    corr = torch.ones_like(advantages)
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_by_origin = (pg_losses2 > pg_losses1) & (response_mask > 0)

    cov_all = (advantages - verl_F.masked_mean(advantages, response_mask)) * (
        log_prob - verl_F.masked_mean(log_prob.detach(), response_mask)
    )
    cov_all[response_mask == 0] = -torch.inf
    cov_all[clip_by_origin] = -torch.inf

    clip_num = max(int(clip_cov_ratio * response_mask.sum().item()), 1)
    top_k_idx = (cov_all < clip_cov_ub) & (cov_all > clip_cov_lb) & (response_mask > 0)
    top_k_idx = torch.nonzero(top_k_idx)

    if len(top_k_idx) > 0:
        perm = torch.randperm(len(top_k_idx))
        top_k_idx = top_k_idx[perm[: min(clip_num, len(top_k_idx))]]
    else:
        top_k_idx = torch.empty((0, 2), device=cov_all.device, dtype=torch.long)

    corr[top_k_idx[:, 0], top_k_idx[:, 1]] = 0

    pg_clipfrac = verl_F.masked_mean((corr == 0).float(), response_mask)

    pg_losses = torch.maximum(pg_losses1, pg_losses2) * corr
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, torch.tensor(0.0)


@register_policy_loss("kl_cov")
def compute_policy_loss_kl_cov(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    loss_agg_mode="token-mean",
    config=None,
):
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        kl_cov_ratio (float, optional):
            Ratio for selecting the top-k covariance values. Defaults to 0.0002.
        ppo_kl_coef (float, optional):
            Coefficient for the KL penalty term in the loss. Defaults to 1.
    """
    kl_cov_ratio = config.policy_loss.kl_cov_ratio if config.policy_loss.kl_cov_ratio is not None else 0.0002
    ppo_kl_coef = config.policy_loss.ppo_kl_coef if config.policy_loss.ppo_kl_coef is not None else 1.0

    assert kl_cov_ratio > 0, "kl_cov_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    abs_kl = negative_approx_kl.abs()
    ratio = torch.exp(negative_approx_kl)
    ppo_kl_abs = verl_F.masked_mean(negative_approx_kl.abs(), response_mask)
    pg_losses1 = -advantages * ratio
    pg_losses_kl = -advantages * ratio + ppo_kl_coef * abs_kl
    pg_losses = pg_losses1

    all_valid = response_mask > 0
    all_valid_idx = torch.nonzero(all_valid.reshape(-1), as_tuple=True)[0]
    all_valid_adv = advantages[all_valid].detach().reshape(-1).cpu()
    all_valid_logp = log_prob[all_valid].detach().reshape(-1).cpu()

    k = min(kl_cov_ratio, len(all_valid_adv))

    if k != 0:
        cov_lst_all = (all_valid_adv - all_valid_adv.mean()) * (all_valid_logp - all_valid_logp.mean())
        k_percent_nums = max(1, int(len(cov_lst_all) * kl_cov_ratio))
        large_cov_idxs = torch.topk(cov_lst_all, k_percent_nums, largest=True).indices

        if len(large_cov_idxs) != 0:
            large_cov_idxs = all_valid_idx[large_cov_idxs]
            pg_losses[
                large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]
            ] = pg_losses_kl[large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]]

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, torch.tensor(0.0), ppo_kl_abs, torch.tensor(0.0)


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = 0.5 * agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data


# =====================================================================
# CDE baseline: actor-side perplexity (PPL) bonus.
#
# Dai et al., "CDE: Curiosity-Driven Exploration for Efficient
# Reinforcement Learning in Large Language Models", ICLR 2026
# (arXiv:2509.09675). Only the actor-side bonus applies to critic-free
# estimators; the multi-head critic bonus is PPO-specific and is not
# ported. Dedicated code path: the shared estimators above are untouched
# and are only called.
#
#   B(q, o)       = -(1/T) sum_t log pi(o_t | o_<t, q)                (Eq. 1)
#   r~(q, o)      = r(q, o) + omega_t * min(|r(q, o)| / kappa, alpha * B)   (Eq. 2)
#
# The shaped reward r~ then goes through the unmodified GRPO group
# normalization. Paper configuration for the PPL bonus (Table 4b):
# kappa = 3, alpha = 1, omega_t = staircase decay (Figure 7: omega = 1
# for the first half of training, 0 afterwards). B is computed from the
# behavior-policy log-probs already materialized for the importance
# ratios (`old_log_probs`), the same source as SCARL's confidence.
# =====================================================================


def cde_bonus_weight(
    step_fraction: float,
    omega: float = 1.0,
    schedule: str = "staircase",
    decay_fraction: float = 0.5,
) -> float:
    """Bonus weight omega_t of CDE's PPL bonus at a point of training.

    Args:
        step_fraction: fraction of training completed before the current
            step, in [0, 1).
        omega: initial bonus weight (1.0 in the paper's Figure 7).
        schedule: "staircase" (paper's recommended setting), "linear",
            "cosine", or "none" (the paper's "No decay").
        decay_fraction: fraction of training over which omega decays;
            Figure 7 decays over the first half (step 150 of 300).
    """
    schedule = (schedule or "staircase").lower()
    if schedule in ("none", "constant", "no_decay"):
        return float(omega)
    if decay_fraction <= 0:
        return 0.0
    progress = min(max(step_fraction / decay_fraction, 0.0), 1.0)
    if schedule == "staircase":
        return float(omega) if progress < 1.0 else 0.0
    if schedule == "linear":
        return float(omega) * (1.0 - progress)
    if schedule == "cosine":
        return float(omega) * 0.5 * (1.0 + float(np.cos(np.pi * progress)))
    raise ValueError(f"Unknown CDE omega schedule: {schedule!r}")


def compute_cde_ppl_bonus(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    old_log_probs: torch.Tensor,
    omega_t: float,
    kappa: float = 3.0,
    alpha: float = 1.0,
):
    """Per-response PPL bonus of CDE (Eqs. 1-2).

    Returns:
        bonus: `(bs,)` omega_t * min(|r| / kappa, alpha * B)
        ppl:   `(bs,)` B, the mean negative token log-probability of the
               response under the behavior policy (log-perplexity)
        cap:   `(bs,)` |r| / kappa
    """
    scores = token_level_rewards.sum(dim=-1)
    mask = response_mask.to(old_log_probs.dtype)
    n_tokens = mask.sum(dim=-1).clamp(min=1.0)
    ppl = -(old_log_probs * mask).sum(dim=-1) / n_tokens
    cap = scores.abs() / kappa
    bonus = omega_t * torch.minimum(cap, alpha * ppl)
    return bonus, ppl, cap


@register_adv_est(AdvantageEstimator.CDE)
def compute_cde_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    old_log_probs: torch.Tensor,
    step_fraction: float = 0.0,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
    **kwargs,
):
    """GRPO outcome advantage on CDE's bonus-shaped reward.

    The bonus is added to each response's outcome reward at its last
    valid token, and the shaped rewards are passed through the unmodified
    `compute_grpo_outcome_advantage`, so the group normalization is
    exactly GRPO's.

    Config keys (read with the paper's values as defaults):
        cde_kappa (3.0), cde_alpha (1.0), cde_omega (1.0),
        cde_omega_schedule ("staircase"), cde_omega_decay_fraction (0.5).

    Returns:
        advantages, returns: `(bs, response_length)`
        metrics: dict of "cde/..." diagnostics
    """
    get = (lambda k, d: config.get(k, d)) if config is not None else (lambda k, d: d)
    kappa = float(get("cde_kappa", 3.0))
    alpha = float(get("cde_alpha", 1.0))
    omega = float(get("cde_omega", 1.0))
    schedule = str(get("cde_omega_schedule", "staircase"))
    decay_fraction = float(get("cde_omega_decay_fraction", 0.5))
    omega_t = cde_bonus_weight(step_fraction, omega=omega, schedule=schedule, decay_fraction=decay_fraction)

    with torch.no_grad():
        bonus, ppl, cap = compute_cde_ppl_bonus(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            old_log_probs=old_log_probs,
            omega_t=omega_t,
            kappa=kappa,
            alpha=alpha,
        )
        shaped_rewards = token_level_rewards.clone()
        last_valid = (response_mask.sum(dim=-1).long() - 1).clamp(min=0)
        rows = torch.arange(shaped_rewards.shape[0], device=shaped_rewards.device)
        shaped_rewards[rows, last_valid] += bonus.to(shaped_rewards.dtype)

        advantages, returns = compute_grpo_outcome_advantage(
            token_level_rewards=shaped_rewards,
            response_mask=response_mask,
            index=index,
            epsilon=epsilon,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )

        scores = token_level_rewards.sum(dim=-1)
        pos = scores > 0
        neg = ~pos

        def _mean(x, m):
            return x[m].mean().item() if bool(m.any()) else 0.0

        metrics = {
            "cde/omega_t": omega_t,
            "cde/bonus_mean": bonus.mean().item(),
            "cde/bonus_mean_pos": _mean(bonus, pos),
            "cde/bonus_mean_neg": _mean(bonus, neg),
            "cde/ppl_mean": ppl.mean().item(),
            "cde/ppl_mean_pos": _mean(ppl, pos),
            "cde/ppl_mean_neg": _mean(ppl, neg),
            "cde/cap_active_frac": (cap < alpha * ppl).float().mean().item(),
        }

    return advantages, returns, metrics
