"""
finetune.py

Fine-tunes OpenVLA via LoRA.
"""

import hashlib
import json
import math
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Type

import draccus
import numpy as np
import tensorflow as tf
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
from accelerate import PartialState
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb

STOP_PROGRESS_HORIZONS = (0, 1, 2, 4)

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map,
)

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import DiffusionActionHead, GaussianActionHead, L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.uav_condition_adapter import IndoorUAVConditionAdapter
from prismatic.models.uav_progress_stop_head import IndoorUAVProgressStopHead
from prismatic.models.uav_stop_head import IndoorUAVStopHead
from prismatic.models.projectors import (
    NoisyActionProjector,
    ProprioProjector,
)
from prismatic.training.train_utils import (
    compute_actions_l1_loss,
    compute_token_accuracy,
    get_current_action_mask,
    get_next_actions_mask,
)
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.condition_matching import (
    CrossEpisodeImageQueue,
    center_condition_branches,
    center_visual_patches,
    condition_branch_contrastive_loss,
    condition_to_patch_similarity,
    condition_to_image_logits,
    projected_condition_to_patch_similarity,
)
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
    get_act_token,
    get_cond_action_tokens,
    get_cond_token,
)
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _shape(value) -> str:
    if value is None:
        return "None"
    if hasattr(value, "shape"):
        return str(tuple(value.shape))
    return type(value).__name__


def _device(value) -> str:
    if value is None:
        return "None"
    if hasattr(value, "device"):
        return str(value.device)
    return "n/a"


def _set_torch_seed(seed: int, label: str) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Set {label} torch seed to {seed}")


def _module_grad_norm(module: Optional[nn.Module]) -> Optional[float]:
    if module is None:
        return None

    total_sq_norm = 0.0
    has_grad = False
    for param in module.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        total_sq_norm += grad.norm(2).item() ** 2
        has_grad = True

    if not has_grad:
        return None
    return total_sq_norm ** 0.5


def _print_dataset_statistics(dataset_statistics: dict) -> None:
    print("Dataset statistics summary:")
    for dataset_name, stats in dataset_statistics.items():
        action_stats = stats.get("action", {})
        proprio_stats = stats.get("proprio", {})
        print(f"  dataset: {dataset_name}")
        if "mean" in action_stats:
            print(f"    action_dim: {len(action_stats['mean'])}")
        if "representation" in action_stats:
            print(f"    action_representation: {action_stats['representation']}")
        if "stride" in action_stats:
            print(f"    future_action_stride: {action_stats['stride']}")
        if "yaw_delta_wrapped" in action_stats:
            print(f"    relative_action_wrap_yaw: {action_stats['yaw_delta_wrapped']}")
        if "mean" in proprio_stats:
            print(f"    proprio_dim: {len(proprio_stats['mean'])}")
        for key in ("num_trajectories", "num_transitions"):
            if key in stats:
                print(f"    {key}: {stats[key]}")


def _get_action_norm_stats(dataset_statistics: dict, dataset_name: str) -> Optional[dict]:
    if not dataset_statistics:
        return None
    if dataset_name in dataset_statistics:
        return dataset_statistics[dataset_name].get("action")
    if len(dataset_statistics) == 1:
        return next(iter(dataset_statistics.values())).get("action")
    return None


def _get_proprio_norm_stats(dataset_statistics: dict, dataset_name: str) -> Optional[dict]:
    if not dataset_statistics:
        return None
    if dataset_name in dataset_statistics:
        return dataset_statistics[dataset_name].get("proprio")
    if len(dataset_statistics) == 1:
        return next(iter(dataset_statistics.values())).get("proprio")
    return None


def get_model_proprio_dim(cfg) -> int:
    return 5 if cfg.cyclic_yaw_proprio else PROPRIO_DIM


def save_policy_contract(cfg, dataset_statistics: dict, output_dir: Path) -> None:
    """Persist the data/model/execution contract required by Stage20 inference."""
    if not cfg.use_indoor_uav_condition_adapter:
        return
    action_stats = _get_action_norm_stats(dataset_statistics, cfg.dataset_name)
    proprio_stats = _get_proprio_norm_stats(dataset_statistics, cfg.dataset_name)
    normalization_values = np.concatenate(
        [
            np.asarray(stats[key], dtype=np.float32).reshape(-1)
            for stats in (action_stats, proprio_stats)
            for key in ("normalization_low", "normalization_high")
        ]
    )
    contract = {
        "schema_version": (
            4 if cfg.use_indoor_uav_progress_stop_head else (3 if cfg.use_indoor_uav_stop_head else 2)
        ),
        "training_objective": "sft",
        "initial_vla_path": cfg.vla_path,
        "uav_modules_initialized_fresh": not cfg.resume and cfg.auxiliary_init_checkpoint_path is None,
        "input_roles": ["reference", "previous", "current"],
        "image_valid_mask": "1 means the image and all of its patches may attend",
        "horizon": NUM_ACTIONS_CHUNK,
        "num_action_branches": cfg.num_action_branches,
        "action_dim": ACTION_DIM,
        "source_proprio_dim": PROPRIO_DIM,
        "model_proprio_dim": get_model_proprio_dim(cfg),
        "proprio_representation": "xyz_sin_yaw_cos_yaw_v1",
        "condition_match_dim": cfg.condition_match_dim,
        "condition_patch_topk": cfg.condition_patch_topk,
        "condition_branch_assignment": "action_error_only",
        "condition_objective": "per_time_k_way_plus_temporal_plus_cross_episode_queue",
        "condition_branch_weight": cfg.condition_contrastive_weight,
        "condition_temporal_weight": cfg.condition_temporal_weight,
        "condition_queue_weight": cfg.condition_queue_weight,
        "condition_queue_size": cfg.condition_queue_size,
        "condition_queue_min_negatives": cfg.condition_queue_min_negatives,
        "condition_alignment": "slot j condition is image I_(t+j), j>=1",
        "action_alignment": "slot j action maps pose p_(t+j) to p_(t+j+1)",
        "action_representation": "body_delta_one_step_v1",
        "action_axes": ["forward", "right", "up", "yaw"],
        "yaw_zero_forward_world": [0.0, -1.0, 0.0],
        "yaw_zero_right_world": [1.0, 0.0, 0.0],
        "positive_yaw_turn": "right",
        "negative_yaw_turn": "left",
        "yaw_unit": "radian",
        "yaw_delta_formula": "((yaw_next-yaw_current+pi) mod (2*pi))-pi",
        "yaw_delta_range": "[-pi, pi)",
        "future_action_stride": cfg.future_action_stride,
        "action_normalization": "per_axis_symmetric_minmax_v1",
        "proprio_normalization": "xyz_q01_q99_and_sincos_unit_bounds_v1",
        "normalization_sha256": hashlib.sha256(normalization_values.tobytes()).hexdigest(),
        "condition_threshold": None,
        "stop_after_action": cfg.use_indoor_uav_stop_head,
        "stop_target_semantics": (
            "execute root action, then terminate instruction"
            if cfg.use_indoor_uav_stop_head
            else None
        ),
        "stop_loss_weight": cfg.stop_loss_weight,
        "stop_positive_weight": cfg.stop_positive_weight,
        "stop_threshold": cfg.stop_threshold,
        "stop_head_type": (
            "act_cond_ordinal_progress_v1"
            if cfg.use_indoor_uav_progress_stop_head
            else ("root_act_binary_v1" if cfg.use_indoor_uav_stop_head else None)
        ),
        "stop_progress_horizons": (
            list(STOP_PROGRESS_HORIZONS) if cfg.use_indoor_uav_progress_stop_head else None
        ),
        "stop_progress_loss_weight": cfg.stop_progress_loss_weight,
        "stop_progress_positive_weights": getattr(cfg, "stop_progress_positive_weights", None),
        "launch_argv": sys.argv,
    }
    output_path = Path(output_dir) / "policy_contract.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(contract, handle, indent=2, ensure_ascii=False)


def _stats_tensor(values, device, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(values, device=device, dtype=dtype)


def add_cond_action_tokens(tokenizer, model, num_action_branches: int) -> None:
    tokens = get_cond_action_tokens(NUM_ACTIONS_CHUNK, num_action_branches)
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": tokens})
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    print(f"COND/ACT special tokens ready: {len(tokens)} tokens ({num_added} newly added)")


def get_cond_action_token_id_tensors(tokenizer, num_action_branches: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    cond_ids = []
    act_ids = []
    for time_idx in range(1, NUM_ACTIONS_CHUNK + 1):
        for branch_idx in range(1, num_action_branches + 1):
            cond_ids.append(tokenizer.convert_tokens_to_ids(get_cond_token(time_idx, branch_idx)))
            act_ids.append(tokenizer.convert_tokens_to_ids(get_act_token(time_idx, branch_idx)))
    return (
        torch.tensor(cond_ids, device=device, dtype=torch.long),
        torch.tensor(act_ids, device=device, dtype=torch.long),
    )


def gather_cond_action_hidden_states(
    text_hidden_states: torch.Tensor,
    shifted_input_ids: torch.Tensor,
    cond_token_ids: torch.Tensor,
    act_token_ids: torch.Tensor,
    num_action_branches: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    cond_mask = torch.isin(shifted_input_ids, cond_token_ids)
    act_mask = torch.isin(shifted_input_ids, act_token_ids)
    batch_size = shifted_input_ids.shape[0]
    expected_count = NUM_ACTIONS_CHUNK * num_action_branches
    cond_counts = cond_mask.sum(dim=1)
    act_counts = act_mask.sum(dim=1)
    if not torch.all(cond_counts == expected_count) or not torch.all(act_counts == expected_count):
        raise ValueError(
            "Incomplete COND/ACT token structure: "
            f"expected {expected_count}, cond_counts={cond_counts.tolist()}, act_counts={act_counts.tolist()}"
        )

    cond_hidden = text_hidden_states[cond_mask].reshape(
        batch_size, NUM_ACTIONS_CHUNK, num_action_branches, -1
    )
    act_hidden = text_hidden_states[act_mask].reshape(
        batch_size, NUM_ACTIONS_CHUNK, num_action_branches, -1
    )
    format_metrics = {
        "format_cond_token_count": cond_counts.float().mean().item(),
        "format_act_token_count": act_counts.float().mean().item(),
        "format_complete_rate": ((cond_counts == expected_count) & (act_counts == expected_count)).float().mean().item(),
    }
    return cond_hidden, act_hidden, format_metrics


def _unwrap_vla_model(vla):
    model = vla.module if hasattr(vla, "module") else vla
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def compute_best_of_k_action_loss(
    predicted_actions: torch.Tensor,
    ground_truth_actions: torch.Tensor,
    assignment_temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Assigns one action branch to each (batch, time) target."""
    if predicted_actions.ndim != 4:
        raise ValueError("best-of-K action loss requires shape (B, T, K, action_dim)")
    if assignment_temperature <= 0:
        raise ValueError("assignment_temperature must be > 0")

    targets = ground_truth_actions.unsqueeze(2).expand_as(predicted_actions)
    per_time_branch_l1 = torch.abs(predicted_actions.float() - targets.float()).mean(dim=-1)
    winner_indices = per_time_branch_l1.detach().argmin(dim=2)
    winner_l1 = per_time_branch_l1.gather(2, winner_indices.unsqueeze(2)).squeeze(2)
    best_of_k_loss = winner_l1.mean()

    assignment_probabilities = torch.softmax(-per_time_branch_l1 / assignment_temperature, dim=2)
    branch_usage = assignment_probabilities.mean(dim=(0, 1))
    uniform_usage = torch.full_like(branch_usage, 1.0 / predicted_actions.shape[2])
    branch_balance_loss = torch.sum(
        branch_usage * torch.log((branch_usage + 1e-8) / uniform_usage)
    )
    assignment_entropy = -torch.sum(
        assignment_probabilities * torch.log(assignment_probabilities + 1e-8), dim=2
    ).mean()

    metrics = {
        "best_of_k_action_loss": best_of_k_loss.item(),
        "branch_balance_loss": branch_balance_loss.item(),
        "branch_assignment_entropy": assignment_entropy.item(),
    }
    hard_assignments = torch.nn.functional.one_hot(
        winner_indices, num_classes=predicted_actions.shape[2]
    ).float()
    for branch_idx in range(predicted_actions.shape[2]):
        metrics[f"branch{branch_idx}_winner_rate"] = hard_assignments[..., branch_idx].mean().item()
        metrics[f"branch{branch_idx}_soft_usage"] = branch_usage[branch_idx].item()

    return best_of_k_loss, branch_balance_loss, winner_indices, metrics


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def _valid_future_episode_ids(episode_ids, future_mask: torch.Tensor) -> list[str]:
    """Expand one episode id per batch item to valid future-image labels."""
    if episode_ids is None or len(episode_ids) != future_mask.shape[0]:
        raise ValueError("cross-episode matching requires one episode_id per batch item")
    valid = future_mask.detach().cpu().tolist()
    output = []
    for episode_id, row_mask in zip(episode_ids, valid):
        if isinstance(episode_id, bytes):
            episode_id = episode_id.decode("utf-8")
        output.extend([str(episode_id)] * sum(row_mask))
    return output


def compute_cross_episode_queue_loss(
    conditions: torch.Tensor,
    positive_images: torch.Tensor,
    query_episode_ids: list[str],
    queued_images: Optional[torch.Tensor],
    queued_episode_ids: list[str],
    *,
    temperature: float,
    patch_topk: int,
    min_negatives: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Contrast each condition with detached images from other episodes only."""
    zero = conditions.sum() * 0.0
    empty_metrics = {
        "condition_queue_loss": 0.0,
        "condition_queue_accuracy": 0.0,
        "condition_queue_random_accuracy": 0.0,
        "condition_queue_margin": 0.0,
        "condition_queue_queries": 0.0,
        "condition_queue_negatives": 0.0,
    }
    if queued_images is None or conditions.shape[0] == 0:
        return zero, empty_metrics
    if len(query_episode_ids) != conditions.shape[0]:
        raise ValueError("one query episode id is required per condition")
    if len(queued_episode_ids) != queued_images.shape[0]:
        raise ValueError("queued episode ids and images must have equal length")
    if min_negatives < 1:
        raise ValueError("condition_queue_min_negatives must be >= 1")

    losses, accuracies, random_accuracies, margins, negative_counts = [], [], [], [], []
    for episode_id in dict.fromkeys(query_episode_ids):
        query_indices = [idx for idx, query_id in enumerate(query_episode_ids) if query_id == episode_id]
        eligible = [idx for idx, queued_id in enumerate(queued_episode_ids) if queued_id != episode_id]
        if len(eligible) < min_negatives:
            continue
        query_conditions = conditions[query_indices]
        positive_scores = condition_to_image_logits(
            query_conditions, positive_images[query_indices], patch_topk
        ).diagonal()
        negative_scores = condition_to_image_logits(
            query_conditions, queued_images[eligible], patch_topk
        )
        logits = torch.cat([positive_scores[:, None], negative_scores], dim=1) / temperature
        labels = logits.new_zeros(logits.shape[0], dtype=torch.long)
        losses.append(torch.nn.functional.cross_entropy(logits, labels, reduction="none"))
        accuracies.append((logits.argmax(dim=1) == 0).float())
        random_accuracies.extend([1.0 / logits.shape[1]] * len(query_indices))
        margins.append(positive_scores - negative_scores.max(dim=1).values)
        negative_counts.extend([float(len(eligible))] * len(query_indices))

    if not losses:
        return zero, empty_metrics
    losses = torch.cat(losses)
    accuracies = torch.cat(accuracies)
    margins = torch.cat(margins)
    loss = losses.mean()
    return loss, {
        "condition_queue_loss": loss.item(),
        "condition_queue_accuracy": accuracies.mean().item(),
        "condition_queue_random_accuracy": sum(random_accuracies) / len(random_accuracies),
        "condition_queue_margin": margins.mean().item(),
        "condition_queue_queries": float(losses.shape[0]),
        "condition_queue_negatives": sum(negative_counts) / len(negative_counts),
    }


def compute_masked_single_branch_sft_loss(
    predicted_actions: torch.Tensor,
    ground_truth_actions: torch.Tensor,
    plan_valid_mask: torch.Tensor,
    *,
    root_action_weight: float,
    future_action_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Stage20 K=1 action baseline with the same masking and loss geometry as Best-of-K SFT."""
    if predicted_actions.ndim == 4:
        if predicted_actions.shape[2] != 1:
            raise ValueError("single-branch SFT expects exactly one action branch")
        predicted_actions = predicted_actions.squeeze(2)
    if predicted_actions.shape != ground_truth_actions.shape:
        raise ValueError("single-branch predictions and targets must have shape (B,T,D)")
    if plan_valid_mask.shape != ground_truth_actions.shape[:2]:
        raise ValueError("plan_valid_mask must have shape (B,T)")

    action_errors = torch.nn.functional.smooth_l1_loss(
        predicted_actions.float(), ground_truth_actions.float(), reduction="none"
    ).mean(dim=-1)
    root_loss = _masked_mean(action_errors[:, 0], plan_valid_mask[:, 0])
    future_loss = _masked_mean(action_errors[:, 1:], plan_valid_mask[:, 1:])
    loss = root_action_weight * root_loss + future_action_weight * future_loss
    return loss, {
        "sft_root_action_loss": root_loss.item(),
        "sft_future_action_loss": future_loss.item(),
        "root_valid_count": plan_valid_mask[:, 0].sum().item(),
        "future_valid_count": plan_valid_mask[:, 1:].sum().item(),
        "plan_valid_ratio": plan_valid_mask.float().mean().item(),
    }


def compute_stop_after_action_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    positive_weight: float,
    threshold: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Balanced BCE for the post-root-action terminal decision."""
    logits = logits.float().reshape(-1)
    targets = targets.float().reshape(-1)
    if logits.shape != targets.shape:
        raise ValueError("stop logits and targets must have the same shape")
    if positive_weight <= 0:
        raise ValueError("positive_weight must be > 0")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must lie in (0,1)")

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=logits.new_tensor(positive_weight),
    )
    probabilities = torch.sigmoid(logits.detach())
    predicted = probabilities >= threshold
    positive = targets.bool()
    negative = ~positive
    return loss, {
        "stop_loss": loss.item(),
        "stop_probability_mean": probabilities.mean().item(),
        "stop_target_rate": targets.mean().item(),
        "stop_predicted_rate": predicted.float().mean().item(),
        "stop_true_positive": float((predicted & positive).sum().item()),
        "stop_true_negative": float(((~predicted) & negative).sum().item()),
        "stop_false_positive": float((predicted & negative).sum().item()),
        "stop_false_negative": float(((~predicted) & positive).sum().item()),
        "stop_sample_count": float(targets.numel()),
        "stop_positive_count": float(positive.sum().item()),
        "stop_negative_count": float(negative.sum().item()),
        # Validation consumes and removes these private fields before scalar
        # aggregation.  Keeping the individual scores is necessary for exact
        # threshold calibration and ROC-AUC rather than guessing from a mean.
        "_stop_probabilities": probabilities.cpu().tolist(),
        "_stop_targets": targets.detach().cpu().tolist(),
    }


def compute_binary_score_diagnostics(probabilities, targets) -> Dict[str, float]:
    """Exact binary ranking and best-balanced-threshold diagnostics."""
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    if probabilities.shape != targets.shape or probabilities.size == 0:
        raise ValueError("probabilities and targets must be equally sized non-empty arrays")
    if not np.isin(targets, [0, 1]).all():
        raise ValueError("binary targets must contain only 0 and 1")
    positives = probabilities[targets == 1]
    negatives = probabilities[targets == 0]
    if positives.size == 0 or negatives.size == 0:
        raise ValueError("binary score diagnostics require both positive and negative samples")

    comparisons = positives[:, None] - negatives[None, :]
    roc_auc = float((comparisons > 0).mean() + 0.5 * (comparisons == 0).mean())

    best = None
    for threshold in np.unique(probabilities):
        predicted = probabilities >= threshold
        tp = int(np.logical_and(predicted, targets == 1).sum())
        tn = int(np.logical_and(~predicted, targets == 0).sum())
        fp = int(np.logical_and(predicted, targets == 0).sum())
        fn = int(np.logical_and(~predicted, targets == 1).sum())
        recall = tp / positives.size
        specificity = tn / negatives.size
        balanced_accuracy = 0.5 * (recall + specificity)
        # Prefer fewer false positives when balanced accuracy ties.
        candidate = (balanced_accuracy, specificity, float(threshold), tp, tn, fp, fn)
        if best is None or candidate[:3] > best[:3]:
            best = candidate

    balanced_accuracy, specificity, threshold, tp, tn, fp, fn = best
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / positives.size
    return {
        "stop_roc_auc": roc_auc,
        "stop_positive_probability_mean": float(positives.mean()),
        "stop_positive_probability_min": float(positives.min()),
        "stop_positive_probability_max": float(positives.max()),
        "stop_negative_probability_mean": float(negatives.mean()),
        "stop_negative_probability_min": float(negatives.min()),
        "stop_negative_probability_max": float(negatives.max()),
        "stop_probability_class_gap": float(positives.mean() - negatives.mean()),
        "stop_best_threshold": threshold,
        "stop_best_balanced_accuracy": float(balanced_accuracy),
        "stop_best_precision": float(precision),
        "stop_best_recall": float(recall),
        "stop_best_specificity": float(specificity),
        "stop_best_true_positive": float(tp),
        "stop_best_true_negative": float(tn),
        "stop_best_false_positive": float(fp),
        "stop_best_false_negative": float(fn),
    }


def compute_progress_stop_loss(
    logits: torch.Tensor,
    actions_remaining_after_root: torch.Tensor,
    positive_weights,
    auxiliary_weight: float,
    threshold: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Terminal BCE plus dense ordinal supervision for remaining trajectory steps."""
    if logits.ndim != 2 or logits.shape[1] != len(STOP_PROGRESS_HORIZONS):
        raise ValueError("progress STOP logits must have shape (B,4)")
    if auxiliary_weight < 0:
        raise ValueError("auxiliary_weight must be non-negative")
    weights = torch.as_tensor(positive_weights, device=logits.device, dtype=torch.float32)
    if weights.shape != (len(STOP_PROGRESS_HORIZONS),) or (weights <= 0).any():
        raise ValueError("positive_weights must provide four positive values")

    remaining = actions_remaining_after_root.to(logits.device).long().reshape(-1)
    horizons = torch.as_tensor(STOP_PROGRESS_HORIZONS, device=logits.device)
    targets = (remaining[:, None] <= horizons[None, :]).float()
    terminal_loss, metrics = compute_stop_after_action_loss(
        logits[:, 0], targets[:, 0], float(weights[0]), threshold=threshold
    )
    auxiliary_losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:, 1:].float(),
        targets[:, 1:],
        pos_weight=weights[1:],
        reduction="none",
    ).mean(dim=0)
    auxiliary_loss = auxiliary_losses.mean()
    total_loss = terminal_loss + auxiliary_weight * auxiliary_loss
    metrics.update(
        {
            "stop_progress_aux_loss": auxiliary_loss.item(),
            "stop_progress_total_loss": total_loss.item(),
        }
    )
    probabilities = torch.sigmoid(logits.detach().float())
    for index, horizon in enumerate(STOP_PROGRESS_HORIZONS):
        metrics[f"stop_progress_h{horizon}_probability_mean"] = probabilities[:, index].mean().item()
        metrics[f"stop_progress_h{horizon}_target_rate"] = targets[:, index].mean().item()
        metrics[f"stop_progress_h{horizon}_accuracy"] = (
            (probabilities[:, index] >= threshold) == targets[:, index].bool()
        ).float().mean().item()
        if index > 0:
            metrics[f"stop_progress_h{horizon}_loss"] = auxiliary_losses[index - 1].item()
    return total_loss, metrics


def compute_indoor_uav_sft_loss(
    predicted_actions: torch.Tensor,
    ground_truth_actions: torch.Tensor,
    projected_conditions: torch.Tensor,
    future_patch_embeddings: torch.Tensor,
    condition_similarities: torch.Tensor,
    plan_valid_mask: torch.Tensor,
    *,
    assignment_temperature: float,
    root_action_weight: float,
    future_action_weight: float,
    condition_alignment_weight: float,
    condition_contrastive_weight: float,
    condition_temporal_weight: float,
    condition_queue_weight: float,
    condition_contrastive_temperature: float,
    branch_balance_weight: float,
    condition_diversity_weight: float,
    condition_diversity_margin: float,
    patch_topk: int,
    query_episode_ids=None,
    queued_image_patches: Optional[torch.Tensor] = None,
    queued_episode_ids: Optional[list[str]] = None,
    condition_queue_min_negatives: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Masked deterministic Best-of-K SFT for the IndoorUAV condition-action policy."""
    if predicted_actions.ndim != 4 or ground_truth_actions.ndim != 3:
        raise ValueError("actions must have shapes (B,T,K,D) and (B,T,D)")
    if predicted_actions.shape[:2] != ground_truth_actions.shape[:2]:
        raise ValueError("predicted and target action batch/time dimensions must match")
    if plan_valid_mask.shape != ground_truth_actions.shape[:2]:
        raise ValueError("plan_valid_mask must have shape (B,T)")

    action_errors = torch.nn.functional.smooth_l1_loss(
        predicted_actions.float(),
        ground_truth_actions[:, :, None, :].float().expand_as(predicted_actions),
        reduction="none",
    ).mean(dim=-1)
    root_loss = _masked_mean(action_errors[:, 0, 0], plan_valid_mask[:, 0])

    future_errors = action_errors[:, 1:]
    future_mask = plan_valid_mask[:, 1:].bool()
    # The expert action, not the condition's own similarity score, assigns the
    # paired branch. Otherwise the K-way label is self-generated by the logits
    # it is supposed to supervise and retrieval accuracy becomes circular.
    winners = future_errors.detach().argmin(dim=-1)
    winner_errors = future_errors.gather(2, winners.unsqueeze(-1)).squeeze(-1)
    future_action_loss = _masked_mean(winner_errors, future_mask)

    # Diagnostics for deciding whether K represents distinct, stable action
    # hypotheses rather than nearly identical branches with arbitrary winners.
    if predicted_actions.shape[2] > 1:
        sorted_action_errors = future_errors.detach().sort(dim=-1).values
        winner_action_gap = sorted_action_errors[..., 1] - sorted_action_errors[..., 0]
        action_pair_distances = []
        for left in range(predicted_actions.shape[2]):
            for right in range(left + 1, predicted_actions.shape[2]):
                action_pair_distances.append(
                    torch.abs(
                        predicted_actions[:, 1:, left].float()
                        - predicted_actions[:, 1:, right].float()
                    ).mean(dim=-1)
                )
        action_pair_distances = torch.stack(action_pair_distances, dim=-1)
        action_branch_pair_l1 = _masked_mean(action_pair_distances.mean(dim=-1), future_mask)
        action_branch_min_pair_l1 = _masked_mean(action_pair_distances.min(dim=-1).values, future_mask)
        action_winner_gap = _masked_mean(winner_action_gap, future_mask)
        action_winner_near_tie_rate = _masked_mean(
            (winner_action_gap < 1e-3).float(), future_mask
        )
    else:
        action_branch_pair_l1 = predicted_actions.sum() * 0.0
        action_branch_min_pair_l1 = predicted_actions.sum() * 0.0
        action_winner_gap = predicted_actions.sum() * 0.0
        action_winner_near_tie_rate = predicted_actions.sum() * 0.0

    winner_similarities = condition_similarities.gather(2, winners.unsqueeze(-1)).squeeze(-1)
    positive_loss = _masked_mean(1.0 - winner_similarities, future_mask)

    selected_conditions = projected_conditions.gather(
        2,
        winners[..., None, None].expand(-1, -1, 1, projected_conditions.shape[-1]),
    ).squeeze(2)
    valid_conditions = selected_conditions[future_mask]
    valid_images = future_patch_embeddings[future_mask]
    if condition_similarities.shape[2] > 1:
        branch_loss, branch_accuracy, branch_margin = condition_branch_contrastive_loss(
            condition_similarities,
            winners,
            temperature=condition_contrastive_temperature,
            loss_start_time_index=0,
            valid_mask=future_mask,
        )
    else:
        branch_loss = predicted_actions.sum() * 0.0
        branch_accuracy = branch_loss.detach()
        branch_margin = branch_loss.detach()

    if valid_conditions.shape[0] > 1:
        retrieval_scores = condition_to_image_logits(valid_conditions, valid_images, patch_topk)
        retrieval_logits = retrieval_scores / condition_contrastive_temperature
        retrieval_labels = torch.arange(retrieval_logits.shape[0], device=retrieval_logits.device)
        temporal_loss = torch.nn.functional.cross_entropy(retrieval_logits, retrieval_labels)
        retrieval_accuracy = (retrieval_logits.argmax(dim=1) == retrieval_labels).float().mean()
        positive_scores = retrieval_scores.diagonal()
        negative_mask = ~torch.eye(
            retrieval_scores.shape[0], dtype=torch.bool, device=retrieval_scores.device
        )
        hardest_negative_scores = retrieval_scores.masked_fill(negative_mask.logical_not(), -torch.inf).max(dim=1).values
        retrieval_positive = positive_scores.mean()
        retrieval_hardest_negative = hardest_negative_scores.mean()
        retrieval_margin = (positive_scores - hardest_negative_scores).mean()
    else:
        temporal_loss = predicted_actions.sum() * 0.0
        retrieval_accuracy = predicted_actions.new_zeros((), dtype=torch.float32)
        retrieval_positive = predicted_actions.new_zeros((), dtype=torch.float32)
        retrieval_hardest_negative = predicted_actions.new_zeros((), dtype=torch.float32)
        retrieval_margin = predicted_actions.new_zeros((), dtype=torch.float32)

    if condition_queue_weight > 0:
        valid_episode_ids = _valid_future_episode_ids(query_episode_ids, future_mask)
        queue_loss, queue_metrics = compute_cross_episode_queue_loss(
            valid_conditions,
            valid_images,
            valid_episode_ids,
            queued_image_patches,
            queued_episode_ids or [],
            temperature=condition_contrastive_temperature,
            patch_topk=patch_topk,
            min_negatives=condition_queue_min_negatives,
        )
    else:
        queue_loss = predicted_actions.sum() * 0.0
        queue_metrics = {
            "condition_queue_loss": 0.0,
            "condition_queue_accuracy": 0.0,
            "condition_queue_random_accuracy": 0.0,
            "condition_queue_margin": 0.0,
            "condition_queue_queries": 0.0,
            "condition_queue_negatives": 0.0,
        }

    soft_assignments = torch.softmax(-future_errors / assignment_temperature, dim=-1)
    valid_soft_assignments = soft_assignments[future_mask]
    if valid_soft_assignments.numel():
        branch_usage = valid_soft_assignments.mean(dim=0)
        uniform = torch.full_like(branch_usage, 1.0 / predicted_actions.shape[2])
        balance_loss = torch.sum(branch_usage * torch.log((branch_usage + 1e-8) / uniform))
    else:
        branch_usage = torch.full(
            (predicted_actions.shape[2],),
            1.0 / predicted_actions.shape[2],
            device=predicted_actions.device,
        )
        balance_loss = predicted_actions.new_zeros((), dtype=torch.float32)

    pair_distances = []
    for left in range(projected_conditions.shape[2]):
        for right in range(left + 1, projected_conditions.shape[2]):
            cosine = (projected_conditions[:, :, left] * projected_conditions[:, :, right]).sum(dim=-1)
            pair_distances.append(1.0 - cosine)
    if pair_distances:
        pair_distances = torch.stack(pair_distances, dim=-1)
        diversity_loss = _masked_mean(
            torch.relu(condition_diversity_margin - pair_distances).mean(dim=-1),
            future_mask,
        )
    else:
        diversity_loss = predicted_actions.new_zeros((), dtype=torch.float32)

    loss = (
        root_action_weight * root_loss
        + future_action_weight * future_action_loss
        + condition_alignment_weight * positive_loss
        + condition_contrastive_weight * branch_loss
        + condition_temporal_weight * temporal_loss
        + condition_queue_weight * queue_loss
        + branch_balance_weight * balance_loss
        + condition_diversity_weight * diversity_loss
    )
    metrics = {
        "sft_root_action_loss": root_loss.item(),
        "sft_future_action_loss": future_action_loss.item(),
        "action_branch_pair_l1": action_branch_pair_l1.item(),
        "action_branch_min_pair_l1": action_branch_min_pair_l1.item(),
        "action_winner_gap": action_winner_gap.item(),
        "action_winner_near_tie_rate": action_winner_near_tie_rate.item(),
        "condition_alignment_loss": positive_loss.item(),
        "condition_contrastive_loss": branch_loss.item(),
        "condition_branch_accuracy": branch_accuracy.item(),
        "condition_branch_random_accuracy": (
            1.0 / condition_similarities.shape[2] if future_mask.any() else 0.0
        ),
        "condition_branch_margin": branch_margin.item(),
        "condition_temporal_loss": temporal_loss.item(),
        "condition_retrieval_accuracy": retrieval_accuracy.item(),
        "condition_temporal_random_accuracy": (
            1.0 / valid_conditions.shape[0] if valid_conditions.shape[0] > 1 else 0.0
        ),
        "condition_retrieval_positive": retrieval_positive.item(),
        "condition_retrieval_hardest_negative": retrieval_hardest_negative.item(),
        "condition_retrieval_margin": retrieval_margin.item(),
        "branch_balance_loss": balance_loss.item(),
        "condition_diversity_loss": diversity_loss.item(),
        "condition_similarity_selected": _masked_mean(winner_similarities, future_mask).item(),
        "root_valid_count": plan_valid_mask[:, 0].sum().item(),
        "future_valid_count": future_mask.sum().item(),
        "condition_temporal_query_count": (
            float(valid_conditions.shape[0]) if valid_conditions.shape[0] > 1 else 0.0
        ),
        "plan_valid_ratio": plan_valid_mask.float().mean().item(),
    }
    metrics.update(queue_metrics)
    hard_assignments = torch.nn.functional.one_hot(winners, num_classes=predicted_actions.shape[2]).float()
    for branch_idx in range(predicted_actions.shape[2]):
        metrics[f"branch{branch_idx}_winner_rate"] = _masked_mean(
            hard_assignments[..., branch_idx], future_mask
        ).item()
        metrics[f"branch{branch_idx}_soft_usage"] = branch_usage[branch_idx].item()
    for future_idx in range(winners.shape[1]):
        slot_mask = future_mask[:, future_idx]
        slot_number = future_idx + 1
        metrics[f"future_slot{slot_number}_valid_count"] = slot_mask.sum().item()
        for branch_idx in range(predicted_actions.shape[2]):
            metrics[f"future_slot{slot_number}_branch{branch_idx}_winner_rate"] = _masked_mean(
                hard_assignments[:, future_idx, branch_idx], slot_mask
            ).item()
    return loss, winners, metrics


def diagonal_gaussian_nll(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Elementwise negative log likelihood for a diagonal Gaussian policy."""
    mean = mean.float()
    log_std = log_std.float()
    target = target.float()
    squared_error = torch.square((target - mean) * torch.exp(-log_std))
    return 0.5 * squared_error + log_std + 0.5 * math.log(2.0 * math.pi)


def match_action_target_shape(predicted_actions: torch.Tensor, ground_truth_actions: torch.Tensor) -> torch.Tensor:
    """Match [B,T,D] targets to either [B,T,D] or explicit [B,T,K,D] policy outputs."""
    if predicted_actions.ndim == 3:
        if predicted_actions.shape != ground_truth_actions.shape:
            raise ValueError(
                f"Action shape mismatch: predictions={tuple(predicted_actions.shape)}, "
                f"targets={tuple(ground_truth_actions.shape)}"
            )
        return ground_truth_actions
    if predicted_actions.ndim == 4:
        if predicted_actions.shape[:2] != ground_truth_actions.shape[:2] or (
            predicted_actions.shape[-1] != ground_truth_actions.shape[-1]
        ):
            raise ValueError(
                f"Action shape mismatch: predictions={tuple(predicted_actions.shape)}, "
                f"targets={tuple(ground_truth_actions.shape)}"
            )
        return ground_truth_actions.unsqueeze(2).expand_as(predicted_actions)
    raise ValueError(f"Expected action predictions with rank 3 or 4, got rank {predicted_actions.ndim}")


def compute_action_regression_loss(
    predicted_actions: torch.Tensor,
    ground_truth_actions: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    targets = match_action_target_shape(predicted_actions, ground_truth_actions)
    if loss_type == "l1":
        return torch.abs(predicted_actions - targets).mean()
    if loss_type == "mse":
        return torch.square(predicted_actions.float() - targets.float()).mean()
    raise ValueError(f"Unsupported action_regression_loss: {loss_type}")


def compute_best_of_k_gaussian_action_loss(
    action_mean: torch.Tensor,
    action_log_std: torch.Tensor,
    ground_truth_actions: torch.Tensor,
    assignment_temperature: float,
    condition_similarities: Optional[torch.Tensor] = None,
    condition_assignment_weight: float = 0.0,
    condition_loss_start_time_index: int = 0,
    initial_action_branch_index: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Assign each (batch, time) target to one paired condition-action branch."""
    if action_mean.ndim != 4 or action_log_std.shape != action_mean.shape:
        raise ValueError("Gaussian best-of-K requires matching (B, T, K, action_dim) tensors")
    if assignment_temperature <= 0:
        raise ValueError("assignment_temperature must be > 0")
    if condition_assignment_weight < 0:
        raise ValueError("condition_assignment_weight must be >= 0")
    if not 0 <= condition_loss_start_time_index < action_mean.shape[1]:
        raise ValueError("condition_loss_start_time_index is outside the action horizon")
    if initial_action_branch_index is not None and not 0 <= initial_action_branch_index < action_mean.shape[2]:
        raise ValueError("initial_action_branch_index is outside the K action branches")

    targets = ground_truth_actions.unsqueeze(2).expand_as(action_mean)
    per_time_branch_nll = diagonal_gaussian_nll(action_mean, action_log_std, targets).mean(dim=-1)
    assignment_cost = per_time_branch_nll
    if condition_assignment_weight > 0:
        if condition_similarities is None:
            raise ValueError("condition similarities are required for joint condition-action assignment")
        if condition_similarities.shape != per_time_branch_nll.shape:
            raise ValueError(
                "condition similarities must match Gaussian branch costs, got "
                f"{tuple(condition_similarities.shape)} and {tuple(per_time_branch_nll.shape)}"
            )
        condition_cost = 1.0 - condition_similarities.float()
        condition_mask = torch.zeros_like(condition_cost)
        condition_mask[:, condition_loss_start_time_index:] = 1.0
        assignment_cost = per_time_branch_nll + condition_assignment_weight * condition_cost * condition_mask

    winner_indices = assignment_cost.detach().argmin(dim=2)
    if initial_action_branch_index is not None:
        winner_indices = winner_indices.clone()
        winner_indices[:, 0] = initial_action_branch_index
    winner_nll = per_time_branch_nll.gather(2, winner_indices.unsqueeze(2)).squeeze(2)
    best_of_k_loss = winner_nll.mean()

    assignment_probabilities = torch.softmax(-assignment_cost / assignment_temperature, dim=2)
    if initial_action_branch_index is not None:
        fixed_initial_assignment = torch.nn.functional.one_hot(
            torch.full(
                (action_mean.shape[0],),
                initial_action_branch_index,
                device=action_mean.device,
                dtype=torch.long,
            ),
            num_classes=action_mean.shape[2],
        ).to(assignment_probabilities.dtype)
        assignment_probabilities = assignment_probabilities.clone()
        assignment_probabilities[:, 0] = fixed_initial_assignment
    branch_usage = assignment_probabilities.mean(dim=(0, 1))
    uniform_usage = torch.full_like(branch_usage, 1.0 / action_mean.shape[2])
    branch_balance_loss = torch.sum(
        branch_usage * torch.log((branch_usage + 1e-8) / uniform_usage)
    )
    assignment_entropy = -torch.sum(
        assignment_probabilities * torch.log(assignment_probabilities + 1e-8), dim=2
    ).mean()

    metrics = {
        "best_of_k_gaussian_nll": best_of_k_loss.item(),
        "joint_assignment_cost": assignment_cost.gather(
            2, winner_indices.unsqueeze(2)
        ).mean().item(),
        "joint_condition_assignment_weight": float(condition_assignment_weight),
        "branch_balance_loss": branch_balance_loss.item(),
        "branch_assignment_entropy": assignment_entropy.item(),
    }
    if initial_action_branch_index is not None:
        metrics["initial_action_branch_index"] = float(initial_action_branch_index)
    hard_assignments = torch.nn.functional.one_hot(
        winner_indices, num_classes=action_mean.shape[2]
    ).float()
    for branch_idx in range(action_mean.shape[2]):
        metrics[f"branch{branch_idx}_winner_rate"] = hard_assignments[..., branch_idx].mean().item()
        metrics[f"branch{branch_idx}_soft_usage"] = branch_usage[branch_idx].item()

    return best_of_k_loss, branch_balance_loss, winner_indices, metrics


def compute_condition_similarity_tensors(
    vla,
    cond_hidden_states: torch.Tensor,
    future_pixel_values: Optional[torch.Tensor],
    patch_topk: int,
    use_film: bool = False,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, float]]:
    """Encode future observations once and return all per-time, per-branch similarities."""
    if cond_hidden_states is None or future_pixel_values is None:
        return None, None, {}
    if use_film:
        return None, None, {"condition_alignment_skipped_film": 1.0}

    base_vla = _unwrap_vla_model(vla)
    vision_backbone = base_vla.vision_backbone
    old_num_images = vision_backbone.get_num_images_in_input()

    with torch.no_grad():
        batch_size, horizon, channels, height, width = future_pixel_values.shape
        future_images = future_pixel_values.reshape(batch_size * horizon, channels, height, width)
        try:
            vision_backbone.set_num_images_in_input(1)
            future_patch_embeddings = base_vla._process_vision_features(future_images, use_film=False)
        finally:
            vision_backbone.set_num_images_in_input(old_num_images)

        future_patch_embeddings = future_patch_embeddings.float().reshape(
            batch_size, horizon, future_patch_embeddings.shape[1], -1
        ).detach()

    similarities = condition_to_patch_similarity(
        cond_hidden_states,
        future_patch_embeddings,
        patch_topk,
    )
    return similarities, future_patch_embeddings, {}


def compute_projected_condition_targets(
    vla,
    condition_adapter: IndoorUAVConditionAdapter,
    cond_hidden_states: torch.Tensor,
    future_pixel_values: torch.Tensor,
    patch_topk: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode future labels and compare them with projected future condition branches."""
    if future_pixel_values.shape[1] != cond_hidden_states.shape[1] - 1:
        raise ValueError("future image labels must correspond to condition slots 1..T-1")

    adapter = condition_adapter.module if hasattr(condition_adapter, "module") else condition_adapter
    base_vla = _unwrap_vla_model(vla)
    vision_backbone = base_vla.vision_backbone
    old_num_images = vision_backbone.get_num_images_in_input()
    batch_size, horizon, channels, height, width = future_pixel_values.shape
    future_images = future_pixel_values.reshape(batch_size * horizon, channels, height, width)
    try:
        vision_backbone.set_num_images_in_input(1)
        # The shared vision encoder is a stable target encoder. The learned vision
        # matching projector below remains outside no_grad and receives gradients.
        with torch.no_grad():
            raw_patch_embeddings = base_vla._process_vision_features(future_images, use_film=False)
    finally:
        vision_backbone.set_num_images_in_input(old_num_images)

    raw_patch_embeddings = raw_patch_embeddings.reshape(
        batch_size,
        horizon,
        raw_patch_embeddings.shape[1],
        raw_patch_embeddings.shape[2],
    )
    projected_conditions = adapter.project_conditions(cond_hidden_states[:, 1:])
    future_patch_embeddings = adapter.project_vision(raw_patch_embeddings)
    similarities, topk_indices = projected_condition_to_patch_similarity(
        projected_conditions,
        future_patch_embeddings,
        patch_topk,
        return_indices=True,
    )
    return projected_conditions, future_patch_embeddings, similarities, topk_indices


def compute_condition_contrastive_loss(
    selected_condition_embeddings: torch.Tensor,
    target_image_embeddings: torch.Tensor,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Matches each selected condition to its own time-indexed future image."""
    if temperature <= 0:
        raise ValueError("condition contrastive temperature must be > 0")
    if selected_condition_embeddings.shape != target_image_embeddings.shape:
        raise ValueError(
            "condition and image embedding shapes must match, got "
            f"{tuple(selected_condition_embeddings.shape)} and {tuple(target_image_embeddings.shape)}"
        )

    flat_conditions = selected_condition_embeddings.flatten(0, 1)
    flat_targets = target_image_embeddings.flatten(0, 1)
    logits = flat_conditions @ flat_targets.transpose(0, 1) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    accuracy = (logits.argmax(dim=1) == labels).float().mean()
    return loss, accuracy


def compute_condition_alignment_loss_and_metrics(
    vla,
    cond_hidden_states: torch.Tensor,
    future_pixel_values: Optional[torch.Tensor],
    similarity_threshold: float,
    diversity_margin: float,
    selected_branch_indices: Optional[torch.Tensor] = None,
    contrastive_temperature: float = 0.07,
    loss_start_time_index: int = 0,
    patch_topk: int = 8,
    use_film: bool = False,
    precomputed_similarities: Optional[torch.Tensor] = None,
    precomputed_future_patch_embeddings: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, float]]:
    if cond_hidden_states is None or future_pixel_values is None:
        return None, None, None, {}
    if use_film:
        return None, None, None, {"condition_alignment_skipped_film": 1.0}

    if (precomputed_similarities is None) != (precomputed_future_patch_embeddings is None):
        raise ValueError("precomputed condition similarities and patch embeddings must be provided together")
    if precomputed_similarities is None:
        similarities, future_patch_embeddings, precompute_metrics = compute_condition_similarity_tensors(
            vla=vla,
            cond_hidden_states=cond_hidden_states,
            future_pixel_values=future_pixel_values,
            patch_topk=patch_topk,
            use_film=use_film,
        )
        if similarities is None:
            return None, None, None, precompute_metrics
    else:
        similarities = precomputed_similarities
        future_patch_embeddings = precomputed_future_patch_embeddings

    cond_embeddings = torch.nn.functional.normalize(cond_hidden_states.float(), dim=-1)
    centered_cond_embeddings = center_condition_branches(cond_hidden_states)
    centered_future_patch_embeddings = center_visual_patches(future_patch_embeddings)
    horizon = similarities.shape[1]
    if not 0 <= loss_start_time_index < horizon:
        raise ValueError(
            f"condition loss start index must be in [0, {horizon}), got {loss_start_time_index}"
        )
    best_similarity, best_branch = similarities.max(dim=2)
    sorted_similarity = similarities.sort(dim=2, descending=True).values
    margin = sorted_similarity[:, :, 0] - sorted_similarity[:, :, 1]
    if selected_branch_indices is None:
        selected_branch_indices = best_branch
        coupled_to_action = False
    else:
        if selected_branch_indices.shape != best_branch.shape:
            raise ValueError(
                f"selected branch shape must be {tuple(best_branch.shape)}, "
                f"got {tuple(selected_branch_indices.shape)}"
            )
        selected_branch_indices = selected_branch_indices.detach()
        coupled_to_action = True

    selected_similarity = similarities.gather(2, selected_branch_indices.unsqueeze(2)).squeeze(2)
    (
        condition_contrastive_loss,
        condition_contrastive_accuracy,
        condition_contrastive_margin,
    ) = condition_branch_contrastive_loss(
        similarities,
        selected_branch_indices,
        temperature=contrastive_temperature,
        loss_start_time_index=loss_start_time_index,
    )

    condition_pair_distances = []
    for left_branch in range(cond_embeddings.shape[2]):
        for right_branch in range(left_branch + 1, cond_embeddings.shape[2]):
            branch_cosine = (
                cond_embeddings[:, loss_start_time_index:, left_branch]
                * cond_embeddings[:, loss_start_time_index:, right_branch]
            ).sum(dim=-1)
            condition_pair_distances.append(1.0 - branch_cosine)
    if condition_pair_distances:
        condition_pair_distances = torch.stack(condition_pair_distances, dim=2)
        condition_mean_distance = condition_pair_distances.mean()
        condition_diversity_loss = torch.relu(diversity_margin - condition_pair_distances).mean()
    else:
        condition_mean_distance = torch.zeros((), device=cond_hidden_states.device)
        condition_diversity_loss = torch.zeros((), device=cond_hidden_states.device)

    condition_alignment_loss = (1.0 - selected_similarity[:, loss_start_time_index:]).mean()

    return condition_alignment_loss, condition_diversity_loss, condition_contrastive_loss, {
        "condition_similarity_mean": similarities.mean().item(),
        "condition_similarity_best": best_similarity.mean().item(),
        "condition_similarity_selected": selected_similarity.mean().item(),
        "condition_similarity_margin": margin.mean().item(),
        "condition_top_branch_mean": best_branch.float().mean().item(),
        "condition_selected_branch_mean": selected_branch_indices.float().mean().item(),
        "condition_action_branch_match_rate": (
            (selected_branch_indices == best_branch).float().mean().item() if coupled_to_action else 1.0
        ),
        "condition_threshold_pass_rate": (best_similarity >= similarity_threshold).float().mean().item(),
        "condition_future_threshold_pass_rate": (
            best_similarity[:, loss_start_time_index:] >= similarity_threshold
        ).float().mean().item(),
        "condition_similarity_t1": best_similarity[:, 0].mean().item(),
        "condition_similarity_future": best_similarity[:, 1:].mean().item() if horizon > 1 else best_similarity.mean().item(),
        "condition_loss_start_time_index": float(loss_start_time_index),
        "condition_patch_topk": float(patch_topk),
        "condition_matching_centered": 1.0,
        "condition_contrastive_num_branches": float(similarities.shape[2]),
        "condition_centered_norm_mean": centered_cond_embeddings.norm(dim=-1).mean().item(),
        "condition_patch_centered_norm_mean": centered_future_patch_embeddings.norm(dim=-1).mean().item(),
        "condition_alignment_loss": condition_alignment_loss.item(),
        "condition_contrastive_loss": condition_contrastive_loss.item(),
        "condition_contrastive_accuracy": condition_contrastive_accuracy.item(),
        "condition_contrastive_margin": condition_contrastive_margin.item(),
        "condition_diversity_loss": condition_diversity_loss.item(),
        "condition_mean_distance": condition_mean_distance.item(),
    }


def _unnormalize_actions_for_reward(actions: torch.Tensor, action_norm_stats: Optional[dict]) -> torch.Tensor:
    """Convert normalized action/state predictions back to real units for offline reward metrics."""
    actions = actions.float()
    if action_norm_stats is None:
        return actions

    if "normalization_low" in action_norm_stats and "normalization_high" in action_norm_stats:
        action_low = _stats_tensor(action_norm_stats["normalization_low"], actions.device)
        action_high = _stats_tensor(action_norm_stats["normalization_high"], actions.device)
    elif "q01" in action_norm_stats and "q99" in action_norm_stats:
        action_low = _stats_tensor(action_norm_stats["q01"], actions.device)
        action_high = _stats_tensor(action_norm_stats["q99"], actions.device)
    elif "min" in action_norm_stats and "max" in action_norm_stats:
        action_low = _stats_tensor(action_norm_stats["min"], actions.device)
        action_high = _stats_tensor(action_norm_stats["max"], actions.device)
    else:
        return actions

    mask = _stats_tensor(action_norm_stats.get("mask", [True] * actions.shape[-1]), actions.device, torch.bool)
    unnormalized = 0.5 * (actions + 1.0) * (action_high - action_low + 1e-8) + action_low
    return torch.where(mask, unnormalized, actions)


def compute_root_action_axis_metrics(
    predicted_actions: torch.Tensor,
    ground_truth_actions: torch.Tensor,
    action_norm_stats: Optional[dict],
) -> Dict[str, float]:
    """Report physical root-action bias and sign accuracy for each body axis."""
    root_predictions = (
        predicted_actions[:, 0, 0] if predicted_actions.ndim == 4 else predicted_actions[:, 0]
    )
    root_targets = ground_truth_actions[:, 0]
    root_predictions = _unnormalize_actions_for_reward(root_predictions.detach(), action_norm_stats)
    root_targets = _unnormalize_actions_for_reward(root_targets.detach(), action_norm_stats)

    metrics = {}
    for axis, name in enumerate(("forward", "right", "up", "yaw")):
        prediction = root_predictions[:, axis]
        target = root_targets[:, axis]
        nonzero = target.abs() > 1e-4
        sign_accuracy = (
            (torch.sign(prediction[nonzero]) == torch.sign(target[nonzero])).float().mean()
            if nonzero.any()
            else prediction.new_zeros(())
        )
        metrics.update(
            {
                f"root_{name}_prediction_mean": prediction.mean().item(),
                f"root_{name}_target_mean": target.mean().item(),
                f"root_{name}_bias": (prediction - target).mean().item(),
                f"root_{name}_abs_error": (prediction - target).abs().mean().item(),
                f"root_{name}_sign_accuracy": sign_accuracy.item(),
                f"root_{name}_nonzero_count": float(nonzero.sum().item()),
            }
        )
    return metrics


def _wrapped_abs_yaw_error(pred_yaw: torch.Tensor, target_yaw: torch.Tensor) -> torch.Tensor:
    diff = torch.remainder(pred_yaw - target_yaw + torch.pi, 2 * torch.pi) - torch.pi
    return diff.abs()


def _oracle_recovery(oracle_loss: float, selected_loss: float, branch0_loss: float) -> float:
    """Fraction of the available branch-0-to-oracle improvement recovered by matching."""
    available_gain = branch0_loss - oracle_loss
    if available_gain <= 1e-8:
        return 1.0 if selected_loss <= oracle_loss + 1e-8 else 0.0
    return (branch0_loss - selected_loss) / available_gain


def compute_condition_selected_action_metrics(
    predicted_actions: torch.Tensor,
    ground_truth_actions: torch.Tensor,
    condition_similarities: torch.Tensor,
    plan_valid_mask: torch.Tensor,
    action_norm_stats: Optional[dict],
) -> Dict[str, float]:
    """Compare inference-time condition selection with oracle and fixed branch 0."""
    if predicted_actions.ndim != 4 or ground_truth_actions.ndim != 3:
        raise ValueError("condition-selected metrics require (B,T,K,D) predictions and (B,T,D) targets")
    if predicted_actions.shape[:2] != ground_truth_actions.shape[:2]:
        raise ValueError("prediction and target batch/time dimensions must match")
    if condition_similarities.shape != predicted_actions[:, 1:, :, 0].shape:
        raise ValueError("condition similarities must have shape (B,T-1,K)")
    if plan_valid_mask.shape != ground_truth_actions.shape[:2]:
        raise ValueError("plan_valid_mask must have shape (B,T)")

    future_predictions = predicted_actions[:, 1:].float()
    future_targets = ground_truth_actions[:, 1:].float()
    future_mask = plan_valid_mask[:, 1:].bool()
    action_errors = torch.nn.functional.smooth_l1_loss(
        future_predictions,
        future_targets[:, :, None, :].expand_as(future_predictions),
        reduction="none",
    ).mean(dim=-1)
    oracle_indices = action_errors.argmin(dim=-1)
    condition_indices = condition_similarities.detach().argmax(dim=-1)

    def gather_branch(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        gather_index = indices.unsqueeze(-1)
        while gather_index.ndim < values.ndim:
            gather_index = gather_index.unsqueeze(-1)
        return values.gather(
            2,
            gather_index.expand(*values.shape[:2], 1, *values.shape[3:]),
        ).squeeze(2)

    oracle_errors = action_errors.gather(2, oracle_indices.unsqueeze(-1)).squeeze(-1)
    selected_errors = action_errors.gather(2, condition_indices.unsqueeze(-1)).squeeze(-1)
    branch0_errors = action_errors[:, :, 0]
    oracle_loss = _masked_mean(oracle_errors, future_mask).item()
    selected_loss = _masked_mean(selected_errors, future_mask).item()
    branch0_loss = _masked_mean(branch0_errors, future_mask).item()

    real_predictions = _unnormalize_actions_for_reward(future_predictions.detach(), action_norm_stats)
    real_targets = _unnormalize_actions_for_reward(future_targets.detach(), action_norm_stats)
    oracle_actions = gather_branch(real_predictions, oracle_indices)
    selected_actions = gather_branch(real_predictions, condition_indices)
    branch0_actions = real_predictions[:, :, 0]

    def physical_errors(actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        position = torch.linalg.vector_norm(actions[..., :3] - real_targets[..., :3], dim=-1)
        yaw = _wrapped_abs_yaw_error(actions[..., 3], real_targets[..., 3])
        return position, yaw

    oracle_position, oracle_yaw = physical_errors(oracle_actions)
    selected_position, selected_yaw = physical_errors(selected_actions)
    branch0_position, branch0_yaw = physical_errors(branch0_actions)
    metrics = {
        "oracle_future_action_loss": oracle_loss,
        "condition_selected_action_loss": selected_loss,
        "branch0_future_action_loss": branch0_loss,
        "condition_selection_regret": selected_loss - oracle_loss,
        "condition_gain_vs_branch0": branch0_loss - selected_loss,
        "condition_oracle_recovery": _oracle_recovery(oracle_loss, selected_loss, branch0_loss),
        "oracle_action_position_error_m": _masked_mean(oracle_position, future_mask).item(),
        "condition_selected_position_error_m": _masked_mean(selected_position, future_mask).item(),
        "branch0_position_error_m": _masked_mean(branch0_position, future_mask).item(),
        "oracle_action_yaw_error_rad": _masked_mean(oracle_yaw, future_mask).item(),
        "condition_selected_yaw_error_rad": _masked_mean(selected_yaw, future_mask).item(),
        "branch0_yaw_error_rad": _masked_mean(branch0_yaw, future_mask).item(),
    }
    selected_one_hot = torch.nn.functional.one_hot(
        condition_indices, num_classes=predicted_actions.shape[2]
    ).float()
    for branch_idx in range(predicted_actions.shape[2]):
        metrics[f"condition_branch{branch_idx}_selected_rate"] = _masked_mean(
            selected_one_hot[..., branch_idx], future_mask
        ).item()
    return metrics


def compute_offline_branch_reward_tensors(
    predicted_actions: torch.Tensor,
    ground_truth_actions: torch.Tensor,
    action_norm_stats: Optional[dict],
) -> Dict[str, torch.Tensor]:
    """Computes offline branch reward tensors in real pose units."""
    with torch.no_grad():
        if predicted_actions.ndim == 3:
            predicted_actions = predicted_actions.unsqueeze(2)

        pred = _unnormalize_actions_for_reward(predicted_actions.detach(), action_norm_stats)
        target = _unnormalize_actions_for_reward(ground_truth_actions.detach(), action_norm_stats).unsqueeze(2)

        pos_error = torch.linalg.vector_norm(pred[..., :3] - target[..., :3], dim=-1)
        yaw_error = _wrapped_abs_yaw_error(pred[..., 3], target[..., 3])
        final_pos_error = pos_error[:, -1, :]
        final_yaw_error = yaw_error[:, -1, :]
        traj_pos_error = pos_error.mean(dim=1)
        traj_yaw_error = yaw_error.mean(dim=1)
        representation = action_norm_stats.get("representation") if action_norm_stats else None
        if hasattr(representation, "item"):
            representation = representation.item()
        if representation == "relative_plan_origin":
            # A negative z offset means descending, not crossing the world z=0 plane.
            z_below_zero_rate = torch.zeros_like(final_pos_error)
        else:
            z_below_zero_rate = (pred[..., 2] < 0).float().mean(dim=1)
        success = (final_pos_error < 0.5) & (final_yaw_error < torch.pi / 4)

        rewards = (
            -final_pos_error
            -0.25 * final_yaw_error
            -0.50 * traj_pos_error
            -0.10 * traj_yaw_error
            -2.00 * z_below_zero_rate
        )

        return {
            "rewards": rewards,
            "final_pos_error": final_pos_error,
            "final_yaw_error": final_yaw_error,
            "traj_pos_error": traj_pos_error,
            "traj_yaw_error": traj_yaw_error,
            "z_below_zero_rate": z_below_zero_rate,
            "success": success.float(),
        }


def compute_offline_branch_rewards(
    predicted_actions: torch.Tensor,
    ground_truth_actions: torch.Tensor,
    action_norm_stats: Optional[dict],
) -> Dict[str, float]:
    """
    Computes offline reward diagnostics in real pose units.

    This only logs reward-like metrics. It does not participate in the training loss yet.
    """
    with torch.no_grad():
        reward_tensors = compute_offline_branch_reward_tensors(
            predicted_actions, ground_truth_actions, action_norm_stats
        )
        rewards = reward_tensors["rewards"]
        final_pos_error = reward_tensors["final_pos_error"]
        final_yaw_error = reward_tensors["final_yaw_error"]
        traj_pos_error = reward_tensors["traj_pos_error"]
        traj_yaw_error = reward_tensors["traj_yaw_error"]
        z_below_zero_rate = reward_tensors["z_below_zero_rate"]
        success = reward_tensors["success"]
        best_rewards, best_branches = rewards.max(dim=1)

        reward_metrics = {
            "offline_reward_mean": rewards.mean().item(),
            "offline_reward_best": best_rewards.mean().item(),
            "offline_best_branch_mean": best_branches.float().mean().item(),
            "offline_final_pos_error": final_pos_error.mean().item(),
            "offline_final_yaw_error": final_yaw_error.mean().item(),
            "offline_traj_pos_error": traj_pos_error.mean().item(),
            "offline_traj_yaw_error": traj_yaw_error.mean().item(),
            "offline_z_below_zero_rate": z_below_zero_rate.mean().item(),
            "offline_success_rate": success.float().mean().item(),
        }

        for branch_idx in range(rewards.shape[1]):
            reward_metrics[f"offline_branch{branch_idx}_reward"] = rewards[:, branch_idx].mean().item()
            reward_metrics[f"offline_branch{branch_idx}_final_pos_error"] = final_pos_error[:, branch_idx].mean().item()

        return reward_metrics


def compute_grpo_branch_loss(
    predicted_actions: torch.Tensor,
    ground_truth_actions: torch.Tensor,
    action_norm_stats: Optional[dict],
    advantage_eps: float,
    advantage_clip: float,
    policy_sigma: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes a GRPO-style loss for deterministic continuous action branches.

    Rewards are detached and converted to group-relative advantages across branches. The branch policy surrogate uses
    a fixed-variance Gaussian log-likelihood around the ground-truth normalized action chunk.
    """
    if predicted_actions.ndim != 4:
        raise ValueError("GRPO branch loss requires predicted_actions with shape (B, T, branches, action_dim)")

    reward_tensors = compute_offline_branch_reward_tensors(predicted_actions, ground_truth_actions, action_norm_stats)
    rewards = reward_tensors["rewards"]
    reward_mean = rewards.mean(dim=1, keepdim=True)
    reward_std = rewards.std(dim=1, keepdim=True, unbiased=False)
    advantages = (rewards - reward_mean) / (reward_std + advantage_eps)
    advantages = advantages.clamp(min=-advantage_clip, max=advantage_clip).detach()

    branch_targets = ground_truth_actions.unsqueeze(2).expand_as(predicted_actions)
    per_branch_mse = ((predicted_actions.float() - branch_targets.float()) ** 2).mean(dim=(1, 3))
    gaussian_nll = per_branch_mse / (2.0 * policy_sigma * policy_sigma)
    grpo_loss = (advantages * gaussian_nll).mean()

    best_branch = rewards.argmax(dim=1).float()
    metrics = {
        "grpo_loss": grpo_loss.item(),
        "grpo_advantage_mean": advantages.mean().item(),
        "grpo_advantage_std": advantages.std(unbiased=False).item(),
        "grpo_policy_mse": per_branch_mse.mean().item(),
        "grpo_best_branch_mean": best_branch.mean().item(),
    }
    return grpo_loss, metrics


def compute_gaussian_group_relative_policy_loss(
    action_mean: torch.Tensor,
    action_log_std: torch.Tensor,
    ground_truth_actions: torch.Tensor,
    action_norm_stats: Optional[dict],
    selected_branch_indices: torch.Tensor,
    group_size: int,
    advantage_eps: float,
    advantage_clip: float,
    clip_epsilon: float,
    safety_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """On-policy group-relative objective using exact diagonal-Gaussian log probabilities.

    G samples are drawn independently from the selected policy at every (batch, time)
    slot. G is deliberately independent of K: K is the number of structured
    condition-action alternatives, while G is the policy-optimization sample group.
    """
    if action_mean.ndim != 4 or action_log_std.shape != action_mean.shape:
        raise ValueError("Gaussian group-relative loss requires matching (B,T,K,D) tensors")
    if selected_branch_indices.shape != action_mean.shape[:2]:
        raise ValueError("selected branch indices must have shape (B,T)")
    if group_size < 2:
        raise ValueError("GRPO group_size must be >= 2")
    if advantage_eps <= 0 or advantage_clip <= 0:
        raise ValueError("GRPO advantage constants must be > 0")
    if not 0 < clip_epsilon < 1:
        raise ValueError("GRPO clip_epsilon must lie in (0, 1)")
    if safety_weight < 0:
        raise ValueError("GRPO safety_weight must be >= 0")

    gather_index = selected_branch_indices.detach().long().unsqueeze(2).unsqueeze(3).expand(
        -1, -1, 1, action_mean.shape[-1]
    )
    selected_mean = action_mean.float().gather(2, gather_index).squeeze(2)
    selected_log_std = action_log_std.float().gather(2, gather_index).squeeze(2)

    sample_noise = torch.randn(
        *selected_mean.shape[:2],
        group_size,
        selected_mean.shape[-1],
        device=selected_mean.device,
        dtype=selected_mean.dtype,
    )
    sampled_actions = selected_mean.unsqueeze(2) + torch.exp(selected_log_std).unsqueeze(2) * sample_noise
    policy_samples = sampled_actions.detach()
    expanded_mean = selected_mean.unsqueeze(2).expand_as(policy_samples)
    expanded_log_std = selected_log_std.unsqueeze(2).expand_as(policy_samples)
    log_prob = -diagonal_gaussian_nll(
        expanded_mean,
        expanded_log_std,
        policy_samples,
    ).sum(dim=-1)

    with torch.no_grad():
        sampled_real = _unnormalize_actions_for_reward(policy_samples, action_norm_stats)
        target_real = _unnormalize_actions_for_reward(
            ground_truth_actions.float(), action_norm_stats
        ).unsqueeze(2)
        position_error = torch.linalg.vector_norm(
            sampled_real[..., :3] - target_real[..., :3], dim=-1
        )
        yaw_error = _wrapped_abs_yaw_error(sampled_real[..., 3], target_real[..., 3])
        # Normalized values beyond [-1, 1] leave the robust training envelope.
        safety_violation = torch.relu(policy_samples.abs() - 1.0).mean(dim=-1)
        rewards = -position_error - 0.25 * yaw_error - safety_weight * safety_violation
        reward_mean = rewards.mean(dim=2, keepdim=True)
        reward_std = rewards.std(dim=2, keepdim=True, unbiased=False)
        advantages = (rewards - reward_mean) / (reward_std + advantage_eps)
        advantages = advantages.clamp(min=-advantage_clip, max=advantage_clip)

    old_log_prob = log_prob.detach()
    ratio = torch.exp(log_prob - old_log_prob)
    unclipped_objective = ratio * advantages
    clipped_objective = torch.clamp(
        ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon
    ) * advantages
    policy_loss = -torch.minimum(unclipped_objective, clipped_objective).mean()

    metrics = {
        "grpo_loss": policy_loss.item(),
        # The clipped on-policy surrogate has value ~0 because group advantages
        # are centered, while its gradient is nonzero. This detached proxy is
        # easier to interpret in logs without changing the optimized objective.
        "grpo_logprob_objective_proxy": (-(advantages * log_prob.detach()).mean()).item(),
        "grpo_group_size": float(group_size),
        "grpo_reward_mean": rewards.mean().item(),
        "grpo_reward_best": rewards.max(dim=2).values.mean().item(),
        "grpo_reward_std": rewards.std(unbiased=False).item(),
        "grpo_advantage_mean": advantages.mean().item(),
        "grpo_advantage_std": advantages.std(unbiased=False).item(),
        "grpo_exact_log_prob_mean": log_prob.mean().item(),
        "grpo_probability_ratio_mean": ratio.mean().item(),
        "grpo_position_error_mean": position_error.mean().item(),
        "grpo_yaw_error_mean": yaw_error.mean().item(),
        "grpo_safety_violation_rate": (safety_violation > 0).float().mean().item(),
    }
    return policy_loss, metrics


def _distributed_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _distributed_barrier() -> None:
    if _distributed_is_initialized():
        dist.barrier()


class SingleProcessModuleWrapper(nn.Module):
    """Matches DDP's .module interface when running without torch.distributed."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def select_overfit_batch(batch_idx: int, incoming_batch: dict, fixed_batches: list, batch_count: int):
    """Cache and cycle a small fixed batch set for an explicit memorization diagnostic."""
    if batch_count <= 0:
        return incoming_batch
    if len(fixed_batches) < batch_count:
        fixed_batches.append(incoming_batch)
        return incoming_batch
    return fixed_batches[batch_idx % batch_count]


def set_module_trainable(module: Optional[nn.Module], trainable: bool) -> None:
    if module is None:
        return
    for parameter in module.parameters():
        parameter.requires_grad_(trainable)


@dataclass
class FinetuneConfig:
    # fmt: off
    # 这里执行的时候需要换成实际路径--vla_path /VLM/base-model/openvla-7b
    vla_path: str = "openvla/openvla-7b"             # Path to OpenVLA model (on HuggingFace Hub or stored locally)

    # Dataset
    data_root_dir: Path = Path("datasets/rlds")      # Directory containing RLDS datasets
    dataset_name: str = "aloha_scoop_x_into_bowl"    # Name of fine-tuning dataset (e.g., `aloha_scoop_x_into_bowl`)
    run_root_dir: Path = Path("runs")                # Path to directory to store logs & checkpoints
    shuffle_buffer_size: int = 100_000               # Dataloader shuffle buffer size (can reduce if OOM errors occur)
    relative_action_targets: bool = False            # Predict cumulative pose offsets from the current UAV state
    future_action_stride: int = 1                    # Raw RLDS step spacing between the T future targets
    relative_action_wrap_yaw: bool = False           # False matches PAI-0 DeltaActions (plain yaw subtraction)
    body_delta_action_targets: bool = False           # IndoorUAV one-step body-frame deltas from absolute poses
    cyclic_yaw_proprio: bool = False                 # Model state is [x,y,z,sin(yaw),cos(yaw)] after action conversion

    # Algorithm and architecture
    use_l1_regression: bool = True                   # If True, trains continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, trains continuous action head with diffusion modeling objective (DDIM)
    use_gaussian_action_head: bool = False           # If True, optimize a diagonal Gaussian policy with NLL
    action_regression_loss: str = "l1"               # Point-regression objective: l1 or mse
    gaussian_log_std_min: float = -5.0               # Lower bound for learned Gaussian log standard deviation
    gaussian_log_std_max: float = 1.0                # Upper bound for learned Gaussian log standard deviation
    gaussian_initial_log_std: float = -0.5           # Initial log standard deviation before policy training
    gaussian_learn_log_std: bool = True              # False keeps BC exploration variance fixed and trains only means
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_action_branches: int = 1                     # Number of supervised action branches to predict for L1 regression
    use_best_of_k_action_loss: bool = False          # Assign one winning action branch independently at each future time
    branch_assignment_temperature: float = 0.1       # Soft assignment temperature used by branch balancing
    branch_balance_weight: float = 0.0               # Weight for uniform branch utilization regularization
    condition_assignment_weight: float = 0.0         # Legacy joint assignment; must be 0 for IndoorUAV action-supervised pairing
    initial_action_branch_index: int = -1             # Fixed branch at t=0; -1 keeps unconstrained best-of-K
    use_cond_action_tokens: bool = False             # If True, use explicit T x K COND/ACT placeholder tokens
    couple_condition_to_action_branch: bool = False  # Align the condition that belongs to the winning action branch
    condition_similarity_threshold: float = 0.2      # Threshold used only for condition-alignment diagnostics
    condition_alignment_weight: float = 0.0          # Weight for selected condition/future-image alignment loss
    condition_contrastive_weight: float = 0.0        # Weight for inference-aligned per-time K-way branch selection
    condition_temporal_weight: float = 0.0           # Weight for matching each selected condition to its future time image
    condition_queue_weight: float = 0.0              # Weight for cross-episode image-queue contrastive learning
    condition_queue_size: int = 256                  # Number of detached future-image patch sets retained on device
    condition_queue_min_negatives: int = 32          # Eligible other-episode images required before queue loss starts
    condition_contrastive_temperature: float = 0.07  # Softmax temperature for condition branch selection
    condition_loss_start_time_index: int = 0         # First condition time supervised; use 1 when condition at t=0 is unused
    condition_patch_topk: int = 8                    # Strongest visual-token matches averaged per condition
    condition_diversity_weight: float = 0.0          # Weight for condition branch diversity loss
    condition_diversity_margin: float = 0.05         # Minimum desired cosine distance between condition branches
    use_indoor_uav_condition_adapter: bool = False    # Learned image roles and 512-D condition/vision matching space
    condition_match_dim: int = 512                   # Shared condition/vision matching dimension
    condition_match_hidden_dim: int = 1024           # Hidden width of both matching projectors
    use_indoor_uav_stop_head: bool = False           # Stop after executing the predicted root action
    use_indoor_uav_progress_stop_head: bool = False  # Fuse root ACT/COND and add remaining-step supervision
    stop_head_hidden_dim: int = 1024                 # Hidden width of the root STOP classifier
    stop_progress_projection_dim: int = 512          # Per-token projection before ACT/COND fusion
    stop_progress_loss_weight: float = 0.25          # Weight for <=1/2/4 remaining-step auxiliary BCE
    stop_initial_positive_rate: float = 0.05         # Classifier prior before STOP training
    stop_loss_weight: float = 0.1                    # Weight of terminal BCE in the SFT objective
    stop_positive_weight: float = 0.0                # 0 derives the BCE weight from dataset counts
    stop_threshold: float = 0.5                      # Inference probability threshold
    root_action_weight: float = 1.0                  # SFT weight for slot-0 branch-0 action
    future_action_weight: float = 1.0                # SFT weight for selected future actions
    branch_diversity_weight: float = 0.0             # Weight for multi-branch diversity regularization
    branch_diversity_margin: float = 0.05            # Minimum desired mean L1 distance between action branches
    grpo_reward_weight: float = 0.0                  # Weight for GRPO-style branch reward optimization
    grpo_policy_sigma: float = 1.0                   # Deprecated legacy fixed-sigma setting; retained for old configs
    grpo_group_size: int = 4                         # Policy samples G per selected (time, branch), independent of K
    grpo_clip_epsilon: float = 0.2                   # PPO-style clipping range for exact Gaussian log-prob ratios
    grpo_safety_weight: float = 0.2                  # Penalty for sampled actions outside the normalized data envelope
    grpo_advantage_eps: float = 1e-4                 # Numerical stability constant for group advantage normalization
    grpo_advantage_clip: float = 5.0                 # Clips group-relative advantages before applying GRPO loss
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 1                     # Number of images in the VLA input (default: 1)
    use_image_history: bool = False                  # If True, uses num_images_in_input primary-camera history frames
    require_full_image_history: bool = True          # If True, skips chunks with padded history frames
    use_reference_previous_current: bool = False     # Input roles are [ref_image, previous, current]
    use_proprio: bool = False                        # If True, includes robot proprioceptive state in input

    # Training configuration
    batch_size: int = 8                              # Batch size per device (total batch size = batch_size * num GPUs)
    learning_rate: float = 5e-4                      # Learning rate
    lr_warmup_steps: int = 0                         # Number of steps to warm up learning rate (from 10% to 100%)
    num_steps_before_decay: int = 100_000            # Number of steps before LR decays by 10x
    grad_accumulation_steps: int = 1                 # Number of gradient accumulation steps
    max_grad_norm: Optional[float] = None             # Global gradient clipping threshold; disabled when None
    seed: int = 17                                   # Shared RNG seed for reproducible A/B experiments
    max_steps: int = 200_000                         # Max number of training steps
    use_val_set: bool = False                        # If True, uses validation set and log validation metrics
    train_tfds_split: Optional[str] = None            # Explicit training split, e.g. train[:95%]
    val_tfds_split: Optional[str] = None              # Explicit validation split, e.g. train[95%:]
    val_freq: int = 10_000                           # (When `use_val_set==True`) Validation set logging frequency in steps
    val_time_limit: int = 180                        # (When `use_val_set==True`) Time limit for computing validation metrics
    val_max_batches: int = 0                         # 0 disables; otherwise cap validation at this exact batch count
    save_freq: int = 10_000                          # Checkpoint saving frequency in steps
    save_latest_checkpoint_only: bool = False        # If True, saves only 1 checkpoint, overwriting latest checkpoint
                                                     #   (If False, saves all checkpoints)
    resume: bool = False                             # If True, resumes from checkpoint 断点重训，从checkpoint继续训练
    resume_step: Optional[int] = None                # (When `resume==True`) Step number that we are resuming from
    auxiliary_init_checkpoint_path: Optional[Path] = None  # Load external projectors/heads without resuming LoRA
    auxiliary_init_checkpoint_step: Optional[int] = None   # Component step inside auxiliary_init_checkpoint_path
    reset_action_head: bool = False                  # Do not load action_head from auxiliary initialization checkpoint
    reset_proprio_projector: bool = False            # Reinitialize projector when the proprio representation changes
    image_aug: bool = True                           # If True, trains with image augmentations (HIGHLY RECOMMENDED)
    diffusion_sample_freq: int = 50                  # (When `use_diffusion==True`) Frequency for sampling in steps

    # LoRA
    use_lora: bool = True                            # If True, uses LoRA fine-tuning
    lora_rank: int = 32                              # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                        # Dropout applied to LoRA weights
    merge_lora_during_training: bool = True          # If True, merges LoRA weights and saves result during training
                                                     #   Note: Merging can be very slow on some machines. If so, set to
                                                     #         False and merge final checkpoint offline!

    # WandB ≈ 深度学习版 TensorBoard + 实验管理系统 + 云端仪表盘。这里需要自己的wandb账号
    # Logging
    wandb_entity: str = "3244403140"          # Name of WandB entity
    wandb_project: str = "openvla-uav"        # Name of WandB project
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    run_id_override: Optional[str] = None            # Optional string to override the run ID with
    wandb_log_freq: int = 10                         # WandB logging frequency in steps
    train_report_freq: int = 50                      # Local stdout diagnostic frequency in optimizer steps
    debug_batch_shapes: bool = False                 # If True, print batch/action/mask shapes for initial batches
    debug_grad_norm: bool = False                    # If True, print gradient norms for trainable components
    debug_num_batches: int = 2                       # Number of initial batches to print when debug flags are enabled
    overfit_fixed_batch_count: int = 0               # If > 0, repeatedly train on the first N batches (diagnostic only)
    overfit_report_freq: int = 25                    # Step interval for fixed-batch overfit loss reports
    freeze_vla: bool = False                         # Freeze the VLA backbone and any attached LoRA parameters
    freeze_proprio_projector: bool = False           # Freeze the proprio projector while training other components
    freeze_action_head: bool = False                 # Freeze continuous actions while fitting auxiliary heads
    freeze_condition_adapter: bool = False           # Freeze image roles and condition projectors

    # fmt: on

# 去掉 DDP 自动添加的 "module." 前缀。DDP:Distributed Data Parallel
def remove_ddp_in_checkpoint(state_dict) -> dict:
    """
    Removes the 'module.' prefix from parameter names in a PyTorch model state dictionary that was saved using
    DistributedDataParallel (DDP).

    When a model is trained using PyTorch's DistributedDataParallel, the saved state dictionary contains parameters
    prefixed with 'module.'. This function removes these prefixes to make the state dictionary compatible when
    loading into models that are not yet wrapped in DDP.

    Args:
        state_dict (dict): PyTorch model state dictionary.

    Returns:
        dict: A new state dictionary with the same contents but with 'module.' prefixes removed from parameter names.
              Parameters without the 'module.' prefix remain unchanged.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k[:7] == "module.":
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


# 根据配置自动生成实验名字
def get_run_id(cfg) -> str:
    """
    Generates or retrieves an identifier string for an experiment run.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        str: Experiment run ID.
    """
    if cfg.run_id_override is not None:
        # Override the run ID with the user-provided ID
        run_id = cfg.run_id_override
    elif cfg.resume:
        # Override run ID with the previous resumed run's ID
        run_id = cfg.vla_path.split("/")[-1]
        # Remove the "--XXX_chkpt" suffix from the run ID if it exists
        if "chkpt" in run_id.split("--")[-1]:
            run_id = "--".join(run_id.split("--")[:-1])
    else:
        run_id = (
            f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
            f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
            f"+lr-{cfg.learning_rate}"
        )
        if cfg.use_lora:
            run_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
        if cfg.image_aug:
            run_id += "--image_aug"
        if cfg.run_id_note is not None:
            run_id += f"--{cfg.run_id_note}"
    return run_id


# 加载 checkpoint
def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    """
    Loads a checkpoint for a given module.

    Args:
        module_name (str): Name of model component to load checkpoint for.
        path (str): Path to checkpoint directory.
        step (int): Gradient step number of saved checkpoint.
        device (str): String specifying how to remap storage locations (default = "cpu").

    Returns:
        dict: PyTorch model state dictionary.
    """
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)


# 把模型包装成多 GPU 模型
def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> DDP:
    """
    Wrap a module with DistributedDataParallel.

    Args:
        module (nn.Module): PyTorch module.
        device_id (str): Device ID.
        find_unused (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        DistributedDataParallel: PyTorch module wrapped with DDP.
    """
    if not _distributed_is_initialized():
        return SingleProcessModuleWrapper(module)
    return DDP(module, device_ids=[device_id], find_unused_parameters=find_unused, gradient_as_bucket_view=True)


# 统计可训练参数数量
def count_parameters(module: nn.Module, name: str) -> None:
    """
    Counts and prints the number of trainable parameters in a module.

    Args:
        module (nn.Module): PyTorch module.
        module_name (str): Name of model component.

    Returns:
        None.
    """
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")


def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: FinetuneConfig,
    device_id: int,
    module_args: dict,
    to_bf16: bool = False,
    find_unused_params: bool = False,
    allow_missing_auxiliary: bool = False,
) -> DDP:
    """
    Initializes a module, optionally loads checkpoint, moves to device, and wraps with DDP.

    Args:
        module_class (Type[nn.Module]): Class of PyTorch module to initialize.
        module_name (str): Name of model component to load checkpoint for.
        cfg (FinetuneConfig): Training configuration.
        device_id (str): Device ID.
        module_args (dict): Args for initializing the module.
        to_bf16 (bool): Whether to convert to torch.bfloat16 data type.
        find_unused_params (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        DistributedDataParallel: PyTorch module wrapped with DDP.
    """
    module = module_class(**module_args)
    count_parameters(module, module_name)

    if cfg.resume:
        state_dict = load_checkpoint(module_name, cfg.vla_path, cfg.resume_step)
        module.load_state_dict(state_dict)
        print(f"Initialized {module_name} from resume checkpoint step {cfg.resume_step}")
    elif cfg.auxiliary_init_checkpoint_path is not None:
        reset_from_auxiliary = (
            (module_name == "action_head" and cfg.reset_action_head)
            or (module_name == "proprio_projector" and cfg.reset_proprio_projector)
        )
        if reset_from_auxiliary:
            print(f"Resetting {module_name}; auxiliary weights were intentionally not loaded")
        else:
            try:
                state_dict = load_checkpoint(
                    module_name,
                    str(cfg.auxiliary_init_checkpoint_path),
                    cfg.auxiliary_init_checkpoint_step,
                )
            except FileNotFoundError:
                if not allow_missing_auxiliary:
                    raise
                print(f"Initializing new {module_name}; no component exists in the auxiliary checkpoint")
            else:
                module.load_state_dict(state_dict)
                print(
                    f"Initialized {module_name} from auxiliary checkpoint "
                    f"{cfg.auxiliary_init_checkpoint_path} step {cfg.auxiliary_init_checkpoint_step}"
                )

    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)

    return wrap_ddp(module, device_id, find_unused_params)

# 它把一个 batch 喂给 VLA，拿到 hidden states，再用 action head 预测动作，最后算 loss 和日志指标
def run_forward_pass(
    vla,
    action_head,
    condition_adapter,
    stop_head,
    noisy_action_projector,
    proprio_projector,
    batch,
    action_tokenizer,
    device_id,
    use_l1_regression,
    use_diffusion,
    use_gaussian_action_head,
    action_regression_loss,
    num_action_branches,
    use_best_of_k_action_loss,
    branch_assignment_temperature,
    branch_balance_weight,
    condition_assignment_weight,
    initial_action_branch_index,
    branch_diversity_weight,
    branch_diversity_margin,
    grpo_reward_weight,
    grpo_policy_sigma,
    grpo_group_size,
    grpo_clip_epsilon,
    grpo_safety_weight,
    grpo_advantage_eps,
    grpo_advantage_clip,
    use_proprio,
    use_film,
    num_patches,
    action_norm_stats=None,
    use_cond_action_tokens=False,
    cond_token_ids=None,
    act_token_ids=None,
    couple_condition_to_action_branch=False,
    condition_similarity_threshold=0.2,
    condition_alignment_weight=0.0,
    condition_contrastive_weight=0.0,
    condition_temporal_weight=0.0,
    condition_queue_weight=0.0,
    condition_queue_min_negatives=1,
    condition_negative_queue: Optional[CrossEpisodeImageQueue] = None,
    condition_contrastive_temperature=0.07,
    condition_loss_start_time_index=0,
    condition_patch_topk=8,
    condition_diversity_weight=0.0,
    condition_diversity_margin=0.05,
    root_action_weight=1.0,
    future_action_weight=1.0,
    stop_loss_weight=0.0,
    stop_positive_weight=1.0,
    stop_threshold=0.5,
    use_progress_stop_head=False,
    stop_progress_positive_weights=None,
    stop_progress_loss_weight=0.0,
    compute_diffusion_l1=False,
    num_diffusion_steps_train=None,
    debug_batch_shapes=False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute model forward pass and metrics for both training and validation.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        batch (dict): Input batch.
        action_tokenizer (ActionTokenizer): Action tokenizer.
        device_id (str): Device ID.
        use_l1_regression (bool): Whether to use L1 regression.
        use_diffusion (bool): Whether to use diffusion.
        use_proprio (bool): Whether to use proprioceptive state as input.
        use_film (bool): Whether to use FiLM for better language following.
        num_patches (int): Number of vision patches.
        compute_diffusion_l1 (bool): Whether to sample actions and compute L1 loss for diffusion (do this once every
                                    diffusion_sample_freq steps during training; do it every batch for validation)
        num_diffusion_steps_train (int): Number of diffusion steps for training (only used for diffusion).

    Returns:
        tuple: (loss, metrics_dict)
            loss: The loss tensor with gradient for backpropagation.
            metrics_dict: Dictionary of computed metrics (detached values for logging).
    """
    metrics = {}

    # Get ground-truth action labels
    input_ids = batch["input_ids"].to(device_id)
    attention_mask = batch["attention_mask"].to(device_id)
    pixel_values = batch["pixel_values"].to(torch.bfloat16).to(device_id)
    future_pixel_values = batch.get("future_pixel_values")
    if future_pixel_values is not None:
        future_pixel_values = future_pixel_values.to(torch.bfloat16).to(device_id)
    ground_truth_actions = batch["actions"].to(device_id).to(torch.bfloat16)
    image_valid_mask = batch.get("image_valid_mask")
    if image_valid_mask is not None:
        image_valid_mask = image_valid_mask.to(device_id)
    plan_valid_mask = batch.get("plan_valid_mask")
    if plan_valid_mask is not None:
        plan_valid_mask = plan_valid_mask.to(device_id)
    stop_after_action = batch.get("stop_after_action")
    if stop_after_action is not None:
        stop_after_action = stop_after_action.to(device_id)
    actions_remaining_after_root = batch.get("actions_remaining_after_root")
    if actions_remaining_after_root is not None:
        actions_remaining_after_root = actions_remaining_after_root.to(device_id)
    proprio = batch["proprio"].to(device_id).to(torch.bfloat16) if use_proprio else None
    labels = batch["labels"].to(device_id)
    debug_info = {}
    condition_alignment_loss = None
    condition_contrastive_loss = None
    condition_diversity_loss = None
    condition_similarities = None
    future_patch_embeddings = None
    action_winner_indices = None

    # [Only for diffusion] Sample noisy actions used as input for noise predictor network. 如果使用diffusion，先给动作加噪声
    if use_diffusion:
        noisy_dict = action_head.module.sample_noisy_actions(ground_truth_actions)
        noise, noisy_actions, diffusion_timestep_embeddings = (
            noisy_dict["noise"],
            noisy_dict["noisy_actions"],
            noisy_dict["diffusion_timestep_embeddings"],
        )
    else:
        noise, noisy_actions, diffusion_timestep_embeddings = None, None, None

    # VLA forward pass 前向传播，就是把图像，语言指令，机器人状态，动作等输入到VLA模型中，得到输出
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output: CausalLMOutputWithPast = vla(
            input_ids=input_ids,     #文本token，包括 prompt 和动作占位 token
            attention_mask=attention_mask,       # 哪些 token 有效
            pixel_values=pixel_values,        # 图像特征
            labels=labels,         # 语言模型训练时的动作 token label，里面包含动作token位置
            output_hidden_states=True,      #因为后面不是只要output.logits，而是要拿LLM最后一层hidden states去预测动作，所以要设置为True
            proprio=proprio,      # 机器人本体状态
            proprio_projector=proprio_projector if use_proprio else None,
            noisy_actions=noisy_actions if use_diffusion else None,
            noisy_action_projector=noisy_action_projector if use_diffusion else None,
            diffusion_timestep_embeddings=diffusion_timestep_embeddings if use_diffusion else None,
            use_film=use_film,
            image_valid_mask=image_valid_mask,
            image_role_embeddings=(
                (condition_adapter.module if hasattr(condition_adapter, "module") else condition_adapter)
                .image_role_embeddings
                if condition_adapter is not None
                else None
            ),
        )

    # Get action masks needed for logging，找到哪些token对应当前动作，哪些token位置对应未来动作，生成action masks
    ground_truth_token_ids = labels[:, 1:]
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)
    shifted_input_ids = input_ids[:, 1:]
    if debug_batch_shapes:
        debug_info.update(
            {
                "input_ids": _shape(input_ids),
                "input_ids_device": _device(input_ids),
                "attention_mask": _shape(attention_mask),
                "attention_mask_device": _device(attention_mask),
                "pixel_values": _shape(pixel_values),
                "pixel_values_device": _device(pixel_values),
                "future_pixel_values": _shape(future_pixel_values),
                "future_pixel_values_device": _device(future_pixel_values),
                "proprio": _shape(proprio),
                "proprio_device": _device(proprio),
                "labels": _shape(labels),
                "labels_device": _device(labels),
                "ground_truth_actions": _shape(ground_truth_actions),
                "ground_truth_actions_device": _device(ground_truth_actions),
                "image_history_pad_mask": (
                    batch["image_history_pad_mask"].tolist() if "image_history_pad_mask" in batch else "None"
                ),
                "image_valid_mask": image_valid_mask.tolist() if image_valid_mask is not None else "None",
                "plan_valid_mask": plan_valid_mask.tolist() if plan_valid_mask is not None else "None",
                "stop_after_action": (
                    stop_after_action.tolist() if stop_after_action is not None else "None"
                ),
                "actions_remaining_after_root": (
                    actions_remaining_after_root.tolist()
                    if actions_remaining_after_root is not None
                    else "None"
                ),
                "current_action_mask_sum": int(current_action_mask.sum().item()),
                "current_action_mask_device": _device(current_action_mask),
                "next_actions_mask_sum": int(next_actions_mask.sum().item()),
                "next_actions_mask_device": _device(next_actions_mask),
                "num_patches": int(num_patches),
            }
        )

    # Compute metrics for discrete action representation (next-token prediction)
    if not (use_l1_regression or use_diffusion):
        loss = output.loss
        predicted_token_ids = output.logits[:, num_patches:-1].argmax(dim=2)
        curr_action_accuracy = compute_token_accuracy(
            predicted_token_ids, ground_truth_token_ids, mask=current_action_mask
        )
        curr_action_l1_loss = compute_actions_l1_loss(
            action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=current_action_mask
        )
        next_actions_accuracy = compute_token_accuracy(
            predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask
        )
        next_actions_l1_loss = compute_actions_l1_loss(
            action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask
        )
        metrics.update(
            {
                "loss_value": loss.item(),  # Detached value for logging
                "curr_action_accuracy": curr_action_accuracy.item(),
                "curr_action_l1_loss": curr_action_l1_loss.item(),
                "next_actions_accuracy": next_actions_accuracy.item(),
                "next_actions_l1_loss": next_actions_l1_loss.item(),
            }
        )
    # Compute metrics for continuous action representations (L1 regression | diffusion)
    else:
        # Get last layer hidden states
        last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
        # Get hidden states for text portion of prompt+response (after the vision patches)
        text_hidden_states = last_hidden_states[:, num_patches:-1]
        # Get hidden states for action portion of response
        batch_size = input_ids.shape[0]
        if use_cond_action_tokens:
            cond_hidden_states, actions_hidden_states, format_metrics = gather_cond_action_hidden_states(
                text_hidden_states=text_hidden_states,
                shifted_input_ids=shifted_input_ids,
                cond_token_ids=cond_token_ids,
                act_token_ids=act_token_ids,
                num_action_branches=num_action_branches,
            )
            metrics.update(format_metrics)
            actions_hidden_states = actions_hidden_states.to(torch.bfloat16)
        else:
            cond_hidden_states = None
            actions_hidden_states = (
                text_hidden_states[current_action_mask | next_actions_mask]
                .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
                .to(torch.bfloat16)
            )  # (B, act_chunk_len, D)=(B,56,D)        act_chunk_len=NUM_ACTIONS_CHUNK * ACTION_DIM=8*7=56
        if debug_batch_shapes:
            debug_info.update(
                {
                    "last_hidden_states": _shape(last_hidden_states),
                    "last_hidden_states_device": _device(last_hidden_states),
                    "text_hidden_states": _shape(text_hidden_states),
                    "text_hidden_states_device": _device(text_hidden_states),
                    "cond_hidden_states": _shape(cond_hidden_states),
                    "actions_hidden_states": _shape(actions_hidden_states),
                    "actions_hidden_states_device": _device(actions_hidden_states),
                }
            )

        if use_l1_regression:
            if use_gaussian_action_head:
                predicted_actions, action_log_std = action_head.module.predict_distribution(actions_hidden_states)
            else:
                predicted_actions = action_head.module.predict_action(actions_hidden_states)
                action_log_std = None
            if debug_batch_shapes:
                debug_info["predicted_actions"] = _shape(predicted_actions)
                debug_info["predicted_actions_device"] = _device(predicted_actions)
                debug_info["action_log_std"] = _shape(action_log_std)
                debug_info["action_log_std_device"] = _device(action_log_std)

            projected_conditions = None
            topk_patch_indices = None
            if condition_adapter is not None:
                if cond_hidden_states is None or future_pixel_values is None or plan_valid_mask is None:
                    raise ValueError(
                        "IndoorUAV condition SFT requires COND tokens, future images, and plan_valid_mask"
                    )
                (
                    projected_conditions,
                    future_patch_embeddings,
                    condition_similarities,
                    topk_patch_indices,
                ) = compute_projected_condition_targets(
                    vla=vla,
                    condition_adapter=condition_adapter,
                    cond_hidden_states=cond_hidden_states,
                    future_pixel_values=future_pixel_values,
                    patch_topk=condition_patch_topk,
                )
                metrics.update(
                    {
                        "condition_similarity_mean": condition_similarities.mean().item(),
                        "condition_patch_topk": float(condition_patch_topk),
                        "condition_match_dim": float(projected_conditions.shape[-1]),
                    }
                )
                if debug_batch_shapes:
                    debug_info.update(
                        {
                            "projected_conditions": _shape(projected_conditions),
                            "future_patch_embeddings": _shape(future_patch_embeddings),
                            "condition_similarities": _shape(condition_similarities),
                            "topk_patch_indices": _shape(topk_patch_indices),
                        }
                    )
            elif cond_hidden_states is not None and future_pixel_values is not None:
                (
                    condition_similarities,
                    future_patch_embeddings,
                    condition_precompute_metrics,
                ) = compute_condition_similarity_tensors(
                    vla=vla,
                    cond_hidden_states=cond_hidden_states,
                    future_pixel_values=future_pixel_values,
                    patch_topk=condition_patch_topk,
                    use_film=use_film,
                )
                metrics.update(condition_precompute_metrics)

            if condition_adapter is not None:
                queued_image_patches, queued_episode_ids = (
                    condition_negative_queue.entries()
                    if condition_negative_queue is not None
                    else (None, [])
                )
                loss, future_winners, assignment_metrics = compute_indoor_uav_sft_loss(
                    predicted_actions=predicted_actions,
                    ground_truth_actions=ground_truth_actions,
                    projected_conditions=projected_conditions,
                    future_patch_embeddings=future_patch_embeddings,
                    condition_similarities=condition_similarities,
                    plan_valid_mask=plan_valid_mask,
                    assignment_temperature=branch_assignment_temperature,
                    root_action_weight=root_action_weight,
                    future_action_weight=future_action_weight,
                    condition_alignment_weight=condition_alignment_weight,
                    condition_contrastive_weight=condition_contrastive_weight,
                    condition_temporal_weight=condition_temporal_weight,
                    condition_queue_weight=condition_queue_weight,
                    condition_contrastive_temperature=condition_contrastive_temperature,
                    branch_balance_weight=branch_balance_weight,
                    condition_diversity_weight=condition_diversity_weight,
                    condition_diversity_margin=condition_diversity_margin,
                    patch_topk=condition_patch_topk,
                    query_episode_ids=batch.get("episode_ids"),
                    queued_image_patches=queued_image_patches,
                    queued_episode_ids=queued_episode_ids,
                    condition_queue_min_negatives=condition_queue_min_negatives,
                )
                assignment_metrics.update(
                    compute_condition_selected_action_metrics(
                        predicted_actions=predicted_actions,
                        ground_truth_actions=ground_truth_actions,
                        condition_similarities=condition_similarities,
                        plan_valid_mask=plan_valid_mask,
                        action_norm_stats=action_norm_stats,
                    )
                )
                if condition_negative_queue is not None:
                    valid_future_mask = plan_valid_mask[:, 1:].bool()
                    current_episode_ids = _valid_future_episode_ids(
                        batch.get("episode_ids"), valid_future_mask
                    )
                    condition_negative_queue.enqueue(
                        future_patch_embeddings[valid_future_mask], current_episode_ids
                    )
                    assignment_metrics["condition_queue_size"] = float(len(condition_negative_queue))
                action_winner_indices = torch.cat(
                    [
                        torch.zeros(
                            future_winners.shape[0],
                            1,
                            dtype=future_winners.dtype,
                            device=future_winners.device,
                        ),
                        future_winners,
                    ],
                    dim=1,
                )
                metrics.update(assignment_metrics)
            elif (
                predicted_actions.ndim == 4
                and predicted_actions.shape[2] == 1
                and plan_valid_mask is not None
                and use_cond_action_tokens
            ):
                loss, single_branch_metrics = compute_masked_single_branch_sft_loss(
                    predicted_actions,
                    ground_truth_actions,
                    plan_valid_mask,
                    root_action_weight=root_action_weight,
                    future_action_weight=future_action_weight,
                )
                action_winner_indices = torch.zeros(
                    predicted_actions.shape[:2], dtype=torch.long, device=predicted_actions.device
                )
                metrics.update(single_branch_metrics)
            elif use_gaussian_action_head and predicted_actions.ndim == 4 and use_best_of_k_action_loss:
                (
                    loss,
                    branch_balance_loss,
                    action_winner_indices,
                    assignment_metrics,
                ) = compute_best_of_k_gaussian_action_loss(
                    predicted_actions,
                    action_log_std,
                    ground_truth_actions,
                    branch_assignment_temperature,
                    condition_similarities=condition_similarities,
                    condition_assignment_weight=condition_assignment_weight,
                    condition_loss_start_time_index=condition_loss_start_time_index,
                    initial_action_branch_index=(
                        initial_action_branch_index if initial_action_branch_index >= 0 else None
                    ),
                )
                loss = loss + branch_balance_weight * branch_balance_loss
                metrics.update(assignment_metrics)
            elif use_gaussian_action_head:
                ground_truth_actions_for_loss = match_action_target_shape(
                    predicted_actions, ground_truth_actions
                )
                gaussian_nll = diagonal_gaussian_nll(
                    predicted_actions,
                    action_log_std,
                    ground_truth_actions_for_loss,
                ).mean()
                loss = gaussian_nll
                metrics["gaussian_nll_loss"] = gaussian_nll.item()
            # Assign one expert target to one deterministic branch at each future time.
            elif predicted_actions.ndim == 4 and use_best_of_k_action_loss:
                (
                    loss,
                    branch_balance_loss,
                    action_winner_indices,
                    assignment_metrics,
                ) = compute_best_of_k_action_loss(
                    predicted_actions,
                    ground_truth_actions,
                    branch_assignment_temperature,
                )
                loss = loss + branch_balance_weight * branch_balance_loss
                metrics.update(assignment_metrics)
            else:
                loss = compute_action_regression_loss(
                    predicted_actions,
                    ground_truth_actions,
                    action_regression_loss,
                )

            if action_log_std is not None:
                metrics.update(
                    {
                        "gaussian_log_std_mean": action_log_std.float().mean().item(),
                        "gaussian_log_std_min": action_log_std.float().min().item(),
                        "gaussian_log_std_max": action_log_std.float().max().item(),
                        "gaussian_policy_std_mean": torch.exp(action_log_std.float()).mean().item(),
                    }
                )

            if condition_adapter is None and cond_hidden_states is not None and future_pixel_values is not None:
                selected_condition_branches = (
                    action_winner_indices if couple_condition_to_action_branch else None
                )
                (
                    condition_alignment_loss,
                    condition_diversity_loss,
                    condition_contrastive_loss,
                    condition_metrics,
                ) = compute_condition_alignment_loss_and_metrics(
                    vla=vla,
                    cond_hidden_states=cond_hidden_states,
                    future_pixel_values=future_pixel_values,
                    similarity_threshold=condition_similarity_threshold,
                    diversity_margin=condition_diversity_margin,
                    selected_branch_indices=selected_condition_branches,
                    contrastive_temperature=condition_contrastive_temperature,
                    loss_start_time_index=condition_loss_start_time_index,
                    patch_topk=condition_patch_topk,
                    use_film=use_film,
                    precomputed_similarities=condition_similarities,
                    precomputed_future_patch_embeddings=future_patch_embeddings,
                )
                metrics.update(condition_metrics)
            if condition_alignment_loss is not None and condition_alignment_weight > 0:
                loss = loss + condition_alignment_weight * condition_alignment_loss
            if condition_contrastive_loss is not None and condition_contrastive_weight > 0:
                loss = loss + condition_contrastive_weight * condition_contrastive_loss
            if condition_diversity_loss is not None and condition_diversity_weight > 0:
                loss = loss + condition_diversity_weight * condition_diversity_loss
            if condition_adapter is None and predicted_actions.ndim == 4 and branch_diversity_weight > 0:
                branch_pair_distances = []
                for left_branch in range(predicted_actions.shape[2]):
                    for right_branch in range(left_branch + 1, predicted_actions.shape[2]):
                        branch_pair_distances.append(
                            torch.abs(
                                predicted_actions[:, :, left_branch] - predicted_actions[:, :, right_branch]
                            ).mean(dim=-1)
                        )
                branch_pair_distances = torch.stack(branch_pair_distances, dim=2)
                branch_mean_distance = branch_pair_distances.mean()
                branch_diversity_loss = torch.relu(branch_diversity_margin - branch_pair_distances).mean()
                loss = loss + branch_diversity_weight * branch_diversity_loss
                metrics.update(
                    {
                        "branch_diversity_loss": branch_diversity_loss.item(),
                        "branch_mean_distance": branch_mean_distance.item(),
                    }
                )
            if condition_adapter is None and predicted_actions.ndim == 4 and grpo_reward_weight > 0:
                if action_log_std is None or action_winner_indices is None:
                    raise ValueError("Gaussian GRPO requires a selected branch and action log standard deviation")
                grpo_loss, grpo_metrics = compute_gaussian_group_relative_policy_loss(
                    action_mean=predicted_actions,
                    action_log_std=action_log_std,
                    ground_truth_actions=ground_truth_actions,
                    action_norm_stats=action_norm_stats,
                    selected_branch_indices=action_winner_indices,
                    group_size=grpo_group_size,
                    advantage_eps=grpo_advantage_eps,
                    advantage_clip=grpo_advantage_clip,
                    clip_epsilon=grpo_clip_epsilon,
                    safety_weight=grpo_safety_weight,
                )
                loss = loss + grpo_reward_weight * grpo_loss
                metrics.update(grpo_metrics)

            if stop_head is not None:
                if stop_after_action is None:
                    raise ValueError("IndoorUAV STOP training requires stop_after_action labels")
                if not use_cond_action_tokens or actions_hidden_states.ndim != 4:
                    raise ValueError("IndoorUAV STOP head requires (B,T,K,D) ACT hidden states")
                if use_progress_stop_head:
                    if cond_hidden_states is None or actions_remaining_after_root is None:
                        raise ValueError("progress STOP requires root COND states and remaining-step labels")
                    stop_logits = stop_head.module(
                        actions_hidden_states[:, 0, 0].float(),
                        cond_hidden_states[:, 0, 0].float(),
                    )
                    stop_loss, stop_metrics = compute_progress_stop_loss(
                        stop_logits,
                        actions_remaining_after_root,
                        stop_progress_positive_weights,
                        stop_progress_loss_weight,
                        threshold=stop_threshold,
                    )
                else:
                    stop_logits = stop_head.module(actions_hidden_states[:, 0, 0].float())
                    stop_loss, stop_metrics = compute_stop_after_action_loss(
                        stop_logits,
                        stop_after_action,
                        stop_positive_weight,
                        threshold=stop_threshold,
                    )
                loss = loss + stop_loss_weight * stop_loss
                metrics.update(stop_metrics)

        if use_diffusion:
            # Predict noise
            noise_pred = action_head.module.predict_noise(actions_hidden_states)
            # Get diffusion noise prediction MSE loss  模型要对 8 步 action chunk 的每一维都预测噪声,用 MSE 训练噪声预测
            noise_pred = noise_pred.reshape(noise.shape)
            loss = nn.functional.mse_loss(noise_pred, noise, reduction="mean")

            # Only sample actions and compute L1 losses if specified,是为了额外评估，因为 diffusion 训练时是预测噪声的 MSE loss，而不是直接预测动作的 L1 loss，所以要额外采样动作来计算 L1 loss
            if compute_diffusion_l1:
                #因为这里不是训练，而是评估，所以不需要梯度计算，节省显存
                with torch.no_grad():
                    predicted_actions = run_diffusion_sampling(
                        vla=vla,
                        action_head=action_head,
                        noisy_action_projector=noisy_action_projector,
                        proprio_projector=proprio_projector,
                        batch=batch,
                        batch_size=batch_size,
                        num_patches=num_patches,
                        actions_shape=ground_truth_actions.shape,
                        device_id=device_id,
                        current_action_mask=current_action_mask,
                        next_actions_mask=next_actions_mask,
                        use_proprio=use_proprio,
                        use_film=use_film,
                    )

        metrics.update(
            {
                "loss_value": loss.item(),  # Detached value for logging
            }
        )
        if debug_batch_shapes:
            metrics.update({f"debug_{key}": value for key, value in debug_info.items()})

        # Get detailed L1 losses for logging
        should_log_l1_loss = not use_diffusion or (use_diffusion and compute_diffusion_l1)
        if should_log_l1_loss:
            if predicted_actions.ndim == 4 and action_winner_indices is not None:
                predicted_actions_for_metrics = predicted_actions.gather(
                    2,
                    action_winner_indices.unsqueeze(2).unsqueeze(3).expand(
                        -1, -1, 1, predicted_actions.shape[-1]
                    ),
                ).squeeze(2)
            elif predicted_actions.ndim == 4:
                predicted_actions_for_metrics = predicted_actions[:, :, 0]
            else:
                predicted_actions_for_metrics = predicted_actions
            #分开的原因是：第一步动作最直接影响当前控制，未来动作更多是为了规划，当前动作更重要，所以单独算
            ground_truth_curr_action = ground_truth_actions[:, 0]
            predicted_curr_action = predicted_actions_for_metrics[:, 0]
            ground_truth_next_actions = ground_truth_actions[:, 1:]
            predicted_next_actions = predicted_actions_for_metrics[:, 1:]
            curr_action_l1_loss = torch.nn.L1Loss()(ground_truth_curr_action, predicted_curr_action)
            next_actions_l1_loss = torch.nn.L1Loss()(ground_truth_next_actions, predicted_next_actions)
            l1_metrics = {
                "curr_action_l1_loss": curr_action_l1_loss.item(),
                "next_actions_l1_loss": next_actions_l1_loss.item(),
            }
            l1_metrics.update(
                compute_root_action_axis_metrics(
                    predicted_actions,
                    ground_truth_actions,
                    action_norm_stats,
                )
            )
            if predicted_actions.ndim == 4:
                branch_targets = ground_truth_actions.unsqueeze(2).expand_as(predicted_actions)
                per_branch_l1 = torch.abs(predicted_actions - branch_targets).mean(dim=(1, 3))
                per_time_branch_l1 = torch.abs(predicted_actions - branch_targets).mean(dim=3)
                l1_metrics["all_branches_l1_loss"] = per_branch_l1.mean().item()
                l1_metrics["best_branch_l1_loss"] = per_branch_l1.min(dim=1).values.mean().item()
                l1_metrics["best_of_k_time_l1_loss"] = per_time_branch_l1.min(dim=2).values.mean().item()
                if predicted_actions.shape[2] > 1 and plan_valid_mask is not None:
                    real_actions = _unnormalize_actions_for_reward(
                        predicted_actions.detach().float(), action_norm_stats
                    )[:, 1:]
                    position_separations = []
                    yaw_separations = []
                    for left in range(predicted_actions.shape[2]):
                        for right in range(left + 1, predicted_actions.shape[2]):
                            position_separations.append(
                                torch.linalg.vector_norm(
                                    real_actions[:, :, left, :3] - real_actions[:, :, right, :3],
                                    dim=-1,
                                )
                            )
                            yaw_separations.append(
                                _wrapped_abs_yaw_error(
                                    real_actions[:, :, left, 3], real_actions[:, :, right, 3]
                                )
                            )
                    future_mask = plan_valid_mask[:, 1:].bool()
                    position_separations = torch.stack(position_separations, dim=-1)
                    yaw_separations = torch.stack(yaw_separations, dim=-1)
                    l1_metrics["action_branch_position_separation_m"] = _masked_mean(
                        position_separations.mean(dim=-1), future_mask
                    ).item()
                    l1_metrics["action_branch_yaw_separation_rad"] = _masked_mean(
                        yaw_separations.mean(dim=-1), future_mask
                    ).item()
                l1_metrics.update(
                    compute_offline_branch_rewards(predicted_actions, ground_truth_actions, action_norm_stats)
                )
            metrics.update(l1_metrics)

    # Return both the loss tensor (with gradients) and the metrics dictionary (with detached values)，其中loss是用来反向传播的，metrics是用来记录日志的
    return loss, metrics

# 从一团随机噪声动作开始，经过多次反向去噪，生成最终动作chunk
# L1只需要一次前向传播就可以得到动作预测，而diffusion需要多次前向传播，每次都要把上一步的噪声动作输入进去，经过VLA和action head预测噪声，然后再去噪，直到最后得到最终动作
# 理论上只多次进入动作头也可以，但是openVLA-oft想让每次diffusion timestep都重新和observation做condition，所以每次都要把observation输入VLA，得到新的hidden states，再去预测噪声
def run_diffusion_sampling(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    batch,
    batch_size,
    num_patches,
    actions_shape,
    device_id,
    current_action_mask,
    next_actions_mask,
    use_proprio,
    use_film,
) -> torch.Tensor:
    """
    Run diffusion sampling (reverse diffusion) to generate actions.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        batch (dict): Input batch.
        batch_size (int): Batch size.
        num_patches (int): Number of vision patches.
        actions_shape (tuple): Shape of ground-truth actions.
        device_id (str): Device ID.
        current_action_mask (torch.Tensor): Mask for current action.
        next_actions_mask (torch.Tensor): Mask for next actions.
        use_proprio (bool): Whether to use proprioceptive state as input.
        use_film (bool): Whether to use FiLM for better language following.

    Returns:
        torch.Tensor: Predicted actions.
    """
    # Sample random noisy action, used as the starting point for reverse diffusion
    noise = torch.randn(
        size=(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM),
        device=device_id,
        dtype=torch.bfloat16,
    )  # (B, chunk_len, action_dim)

    # Set diffusion timestep values 设置反向扩散的时间步长，训练时是50步，采样时是100步
    action_head.module.noise_scheduler.set_timesteps(action_head.module.num_diffusion_steps_train)

    # Reverse diffusion: Iteratively denoise to generate action, conditioned on observation
    curr_noisy_actions = noise
    input_ids = batch["input_ids"].to(device_id)
    attention_mask = batch["attention_mask"].to(device_id)
    pixel_values = batch["pixel_values"].to(torch.bfloat16).to(device_id)
    proprio = batch["proprio"].to(device_id).to(torch.bfloat16) if use_proprio else None
    labels = batch["labels"].to(device_id)
    for t in action_head.module.noise_scheduler.timesteps:
        # Get diffusion model's noise prediction (conditioned on VLA latent embedding, current noisy action embedding,
        # and diffusion timestep embedding)
        timesteps = torch.Tensor([t]).repeat(batch_size).to(device_id)          # 把当前时间步拓展为batch_size个(B,)
        diffusion_timestep_embeddings = (
            action_head.module.time_encoder(timesteps).to(curr_noisy_actions.dtype).to(curr_noisy_actions.device)
        )  # (B, llm_dim) 用 time_encoder 把数字时间步变成向量，因为 VLA 的输入是向量，不能直接输入数字时间步
        diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim) 因为它要作为一个额外 embedding/token 拼进 VLA 的输入序列，所以需要格式对齐

        #VLA前向传播
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = vla(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
                output_hidden_states=True,
                proprio=proprio,
                proprio_projector=proprio_projector if use_proprio else None,
                noisy_actions=curr_noisy_actions,
                noisy_action_projector=noisy_action_projector,
                diffusion_timestep_embeddings=diffusion_timestep_embeddings,
                use_film=use_film,
            )
            # Get last layer hidden states
            last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
            # Get hidden states for text portion of prompt+response (after the vision patches)
            text_hidden_states = last_hidden_states[:, num_patches:-1]
            # Get hidden states for action portion of response
            actions_hidden_states = text_hidden_states[current_action_mask | next_actions_mask].reshape(
                batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1
            )  # (B, act_chunk_len, D)
            actions_hidden_states = actions_hidden_states.to(torch.bfloat16)
            # Predict noise 输入：actions_hidden_states: (B, 56, D). 输出：noise_pred
            noise_pred = action_head.module.predict_noise(actions_hidden_states)

        # Compute the action at the previous diffusion timestep: x_t -> x_{t-1}
        curr_noisy_actions = action_head.module.noise_scheduler.step(noise_pred, t, curr_noisy_actions).prev_sample

    return curr_noisy_actions.reshape(actions_shape)

# 计算最近若干 step 指标的滑动平均值，因为机器人训练时指标波动很大，所以用滑动平均值来平滑指标曲线，便于观察训练趋势
def compute_smoothened_metrics(metrics_deques) -> dict:
    """
    Compute smoothened metrics from recent deques.

    Args:
        metrics_deques (dict): Dictionary of deques containing recent metrics.

    Returns:
        dict: Dictionary of smoothened metrics.
    """
    smoothened_metrics = {}
    for name, deque in metrics_deques.items():
        if deque and len(deque) > 0:
            smoothened_metrics[name] = sum(deque) / len(deque)
    return smoothened_metrics


def completed_optimizer_step(
    batch_idx: int,
    grad_accumulation_steps: int,
    resume_step: int = 0,
) -> Optional[int]:
    """Return the absolute step only when this microbatch completes an optimizer update."""
    if grad_accumulation_steps < 1:
        raise ValueError("grad_accumulation_steps must be >= 1")
    if (batch_idx + 1) % grad_accumulation_steps != 0:
        return None
    return resume_step + (batch_idx + 1) // grad_accumulation_steps


#把指标记录到wandb上
def log_metrics_to_wandb(metrics, prefix, step, wandb_entity) -> None:
    """
    Log metrics to Weights & Biases.

    Args:
        metrics (dict): Dictionary of metrics to log
        prefix (str): Prefix for metric names
        step (int): Training step
        wandb_entity (str): W&B entity instance

    Returns:
        None.
    """
    log_dict = {}
    for name, value in metrics.items():
        # Map loss_value to Loss for better readability in W&B
        if name == "loss_value":
            log_dict[f"{prefix}/Loss"] = value
        # Keep other metrics as is
        else:
            log_dict[f"{prefix}/{name.replace('_', ' ').title()}"] = value
    wandb_entity.log(log_dict, step=step)

# 保存训练成果
# Save all training checkpoints including model components, LoRA adapter, and dataset statistics.
def save_training_checkpoint(
    cfg,
    run_dir,
    log_step,
    vla,
    processor,
    proprio_projector,
    noisy_action_projector,
    action_head,
    condition_adapter,
    stop_head,
    train_dataset,
    distributed_state,
) -> None:
    """
    Save all training checkpoints including model components, LoRA adapter, and dataset statistics.

    Args:
        cfg (FinetuneConfig): Training configuration.
        run_dir (Path): Experiment run directory path.
        log_step (int): Current logging step.
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        processor (PrismaticProcessor): OpenVLA inputs processor.
        proprio_projector (nn.Module): Proprioceptive state projector module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        action_head (nn.Module): Action head module.
        train_dataset (RLDSDataset): Training dataset.
        distributed_state (PartialState): Distributed training state.

    Returns:
        None.
    """
    # Determine checkpoint paths and naming
    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = run_dir
        checkpoint_name_suffix = "latest_checkpoint.pt"
    else:
        checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
        checkpoint_name_suffix = f"{log_step}_checkpoint.pt"

    adapter_dir = checkpoint_dir / "lora_adapter"

    # Create directories and save dataset statistics (main process only)
    if distributed_state.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(adapter_dir, exist_ok=True)
        save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
        save_policy_contract(cfg, train_dataset.dataset_statistics, checkpoint_dir)
        print(f"Saving Model Checkpoint for Step {log_step}")

    # Wait for directories to be created
    _distributed_barrier()

    # Save model components (main process only)
    if distributed_state.is_main_process:
        # Save processor and LoRA adapter
        processor.save_pretrained(checkpoint_dir)
        vla.module.save_pretrained(adapter_dir)

        # Save other components
        if cfg.use_proprio and proprio_projector is not None:
            torch.save(proprio_projector.state_dict(), checkpoint_dir / f"proprio_projector--{checkpoint_name_suffix}")

        if cfg.use_diffusion and noisy_action_projector is not None:
            torch.save(
                noisy_action_projector.state_dict(), checkpoint_dir / f"noisy_action_projector--{checkpoint_name_suffix}"
            )

        if (cfg.use_l1_regression or cfg.use_diffusion) and action_head is not None:
            torch.save(action_head.state_dict(), checkpoint_dir / f"action_head--{checkpoint_name_suffix}")

        if condition_adapter is not None:
            torch.save(
                condition_adapter.state_dict(),
                checkpoint_dir / f"condition_adapter--{checkpoint_name_suffix}",
            )
        if stop_head is not None:
            torch.save(
                stop_head.state_dict(),
                checkpoint_dir / f"stop_head--{checkpoint_name_suffix}",
            )

        if cfg.use_film:
            #如果用了FiLM,因为FiLM会改视觉backbone的参数，所以要保存视觉backbone的参数
            # To be safe, just save the entire vision backbone (not just FiLM components)
            torch.save(
                vla.module.vision_backbone.state_dict(), checkpoint_dir / f"vision_backbone--{checkpoint_name_suffix}"
            )

    # Wait for model components to be saved
    _distributed_barrier()

    # Merge LoRA weights into base model and save resulting model checkpoint
    # Note: Can be very slow on some devices; if so, we recommend merging offline
    if cfg.use_lora and cfg.merge_lora_during_training:
        base_vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
        )
        if cfg.use_cond_action_tokens:
            base_vla.resize_token_embeddings(len(processor.tokenizer), pad_to_multiple_of=64)
        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
        merged_vla = merged_vla.merge_and_unload()
        merged_vla.config.condition_target_fusion = (
            "learned_condition_to_patch_topk"
            if cfg.use_indoor_uav_condition_adapter
            else "centered_condition_to_patch_topk"
        )
        merged_vla.config.condition_patch_topk = cfg.condition_patch_topk
        merged_vla.config.condition_matching_centered = not cfg.use_indoor_uav_condition_adapter
        merged_vla.config.condition_match_dim = cfg.condition_match_dim
        merged_vla.config.condition_match_hidden_dim = cfg.condition_match_hidden_dim
        merged_vla.config.use_indoor_uav_condition_adapter = cfg.use_indoor_uav_condition_adapter
        merged_vla.config.proprio_representation = (
            "xyz_sin_yaw_cos_yaw_v1" if cfg.cyclic_yaw_proprio else "dataset_default"
        )
        merged_vla.config.proprio_dim = get_model_proprio_dim(cfg)
        merged_vla.config.action_normalization = (
            "per_axis_symmetric_minmax_v1" if cfg.body_delta_action_targets else "dataset_default"
        )
        merged_vla.config.image_input_roles = (
            ["reference", "previous", "current"] if cfg.use_reference_previous_current else None
        )
        merged_vla.config.condition_contrastive_mode = (
            "per_time_k_way_plus_temporal_plus_cross_episode_queue"
            if cfg.use_indoor_uav_condition_adapter
            else "per_time_branch_selection"
        )
        merged_vla.config.condition_branch_assignment = "action_error_only"
        merged_vla.config.condition_branch_weight = cfg.condition_contrastive_weight
        merged_vla.config.condition_temporal_weight = cfg.condition_temporal_weight
        merged_vla.config.condition_queue_weight = cfg.condition_queue_weight
        merged_vla.config.condition_queue_size = cfg.condition_queue_size
        merged_vla.config.condition_queue_min_negatives = cfg.condition_queue_min_negatives
        merged_vla.config.use_indoor_uav_stop_head = cfg.use_indoor_uav_stop_head
        merged_vla.config.use_indoor_uav_progress_stop_head = cfg.use_indoor_uav_progress_stop_head
        merged_vla.config.stop_head_hidden_dim = cfg.stop_head_hidden_dim
        merged_vla.config.stop_progress_projection_dim = cfg.stop_progress_projection_dim
        merged_vla.config.stop_progress_horizons = list(STOP_PROGRESS_HORIZONS)
        merged_vla.config.stop_threshold = cfg.stop_threshold
        merged_vla.config.stop_target_semantics = (
            "execute_root_action_then_stop" if cfg.use_indoor_uav_stop_head else None
        )
        merged_vla.config.action_head_type = (
            "gaussian" if cfg.use_gaussian_action_head else ("diffusion" if cfg.use_diffusion else "l1")
        )
        merged_vla.config.num_action_branches = cfg.num_action_branches
        merged_vla.config.use_cond_action_tokens = cfg.use_cond_action_tokens
        merged_vla.config.condition_action_pairing = "per_time_action_error_assignment"
        merged_vla.config.condition_assignment_weight = cfg.condition_assignment_weight
        merged_vla.config.initial_action_branch_index = cfg.initial_action_branch_index
        merged_vla.config.condition_loss_start_time_index = cfg.condition_loss_start_time_index
        merged_vla.config.grpo_group_size = cfg.grpo_group_size
        merged_vla.config.grpo_exact_gaussian_log_prob = cfg.grpo_reward_weight > 0
        if cfg.body_delta_action_targets:
            merged_vla.config.action_representation = "body_delta_one_step_v1"
        elif cfg.relative_action_targets:
            merged_vla.config.action_representation = "relative_plan_origin"
        else:
            merged_vla.config.action_representation = "dataset_default"
        merged_vla.config.training_objective = (
            "sft" if cfg.grpo_reward_weight == 0 else "sft_plus_grpo"
        )
        if cfg.use_gaussian_action_head:
            merged_vla.config.gaussian_log_std_min = cfg.gaussian_log_std_min
            merged_vla.config.gaussian_log_std_max = cfg.gaussian_log_std_max
            merged_vla.config.gaussian_initial_log_std = cfg.gaussian_initial_log_std
            merged_vla.config.gaussian_learn_log_std = cfg.gaussian_learn_log_std

        if distributed_state.is_main_process:
            merged_vla.save_pretrained(checkpoint_dir)
            print(f"Saved merged model for Step {log_step} at: {checkpoint_dir}")

        # Wait for merged model to be saved
        _distributed_barrier()


def aggregate_validation_metrics(all_metrics: list[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate validation metrics by their actual number of valid queries."""
    if not all_metrics:
        raise ValueError("cannot aggregate an empty validation result")

    count_metrics = {
        "root_valid_count",
        "future_valid_count",
        "condition_temporal_query_count",
        "condition_queue_queries",
        "stop_true_positive",
        "stop_true_negative",
        "stop_false_positive",
        "stop_false_negative",
        "stop_sample_count",
        "stop_positive_count",
        "stop_negative_count",
    }
    root_metrics = {
        "sft_root_action_loss",
        "stop_loss",
        "stop_probability_mean",
        "stop_target_rate",
        "stop_predicted_rate",
        "stop_progress_aux_loss",
        "stop_progress_total_loss",
    }
    future_metrics = {
        "sft_future_action_loss",
        "action_branch_pair_l1",
        "action_branch_min_pair_l1",
        "action_branch_position_separation_m",
        "action_branch_yaw_separation_rad",
        "action_winner_gap",
        "action_winner_near_tie_rate",
        "oracle_future_action_loss",
        "condition_selected_action_loss",
        "branch0_future_action_loss",
        "condition_selection_regret",
        "condition_gain_vs_branch0",
        "condition_oracle_recovery",
        "oracle_action_position_error_m",
        "condition_selected_position_error_m",
        "branch0_position_error_m",
        "oracle_action_yaw_error_rad",
        "condition_selected_yaw_error_rad",
        "branch0_yaw_error_rad",
        "condition_alignment_loss",
        "condition_contrastive_loss",
        "condition_branch_accuracy",
        "condition_branch_random_accuracy",
        "condition_branch_margin",
        "condition_diversity_loss",
        "condition_similarity_selected",
        "branch_balance_loss",
    }
    temporal_metrics = {
        "condition_temporal_loss",
        "condition_retrieval_accuracy",
        "condition_temporal_random_accuracy",
        "condition_retrieval_positive",
        "condition_retrieval_hardest_negative",
        "condition_retrieval_margin",
    }
    queue_metrics = {
        "condition_queue_loss",
        "condition_queue_accuracy",
        "condition_queue_random_accuracy",
        "condition_queue_margin",
        "condition_queue_negatives",
    }

    output = {}
    metric_names = {
        name for metrics in all_metrics for name in metrics if not name.startswith("_")
    }
    for metric_name in metric_names:
        rows = [metrics for metrics in all_metrics if metric_name in metrics]
        is_slot_count = metric_name.startswith("future_slot") and metric_name.endswith("_valid_count")
        is_axis_nonzero_count = metric_name.startswith("root_") and metric_name.endswith("_nonzero_count")
        if metric_name in count_metrics or is_slot_count or is_axis_nonzero_count:
            output[metric_name] = sum(metrics[metric_name] for metrics in rows)
            continue

        weight_name = None
        if metric_name in root_metrics:
            weight_name = "root_valid_count"
        elif metric_name.startswith("root_") and metric_name.endswith("_sign_accuracy"):
            axis_name = metric_name.removeprefix("root_").removesuffix("_sign_accuracy")
            weight_name = f"root_{axis_name}_nonzero_count"
        elif metric_name.startswith("root_") and metric_name.endswith(
            ("_prediction_mean", "_target_mean", "_bias", "_abs_error")
        ):
            weight_name = "root_valid_count"
        elif metric_name in future_metrics or (
            metric_name.startswith("branch")
            and (metric_name.endswith("_winner_rate") or metric_name.endswith("_soft_usage"))
        ) or (
            metric_name.startswith("condition_branch")
            and metric_name.endswith("_selected_rate")
        ):
            weight_name = "future_valid_count"
        elif metric_name in temporal_metrics:
            weight_name = "condition_temporal_query_count"
        elif metric_name in queue_metrics:
            weight_name = "condition_queue_queries"
        elif metric_name.startswith("future_slot") and "_branch" in metric_name:
            slot_prefix = metric_name.split("_branch", 1)[0]
            weight_name = f"{slot_prefix}_valid_count"

        if weight_name is None:
            output[metric_name] = sum(metrics[metric_name] for metrics in rows) / len(rows)
            continue

        weighted_rows = [metrics for metrics in rows if metrics.get(weight_name, 0.0) > 0]
        total_weight = sum(metrics[weight_name] for metrics in weighted_rows)
        output[metric_name] = (
            sum(metrics[metric_name] * metrics[weight_name] for metrics in weighted_rows) / total_weight
            if total_weight > 0
            else 0.0
        )
    required = (
        "oracle_future_action_loss",
        "condition_selected_action_loss",
        "branch0_future_action_loss",
    )
    if all(name in output for name in required):
        oracle_loss, selected_loss, branch0_loss = (output[name] for name in required)
        output["condition_selection_regret"] = selected_loss - oracle_loss
        output["condition_gain_vs_branch0"] = branch0_loss - selected_loss
        output["condition_oracle_recovery"] = _oracle_recovery(
            oracle_loss, selected_loss, branch0_loss
        )
    if output.get("stop_sample_count", 0.0) > 0:
        tp = output.get("stop_true_positive", 0.0)
        tn = output.get("stop_true_negative", 0.0)
        fp = output.get("stop_false_positive", 0.0)
        fn = output.get("stop_false_negative", 0.0)
        sample_count = output["stop_sample_count"]
        positive_count = tp + fn
        negative_count = tn + fp
        recall = tp / positive_count if positive_count else 0.0
        specificity = tn / negative_count if negative_count else 0.0
        output["stop_accuracy"] = (tp + tn) / sample_count
        output["stop_precision"] = tp / (tp + fp) if tp + fp else 0.0
        output["stop_recall"] = recall
        output["stop_specificity"] = specificity
        output["stop_balanced_accuracy"] = 0.5 * (recall + specificity)
        output["stop_target_rate"] = positive_count / sample_count
        output["stop_predicted_rate"] = (tp + fp) / sample_count
    return output


# 在验证集上计算指标
def run_validation(
    vla,
    action_head,
    condition_adapter,
    stop_head,
    noisy_action_projector,
    proprio_projector,
    val_dataloader,
    action_tokenizer,
    device_id,
    cfg,
    num_patches,
    log_step,
    distributed_state,
    val_time_limit,
    action_norm_stats=None,
    cond_token_ids=None,
    act_token_ids=None,
    run_dir: Optional[Path] = None,
) -> None:
    """
    Compute validation set metrics for logging.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        val_dataloader (DataLoader): Validation data loader.
        action_tokenizer (ActionTokenizer): Action tokenizer.
        device_id (str): Device ID.
        cfg (FinetuneConfig): Training configuration.
        num_patches (int): Number of vision patches.
        log_step (int): Current logging step.
        distributed_state (PartialState): Distributed training state.
        val_time_limit (int): Time limit for computing validation metrics.

    Returns:
        None.
    """
    val_start_time = time.time()
    vla.eval()
    val_batches_count = 0

    # List to store validation metrics
    all_val_metrics = []
    stop_probabilities = []
    stop_targets = []
    validation_condition_queue = (
        CrossEpisodeImageQueue(cfg.condition_queue_size)
        if condition_adapter is not None and cfg.condition_queue_weight > 0
        else None
    )

    with torch.no_grad():
        for batch in val_dataloader:
            # Always compute L1 loss for validation, even for diffusion
            _, metrics = run_forward_pass(
                vla=vla,
                action_head=action_head,
                condition_adapter=condition_adapter,
                stop_head=stop_head,
                noisy_action_projector=noisy_action_projector,
                proprio_projector=proprio_projector,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_diffusion=cfg.use_diffusion,
                use_gaussian_action_head=cfg.use_gaussian_action_head,
                action_regression_loss=cfg.action_regression_loss,
                num_action_branches=cfg.num_action_branches,
                use_best_of_k_action_loss=cfg.use_best_of_k_action_loss,
                branch_assignment_temperature=cfg.branch_assignment_temperature,
                branch_balance_weight=cfg.branch_balance_weight,
                condition_assignment_weight=cfg.condition_assignment_weight,
                initial_action_branch_index=cfg.initial_action_branch_index,
                branch_diversity_weight=cfg.branch_diversity_weight,
                branch_diversity_margin=cfg.branch_diversity_margin,
                grpo_reward_weight=cfg.grpo_reward_weight,
                grpo_policy_sigma=cfg.grpo_policy_sigma,
                grpo_group_size=cfg.grpo_group_size,
                grpo_clip_epsilon=cfg.grpo_clip_epsilon,
                grpo_safety_weight=cfg.grpo_safety_weight,
                grpo_advantage_eps=cfg.grpo_advantage_eps,
                grpo_advantage_clip=cfg.grpo_advantage_clip,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=num_patches,
                action_norm_stats=action_norm_stats,
                use_cond_action_tokens=cfg.use_cond_action_tokens,
                cond_token_ids=cond_token_ids,
                act_token_ids=act_token_ids,
                couple_condition_to_action_branch=cfg.couple_condition_to_action_branch,
                condition_similarity_threshold=cfg.condition_similarity_threshold,
                condition_alignment_weight=cfg.condition_alignment_weight,
                condition_contrastive_weight=cfg.condition_contrastive_weight,
                condition_temporal_weight=cfg.condition_temporal_weight,
                condition_queue_weight=cfg.condition_queue_weight,
                condition_queue_min_negatives=cfg.condition_queue_min_negatives,
                condition_negative_queue=validation_condition_queue,
                condition_contrastive_temperature=cfg.condition_contrastive_temperature,
                condition_loss_start_time_index=cfg.condition_loss_start_time_index,
                condition_patch_topk=cfg.condition_patch_topk,
                condition_diversity_weight=cfg.condition_diversity_weight,
                condition_diversity_margin=cfg.condition_diversity_margin,
                root_action_weight=cfg.root_action_weight,
                future_action_weight=cfg.future_action_weight,
                stop_loss_weight=cfg.stop_loss_weight,
                stop_positive_weight=cfg.stop_positive_weight,
                stop_threshold=cfg.stop_threshold,
                use_progress_stop_head=cfg.use_indoor_uav_progress_stop_head,
                stop_progress_positive_weights=cfg.stop_progress_positive_weights,
                stop_progress_loss_weight=cfg.stop_progress_loss_weight,
                compute_diffusion_l1=True,
                num_diffusion_steps_train=cfg.num_diffusion_steps_train if cfg.use_diffusion else None,
            )

            # Add the loss value to the metrics
            stop_probabilities.extend(metrics.pop("_stop_probabilities", []))
            stop_targets.extend(metrics.pop("_stop_targets", []))
            metrics["loss"] = metrics["loss_value"]
            all_val_metrics.append(metrics)
            val_batches_count += 1

            # Prefer a reproducible batch cap; retain the time limit as a safety bound.
            if cfg.val_max_batches > 0 and val_batches_count >= cfg.val_max_batches:
                break
            if time.time() - val_start_time > val_time_limit:
                break

    # Compute average validation metrics
    avg_val_metrics = aggregate_validation_metrics(all_val_metrics)
    stop_score_diagnostics = None
    if stop_probabilities:
        stop_score_diagnostics = compute_binary_score_diagnostics(
            stop_probabilities,
            stop_targets,
        )
        avg_val_metrics.update(stop_score_diagnostics)

    # Add batch count to metrics
    avg_val_metrics["val_batches_count"] = val_batches_count

    # Log validation metrics to W&B
    if distributed_state.is_main_process:
        summary_keys = (
            "loss",
            "stop_loss",
            "stop_accuracy",
            "stop_balanced_accuracy",
            "stop_precision",
            "stop_recall",
            "stop_specificity",
            "stop_target_rate",
            "stop_predicted_rate",
            "stop_probability_mean",
            "stop_progress_aux_loss",
            "stop_progress_total_loss",
            "stop_progress_h1_accuracy",
            "stop_progress_h2_accuracy",
            "stop_progress_h4_accuracy",
            "stop_roc_auc",
            "stop_positive_probability_mean",
            "stop_negative_probability_mean",
            "stop_probability_class_gap",
            "stop_best_threshold",
            "stop_best_balanced_accuracy",
            "stop_best_precision",
            "stop_best_recall",
            "stop_best_specificity",
            "sft_root_action_loss",
            "sft_future_action_loss",
            "root_forward_bias",
            "root_forward_abs_error",
            "root_forward_sign_accuracy",
            "root_right_bias",
            "root_right_abs_error",
            "root_right_sign_accuracy",
            "root_up_bias",
            "root_up_abs_error",
            "root_up_sign_accuracy",
            "root_yaw_bias",
            "root_yaw_abs_error",
            "root_yaw_sign_accuracy",
            "action_branch_pair_l1",
            "action_branch_min_pair_l1",
            "action_branch_position_separation_m",
            "action_branch_yaw_separation_rad",
            "action_winner_gap",
            "action_winner_near_tie_rate",
            "oracle_future_action_loss",
            "condition_selected_action_loss",
            "branch0_future_action_loss",
            "condition_selection_regret",
            "condition_gain_vs_branch0",
            "condition_oracle_recovery",
            "oracle_action_position_error_m",
            "condition_selected_position_error_m",
            "branch0_position_error_m",
            "oracle_action_yaw_error_rad",
            "condition_selected_yaw_error_rad",
            "branch0_yaw_error_rad",
            "condition_contrastive_loss",
            "condition_branch_accuracy",
            "condition_branch_random_accuracy",
            "condition_branch_margin",
            "condition_temporal_loss",
            "condition_retrieval_accuracy",
            "condition_temporal_random_accuracy",
            "condition_retrieval_margin",
            "condition_queue_loss",
            "condition_queue_accuracy",
            "condition_queue_random_accuracy",
            "condition_queue_margin",
            "branch0_winner_rate",
            "branch1_winner_rate",
            "branch2_winner_rate",
        )
        summary = ", ".join(
            f"{key}={avg_val_metrics[key]:.6f}" for key in summary_keys if key in avg_val_metrics
        )
        print(f"[Validation] step={log_step}, batches={val_batches_count}, {summary}")
        if stop_score_diagnostics is not None and run_dir is not None:
            diagnostic_path = Path(run_dir) / f"stop_validation_step{log_step}.json"
            with diagnostic_path.open("w", encoding="utf-8") as diagnostic_file:
                json.dump(
                    {
                        "step": log_step,
                        "num_samples": len(stop_targets),
                        "probabilities": stop_probabilities,
                        "targets": stop_targets,
                        "diagnostics": stop_score_diagnostics,
                    },
                    diagnostic_file,
                    indent=2,
                )
            print(f"[Validation] Saved STOP score audit to {diagnostic_path}")
        log_metrics_to_wandb(avg_val_metrics, "VLA Val", log_step, wandb)


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    """
    Fine-tunes base VLA on demonstration dataset via LoRA.

    Allows toggling different action representations (discrete vs. continuous), different learning objectives
    (next-token prediction vs. L1 regression vs. diffusion), FiLM. Also allows for additional model inputs,
    such as additional camera images and robot proprioceptive state. Assumes parallel action generation with
    action chunking.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        None.
    """
    assert cfg.use_lora, "Only LoRA fine-tuning is supported. Please set --use_lora=True!"
    assert not (cfg.use_l1_regression and cfg.use_diffusion), (
        "Cannot do both L1 regression and diffusion. Please pick one of them!"
    )
    if cfg.use_gaussian_action_head and not cfg.use_l1_regression:
        raise ValueError("use_gaussian_action_head requires use_l1_regression=True")
    if cfg.action_regression_loss not in {"l1", "mse"}:
        raise ValueError("action_regression_loss must be one of: l1, mse")
    if cfg.action_regression_loss != "l1" and (
        cfg.use_gaussian_action_head or cfg.use_best_of_k_action_loss
    ):
        raise ValueError("mse action regression currently supports only a single deterministic branch")
    if cfg.gaussian_log_std_min >= cfg.gaussian_log_std_max:
        raise ValueError("gaussian_log_std_min must be smaller than gaussian_log_std_max")
    if not cfg.gaussian_log_std_min < cfg.gaussian_initial_log_std < cfg.gaussian_log_std_max:
        raise ValueError("gaussian_initial_log_std must lie strictly inside the configured bounds")
    auxiliary_path_set = cfg.auxiliary_init_checkpoint_path is not None
    auxiliary_step_set = cfg.auxiliary_init_checkpoint_step is not None
    if auxiliary_path_set != auxiliary_step_set:
        raise ValueError(
            "auxiliary_init_checkpoint_path and auxiliary_init_checkpoint_step must be provided together"
        )
    if cfg.resume and auxiliary_path_set:
        raise ValueError("resume and auxiliary_init_checkpoint_path cannot be used together")
    if cfg.cyclic_yaw_proprio and not cfg.body_delta_action_targets:
        raise ValueError("cyclic_yaw_proprio requires body_delta_action_targets=True")
    if cfg.overfit_fixed_batch_count < 0:
        raise ValueError("overfit_fixed_batch_count must be >= 0")
    if cfg.overfit_report_freq < 1:
        raise ValueError("overfit_report_freq must be >= 1")
    if cfg.freeze_proprio_projector and not cfg.use_proprio:
        raise ValueError("freeze_proprio_projector requires use_proprio=True")
    if cfg.freeze_action_head and not cfg.use_l1_regression:
        raise ValueError("freeze_action_head requires use_l1_regression=True")
    if cfg.freeze_condition_adapter and not cfg.use_indoor_uav_condition_adapter:
        raise ValueError("freeze_condition_adapter requires use_indoor_uav_condition_adapter=True")
    if cfg.use_indoor_uav_stop_head:
        if not (cfg.use_l1_regression and cfg.use_cond_action_tokens):
            raise ValueError("IndoorUAV STOP head requires continuous COND/ACT-token SFT")
        if not cfg.use_indoor_uav_condition_adapter:
            raise ValueError("IndoorUAV STOP head requires the IndoorUAV condition adapter contract")
        if cfg.stop_head_hidden_dim < 1:
            raise ValueError("stop_head_hidden_dim must be positive")
        if cfg.stop_loss_weight <= 0:
            raise ValueError("stop_loss_weight must be > 0 when STOP is enabled")
        if cfg.stop_positive_weight < 0:
            raise ValueError("stop_positive_weight must be >= 0")
        if not 0 < cfg.stop_initial_positive_rate < 1:
            raise ValueError("stop_initial_positive_rate must lie in (0,1)")
        if not 0 < cfg.stop_threshold < 1:
            raise ValueError("stop_threshold must lie in (0,1)")
        if cfg.use_indoor_uav_progress_stop_head:
            if cfg.stop_progress_projection_dim < 1:
                raise ValueError("stop_progress_projection_dim must be positive")
            if cfg.stop_progress_loss_weight < 0:
                raise ValueError("stop_progress_loss_weight must be non-negative")
    elif cfg.use_indoor_uav_progress_stop_head:
        raise ValueError("progress STOP requires use_indoor_uav_stop_head=True")
    if cfg.num_action_branches < 1:
        raise ValueError("num_action_branches must be >= 1")
    if cfg.num_action_branches > 1 and not cfg.use_l1_regression:
        raise ValueError("num_action_branches > 1 is currently supported only with use_l1_regression=True")
    if cfg.use_best_of_k_action_loss and cfg.num_action_branches < 2:
        raise ValueError("use_best_of_k_action_loss requires num_action_branches >= 2")
    if cfg.branch_assignment_temperature <= 0:
        raise ValueError("branch_assignment_temperature must be > 0")
    if cfg.branch_balance_weight < 0:
        raise ValueError("branch_balance_weight must be >= 0")
    if cfg.condition_assignment_weight < 0:
        raise ValueError("condition_assignment_weight must be >= 0")
    if cfg.initial_action_branch_index < -1 or cfg.initial_action_branch_index >= cfg.num_action_branches:
        raise ValueError("initial_action_branch_index must be -1 or a valid branch index")
    if cfg.use_cond_action_tokens and not cfg.use_l1_regression:
        raise ValueError("use_cond_action_tokens currently requires use_l1_regression=True")
    if cfg.use_cond_action_tokens and cfg.use_diffusion:
        raise ValueError("use_cond_action_tokens is not yet implemented for diffusion")
    if cfg.branch_diversity_weight < 0:
        raise ValueError("branch_diversity_weight must be >= 0")
    if cfg.branch_diversity_margin < 0:
        raise ValueError("branch_diversity_margin must be >= 0")
    if cfg.grpo_reward_weight < 0:
        raise ValueError("grpo_reward_weight must be >= 0")
    if cfg.condition_alignment_weight < 0:
        raise ValueError("condition_alignment_weight must be >= 0")
    if cfg.condition_contrastive_weight < 0:
        raise ValueError("condition_contrastive_weight must be >= 0")
    if cfg.condition_temporal_weight < 0:
        raise ValueError("condition_temporal_weight must be >= 0")
    if cfg.condition_queue_weight < 0:
        raise ValueError("condition_queue_weight must be >= 0")
    if cfg.condition_queue_size < 1:
        raise ValueError("condition_queue_size must be >= 1")
    if cfg.condition_queue_min_negatives < 1:
        raise ValueError("condition_queue_min_negatives must be >= 1")
    if (cfg.condition_temporal_weight > 0 or cfg.condition_queue_weight > 0) and not cfg.use_indoor_uav_condition_adapter:
        raise ValueError("temporal and queue condition losses require use_indoor_uav_condition_adapter=True")
    if cfg.condition_contrastive_temperature <= 0:
        raise ValueError("condition_contrastive_temperature must be > 0")
    if not 0 <= cfg.condition_loss_start_time_index < NUM_ACTIONS_CHUNK:
        raise ValueError(
            f"condition_loss_start_time_index must be in [0, {NUM_ACTIONS_CHUNK})"
        )
    if cfg.condition_patch_topk < 1:
        raise ValueError("condition_patch_topk must be >= 1")
    if cfg.couple_condition_to_action_branch and (
        not cfg.use_cond_action_tokens or not cfg.use_best_of_k_action_loss
    ):
        raise ValueError(
            "couple_condition_to_action_branch requires use_cond_action_tokens "
            "and use_best_of_k_action_loss"
        )
    k1_condition_sft = cfg.use_indoor_uav_condition_adapter and cfg.num_action_branches == 1
    if (
        cfg.condition_contrastive_weight > 0
        and not cfg.couple_condition_to_action_branch
        and not k1_condition_sft
    ):
        raise ValueError(
            "condition_contrastive_weight > 0 requires couple_condition_to_action_branch=True"
        )
    if cfg.condition_assignment_weight > 0 and (
        not cfg.use_best_of_k_action_loss
        or not cfg.couple_condition_to_action_branch
        or not (cfg.use_gaussian_action_head or cfg.use_indoor_uav_condition_adapter)
    ):
        raise ValueError(
            "condition_assignment_weight > 0 requires coupled Gaussian or IndoorUAV Best-of-K branches"
        )
    if cfg.condition_assignment_weight > 0 and (
        cfg.condition_alignment_weight <= 0 and cfg.condition_contrastive_weight <= 0
    ):
        raise ValueError(
            "joint condition-action assignment requires a nonzero condition alignment or contrastive weight"
        )
    if cfg.condition_diversity_weight < 0:
        raise ValueError("condition_diversity_weight must be >= 0")
    if cfg.condition_diversity_margin < 0:
        raise ValueError("condition_diversity_margin must be >= 0")
    if cfg.grpo_reward_weight > 0 and cfg.num_action_branches < 2:
        raise ValueError("grpo_reward_weight > 0 requires num_action_branches >= 2")
    if cfg.grpo_reward_weight > 0 and not (
        cfg.use_gaussian_action_head and cfg.use_best_of_k_action_loss
    ):
        raise ValueError(
            "grpo_reward_weight > 0 requires Gaussian per-time best-of-K action policies"
        )
    if cfg.grpo_group_size < 2:
        raise ValueError("grpo_group_size must be >= 2")
    if not 0 < cfg.grpo_clip_epsilon < 1:
        raise ValueError("grpo_clip_epsilon must lie in (0, 1)")
    if cfg.grpo_safety_weight < 0:
        raise ValueError("grpo_safety_weight must be >= 0")
    if cfg.grpo_policy_sigma <= 0:
        raise ValueError("grpo_policy_sigma must be > 0")
    if cfg.grpo_advantage_eps <= 0:
        raise ValueError("grpo_advantage_eps must be > 0")
    if cfg.grpo_advantage_clip <= 0:
        raise ValueError("grpo_advantage_clip must be > 0")
    if cfg.max_grad_norm is not None and cfg.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be > 0 when provided")
    if cfg.root_action_weight < 0 or cfg.future_action_weight < 0:
        raise ValueError("root_action_weight and future_action_weight must be >= 0")
    if cfg.condition_match_dim < 1 or cfg.condition_match_hidden_dim < 1:
        raise ValueError("condition matching dimensions must be positive")
    if cfg.relative_action_targets and cfg.body_delta_action_targets:
        raise ValueError("relative_action_targets and body_delta_action_targets are mutually exclusive")

    if cfg.use_indoor_uav_condition_adapter:
        required_flags = {
            "use_l1_regression": cfg.use_l1_regression,
            "use_cond_action_tokens": cfg.use_cond_action_tokens,
            "use_reference_previous_current": cfg.use_reference_previous_current,
            "body_delta_action_targets": cfg.body_delta_action_targets,
            "cyclic_yaw_proprio": cfg.cyclic_yaw_proprio,
            "use_proprio": cfg.use_proprio,
        }
        missing = [name for name, enabled in required_flags.items() if not enabled]
        if missing:
            raise ValueError(f"IndoorUAV condition SFT requires: {', '.join(missing)}")
        if cfg.num_action_branches > 1 and not (
            cfg.use_best_of_k_action_loss and cfg.couple_condition_to_action_branch
        ):
            raise ValueError("multi-branch IndoorUAV condition SFT requires coupled Best-of-K assignment")
        if cfg.condition_assignment_weight != 0:
            raise ValueError(
                "IndoorUAV K-way condition labels must use action-only branch assignment; "
                "set condition_assignment_weight=0"
            )
        if cfg.num_action_branches == 1 and cfg.condition_contrastive_weight > 0:
            raise ValueError("K-way condition contrastive loss requires num_action_branches > 1")
        if cfg.use_diffusion or cfg.use_gaussian_action_head or cfg.grpo_reward_weight > 0:
            raise ValueError(
                "IndoorUAV condition training is deterministic SFT; "
                "diffusion, Gaussian, and GRPO are disabled"
            )
        if cfg.num_images_in_input != 3:
            raise ValueError("IndoorUAV condition SFT requires exactly [reference, previous, current] images")
        if cfg.future_action_stride != 1:
            raise ValueError("one-step body-delta supervision requires future_action_stride=1")
        if cfg.initial_action_branch_index != 0:
            raise ValueError("slot 0 is supervised and executed only through branch 0")
        if cfg.condition_loss_start_time_index != 1:
            raise ValueError("future visual conditions start at slot 1")
        if NUM_ACTIONS_CHUNK != 5 or ACTION_DIM != 4 or PROPRIO_DIM != 4:
            raise ValueError("IndoorUAV condition SFT requires T=5 and 4D source pose/action constants")
        if not cfg.reset_proprio_projector and not cfg.resume and auxiliary_path_set:
            auxiliary_contract_path = Path(cfg.auxiliary_init_checkpoint_path) / "policy_contract.json"
            if not auxiliary_contract_path.is_file():
                raise ValueError(
                    "loading a cyclic proprio projector requires an auxiliary policy_contract.json"
                )
            with auxiliary_contract_path.open("r", encoding="utf-8") as contract_file:
                auxiliary_contract = json.load(contract_file)
            expected_auxiliary_contract = {
                "model_proprio_dim": 5,
                "proprio_representation": "xyz_sin_yaw_cos_yaw_v1",
                "action_representation": "body_delta_one_step_v1",
                "horizon": NUM_ACTIONS_CHUNK,
                "num_action_branches": cfg.num_action_branches,
            }
            mismatches = {
                key: (auxiliary_contract.get(key), expected)
                for key, expected in expected_auxiliary_contract.items()
                if auxiliary_contract.get(key) != expected
            }
            if mismatches:
                raise ValueError(f"incompatible auxiliary IndoorUAV policy contract: {mismatches}")

    # Trim trailing forward slash ('/') in VLA path if it exists
    cfg.vla_path = cfg.vla_path.rstrip("/")
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # Get experiment run ID
    run_id = get_run_id(cfg)

    # Create experiment run directory
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    _set_torch_seed(cfg.seed, "global")
    tf.random.set_seed(cfg.seed)

    # GPU setup 初始化GPU
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    # Initialize wandb logging 初始化WandB，只有主进程上传
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{run_id}")

    # Print detected constants
    print(
        "Detected constants:\n"
        f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
        f"\tACTION_DIM: {ACTION_DIM}\n"
        f"\tPROPRIO_DIM: {PROPRIO_DIM}\n"
        f"\tMODEL_PROPRIO_DIM: {get_model_proprio_dim(cfg)}\n"
        f"\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}"
    )

    # Two options:
    # (1) Base model is on Hugging Face Hub
    #   - Then download it and record the path to the download directory
    # (2) Base model is stored locally
    #   - Then register model config in HF Auto Classes
    # In both cases, we want to check whether any changes have been made to
    # the `modeling_prismatic.py` file in this codebase; if so, we will copy
    # the file to the downloaded or locally stored checkpoint directory so
    # that the user's changes to the VLA class logic go into effect
    if model_is_on_hf_hub(cfg.vla_path):
        # Download model directly from Hugging Face Hub
        vla_download_path = snapshot_download(repo_id=cfg.vla_path)
        # Overwrite VLA path
        cfg.vla_path = vla_download_path
    else:
        # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Update config.json and sync model files
    if distributed_state.is_main_process:
        update_auto_map(cfg.vla_path)
        check_model_logic_mismatch(cfg.vla_path)

    # Wait for model files to be synced
    _distributed_barrier()

    # Load processor and VLA 真正加载模型和处理器
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device_id)
    if cfg.use_cond_action_tokens:
        add_cond_action_tokens(processor.tokenizer, vla, cfg.num_action_branches)

    # Set number of images in VLA input
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    # LoRA setup
    if cfg.use_lora:
        # Keep LoRA initialization identical when comparing base checkpoints that
        # take different token-resize paths before PEFT wrapping.
        _set_torch_seed(cfg.seed + 1, "LoRA initialization")
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # FiLM setup
    if cfg.use_film:
        count_parameters(vla.vision_backbone, "vla.vision_backbone (original)")
        # Wrap vision backbone with FiLM wrapper
        # Important: For this, must specify `vla.model.vision_backbone` instead of just `vla.vision_backbone`, since the
        # latter would cause the new wrapped backbone to be saved as a new attribute of `vla` instead of overwriting the
        # original one (due to the LoRA wrapper)
        vla.model.vision_backbone = FiLMedPrismaticVisionBackbone(
            vision_backbone=vla.model.vision_backbone,
            llm_dim=vla.llm_dim,
        )
        count_parameters(vla.vision_backbone, "vla.vision_backbone (post-wrap)")
        if cfg.resume:
            state_dict = load_checkpoint("vision_backbone", cfg.vla_path, cfg.resume_step)
            vla.model.vision_backbone.load_state_dict(state_dict)
        vla.model.vision_backbone = vla.model.vision_backbone.to(device_id)

    # Wrap VLA with DDP 多卡训练
    vla = wrap_ddp(vla, device_id, find_unused=True)

    # If applicable, instantiate proprio projector 创建额外模块
    if cfg.use_proprio:
        proprio_projector = init_module(
            ProprioProjector,
            "proprio_projector",
            cfg,
            device_id,
            {"llm_dim": vla.module.llm_dim, "proprio_dim": get_model_proprio_dim(cfg)},
        )

    # If applicable, instantiate continuous action head for L1 regression
    if cfg.use_l1_regression:
        # The action policy is reset for Stage17 A/B, so isolate its initialization
        # from any random numbers consumed while loading the selected VLA backbone.
        _set_torch_seed(cfg.seed + 2, "action-head initialization")
        action_head_class = GaussianActionHead if cfg.use_gaussian_action_head else L1RegressionActionHead
        action_head_args = {
            "input_dim": vla.module.llm_dim,
            "hidden_dim": vla.module.llm_dim,
            "action_dim": ACTION_DIM,
            "num_action_branches": cfg.num_action_branches,
            "use_cond_action_tokens": cfg.use_cond_action_tokens,
        }
        if cfg.use_gaussian_action_head:
            action_head_args.update(
                {
                    "log_std_min": cfg.gaussian_log_std_min,
                    "log_std_max": cfg.gaussian_log_std_max,
                    "initial_log_std": cfg.gaussian_initial_log_std,
                    "learn_log_std": cfg.gaussian_learn_log_std,
                }
            )
        action_head = init_module(
            action_head_class,
            "action_head",
            cfg,
            device_id,
            action_head_args,
            to_bf16=True,
        )

    condition_adapter = None
    if cfg.use_indoor_uav_condition_adapter:
        _set_torch_seed(cfg.seed + 3, "condition-adapter initialization")
        condition_adapter = init_module(
            IndoorUAVConditionAdapter,
            "condition_adapter",
            cfg,
            device_id,
            {
                "llm_dim": vla.module.llm_dim,
                "hidden_dim": cfg.condition_match_hidden_dim,
                "match_dim": cfg.condition_match_dim,
            },
            allow_missing_auxiliary=True,
        )

    stop_head = None
    if cfg.use_indoor_uav_stop_head:
        _set_torch_seed(cfg.seed + 4, "stop-head initialization")
        stop_head_class = (
            IndoorUAVProgressStopHead
            if cfg.use_indoor_uav_progress_stop_head
            else IndoorUAVStopHead
        )
        stop_head_args = (
            {
                "input_dim": vla.module.llm_dim,
                "projection_dim": cfg.stop_progress_projection_dim,
                "hidden_dim": cfg.stop_head_hidden_dim,
                "initial_positive_rates": tuple(
                    min(0.95, cfg.stop_initial_positive_rate * (horizon + 1))
                    for horizon in STOP_PROGRESS_HORIZONS
                ),
            }
            if cfg.use_indoor_uav_progress_stop_head
            else {
                "input_dim": vla.module.llm_dim,
                "hidden_dim": cfg.stop_head_hidden_dim,
                "initial_positive_rate": cfg.stop_initial_positive_rate,
            }
        )
        stop_head = init_module(
            stop_head_class,
            "stop_head",
            cfg,
            device_id,
            stop_head_args,
            allow_missing_auxiliary=True,
        )

    # If applicable, instantiate diffusion action head and noisy action projector
    if cfg.use_diffusion:
        action_head = init_module(
            DiffusionActionHead,
            "action_head",
            cfg,
            device_id,
            {
                "input_dim": vla.module.llm_dim,
                "hidden_dim": vla.module.llm_dim,
                "action_dim": ACTION_DIM,
                "num_diffusion_steps_train": cfg.num_diffusion_steps_train,
            },
            to_bf16=True,
        )
        noisy_action_projector = init_module(
            NoisyActionProjector, "noisy_action_projector", cfg, device_id, {"llm_dim": vla.module.llm_dim}
        )

    # Get number of vision patches 类似于把[vision patches] + [proprio] + [diffusion timestep]拼成一个序列，得到总长度
    NUM_PATCHES = vla.module.vision_backbone.get_num_patches() * vla.module.vision_backbone.get_num_images_in_input()
    # If we have proprio inputs, a single proprio embedding is appended to the end of the vision patch embeddings
    if cfg.use_proprio:
        NUM_PATCHES += 1
    # For diffusion, a single diffusion timestep embedding is appended to the end of the vision patch embeddings
    if cfg.use_diffusion:
        NUM_PATCHES += 1

    if cfg.freeze_vla:
        set_module_trainable(vla, False)
        print("[Training] Frozen VLA backbone and LoRA parameters")
    if cfg.freeze_proprio_projector:
        set_module_trainable(proprio_projector if cfg.use_proprio else None, False)
        print("[Training] Frozen proprio projector")
    if cfg.freeze_action_head:
        set_module_trainable(action_head if cfg.use_l1_regression else None, False)
        print("[Training] Frozen action head")
    if cfg.freeze_condition_adapter:
        set_module_trainable(condition_adapter, False)
        print("[Training] Frozen condition adapter")

    # Instantiate optimizer 收集所有可训练参数
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    if cfg.use_l1_regression or cfg.use_diffusion:
        trainable_params += [param for param in action_head.parameters() if param.requires_grad]
    if cfg.use_diffusion:
        trainable_params += [param for param in noisy_action_projector.parameters() if param.requires_grad]
    if cfg.use_proprio:
        trainable_params += [param for param in proprio_projector.parameters() if param.requires_grad]
    if condition_adapter is not None:
        trainable_params += [param for param in condition_adapter.parameters() if param.requires_grad]
    if stop_head is not None:
        trainable_params += [param for param in stop_head.parameters() if param.requires_grad]
    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Record original learning rate
    original_lr = optimizer.param_groups[0]["lr"]

    # Create learning rate scheduler
    scheduler = MultiStepLR(
        optimizer,
        milestones=[cfg.num_steps_before_decay],  # Number of steps after which LR will change
        gamma=0.1,  # Multiplicative factor of learning rate decay
    )

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # train_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder,
    # )
    # ---

    if cfg.use_image_history and cfg.num_images_in_input < 1:
        raise ValueError("num_images_in_input must be >= 1 when use_image_history=True")
    if cfg.future_action_stride < 1:
        raise ValueError("future_action_stride must be >= 1")
    if cfg.relative_action_targets and (not cfg.use_proprio or ACTION_DIM != 4 or PROPRIO_DIM != 4):
        raise ValueError("relative_action_targets requires 4D UAV action/proprio and use_proprio=True")
    if cfg.body_delta_action_targets and (not cfg.use_proprio or ACTION_DIM != 4 or PROPRIO_DIM != 4):
        raise ValueError("body_delta_action_targets requires 4D UAV action/proprio and use_proprio=True")

    # The condition policy uses two dynamic frames; the third image is the
    # episode reference and therefore must not enlarge the temporal window.
    use_wrist_image = (
        cfg.num_images_in_input > 1
        and not cfg.use_image_history
        and not cfg.use_reference_previous_current
    )
    window_size = 2 if cfg.use_reference_previous_current else (
        cfg.num_images_in_input if cfg.use_image_history else 1
    )

    # Create training and optional validation datasets
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
        use_image_history=cfg.use_image_history,
        num_images_in_input=cfg.num_images_in_input,
        require_full_image_history=cfg.require_full_image_history,
        use_cond_action_tokens=cfg.use_cond_action_tokens,
        load_future_images=(
            cfg.use_cond_action_tokens
            and (
                cfg.use_indoor_uav_condition_adapter
                or cfg.condition_alignment_weight > 0
                or cfg.condition_contrastive_weight > 0
                or cfg.condition_temporal_weight > 0
                or cfg.condition_queue_weight > 0
                or cfg.condition_diversity_weight > 0
                or cfg.condition_assignment_weight > 0
            )
        ),
        num_action_branches=cfg.num_action_branches,
        use_reference_previous_current=cfg.use_reference_previous_current,
        body_delta_action_targets=cfg.body_delta_action_targets,
    )
    train_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        tfds_split=cfg.train_tfds_split,
        window_size=window_size,
        relative_action_targets=cfg.relative_action_targets,
        future_action_stride=cfg.future_action_stride,
        relative_action_wrap_yaw=cfg.relative_action_wrap_yaw,
        body_delta_action_targets=cfg.body_delta_action_targets,
        cyclic_yaw_proprio=cfg.cyclic_yaw_proprio,
        use_reference_previous_current=cfg.use_reference_previous_current,
    )
    if cfg.use_val_set:
        val_dataset = RLDSDataset(
            cfg.data_root_dir,
            cfg.dataset_name,
            batch_transform,
            resize_resolution=tuple(vla.module.config.image_sizes),
            shuffle_buffer_size=cfg.shuffle_buffer_size // 10,
            image_aug=False,
            tfds_split=cfg.val_tfds_split,
            train=False,
            window_size=window_size,
            relative_action_targets=cfg.relative_action_targets,
            future_action_stride=cfg.future_action_stride,
            relative_action_wrap_yaw=cfg.relative_action_wrap_yaw,
            body_delta_action_targets=cfg.body_delta_action_targets,
            cyclic_yaw_proprio=cfg.cyclic_yaw_proprio,
            use_reference_previous_current=cfg.use_reference_previous_current,
        )

    if cfg.use_indoor_uav_stop_head and cfg.stop_positive_weight == 0:
        stop_stats = train_dataset.dataset_statistics[cfg.dataset_name]
        num_transitions = float(stop_stats["num_transitions"])
        num_trajectories = float(stop_stats["num_trajectories"])
        if not 0 < num_trajectories < num_transitions:
            raise ValueError("invalid transition/trajectory counts for STOP class balancing")
        cfg.stop_positive_weight = (num_transitions - num_trajectories) / num_trajectories
    cfg.stop_progress_positive_weights = None
    if cfg.use_indoor_uav_progress_stop_head:
        stop_stats = train_dataset.dataset_statistics[cfg.dataset_name]
        num_transitions = float(stop_stats["num_transitions"])
        num_trajectories = float(stop_stats["num_trajectories"])
        positive_counts = [
            min(num_transitions, (horizon + 1) * num_trajectories)
            for horizon in STOP_PROGRESS_HORIZONS
        ]
        cfg.stop_progress_positive_weights = tuple(
            (num_transitions - count) / count for count in positive_counts
        )
    if cfg.use_indoor_uav_stop_head:
        print(
            "[STOP] target=terminate after root action, "
            f"positive_weight={cfg.stop_positive_weight:.6f}, "
            f"loss_weight={cfg.stop_loss_weight:.6f}, threshold={cfg.stop_threshold:.3f}"
        )
        if cfg.use_indoor_uav_progress_stop_head:
            print(
                f"[STOP] progress_horizons={STOP_PROGRESS_HORIZONS}, "
                f"progress_positive_weights={cfg.stop_progress_positive_weights}, "
                f"progress_loss_weight={cfg.stop_progress_loss_weight:.6f}"
            )

    # [Important] Save dataset statistics so that we can unnormalize actions during inference
    if distributed_state.is_main_process:
        _print_dataset_statistics(train_dataset.dataset_statistics)
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)
        save_policy_contract(cfg, train_dataset.dataset_statistics, run_dir)
    action_norm_stats = _get_action_norm_stats(train_dataset.dataset_statistics, cfg.dataset_name)
    if cfg.use_cond_action_tokens:
        cond_token_ids, act_token_ids = get_cond_action_token_id_tensors(
            processor.tokenizer, cfg.num_action_branches, device_id
        )
    else:
        cond_token_ids, act_token_ids = None, None

    # Create collator and dataloader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
    )
    if cfg.use_val_set:
        val_batch_size = cfg.batch_size
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
        )

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_metrics = {
        "loss_value": deque(maxlen=cfg.grad_accumulation_steps),
        "stop_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "stop_probability_mean": deque(maxlen=cfg.grad_accumulation_steps),
        "stop_target_rate": deque(maxlen=cfg.grad_accumulation_steps),
        "stop_predicted_rate": deque(maxlen=cfg.grad_accumulation_steps),
        "stop_progress_aux_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "stop_progress_total_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "all_branches_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "best_branch_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "best_of_k_time_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "best_of_k_action_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "best_of_k_gaussian_nll": deque(maxlen=cfg.grad_accumulation_steps),
        "sft_root_action_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "sft_future_action_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "action_branch_pair_l1": deque(maxlen=cfg.grad_accumulation_steps),
        "action_branch_min_pair_l1": deque(maxlen=cfg.grad_accumulation_steps),
        "action_branch_position_separation_m": deque(maxlen=cfg.grad_accumulation_steps),
        "action_branch_yaw_separation_rad": deque(maxlen=cfg.grad_accumulation_steps),
        "action_winner_gap": deque(maxlen=cfg.grad_accumulation_steps),
        "action_winner_near_tie_rate": deque(maxlen=cfg.grad_accumulation_steps),
        "oracle_future_action_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "condition_selected_action_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "branch0_future_action_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "condition_selection_regret": deque(maxlen=cfg.grad_accumulation_steps),
        "condition_gain_vs_branch0": deque(maxlen=cfg.grad_accumulation_steps),
        "condition_oracle_recovery": deque(maxlen=cfg.grad_accumulation_steps),
        "oracle_action_position_error_m": deque(maxlen=cfg.grad_accumulation_steps),
        "condition_selected_position_error_m": deque(maxlen=cfg.grad_accumulation_steps),
        "branch0_position_error_m": deque(maxlen=cfg.grad_accumulation_steps),
        "oracle_action_yaw_error_rad": deque(maxlen=cfg.grad_accumulation_steps),
        "condition_selected_yaw_error_rad": deque(maxlen=cfg.grad_accumulation_steps),
        "branch0_yaw_error_rad": deque(maxlen=cfg.grad_accumulation_steps),
        "plan_valid_ratio": deque(maxlen=cfg.grad_accumulation_steps),
        "gaussian_nll_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "gaussian_log_std_mean": deque(maxlen=cfg.grad_accumulation_steps),
        "gaussian_log_std_min": deque(maxlen=cfg.grad_accumulation_steps),
        "gaussian_log_std_max": deque(maxlen=cfg.grad_accumulation_steps),
        "gaussian_policy_std_mean": deque(maxlen=cfg.grad_accumulation_steps),
        "gradient_norm_before_clip": deque(maxlen=cfg.grad_accumulation_steps),
        "branch_balance_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "branch_assignment_entropy": deque(maxlen=cfg.grad_accumulation_steps),
        "branch0_winner_rate": deque(maxlen=cfg.grad_accumulation_steps),
        "branch1_winner_rate": deque(maxlen=cfg.grad_accumulation_steps),
        "branch2_winner_rate": deque(maxlen=cfg.grad_accumulation_steps),
        "branch0_soft_usage": deque(maxlen=cfg.grad_accumulation_steps),
        "branch1_soft_usage": deque(maxlen=cfg.grad_accumulation_steps),
        "branch2_soft_usage": deque(maxlen=cfg.grad_accumulation_steps),
        "branch_diversity_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "branch_mean_distance": deque(maxlen=cfg.grad_accumulation_steps),
        "format_cond_token_count": deque(maxlen=cfg.grad_accumulation_steps),
        "format_act_token_count": deque(maxlen=cfg.grad_accumulation_steps),
        "format_complete_rate": deque(maxlen=cfg.grad_accumulation_steps),
        "grpo_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "grpo_advantage_mean": deque(maxlen=cfg.grad_accumulation_steps),
        "grpo_advantage_std": deque(maxlen=cfg.grad_accumulation_steps),
        "grpo_policy_mse": deque(maxlen=cfg.grad_accumulation_steps),
        "grpo_best_branch_mean": deque(maxlen=cfg.grad_accumulation_steps),
        "offline_reward_mean": deque(maxlen=cfg.grad_accumulation_steps),
        "offline_reward_best": deque(maxlen=cfg.grad_accumulation_steps),
        "offline_best_branch_mean": deque(maxlen=cfg.grad_accumulation_steps),
        "offline_final_pos_error": deque(maxlen=cfg.grad_accumulation_steps),
        "offline_final_yaw_error": deque(maxlen=cfg.grad_accumulation_steps),
        "offline_traj_pos_error": deque(maxlen=cfg.grad_accumulation_steps),
        "offline_traj_yaw_error": deque(maxlen=cfg.grad_accumulation_steps),
        "offline_z_below_zero_rate": deque(maxlen=cfg.grad_accumulation_steps),
        "offline_success_rate": deque(maxlen=cfg.grad_accumulation_steps),
        "offline_branch0_reward": deque(maxlen=cfg.grad_accumulation_steps),
        "offline_branch1_reward": deque(maxlen=cfg.grad_accumulation_steps),
        "offline_branch2_reward": deque(maxlen=cfg.grad_accumulation_steps),
        "condition_similarity_mean": deque(maxlen=cfg.grad_accumulation_steps),
        "condition_similarity_best": deque(maxlen=cfg.grad_accumulation_steps),
        "condition_similarity_selected": deque(maxlen=cfg.grad_accumulation_steps),
        "condition_similarity_margin": deque(maxlen=cfg.grad_accumulation_steps),
        "condition_selected_branch_mean": deque(maxlen=cfg.grad_accumulation_steps),
        "condition_action_branch_match_rate": deque(maxlen=cfg.grad_accumulation_steps),
        "condition_threshold_pass_rate": deque(maxlen=cfg.grad_accumulation_steps),
        "condition_future_threshold_pass_rate": deque(maxlen=cfg.grad_accumulation_steps),
        "condition_loss_start_time_index": deque(maxlen=cfg.grad_accumulation_steps),
        "condition_patch_topk": deque(maxlen=cfg.grad_accumulation_steps),
        "condition_matching_centered": deque(maxlen=cfg.grad_accumulation_steps),
        "condition_contrastive_num_branches": deque(maxlen=cfg.grad_accumulation_steps),
        "condition_centered_norm_mean": deque(maxlen=cfg.grad_accumulation_steps),
        "condition_patch_centered_norm_mean": deque(maxlen=cfg.grad_accumulation_steps),
        "condition_alignment_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "condition_contrastive_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "condition_branch_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "condition_branch_margin": deque(maxlen=cfg.grad_accumulation_steps),
        "condition_temporal_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "condition_retrieval_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "condition_retrieval_positive": deque(maxlen=cfg.grad_accumulation_steps),
        "condition_retrieval_hardest_negative": deque(maxlen=cfg.grad_accumulation_steps),
        "condition_retrieval_margin": deque(maxlen=cfg.grad_accumulation_steps),
        "condition_queue_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "condition_queue_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "condition_queue_margin": deque(maxlen=cfg.grad_accumulation_steps),
        "condition_queue_queries": deque(maxlen=cfg.grad_accumulation_steps),
        "condition_queue_negatives": deque(maxlen=cfg.grad_accumulation_steps),
        "condition_queue_size": deque(maxlen=cfg.grad_accumulation_steps),
        "condition_contrastive_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "condition_contrastive_margin": deque(maxlen=cfg.grad_accumulation_steps),
        "condition_diversity_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "condition_mean_distance": deque(maxlen=cfg.grad_accumulation_steps),
    }
    for horizon in STOP_PROGRESS_HORIZONS:
        for suffix in ("probability_mean", "target_rate", "accuracy"):
            recent_metrics[f"stop_progress_h{horizon}_{suffix}"] = deque(
                maxlen=cfg.grad_accumulation_steps
            )
        if horizon > 0:
            recent_metrics[f"stop_progress_h{horizon}_loss"] = deque(
                maxlen=cfg.grad_accumulation_steps
            )
    for axis_name in ("forward", "right", "up", "yaw"):
        for suffix in ("prediction_mean", "target_mean", "bias", "abs_error", "sign_accuracy"):
            recent_metrics[f"root_{axis_name}_{suffix}"] = deque(
                maxlen=cfg.grad_accumulation_steps
            )
    for branch_idx in range(cfg.num_action_branches):
        recent_metrics.setdefault(
            f"branch{branch_idx}_winner_rate", deque(maxlen=cfg.grad_accumulation_steps)
        )
        recent_metrics.setdefault(
            f"branch{branch_idx}_soft_usage", deque(maxlen=cfg.grad_accumulation_steps)
        )
        recent_metrics.setdefault(
            f"condition_branch{branch_idx}_selected_rate",
            deque(maxlen=cfg.grad_accumulation_steps),
        )
        recent_metrics.setdefault(
            f"offline_branch{branch_idx}_reward", deque(maxlen=cfg.grad_accumulation_steps)
        )
        recent_metrics.setdefault(
            f"offline_branch{branch_idx}_final_pos_error",
            deque(maxlen=cfg.grad_accumulation_steps),
        )

    # Start training 真正开始训练（核心）
    fixed_overfit_batches = []
    overfit_loss_window = deque(maxlen=max(cfg.overfit_fixed_batch_count, 1))
    training_condition_queue = (
        CrossEpisodeImageQueue(cfg.condition_queue_size)
        if condition_adapter is not None and cfg.condition_queue_weight > 0
        else None
    )
    if cfg.overfit_fixed_batch_count > 0 and distributed_state.is_main_process:
        print(
            "[Overfit diagnostic] Repeating the first "
            f"{cfg.overfit_fixed_batch_count} batches; this run does not measure generalization."
        )
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        if cfg.freeze_vla:
            vla.eval()
        else:
            vla.train()
        if cfg.use_proprio and cfg.freeze_proprio_projector:
            proprio_projector.eval()
        if cfg.use_l1_regression or cfg.use_diffusion:
            action_head.eval() if cfg.freeze_action_head else action_head.train()
        if condition_adapter is not None:
            condition_adapter.eval() if cfg.freeze_condition_adapter else condition_adapter.train()
        if stop_head is not None:
            stop_head.train()
        optimizer.zero_grad()
        for batch_idx, incoming_batch in enumerate(dataloader):
            batch = select_overfit_batch(
                batch_idx,
                incoming_batch,
                fixed_overfit_batches,
                cfg.overfit_fixed_batch_count,
            )
            # Compute training metrics and loss
            compute_diffusion_l1 = cfg.use_diffusion and batch_idx % cfg.diffusion_sample_freq == 0
            loss, metrics = run_forward_pass(
                vla=vla,
                action_head=action_head,
                condition_adapter=condition_adapter,
                stop_head=stop_head,
                noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                proprio_projector=proprio_projector if cfg.use_proprio else None,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_diffusion=cfg.use_diffusion,
                use_gaussian_action_head=cfg.use_gaussian_action_head,
                action_regression_loss=cfg.action_regression_loss,
                num_action_branches=cfg.num_action_branches,
                use_best_of_k_action_loss=cfg.use_best_of_k_action_loss,
                branch_assignment_temperature=cfg.branch_assignment_temperature,
                branch_balance_weight=cfg.branch_balance_weight,
                condition_assignment_weight=cfg.condition_assignment_weight,
                initial_action_branch_index=cfg.initial_action_branch_index,
                branch_diversity_weight=cfg.branch_diversity_weight,
                branch_diversity_margin=cfg.branch_diversity_margin,
                grpo_reward_weight=cfg.grpo_reward_weight,
                grpo_policy_sigma=cfg.grpo_policy_sigma,
                grpo_group_size=cfg.grpo_group_size,
                grpo_clip_epsilon=cfg.grpo_clip_epsilon,
                grpo_safety_weight=cfg.grpo_safety_weight,
                grpo_advantage_eps=cfg.grpo_advantage_eps,
                grpo_advantage_clip=cfg.grpo_advantage_clip,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=NUM_PATCHES,
                action_norm_stats=action_norm_stats,
                use_cond_action_tokens=cfg.use_cond_action_tokens,
                cond_token_ids=cond_token_ids,
                act_token_ids=act_token_ids,
                couple_condition_to_action_branch=cfg.couple_condition_to_action_branch,
                condition_similarity_threshold=cfg.condition_similarity_threshold,
                condition_alignment_weight=cfg.condition_alignment_weight,
                condition_contrastive_weight=cfg.condition_contrastive_weight,
                condition_temporal_weight=cfg.condition_temporal_weight,
                condition_queue_weight=cfg.condition_queue_weight,
                condition_queue_min_negatives=cfg.condition_queue_min_negatives,
                condition_negative_queue=training_condition_queue,
                condition_contrastive_temperature=cfg.condition_contrastive_temperature,
                condition_loss_start_time_index=cfg.condition_loss_start_time_index,
                condition_patch_topk=cfg.condition_patch_topk,
                condition_diversity_weight=cfg.condition_diversity_weight,
                condition_diversity_margin=cfg.condition_diversity_margin,
                root_action_weight=cfg.root_action_weight,
                future_action_weight=cfg.future_action_weight,
                stop_loss_weight=cfg.stop_loss_weight,
                stop_positive_weight=cfg.stop_positive_weight,
                stop_threshold=cfg.stop_threshold,
                use_progress_stop_head=cfg.use_indoor_uav_progress_stop_head,
                stop_progress_positive_weights=cfg.stop_progress_positive_weights,
                stop_progress_loss_weight=cfg.stop_progress_loss_weight,
                compute_diffusion_l1=compute_diffusion_l1,
                num_diffusion_steps_train=cfg.num_diffusion_steps_train if cfg.use_diffusion else None,
                debug_batch_shapes=cfg.debug_batch_shapes and batch_idx < cfg.debug_num_batches,
            )

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()

            gradient_step_boundary = (batch_idx + 1) % cfg.grad_accumulation_steps == 0
            if gradient_step_boundary and cfg.max_grad_norm is not None:
                total_grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
                metrics["gradient_norm_before_clip"] = float(total_grad_norm)

            if (
                cfg.debug_batch_shapes
                and distributed_state.is_main_process
                and batch_idx < cfg.debug_num_batches
            ):
                print(f"\n[Debug] Batch {batch_idx} diagnostics:")
                for key, value in metrics.items():
                    if key.startswith("debug_"):
                        print(f"  {key.removeprefix('debug_')}: {value}")
                print(f"\n[Debug] Batch {batch_idx} scalar metrics:")
                for key, value in metrics.items():
                    if not key.startswith("debug_"):
                        print(f"  {key}: {value}")

            if cfg.debug_grad_norm and distributed_state.is_main_process and batch_idx < cfg.debug_num_batches:
                grad_norms = {
                    "vla": _module_grad_norm(vla),
                    "action_head": _module_grad_norm(
                        action_head if (cfg.use_l1_regression or cfg.use_diffusion) else None
                    ),
                    "condition_adapter": _module_grad_norm(condition_adapter),
                    "stop_head": _module_grad_norm(stop_head),
                    "proprio_projector": _module_grad_norm(proprio_projector if cfg.use_proprio else None),
                    "noisy_action_projector": _module_grad_norm(noisy_action_projector if cfg.use_diffusion else None),
                }
                print(f"\n[Debug] Batch {batch_idx} grad norms:")
                for key, value in grad_norms.items():
                    print(f"  {key}: {value}")

            # Store recent train metrics
            for metric_name, value in metrics.items():
                if metric_name in recent_metrics:
                    recent_metrics[metric_name].append(value)
            if cfg.overfit_fixed_batch_count > 0:
                overfit_loss_window.append(float(metrics["loss_value"]))

            # Compute gradient step index
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            if (
                cfg.overfit_fixed_batch_count > 0
                and gradient_step_boundary
                and distributed_state.is_main_process
            ):
                overfit_step = gradient_step_idx + 1
                if overfit_step == 1 or overfit_step % cfg.overfit_report_freq == 0:
                    diagnostic_names = (
                        "stop_loss",
                        "stop_probability_mean",
                        "stop_target_rate",
                        "stop_predicted_rate",
                        "stop_progress_aux_loss",
                        "stop_progress_total_loss",
                        "sft_root_action_loss",
                        "sft_future_action_loss",
                        "condition_alignment_loss",
                        "condition_contrastive_loss",
                        "condition_branch_accuracy",
                        "condition_branch_margin",
                        "condition_temporal_loss",
                        "condition_retrieval_accuracy",
                        "condition_retrieval_margin",
                        "condition_queue_loss",
                        "condition_queue_accuracy",
                        "condition_queue_margin",
                        "condition_similarity_selected",
                    ) + tuple(
                        metric_name
                        for branch_idx in range(cfg.num_action_branches)
                        for metric_name in (
                            f"branch{branch_idx}_winner_rate",
                            f"branch{branch_idx}_soft_usage",
                        )
                    )
                    diagnostic_values = " ".join(
                        f"{name}={sum(recent_metrics[name]) / len(recent_metrics[name]):.6f}"
                        for name in diagnostic_names
                        if recent_metrics.get(name)
                    )
                    print(
                        f"[Overfit diagnostic] step={overfit_step} "
                        f"cached_batches={len(fixed_overfit_batches)} "
                        f"recent_mean_loss={sum(overfit_loss_window) / len(overfit_loss_window):.6f} "
                        f"{diagnostic_values}"
                    )

            log_step = completed_optimizer_step(
                batch_idx,
                cfg.grad_accumulation_steps,
                cfg.resume_step if cfg.resume else 0,
            )
            if log_step is None:
                continue

            # Apply warmup once per optimizer step, immediately before updating parameters.
            if cfg.lr_warmup_steps > 0:
                lr_progress = min(log_step / cfg.lr_warmup_steps, 1.0)
                current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.update()

            # Log/save/validate exactly once per completed optimizer step.
            smoothened_metrics = compute_smoothened_metrics(recent_metrics)
            if distributed_state.is_main_process and log_step % cfg.wandb_log_freq == 0:
                log_metrics_to_wandb(smoothened_metrics, "VLA Train", log_step, wandb)
                wandb.log(
                    {"VLA Train/Learning Rate": optimizer.param_groups[0]["lr"]},
                    step=log_step,
                )
            if distributed_state.is_main_process and (
                log_step == 1 or log_step % cfg.train_report_freq == 0
            ):
                report_keys = (
                    "loss_value",
                    "stop_loss",
                    "stop_probability_mean",
                    "stop_target_rate",
                    "stop_predicted_rate",
                    "stop_progress_aux_loss",
                    "stop_progress_total_loss",
                    "sft_root_action_loss",
                    "sft_future_action_loss",
                    "root_forward_bias",
                    "root_right_bias",
                    "root_up_bias",
                    "root_yaw_bias",
                    "condition_selected_action_loss",
                    "branch0_future_action_loss",
                    "condition_selection_regret",
                    "condition_gain_vs_branch0",
                    "condition_oracle_recovery",
                    "condition_contrastive_loss",
                    "condition_branch_accuracy",
                    "condition_branch_margin",
                    "condition_temporal_loss",
                    "condition_retrieval_accuracy",
                    "condition_retrieval_margin",
                    "condition_queue_loss",
                    "condition_queue_accuracy",
                    "condition_queue_margin",
                    "condition_queue_queries",
                    "condition_queue_negatives",
                    "condition_queue_size",
                    "branch0_soft_usage",
                    "branch1_soft_usage",
                    "branch2_soft_usage",
                    "gradient_norm_before_clip",
                )
                report = ", ".join(
                    f"{key}={smoothened_metrics[key]:.6f}"
                    for key in report_keys
                    if key in smoothened_metrics
                )
                print(f"[Train] step={log_step}, lr={optimizer.param_groups[0]['lr']:.8f}, {report}")

            # Save model checkpoint: either keep latest checkpoint only or all checkpoints
            if log_step % cfg.save_freq == 0:
                save_training_checkpoint(
                    cfg=cfg,
                    run_dir=run_dir,
                    log_step=log_step,
                    vla=vla,
                    processor=processor,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                    action_head=action_head if (cfg.use_l1_regression or cfg.use_diffusion) else None,
                    condition_adapter=condition_adapter,
                    stop_head=stop_head,
                    train_dataset=train_dataset,
                    distributed_state=distributed_state,
                )

            # Test model on validation set
            if cfg.use_val_set and log_step % cfg.val_freq == 0:
                run_validation(
                    vla=vla,
                    action_head=action_head,
                    condition_adapter=condition_adapter,
                    stop_head=stop_head,
                    noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    val_dataloader=val_dataloader,
                    action_tokenizer=action_tokenizer,
                    device_id=device_id,
                    cfg=cfg,
                    num_patches=NUM_PATCHES,
                    log_step=log_step,
                    distributed_state=distributed_state,
                    val_time_limit=cfg.val_time_limit,
                    action_norm_stats=action_norm_stats,
                    cond_token_ids=cond_token_ids,
                    act_token_ids=act_token_ids,
                    run_dir=run_dir,
                )
                # Restore the exact modes used by the optimizer.  A frozen VLA
                # must stay in eval mode so dropout cannot move STOP features.
                vla.eval() if cfg.freeze_vla else vla.train()
                if cfg.use_proprio:
                    proprio_projector.eval() if cfg.freeze_proprio_projector else proprio_projector.train()
                if cfg.use_l1_regression or cfg.use_diffusion:
                    action_head.eval() if cfg.freeze_action_head else action_head.train()
                if condition_adapter is not None:
                    condition_adapter.eval() if cfg.freeze_condition_adapter else condition_adapter.train()
                if stop_head is not None:
                    stop_head.train()

            # Stop training when max_steps is reached
            if log_step >= cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune()
