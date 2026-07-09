"""
finetune.py

Fine-tunes OpenVLA via LoRA.
"""

import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Type

import draccus
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

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map,
)

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import DiffusionActionHead, L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
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


def _unnormalize_actions_for_reward(actions: torch.Tensor, action_norm_stats: Optional[dict]) -> torch.Tensor:
    """Convert normalized action/state predictions back to real units for offline reward metrics."""
    actions = actions.float()
    if action_norm_stats is None:
        return actions

    if "q01" in action_norm_stats and "q99" in action_norm_stats:
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


def _wrapped_abs_yaw_error(pred_yaw: torch.Tensor, target_yaw: torch.Tensor) -> torch.Tensor:
    diff = torch.remainder(pred_yaw - target_yaw + torch.pi, 2 * torch.pi) - torch.pi
    return diff.abs()


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

    # Algorithm and architecture
    use_l1_regression: bool = True                   # If True, trains continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, trains continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_action_branches: int = 1                     # Number of supervised action branches to predict for L1 regression
    use_cond_action_tokens: bool = False             # If True, use explicit T x K COND/ACT placeholder tokens
    branch_diversity_weight: float = 0.0             # Weight for multi-branch diversity regularization
    branch_diversity_margin: float = 0.05            # Minimum desired mean L1 distance between action branches
    grpo_reward_weight: float = 0.0                  # Weight for GRPO-style branch reward optimization
    grpo_policy_sigma: float = 1.0                   # Fixed Gaussian sigma for continuous-action GRPO surrogate
    grpo_advantage_eps: float = 1e-4                 # Numerical stability constant for group advantage normalization
    grpo_advantage_clip: float = 5.0                 # Clips group-relative advantages before applying GRPO loss
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 1                     # Number of images in the VLA input (default: 1)
    use_image_history: bool = False                  # If True, uses num_images_in_input primary-camera history frames
    require_full_image_history: bool = True          # If True, skips chunks with padded history frames
    use_proprio: bool = False                        # If True, includes robot proprioceptive state in input

    # Training configuration
    batch_size: int = 8                              # Batch size per device (total batch size = batch_size * num GPUs)
    learning_rate: float = 5e-4                      # Learning rate
    lr_warmup_steps: int = 0                         # Number of steps to warm up learning rate (from 10% to 100%)
    num_steps_before_decay: int = 100_000            # Number of steps before LR decays by 10x
    grad_accumulation_steps: int = 1                 # Number of gradient accumulation steps
    max_steps: int = 200_000                         # Max number of training steps
    use_val_set: bool = False                        # If True, uses validation set and log validation metrics
    val_freq: int = 10_000                           # (When `use_val_set==True`) Validation set logging frequency in steps
    val_time_limit: int = 180                        # (When `use_val_set==True`) Time limit for computing validation metrics
    save_freq: int = 10_000                          # Checkpoint saving frequency in steps
    save_latest_checkpoint_only: bool = False        # If True, saves only 1 checkpoint, overwriting latest checkpoint
                                                     #   (If False, saves all checkpoints)
    resume: bool = False                             # If True, resumes from checkpoint 断点重训，从checkpoint继续训练
    resume_step: Optional[int] = None                # (When `resume==True`) Step number that we are resuming from
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
    debug_batch_shapes: bool = False                 # If True, print batch/action/mask shapes for initial batches
    debug_grad_norm: bool = False                    # If True, print gradient norms for trainable components
    debug_num_batches: int = 2                       # Number of initial batches to print when debug flags are enabled

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

    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)

    return wrap_ddp(module, device_id, find_unused_params)

# 它把一个 batch 喂给 VLA，拿到 hidden states，再用 action head 预测动作，最后算 loss 和日志指标
def run_forward_pass(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    batch,
    action_tokenizer,
    device_id,
    use_l1_regression,
    use_diffusion,
    num_action_branches,
    branch_diversity_weight,
    branch_diversity_margin,
    grpo_reward_weight,
    grpo_policy_sigma,
    grpo_advantage_eps,
    grpo_advantage_clip,
    use_proprio,
    use_film,
    num_patches,
    action_norm_stats=None,
    use_cond_action_tokens=False,
    cond_token_ids=None,
    act_token_ids=None,
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
    proprio = batch["proprio"].to(device_id).to(torch.bfloat16) if use_proprio else None
    labels = batch["labels"].to(device_id)
    debug_info = {}

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
            # Predict action,输出的是(B, NUM_ACTIONS_CHUNK, ACTION_DIM)的连续动作=(B,8,7)
            predicted_actions = action_head.module.predict_action(actions_hidden_states)
            if debug_batch_shapes:
                debug_info["predicted_actions"] = _shape(predicted_actions)
                debug_info["predicted_actions_device"] = _device(predicted_actions)
            # Get full L1 loss,和专家动作的L1 loss
            if num_action_branches > 1:
                ground_truth_actions_for_loss = ground_truth_actions.unsqueeze(2).expand_as(predicted_actions)
            else:
                ground_truth_actions_for_loss = ground_truth_actions
            loss = torch.nn.L1Loss()(ground_truth_actions_for_loss, predicted_actions)
            if predicted_actions.ndim == 4 and branch_diversity_weight > 0:
                branch_pair_distances = []
                for left_branch in range(predicted_actions.shape[2]):
                    for right_branch in range(left_branch + 1, predicted_actions.shape[2]):
                        branch_pair_distances.append(
                            torch.abs(
                                predicted_actions[:, :, left_branch] - predicted_actions[:, :, right_branch]
                            ).mean(dim=(1, 2))
                        )
                branch_pair_distances = torch.stack(branch_pair_distances, dim=1)
                branch_mean_distance = branch_pair_distances.mean()
                branch_diversity_loss = torch.relu(branch_diversity_margin - branch_pair_distances).mean()
                loss = loss + branch_diversity_weight * branch_diversity_loss
                metrics.update(
                    {
                        "branch_diversity_loss": branch_diversity_loss.item(),
                        "branch_mean_distance": branch_mean_distance.item(),
                    }
                )
            if predicted_actions.ndim == 4 and grpo_reward_weight > 0:
                grpo_loss, grpo_metrics = compute_grpo_branch_loss(
                    predicted_actions=predicted_actions,
                    ground_truth_actions=ground_truth_actions,
                    action_norm_stats=action_norm_stats,
                    advantage_eps=grpo_advantage_eps,
                    advantage_clip=grpo_advantage_clip,
                    policy_sigma=grpo_policy_sigma,
                )
                loss = loss + grpo_reward_weight * grpo_loss
                metrics.update(grpo_metrics)

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
            predicted_actions_for_metrics = predicted_actions[:, :, 0] if predicted_actions.ndim == 4 else predicted_actions
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
            if predicted_actions.ndim == 4:
                branch_targets = ground_truth_actions.unsqueeze(2).expand_as(predicted_actions)
                per_branch_l1 = torch.abs(predicted_actions - branch_targets).mean(dim=(1, 3))
                l1_metrics["all_branches_l1_loss"] = per_branch_l1.mean().item()
                l1_metrics["best_branch_l1_loss"] = per_branch_l1.min(dim=1).values.mean().item()
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

        if distributed_state.is_main_process:
            merged_vla.save_pretrained(checkpoint_dir)
            print(f"Saved merged model for Step {log_step} at: {checkpoint_dir}")

        # Wait for merged model to be saved
        _distributed_barrier()


# 在验证集上计算指标
def run_validation(
    vla,
    action_head,
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

    with torch.no_grad():
        for batch in val_dataloader:
            # Always compute L1 loss for validation, even for diffusion
            _, metrics = run_forward_pass(
                vla=vla,
                action_head=action_head,
                noisy_action_projector=noisy_action_projector,
                proprio_projector=proprio_projector,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_diffusion=cfg.use_diffusion,
                num_action_branches=cfg.num_action_branches,
                branch_diversity_weight=cfg.branch_diversity_weight,
                branch_diversity_margin=cfg.branch_diversity_margin,
                grpo_reward_weight=cfg.grpo_reward_weight,
                grpo_policy_sigma=cfg.grpo_policy_sigma,
                grpo_advantage_eps=cfg.grpo_advantage_eps,
                grpo_advantage_clip=cfg.grpo_advantage_clip,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=num_patches,
                action_norm_stats=action_norm_stats,
                use_cond_action_tokens=cfg.use_cond_action_tokens,
                cond_token_ids=cond_token_ids,
                act_token_ids=act_token_ids,
                compute_diffusion_l1=True,
                num_diffusion_steps_train=cfg.num_diffusion_steps_train if cfg.use_diffusion else None,
            )

            # Add the loss value to the metrics
            metrics["loss"] = metrics["loss_value"]
            all_val_metrics.append(metrics)
            val_batches_count += 1

            # Cut testing on validation set short if it exceeds time limit
            if time.time() - val_start_time > val_time_limit:
                break

    # Compute average validation metrics
    avg_val_metrics = {}
    for metric_name in all_val_metrics[0].keys():
        values = [metrics[metric_name] for metrics in all_val_metrics if metric_name in metrics]
        if values:
            avg_val_metrics[metric_name] = sum(values) / len(values)

    # Add batch count to metrics
    avg_val_metrics["val_batches_count"] = val_batches_count

    # Log validation metrics to W&B
    if distributed_state.is_main_process:
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
    if cfg.num_action_branches < 1:
        raise ValueError("num_action_branches must be >= 1")
    if cfg.num_action_branches > 1 and not cfg.use_l1_regression:
        raise ValueError("num_action_branches > 1 is currently supported only with use_l1_regression=True")
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
    if cfg.grpo_reward_weight > 0 and cfg.num_action_branches < 2:
        raise ValueError("grpo_reward_weight > 0 requires num_action_branches >= 2")
    if cfg.grpo_policy_sigma <= 0:
        raise ValueError("grpo_policy_sigma must be > 0")
    if cfg.grpo_advantage_eps <= 0:
        raise ValueError("grpo_advantage_eps must be > 0")
    if cfg.grpo_advantage_clip <= 0:
        raise ValueError("grpo_advantage_clip must be > 0")

    # Trim trailing forward slash ('/') in VLA path if it exists
    cfg.vla_path = cfg.vla_path.rstrip("/")
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # Get experiment run ID
    run_id = get_run_id(cfg)

    # Create experiment run directory
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

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
            {"llm_dim": vla.module.llm_dim, "proprio_dim": PROPRIO_DIM},
        )

    # If applicable, instantiate continuous action head for L1 regression
    if cfg.use_l1_regression:
        action_head = init_module(
            L1RegressionActionHead,
            "action_head",
            cfg,
            device_id,
            {
                "input_dim": vla.module.llm_dim,
                "hidden_dim": vla.module.llm_dim,
                "action_dim": ACTION_DIM,
                "num_action_branches": cfg.num_action_branches,
                "use_cond_action_tokens": cfg.use_cond_action_tokens,
            },
            to_bf16=True,
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

    # Instantiate optimizer 收集所有可训练参数
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    if cfg.use_l1_regression or cfg.use_diffusion:
        trainable_params += [param for param in action_head.parameters() if param.requires_grad]
    if cfg.use_diffusion:
        trainable_params += [param for param in noisy_action_projector.parameters() if param.requires_grad]
    if cfg.use_proprio:
        trainable_params += [param for param in proprio_projector.parameters() if param.requires_grad]
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

    # Multi-image IndoorUAV uses primary-camera history, not wrist cameras.
    use_wrist_image = cfg.num_images_in_input > 1 and not cfg.use_image_history
    window_size = cfg.num_images_in_input if cfg.use_image_history else 1

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
        num_action_branches=cfg.num_action_branches,
    )
    train_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        window_size=window_size,
    )
    if cfg.use_val_set:
        val_dataset = RLDSDataset(
            cfg.data_root_dir,
            cfg.dataset_name,
            batch_transform,
            resize_resolution=tuple(vla.module.config.image_sizes),
            shuffle_buffer_size=cfg.shuffle_buffer_size // 10,
            image_aug=cfg.image_aug,
            train=False,
            window_size=window_size,
        )

    # [Important] Save dataset statistics so that we can unnormalize actions during inference
    if distributed_state.is_main_process:
        _print_dataset_statistics(train_dataset.dataset_statistics)
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)
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
        "curr_action_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "all_branches_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "best_branch_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
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
    }

    # Start training 真正开始训练（核心）
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            # Compute training metrics and loss
            compute_diffusion_l1 = cfg.use_diffusion and batch_idx % cfg.diffusion_sample_freq == 0
            loss, metrics = run_forward_pass(
                vla=vla,
                action_head=action_head,
                noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                proprio_projector=proprio_projector if cfg.use_proprio else None,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_diffusion=cfg.use_diffusion,
                num_action_branches=cfg.num_action_branches,
                branch_diversity_weight=cfg.branch_diversity_weight,
                branch_diversity_margin=cfg.branch_diversity_margin,
                grpo_reward_weight=cfg.grpo_reward_weight,
                grpo_policy_sigma=cfg.grpo_policy_sigma,
                grpo_advantage_eps=cfg.grpo_advantage_eps,
                grpo_advantage_clip=cfg.grpo_advantage_clip,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=NUM_PATCHES,
                action_norm_stats=action_norm_stats,
                use_cond_action_tokens=cfg.use_cond_action_tokens,
                cond_token_ids=cond_token_ids,
                act_token_ids=act_token_ids,
                compute_diffusion_l1=compute_diffusion_l1,
                num_diffusion_steps_train=cfg.num_diffusion_steps_train if cfg.use_diffusion else None,
                debug_batch_shapes=cfg.debug_batch_shapes and batch_idx < cfg.debug_num_batches,
            )

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()

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
                    "action_head": _module_grad_norm(action_head if (cfg.use_l1_regression or cfg.use_diffusion) else None),
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

            # Compute gradient step index
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            # Compute smoothened train metrics
            smoothened_metrics = compute_smoothened_metrics(recent_metrics)

            # Push Metrics to W&B (every wandb_log_freq gradient steps)
            log_step = gradient_step_idx if not cfg.resume else cfg.resume_step + gradient_step_idx
            if distributed_state.is_main_process and log_step % cfg.wandb_log_freq == 0:
                log_metrics_to_wandb(smoothened_metrics, "VLA Train", log_step, wandb)

            # [If applicable] Linearly warm up learning rate from 10% to 100% of original
            if cfg.lr_warmup_steps > 0:
                lr_progress = min((gradient_step_idx + 1) / cfg.lr_warmup_steps, 1.0)  # Cap at 1.0
                current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

            if distributed_state.is_main_process and gradient_step_idx % cfg.wandb_log_freq == 0:
                # Log the learning rate
                # Make sure to do this AFTER any learning rate modifications (e.g., warmup/decay)
                wandb.log(
                    {
                        "VLA Train/Learning Rate": optimizer.param_groups[0]["lr"],
                    },
                    step=log_step,
                )

            # Optimizer and LR scheduler step 真正更新参数
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.update()

            # Save model checkpoint: either keep latest checkpoint only or all checkpoints
            if gradient_step_idx > 0 and log_step % cfg.save_freq == 0:
                save_training_checkpoint(
                    cfg=cfg,
                    run_dir=run_dir,
                    log_step=log_step,
                    vla=vla,
                    processor=processor,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                    action_head=action_head if (cfg.use_l1_regression or cfg.use_diffusion) else None,
                    train_dataset=train_dataset,
                    distributed_state=distributed_state,
                )

            # Test model on validation set
            if cfg.use_val_set and log_step > 0 and log_step % cfg.val_freq == 0:
                run_validation(
                    vla=vla,
                    action_head=action_head,
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
                )
                # Set model back to training mode after validation
                vla.train()

            # Stop training when max_steps is reached
            if log_step == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune()
