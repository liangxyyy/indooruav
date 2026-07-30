"""Shared patch-level condition matching for training and online evaluation."""

import torch
import torch.nn.functional as F


def center_condition_branches(condition_embeddings: torch.Tensor) -> torch.Tensor:
    """Remove the shared direction across the K condition branches at each time step."""
    if condition_embeddings.ndim != 4:
        raise ValueError("condition embeddings must have shape (B,T,K,D)")
    if condition_embeddings.shape[2] < 2:
        raise ValueError("condition branch centering requires K >= 2")
    conditions = condition_embeddings.float()
    return conditions - conditions.mean(dim=2, keepdim=True)


def center_visual_patches(patch_embeddings: torch.Tensor) -> torch.Tensor:
    """Remove the image-level common direction while retaining local patch residuals."""
    if patch_embeddings.ndim != 4:
        raise ValueError("patch embeddings must have shape (B,T,N,D)")
    if patch_embeddings.shape[2] < 2:
        raise ValueError("visual patch centering requires N >= 2")
    patches = patch_embeddings.float()
    return patches - patches.mean(dim=2, keepdim=True)


def condition_to_patch_similarity(
    condition_embeddings: torch.Tensor,
    patch_embeddings: torch.Tensor,
    topk_patches: int,
) -> torch.Tensor:
    """
    Score each condition against an image by averaging its strongest patch matches.

    Args:
        condition_embeddings: Tensor with shape (B, T, K, D).
        patch_embeddings: Tensor with shape (B, T, N, D).
        topk_patches: Number of strongest visual-token matches to average.
    """
    if condition_embeddings.ndim != 4 or patch_embeddings.ndim != 4:
        raise ValueError("condition and patch embeddings must have shapes (B,T,K,D) and (B,T,N,D)")
    if condition_embeddings.shape[:2] != patch_embeddings.shape[:2]:
        raise ValueError("condition and patch batch/time dimensions must match")
    if condition_embeddings.shape[-1] != patch_embeddings.shape[-1]:
        raise ValueError("condition and patch embedding dimensions must match")
    if topk_patches < 1:
        raise ValueError("topk_patches must be >= 1")

    conditions = F.normalize(center_condition_branches(condition_embeddings), dim=-1)
    patches = F.normalize(center_visual_patches(patch_embeddings), dim=-1)
    patch_similarities = torch.einsum("btkd,btnd->btkn", conditions, patches)
    effective_topk = min(topk_patches, patch_similarities.shape[-1])
    return patch_similarities.topk(effective_topk, dim=-1).values.mean(dim=-1)


def condition_branch_contrastive_loss(
    condition_similarities: torch.Tensor,
    selected_branch_indices: torch.Tensor,
    temperature: float,
    loss_start_time_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Train visual condition matching to recover the action-supervised branch.

    At each future time step, the condition paired with the best-of-K action is
    the positive class and the remaining K-1 conditions are negatives.
    """
    if condition_similarities.ndim != 3 or selected_branch_indices.ndim != 2:
        raise ValueError("condition similarities and branch indices must have shapes (B,T,K) and (B,T)")
    if condition_similarities.shape[:2] != selected_branch_indices.shape:
        raise ValueError("condition similarity and branch-index batch/time dimensions must match")
    if condition_similarities.shape[2] < 2:
        raise ValueError("condition branch contrastive loss requires K >= 2")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    horizon = condition_similarities.shape[1]
    if not 0 <= loss_start_time_index < horizon:
        raise ValueError(
            f"condition loss start index must be in [0, {horizon}), got {loss_start_time_index}"
        )

    supervised_scores = condition_similarities[:, loss_start_time_index:].flatten(0, 1)
    labels = selected_branch_indices[:, loss_start_time_index:].flatten().long()
    if labels.min() < 0 or labels.max() >= supervised_scores.shape[1]:
        raise ValueError("selected branch index is outside the K condition branches")

    logits = supervised_scores / temperature
    loss = F.cross_entropy(logits, labels)
    accuracy = (logits.argmax(dim=1) == labels).float().mean()
    positive_scores = supervised_scores.gather(1, labels.unsqueeze(1)).squeeze(1)
    positive_mask = F.one_hot(labels, num_classes=supervised_scores.shape[1]).bool()
    hardest_negative = supervised_scores.masked_fill(positive_mask, -torch.inf).max(dim=1).values
    positive_margin = (positive_scores - hardest_negative).mean()
    return loss, accuracy, positive_margin
