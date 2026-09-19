"""Shared patch-level condition matching for training and online evaluation."""

from typing import Optional, Sequence

import torch
import torch.nn.functional as F


class CrossEpisodeImageQueue:
    """Device-local FIFO memory bank of detached image-patch embeddings.

    Episode identifiers are stored alongside the embeddings so callers can
    exclude images from the query episode instead of accidentally treating
    nearby windows from the same trajectory as cross-episode negatives.
    """

    def __init__(self, capacity: int, storage_dtype: torch.dtype = torch.bfloat16):
        if capacity < 1:
            raise ValueError("queue capacity must be >= 1")
        self.capacity = capacity
        self.storage_dtype = storage_dtype
        self._patches: Optional[torch.Tensor] = None
        self._episode_ids: list[Optional[str]] = [None] * capacity
        self._next_index = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @staticmethod
    def _normalize_episode_id(episode_id) -> str:
        if isinstance(episode_id, bytes):
            return episode_id.decode("utf-8")
        return str(episode_id)

    def enqueue(self, image_patches: torch.Tensor, episode_ids: Sequence) -> None:
        if image_patches.ndim != 3:
            raise ValueError("queued image patches must have shape (N,P,D)")
        if len(episode_ids) != image_patches.shape[0]:
            raise ValueError("one episode identifier is required per queued image")
        if image_patches.shape[0] == 0:
            return

        detached = image_patches.detach().to(dtype=self.storage_dtype)
        if self._patches is None:
            self._patches = torch.empty(
                (self.capacity, *detached.shape[1:]),
                dtype=self.storage_dtype,
                device=detached.device,
            )
        elif self._patches.shape[1:] != detached.shape[1:]:
            raise ValueError("queued image patch shape changed during the run")
        elif self._patches.device != detached.device:
            raise ValueError("queued image patches changed device during the run")

        for patches, episode_id in zip(detached, episode_ids):
            self._patches[self._next_index].copy_(patches)
            self._episode_ids[self._next_index] = self._normalize_episode_id(episode_id)
            self._next_index = (self._next_index + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def entries(self) -> tuple[Optional[torch.Tensor], list[str]]:
        if self._size == 0:
            return None, []
        if self._patches is None:
            raise RuntimeError("non-empty queue has no embedding storage")
        patches = self._patches[: self._size] if self._size < self.capacity else self._patches
        episode_ids = self._episode_ids[: self._size] if self._size < self.capacity else self._episode_ids
        if any(episode_id is None for episode_id in episode_ids):
            raise RuntimeError("queue contains image patches without episode identifiers")
        return patches, [episode_id for episode_id in episode_ids if episode_id is not None]


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


def projected_condition_to_patch_similarity(
    condition_embeddings: torch.Tensor,
    patch_embeddings: torch.Tensor,
    topk_patches: int,
    return_indices: bool = False,
):
    """Top-k cosine scores in the learned matching space, without legacy centering."""
    if condition_embeddings.ndim != 4 or patch_embeddings.ndim != 4:
        raise ValueError("condition and patch embeddings must have shapes (B,T,K,D) and (B,T,N,D)")
    if condition_embeddings.shape[:2] != patch_embeddings.shape[:2]:
        raise ValueError("condition and patch batch/time dimensions must match")
    if condition_embeddings.shape[-1] != patch_embeddings.shape[-1]:
        raise ValueError("condition and patch embedding dimensions must match")
    if topk_patches < 1:
        raise ValueError("topk_patches must be >= 1")

    conditions = F.normalize(condition_embeddings.float(), dim=-1)
    patches = F.normalize(patch_embeddings.float(), dim=-1)
    patch_similarities = torch.einsum("btkd,btnd->btkn", conditions, patches)
    effective_topk = min(topk_patches, patch_similarities.shape[-1])
    topk = patch_similarities.topk(effective_topk, dim=-1)
    scores = topk.values.mean(dim=-1)
    return (scores, topk.indices) if return_indices else scores


def condition_to_image_logits(
    condition_embeddings: torch.Tensor,
    image_patch_embeddings: torch.Tensor,
    topk_patches: int,
) -> torch.Tensor:
    """Score each ``[N,D]`` condition against every ``[M,P,D]`` candidate image."""
    if condition_embeddings.ndim != 2 or image_patch_embeddings.ndim != 3:
        raise ValueError("conditions and image patches must have shapes (N,D) and (M,P,D)")
    if condition_embeddings.shape[-1] != image_patch_embeddings.shape[-1]:
        raise ValueError("condition and image-patch embedding dimensions must match")
    if topk_patches < 1:
        raise ValueError("topk_patches must be >= 1")

    conditions = F.normalize(condition_embeddings.float(), dim=-1)
    patches = F.normalize(image_patch_embeddings.float(), dim=-1)
    patch_logits = torch.einsum("nd,mpd->nmp", conditions, patches)
    effective_topk = min(topk_patches, patch_logits.shape[-1])
    return patch_logits.topk(effective_topk, dim=-1).values.mean(dim=-1)


def condition_branch_contrastive_loss(
    condition_similarities: torch.Tensor,
    selected_branch_indices: torch.Tensor,
    temperature: float,
    loss_start_time_index: int,
    valid_mask: Optional[torch.Tensor] = None,
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

    supervised_scores = condition_similarities[:, loss_start_time_index:]
    labels = selected_branch_indices[:, loss_start_time_index:].long()
    if valid_mask is not None:
        if valid_mask.shape != selected_branch_indices.shape:
            raise ValueError("valid_mask must match selected branch batch/time dimensions")
        supervised_mask = valid_mask[:, loss_start_time_index:].bool()
        supervised_scores = supervised_scores[supervised_mask]
        labels = labels[supervised_mask]
    else:
        supervised_scores = supervised_scores.flatten(0, 1)
        labels = labels.flatten()
    if supervised_scores.shape[0] == 0:
        zero = condition_similarities.sum() * 0.0
        return zero, zero.detach(), zero.detach()
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
