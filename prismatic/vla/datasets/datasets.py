"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""
# RLDS：所有机器人数据都统一包装成一种格式
# rlds = {observation, action, task, dataset_name}
# 其他无人机原始数据集，可以写RLDS转换器
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    ACTION_TOKEN_BEGIN_IDX,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
    STOP_INDEX,
    get_act_token,
    get_cond_token,
)
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights

# 一条RLDS数据->>一条OpenVLA数据
# RLDS原始数据->读取RGB、读取语言指令、读取动作、读取proprio->prompt构建->tokenizer->image Transformer->labels构造->返回batch
# 决定了最终forward()收到的batch长什么样子
@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False
    use_image_history: bool = False
    num_images_in_input: int = 1
    require_full_image_history: bool = False
    use_cond_action_tokens: bool = False
    load_future_images: bool = False
    num_action_branches: int = 1
    use_reference_previous_current: bool = False
    body_delta_action_targets: bool = False

    def _build_cond_action_string(self) -> str:
        tokens = []
        for time_idx in range(1, NUM_ACTIONS_CHUNK + 1):
            for branch_idx in range(1, self.num_action_branches + 1):
                tokens.extend([get_cond_token(time_idx, branch_idx), get_act_token(time_idx, branch_idx)])
        return "".join(tokens)

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name = rlds_batch["dataset_name"]
        current_obs_index = self.num_images_in_input - 1 if self.use_image_history else 0
        if self.use_reference_previous_current:
            if self.num_images_in_input != 3:
                raise ValueError("reference/previous/current input requires num_images_in_input=3")
            history_mask = rlds_batch["observation"].get("pad_mask")
            if history_mask is None or len(history_mask) != 2:
                raise ValueError("reference/previous/current input requires a two-frame primary history window")
            reference = rlds_batch["observation"]["image_secondary"][1]
            previous, current = rlds_batch["observation"]["image_primary"][:2]
            pixel_values = torch.cat(
                [self.image_transform(Image.fromarray(image)) for image in (reference, previous, current)],
                dim=0,
            )
            image_valid_mask = np.asarray([True, bool(history_mask[0]), True], dtype=np.bool_)
            current_obs_index = 1
        elif self.use_image_history:
            pad_mask = rlds_batch["observation"].get("pad_mask")
            if self.require_full_image_history and (pad_mask is None or not np.all(pad_mask)):
                return None
            image_history = rlds_batch["observation"]["image_primary"][: self.num_images_in_input]
            pixel_values = torch.cat(
                [self.image_transform(Image.fromarray(image)) for image in image_history],
                dim=0,
            )
        else:
            img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
            pixel_values = self.image_transform(img)
            image_valid_mask = None

        action_chunk = (
            rlds_batch["action"]
            if self.body_delta_action_targets
            else rlds_batch["action"][current_obs_index:]
        )
        current_action = action_chunk[0]
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        actions = action_chunk

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")

        if self.use_cond_action_tokens:
            action_chunk_string = self._build_cond_action_string()
            action_chunk_len = 0
        else:
            # Get future action chunk
            future_actions = action_chunk[1:]
            future_actions_string = ''.join(self.action_tokenizer(future_actions))

            # Get action chunk string
            current_action_string = self.action_tokenizer(current_action)
            action_chunk_string = current_action_string + future_actions_string
            action_chunk_len = len(action_chunk_string)

        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

        if self.use_cond_action_tokens:
            labels[:] = IGNORE_INDEX
        else:
            # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
            labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return_dict = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            dataset_name=dataset_name,
            actions=actions,
        )
        if "episode_id" in rlds_batch:
            return_dict["episode_id"] = rlds_batch["episode_id"]
        if "stop_after_action" in rlds_batch:
            return_dict["stop_after_action"] = np.asarray(
                rlds_batch["stop_after_action"], dtype=np.float32
            )
        if "actions_remaining_after_root" in rlds_batch:
            return_dict["actions_remaining_after_root"] = np.asarray(
                rlds_batch["actions_remaining_after_root"], dtype=np.int64
            )

        # Add additional inputs
        if self.use_wrist_image:
            all_wrist_pixels = []
            for k in rlds_batch["observation"].keys():
                if "wrist" in k:
                    img_wrist = Image.fromarray(rlds_batch["observation"][k][0])
                    pixel_values_wrist = self.image_transform(img_wrist)
                    all_wrist_pixels.append(pixel_values_wrist)
            return_dict["pixel_values_wrist"] = torch.cat(all_wrist_pixels, dim=0)
        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            proprio = rlds_batch["observation"]["proprio"][current_obs_index]
            return_dict["proprio"] = proprio
        if self.use_reference_previous_current:
            return_dict["image_valid_mask"] = image_valid_mask
        elif self.use_image_history:
            return_dict["image_history_pad_mask"] = rlds_batch["observation"].get("pad_mask")
        if "plan_valid_mask" in rlds_batch:
            return_dict["plan_valid_mask"] = rlds_batch["plan_valid_mask"][:NUM_ACTIONS_CHUNK]
        if self.load_future_images and "future_observation" in rlds_batch:
            # Slot 0 has no visual-condition loss; only I_(t+1)..I_(t+T-1)
            # are image labels for the future COND tokens.
            future_images = rlds_batch["future_observation"]["image_primary"][1:NUM_ACTIONS_CHUNK]
            return_dict["future_pixel_values"] = torch.stack(
                [self.image_transform(Image.fromarray(image)) for image in future_images],
                dim=0,
            )

        return return_dict

# 构建整个Dataset
class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        tfds_split: Optional[str] = None,
        image_aug: bool = False,
        window_size: int = 1,
        relative_action_targets: bool = False,
        future_action_stride: int = 1,
        relative_action_wrap_yaw: bool = False,
        body_delta_action_targets: bool = False,
        cyclic_yaw_proprio: bool = False,
        use_reference_previous_current: bool = False,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform
        self.window_size = window_size

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        if use_reference_previous_current:
            load_camera_views = ("primary", "secondary")
        elif "aloha" in self.data_mix:
            load_camera_views = ("primary", "left_wrist", "right_wrist")
        else:
            load_camera_views = ("primary", "wrist")

        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
        )
        for dataset_kwargs in per_dataset_kwargs:
            dataset_kwargs.update(
                {
                    "tfds_split": tfds_split,
                    "relative_action_targets": relative_action_targets,
                    "relative_action_horizon": NUM_ACTIONS_CHUNK,
                    "relative_action_stride": future_action_stride,
                    "relative_action_wrap_yaw": relative_action_wrap_yaw,
                    "body_delta_action_targets": body_delta_action_targets,
                    "cyclic_yaw_proprio": cyclic_yaw_proprio,
                }
            )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=self.window_size,                       # Observation history length
                future_action_window_size=NUM_ACTIONS_CHUNK-1,      # For action chunking
                future_action_stride=future_action_stride,          # Raw-step spacing between future targets
                relative_action_targets=relative_action_targets,    # Cumulative offsets from current UAV state
                relative_action_wrap_yaw=relative_action_wrap_yaw,  # Match PAI-0 when False
                body_delta_action_targets=body_delta_action_targets,
                cyclic_yaw_proprio=cyclic_yaw_proprio,
                pad_future_horizon=body_delta_action_targets,
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            transformed = self.batch_transform(rlds_batch)
            if transformed is not None:
                yield transformed

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")

# 返回完整Episodes作为步骤列表，而不是单个转换（对于可视化很有用）。
class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out

# 示例数据集
class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
