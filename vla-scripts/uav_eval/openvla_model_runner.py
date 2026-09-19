import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image


DEFAULT_OPENVLA_ROOT = "/VLM/liangxinyue_25/openvla-oft"
DEFAULT_CHECKPOINT = (
    "/VLM/liangxinyue_25/openvla-oft/runs/uav/"
    "stage6_30k_ckpt+indoor_uav+b1+lr-0.0005+lora-r32+dropout-0.0--image_aug--stage12--30000_chkpt"
)


def parse_args():
    parser = argparse.ArgumentParser(description="OpenVLA-OFT model runner for IndoorUAV online VLA evaluation.")
    parser.add_argument("--openvla_root", default=DEFAULT_OPENVLA_ROOT)
    parser.add_argument("--pretrained_checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--shared_folder", default="shared_folder")
    parser.add_argument("--unnorm_key", default="indoor_uav")
    parser.add_argument("--num_action_branches", type=int, default=3)
    parser.add_argument("--action_branch_index", type=int, default=0)
    parser.add_argument("--use_condition_plan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--condition_selection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Select future plan branches by image-condition similarity; disable for a fixed-branch plan control.",
    )
    parser.add_argument(
        "--condition_plan_steps",
        type=int,
        default=None,
        help="Execute this many steps from each plan before replanning; defaults to the full model horizon.",
    )
    parser.add_argument(
        "--use_cond_action_tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the interleaved <COND>/<ACT> output format independently of condition-based execution.",
    )
    parser.add_argument("--condition_threshold", type=float, default=0.6)
    parser.add_argument(
        "--stop_threshold",
        type=float,
        default=None,
        help="Post-root-action STOP probability threshold; defaults to checkpoint metadata.",
    )
    parser.add_argument(
        "--condition_patch_topk",
        type=int,
        default=None,
        help="Override checkpoint patch aggregation count; defaults to checkpoint metadata or 8.",
    )
    parser.add_argument("--num_images_in_input", type=int, default=3)
    parser.add_argument(
        "--relative_actions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override checkpoint action representation detection.",
    )
    parser.add_argument("--center_crop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--poll_interval", type=float, default=0.1)
    return parser.parse_args()


def build_cfg(args):
    return SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=args.pretrained_checkpoint,
        use_l1_regression=True,
        use_diffusion=False,
        use_gaussian_action_head=False,
        gaussian_log_std_min=-5.0,
        gaussian_log_std_max=1.0,
        gaussian_initial_log_std=-0.5,
        gaussian_learn_log_std=True,
        num_diffusion_steps_train=50,
        num_diffusion_steps_inference=50,
        num_action_branches=args.num_action_branches,
        action_branch_index=args.action_branch_index,
        return_all_action_branches=False,
        use_cond_action_tokens=args.use_cond_action_tokens,
        use_film=False,
        num_images_in_input=args.num_images_in_input,
        use_image_history=True,
        use_proprio=True,
        center_crop=args.center_crop,
        lora_rank=32,
        unnorm_key=args.unnorm_key,
        use_relative_actions=bool(args.relative_actions),
        condition_match_hidden_dim=1024,
        condition_match_dim=512,
        load_in_8bit=False,
        load_in_4bit=False,
        seed=7,
    )


def load_image(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def normalize_coords(coords):
    coords = [] if coords is None else list(coords)
    if len(coords) < 4:
        coords = coords + [0.0] * (4 - len(coords))
    return np.asarray(coords[:4], dtype=np.float32)


def pose_to_cyclic_proprio(coords):
    """Encode raw Habitat ``[x,y,z,yaw]`` without a discontinuity at 0/2pi."""
    x, y, z, yaw = normalize_coords(coords)
    return np.asarray([x, y, z, np.sin(yaw), np.cos(yaw)], dtype=np.float32)


def apply_action(coords, action, relative_actions, plan_origin=None):
    """Legacy action composer retained for pre-Stage20 checkpoints."""
    action = np.asarray(action, dtype=np.float32)[:4]
    if relative_actions:
        if plan_origin is None:
            raise ValueError("plan_origin is required for relative plan actions")
        next_coords = np.asarray(plan_origin, dtype=np.float32)[:4] + action
        next_coords[3] = np.mod(next_coords[3], 2.0 * np.pi)
    else:
        next_coords = action
    return next_coords.astype(float).tolist()


def apply_body_delta(coords, action):
    """Compose an IndoorUAV ``[forward, right, up, dyaw]`` action."""
    x, y, z, yaw = np.asarray(coords, dtype=np.float32)[:4]
    forward, right, up, delta_yaw = np.asarray(action, dtype=np.float32)[:4]
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    return [
        float(x + sin_yaw * forward + cos_yaw * right),
        float(y - cos_yaw * forward + sin_yaw * right),
        float(z + up),
        float(np.mod(yaw + delta_yaw, 2.0 * np.pi)),
    ]


def require_shape(name, value, expected):
    actual = tuple(value.shape)
    if actual != tuple(expected):
        raise RuntimeError(f"{name} shape mismatch: expected {tuple(expected)}, got {actual}")


def load_json_when_ready(file_path, attempts=20, interval=0.05):
    last_error = None
    for _ in range(attempts):
        try:
            if os.path.getsize(file_path) == 0:
                time.sleep(interval)
                continue
            with open(file_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            last_error = exc
            time.sleep(interval)
    raise RuntimeError(f"JSON file is not ready: {file_path} ({last_error})")


def load_policy_contract(checkpoint):
    contract_path = Path(checkpoint) / "policy_contract.json"
    if not contract_path.is_file():
        return None
    with contract_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class OpenVLAModelService:
    def __init__(self, args):
        os.environ.setdefault("ROBOT_PLATFORM", "UAV")
        sys.path.insert(0, args.openvla_root)

        from experiments.robot.openvla_utils import (
            get_action_head,
            get_indoor_uav_condition_adapter,
            get_indoor_uav_stop_head,
            get_processor,
            get_proprio_projector,
            get_vla,
            get_vla_action,
            normalize_proprio,
            prepare_images_for_vla,
        )
        from prismatic.vla.constants import IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, get_act_token, get_cond_token
        from prismatic.vla.condition_matching import (
            condition_to_patch_similarity,
            projected_condition_to_patch_similarity,
        )

        self.args = args
        self.cfg = build_cfg(args)
        self.get_vla_action = get_vla_action
        self.normalize_proprio = normalize_proprio
        self.prepare_images_for_vla = prepare_images_for_vla
        self.num_actions_chunk = NUM_ACTIONS_CHUNK
        self.condition_plan_steps = (
            self.num_actions_chunk
            if args.condition_plan_steps is None
            else args.condition_plan_steps
        )
        if not 1 <= self.condition_plan_steps <= self.num_actions_chunk:
            raise ValueError(
                f"condition_plan_steps must be in [1,{self.num_actions_chunk}], "
                f"got {self.condition_plan_steps}"
            )
        self.ignore_index = IGNORE_INDEX
        self.get_cond_token = get_cond_token
        self.get_act_token = get_act_token
        self.condition_to_patch_similarity = condition_to_patch_similarity
        self.projected_condition_to_patch_similarity = projected_condition_to_patch_similarity

        print("Loading OpenVLA base model...", flush=True)
        self.vla = get_vla(self.cfg)
        action_head_type = getattr(self.vla.config, "action_head_type", "l1")
        self.cfg.use_gaussian_action_head = action_head_type == "gaussian"
        self.cfg.gaussian_log_std_min = getattr(self.vla.config, "gaussian_log_std_min", -5.0)
        self.cfg.gaussian_log_std_max = getattr(self.vla.config, "gaussian_log_std_max", 1.0)
        self.cfg.gaussian_initial_log_std = getattr(self.vla.config, "gaussian_initial_log_std", -0.5)
        self.cfg.gaussian_learn_log_std = getattr(self.vla.config, "gaussian_learn_log_std", True)
        action_stats = self.vla.norm_stats[self.cfg.unnorm_key]["action"]
        proprio_stats = self.vla.norm_stats[self.cfg.unnorm_key]["proprio"]
        self.action_representation = action_stats.get("representation", "absolute_world_pose")
        self.action_normalization = action_stats.get("normalization_representation", "bounds_q99")
        self.proprio_representation = proprio_stats.get("representation", "xyz_yaw_radian")
        self.model_proprio_dim = len(proprio_stats["mean"])
        checkpoint_relative_actions = self.action_representation == "relative_plan_origin"
        self.body_delta_actions = self.action_representation == "body_delta_one_step_v1"
        self.relative_actions = (
            checkpoint_relative_actions if args.relative_actions is None else bool(args.relative_actions)
        )
        if self.body_delta_actions and args.relative_actions is not None:
            raise RuntimeError("--relative_actions cannot override a body-delta checkpoint")
        if args.relative_actions is not None and self.relative_actions != checkpoint_relative_actions:
            print(
                "WARNING: CLI action representation override disagrees with checkpoint metadata: "
                f"checkpoint_relative={checkpoint_relative_actions}, cli_relative={self.relative_actions}",
                flush=True,
            )
        self.cfg.use_relative_actions = self.relative_actions
        self.use_learned_condition_adapter = bool(
            getattr(self.vla.config, "use_indoor_uav_condition_adapter", False)
        )
        self.use_stop_head = bool(getattr(self.vla.config, "use_indoor_uav_stop_head", False))
        self.use_progress_stop_head = bool(
            getattr(self.vla.config, "use_indoor_uav_progress_stop_head", False)
        )
        checkpoint_stop_threshold = float(getattr(self.vla.config, "stop_threshold", 0.5))
        self.stop_threshold = (
            checkpoint_stop_threshold if args.stop_threshold is None else args.stop_threshold
        )
        if not 0.0 < self.stop_threshold < 1.0:
            raise ValueError("stop_threshold must lie in (0,1)")
        if self.body_delta_actions != self.use_learned_condition_adapter:
            raise RuntimeError(
                "Checkpoint contract mismatch: body-delta actions and the learned IndoorUAV adapter must coexist"
            )
        self.cfg.condition_match_hidden_dim = getattr(
            self.vla.config, "condition_match_hidden_dim", 1024
        )
        self.cfg.condition_match_dim = getattr(self.vla.config, "condition_match_dim", 512)
        self.cfg.stop_head_hidden_dim = getattr(self.vla.config, "stop_head_hidden_dim", 1024)
        self.cfg.use_indoor_uav_progress_stop_head = self.use_progress_stop_head
        self.cfg.stop_progress_projection_dim = getattr(
            self.vla.config, "stop_progress_projection_dim", 512
        )
        self.policy_contract = load_policy_contract(args.pretrained_checkpoint)
        if self.use_learned_condition_adapter:
            if args.num_images_in_input != 3 or not args.use_cond_action_tokens:
                raise RuntimeError("Stage20 inference requires three images and COND/ACT tokens")
            if args.action_branch_index != 0:
                raise RuntimeError("Stage20 slot 0 must execute action branch 0")
            if self.policy_contract is None:
                raise RuntimeError("Stage20 checkpoint is missing policy_contract.json")
            expected_contract = {
                "schema_version": (
                    4 if self.use_progress_stop_head else (3 if self.use_stop_head else 2)
                ),
                "training_objective": "sft",
                "input_roles": ["reference", "previous", "current"],
                "horizon": self.num_actions_chunk,
                "num_action_branches": args.num_action_branches,
                "action_dim": 4,
                "source_proprio_dim": 4,
                "model_proprio_dim": 5,
                "proprio_representation": "xyz_sin_yaw_cos_yaw_v1",
                "action_representation": "body_delta_one_step_v1",
                "yaw_zero_forward_world": [0.0, -1.0, 0.0],
                "yaw_zero_right_world": [1.0, 0.0, 0.0],
                "positive_yaw_turn": "right",
                "negative_yaw_turn": "left",
                "yaw_unit": "radian",
                "yaw_delta_formula": "((yaw_next-yaw_current+pi) mod (2*pi))-pi",
                "future_action_stride": 1,
                "action_normalization": "per_axis_symmetric_minmax_v1",
                "proprio_normalization": "xyz_q01_q99_and_sincos_unit_bounds_v1",
            }
            if self.use_stop_head:
                expected_contract.update(
                    {
                        "stop_after_action": True,
                        "stop_target_semantics": "execute root action, then terminate instruction",
                    }
                )
            if self.use_progress_stop_head:
                expected_contract.update(
                    {
                        "stop_head_type": "act_cond_ordinal_progress_v1",
                        "stop_progress_horizons": [0, 1, 2, 4],
                    }
                )
            mismatches = {
                key: (self.policy_contract.get(key), expected)
                for key, expected in expected_contract.items()
                if self.policy_contract.get(key) != expected
            }
            if mismatches:
                raise RuntimeError(f"Stage20 policy contract mismatch: {mismatches}")
            if self.action_normalization != "per_axis_symmetric_minmax_v1":
                raise RuntimeError(
                    f"Stage20 action statistics use an incompatible normalization: {self.action_normalization}"
                )
            if self.proprio_representation != "xyz_sin_yaw_cos_yaw_v1":
                raise RuntimeError(
                    f"Stage20 proprio statistics use an incompatible representation: {self.proprio_representation}"
                )
            checkpoint_proprio_representation = getattr(
                self.vla.config,
                "proprio_representation",
                self.proprio_representation,
            )
            checkpoint_proprio_dim = getattr(
                self.vla.config,
                "proprio_dim",
                self.model_proprio_dim,
            )
            if (
                checkpoint_proprio_representation != self.proprio_representation
                or checkpoint_proprio_dim != self.model_proprio_dim
            ):
                raise RuntimeError(
                    "Stage20 VLA config and dataset proprio metadata disagree: "
                    f"config=({checkpoint_proprio_representation}, {checkpoint_proprio_dim}), "
                    f"stats=({self.proprio_representation}, {self.model_proprio_dim})"
                )
            normalization_values = np.concatenate(
                [
                    np.asarray(stats[key], dtype=np.float32).reshape(-1)
                    for stats in (action_stats, proprio_stats)
                    for key in ("normalization_low", "normalization_high")
                ]
            )
            normalization_sha256 = hashlib.sha256(normalization_values.tobytes()).hexdigest()
            if self.policy_contract.get("normalization_sha256") != normalization_sha256:
                raise RuntimeError("Stage20 normalization statistics do not match policy_contract.json")
        checkpoint_patch_topk = getattr(self.vla.config, "condition_patch_topk", 8)
        checkpoint_matching_centered = getattr(self.vla.config, "condition_matching_centered", False)
        if not checkpoint_matching_centered and not self.use_learned_condition_adapter:
            print(
                "WARNING: checkpoint predates centered condition matching; "
                "online matching will use the current centered rule.",
                flush=True,
            )
        self.condition_patch_topk = (
            checkpoint_patch_topk
            if args.condition_patch_topk is None
            else args.condition_patch_topk
        )
        if self.condition_patch_topk < 1:
            raise ValueError("condition_patch_topk must be >= 1")
        print(f"Using checkpoint: {args.pretrained_checkpoint}", flush=True)
        print(
            "Condition plan config: "
            f"enabled={args.use_condition_plan}, "
            f"condition_selection={args.condition_selection}, "
            f"plan_steps={self.condition_plan_steps}, "
            f"cond_action_tokens={args.use_cond_action_tokens}, "
            f"threshold={args.condition_threshold}, "
            f"patch_topk={self.condition_patch_topk}, "
            f"matching={'learned_512d' if self.use_learned_condition_adapter else 'centered_condition_to_patch'}, "
            f"branches={args.num_action_branches}, "
            f"images={args.num_images_in_input}, "
            f"action_representation={self.action_representation}",
            f"action_normalization={self.action_normalization}",
            f"proprio_representation={self.proprio_representation}",
            f"action_head_type={action_head_type}",
            f"stop_after_action={self.use_stop_head}",
            f"stop_threshold={self.stop_threshold}",
            flush=True,
        )
        print("Loading OpenVLA processor...", flush=True)
        self.processor = get_processor(self.cfg)
        print("Loading proprio projector...", flush=True)
        if self.use_learned_condition_adapter and self.model_proprio_dim != 5:
            raise RuntimeError(f"Stage20 expects 5D model proprio, got {self.model_proprio_dim}")
        self.proprio_projector = get_proprio_projector(
            self.cfg,
            self.vla.llm_dim,
            self.model_proprio_dim if self.use_learned_condition_adapter else PROPRIO_DIM,
        )
        print("Loading action head...", flush=True)
        self.action_head = get_action_head(self.cfg, self.vla.llm_dim)
        self.stop_head = (
            get_indoor_uav_stop_head(self.cfg, self.vla.llm_dim)
            if self.use_stop_head
            else None
        )
        self.condition_adapter = None
        if self.use_learned_condition_adapter:
            print("Loading IndoorUAV condition adapter...", flush=True)
            self.condition_adapter = get_indoor_uav_condition_adapter(self.cfg, self.vla.llm_dim)
        print("OpenVLA model components ready.", flush=True)

        self.current_episode = None
        self.instruction = None
        self.end_coords = None
        self.reference_image = None
        self.histories = {}
        self.plans = {}

        self.shared_folder = args.shared_folder
        self.model_input_dir = os.path.join(self.shared_folder, "model_input")
        self.model_output_dir = os.path.join(self.shared_folder, "model_output")
        self.instructions_dir = os.path.join(self.shared_folder, "instructions")
        os.makedirs(self.model_input_dir, exist_ok=True)
        os.makedirs(self.model_output_dir, exist_ok=True)
        os.makedirs(self.instructions_dir, exist_ok=True)

    def load_instruction(self):
        instruction_file = os.path.join(self.instructions_dir, "current_instruction.json")
        if not os.path.exists(instruction_file):
            return

        try:
            data = load_json_when_ready(instruction_file)
        except RuntimeError as exc:
            print(exc)
            return

        episode_key = data.get("episode_key")
        if self.current_episode != episode_key:
            self.current_episode = episode_key
            self.instruction = data.get("instruction")
            self.end_coords = data.get("end_coords")
            reference_path = data.get("ref_image_path", data.get("start_image_path"))
            if self.use_learned_condition_adapter and not reference_path:
                raise RuntimeError("Stage20 requires ref_image_path or start_image_path in the instruction file")
            self.reference_image = load_image(reference_path) if reference_path else None
            history_size = 2 if self.use_learned_condition_adapter else self.args.num_images_in_input
            self.histories[episode_key] = deque(maxlen=history_size)
            self.plans.pop(episode_key, None)
            print(f"Loaded episode instruction: {episode_key}")

    def build_cond_action_suffix(self):
        tokens = []
        for time_idx in range(1, self.num_actions_chunk + 1):
            for branch_idx in range(1, self.args.num_action_branches + 1):
                tokens.extend([self.get_cond_token(time_idx, branch_idx), self.get_act_token(time_idx, branch_idx)])
        return "".join(tokens)

    def build_prompt(self, include_condition_tokens=True):
        suffix = self.build_cond_action_suffix() if include_condition_tokens else ""
        return f"In: What action should the robot take to {self.instruction.lower()}?\nOut:{suffix}"

    def prepare_vla_inputs(self, prompt, images, image_valid_mask=None):
        all_images = self.prepare_images_for_vla(list(images), self.cfg)
        primary_image = all_images.pop(0)
        inputs = self.processor(prompt, primary_image).to("cuda:0", dtype=torch.bfloat16)
        inputs["attention_mask"] = inputs["attention_mask"].to(torch.long)
        if all_images:
            all_wrist_inputs = [
                self.processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
                for image in all_images
            ]
            inputs["pixel_values"] = torch.cat(
                [inputs["pixel_values"]] + [wrist_inputs["pixel_values"] for wrist_inputs in all_wrist_inputs], dim=1
            )
        if image_valid_mask is not None:
            inputs["image_valid_mask"] = torch.as_tensor(
                image_valid_mask, device="cuda:0", dtype=torch.bool
            ).unsqueeze(0)
        return inputs

    def get_cond_action_token_ids(self, device):
        cond_ids = []
        act_ids = []
        tokenizer = self.processor.tokenizer
        for time_idx in range(1, self.num_actions_chunk + 1):
            for branch_idx in range(1, self.args.num_action_branches + 1):
                cond_ids.append(tokenizer.convert_tokens_to_ids(self.get_cond_token(time_idx, branch_idx)))
                act_ids.append(tokenizer.convert_tokens_to_ids(self.get_act_token(time_idx, branch_idx)))
        return torch.tensor(cond_ids, device=device), torch.tensor(act_ids, device=device)

    def gather_plan_hidden_states(self, text_hidden_states, shifted_input_ids):
        cond_ids, act_ids = self.get_cond_action_token_ids(shifted_input_ids.device)
        cond_mask = torch.isin(shifted_input_ids, cond_ids)
        act_mask = torch.isin(shifted_input_ids, act_ids)
        expected_count = self.num_actions_chunk * self.args.num_action_branches
        if int(cond_mask.sum().item()) != expected_count or int(act_mask.sum().item()) != expected_count:
            raise RuntimeError(
                f"Incomplete COND/ACT tokens: cond={int(cond_mask.sum().item())}, "
                f"act={int(act_mask.sum().item())}, expected={expected_count}"
            )
        cond_hidden = text_hidden_states[cond_mask].reshape(
            1, self.num_actions_chunk, self.args.num_action_branches, -1
        )
        act_hidden = text_hidden_states[act_mask].reshape(
            1, self.num_actions_chunk, self.args.num_action_branches, -1
        )
        return cond_hidden, act_hidden

    def normalize_proprio_for_model(self, coordinates):
        proprio_norm_stats = self.vla.norm_stats[self.cfg.unnorm_key]["proprio"]
        raw_proprio = (
            pose_to_cyclic_proprio(coordinates)
            if self.proprio_representation == "xyz_sin_yaw_cos_yaw_v1"
            else coordinates
        )
        proprio = self.normalize_proprio(raw_proprio, proprio_norm_stats)
        require_shape("normalized_proprio", proprio, (self.model_proprio_dim,))
        return torch.as_tensor(proprio, device="cuda:0", dtype=torch.bfloat16)

    def create_condition_plan(self, episode_key, model_images, image_valid_mask, coordinates):
        prompt = self.build_prompt(include_condition_tokens=True)
        inputs = self.prepare_vla_inputs(prompt, model_images, image_valid_mask)
        proprio = self.normalize_proprio_for_model(coordinates) if self.cfg.use_proprio else None
        labels = torch.full_like(inputs["input_ids"], fill_value=self.ignore_index, dtype=torch.long)
        if inputs["attention_mask"].dtype != torch.long:
            raise RuntimeError(f"attention_mask must be torch.long, got {inputs['attention_mask'].dtype}")
        if labels.dtype != torch.long:
            raise RuntimeError(f"labels must be torch.long, got {labels.dtype}")

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.vla(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
                labels=labels,
                output_hidden_states=True,
                proprio=proprio,
                proprio_projector=self.proprio_projector if self.cfg.use_proprio else None,
                use_film=False,
                image_valid_mask=inputs.get("image_valid_mask"),
                image_role_embeddings=(
                    self.condition_adapter.image_role_embeddings
                    if self.condition_adapter is not None
                    else None
                ),
            )

        num_patches = self.vla.vision_backbone.get_num_patches() * self.vla.vision_backbone.get_num_images_in_input()
        if self.cfg.use_proprio:
            num_patches += 1
        text_hidden_states = output.hidden_states[-1][:, num_patches:-1]
        shifted_input_ids = inputs["input_ids"][:, 1:]
        cond_hidden, act_hidden = self.gather_plan_hidden_states(text_hidden_states, shifted_input_ids)
        hidden_shape = (1, self.num_actions_chunk, self.args.num_action_branches, self.vla.llm_dim)
        require_shape("cond_hidden", cond_hidden, hidden_shape)
        require_shape("act_hidden", act_hidden, hidden_shape)
        with torch.inference_mode():
            normalized_actions = self.action_head.predict_action(act_hidden.to(torch.bfloat16)).squeeze(0)
            if self.stop_head is None:
                stop_probability = None
            elif self.use_progress_stop_head:
                stop_logits = self.stop_head(
                    act_hidden[:, 0, 0].float(), cond_hidden[:, 0, 0].float()
                )
                stop_probability = float(torch.sigmoid(stop_logits[:, 0]).item())
            else:
                stop_probability = float(
                    torch.sigmoid(self.stop_head(act_hidden[:, 0, 0].float())).item()
                )
        action_shape = (self.num_actions_chunk, self.args.num_action_branches, 4)
        require_shape("normalized_actions", normalized_actions, action_shape)
        actions = self.vla._unnormalize_actions(normalized_actions.float().detach().cpu().numpy(), self.cfg.unnorm_key)
        require_shape("actions", np.asarray(actions), action_shape)

        conditions = cond_hidden.squeeze(0)
        if self.condition_adapter is not None:
            with torch.inference_mode():
                conditions = self.condition_adapter.project_conditions(conditions)
        plan = {
            "actions": np.asarray(actions, dtype=np.float32),
            "conditions": conditions.float().cpu().numpy(),
            "origin": np.asarray(coordinates, dtype=np.float32).copy(),
            "step_index": 0,
            "stop_probability": stop_probability,
        }
        self.plans[episode_key] = plan
        return plan

    def encode_observed_condition(self, image_array):
        prompt = self.build_prompt(include_condition_tokens=False)
        image = self.prepare_images_for_vla([image_array], self.cfg)[0]
        inputs = self.processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
        old_num_images = self.vla.vision_backbone.get_num_images_in_input()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            try:
                self.vla.vision_backbone.set_num_images_in_input(1)
                patch_embeddings = self.vla._process_vision_features(inputs["pixel_values"], use_film=False)
            finally:
                self.vla.vision_backbone.set_num_images_in_input(old_num_images)
        if self.condition_adapter is not None:
            with torch.inference_mode():
                patch_embeddings = self.condition_adapter.project_vision(patch_embeddings)
        return patch_embeddings.squeeze(0).float().cpu().numpy()

    def select_from_condition_plan(self, episode_key, image_array, model_images, image_valid_mask, coordinates):
        plan = self.plans.get(episode_key)
        replan_reason = None
        if plan is None or plan["step_index"] >= self.condition_plan_steps:
            replan_reason = "new_plan" if plan is None else "horizon_exhausted"
            plan = self.create_condition_plan(episode_key, model_images, image_valid_mask, coordinates)

        step_index = plan["step_index"]
        if step_index == 0:
            selected_branch = self.args.action_branch_index
            similarities = None
            topk_indices = None
        elif self.args.condition_selection:
            observed_patches = torch.as_tensor(
                self.encode_observed_condition(image_array),
                dtype=torch.float32,
            ).unsqueeze(0).unsqueeze(0)
            cond_embeddings = torch.as_tensor(
                plan["conditions"][step_index],
                dtype=torch.float32,
            ).unsqueeze(0).unsqueeze(0)
            if self.condition_adapter is not None:
                similarity_tensor, index_tensor = self.projected_condition_to_patch_similarity(
                    cond_embeddings,
                    observed_patches,
                    self.condition_patch_topk,
                    return_indices=True,
                )
                topk_indices = index_tensor.squeeze(0).squeeze(0).cpu().numpy()
            else:
                similarity_tensor = self.condition_to_patch_similarity(
                    cond_embeddings,
                    observed_patches,
                    self.condition_patch_topk,
                )
                topk_indices = None
            similarities = similarity_tensor.squeeze(0).squeeze(0).cpu().numpy()
            selected_branch = int(np.argmax(similarities))
            similarity = float(similarities[selected_branch])
            if similarity < self.args.condition_threshold:
                replan_reason = f"condition_below_threshold:{similarity:.4f}"
                plan = self.create_condition_plan(episode_key, model_images, image_valid_mask, coordinates)
                step_index = 0
                selected_branch = self.args.action_branch_index
                similarities = None
                topk_indices = None
        else:
            selected_branch = self.args.action_branch_index
            similarities = None
            topk_indices = None
            replan_reason = "fixed_branch_plan"

        action_chunk = plan["actions"]
        require_shape("planned_actions", action_chunk, (self.num_actions_chunk, self.args.num_action_branches, 4))
        selected_action = action_chunk[step_index, selected_branch]
        stop_probability = plan["stop_probability"] if step_index == 0 else None
        plan["step_index"] = step_index + 1
        return (
            action_chunk,
            selected_action,
            plan["origin"],
            step_index,
            selected_branch,
            similarities,
            topk_indices,
            replan_reason,
            stop_probability,
        )

    def select_fixed_branch_replan(self, episode_key, model_images, image_valid_mask, coordinates):
        """Replan from every observation and execute time zero from one fixed branch."""
        plan = self.create_condition_plan(episode_key, model_images, image_valid_mask, coordinates)
        action_chunk = plan["actions"]
        require_shape(
            "planned_actions",
            action_chunk,
            (self.num_actions_chunk, self.args.num_action_branches, 4),
        )
        selected_branch = self.args.action_branch_index
        selected_action = action_chunk[0, selected_branch]
        return (
            action_chunk,
            selected_action,
            plan["origin"],
            0,
            selected_branch,
            None,
            None,
            "fixed_branch_replan",
            plan["stop_probability"],
        )

    def get_model_images(self, episode_key, image_array):
        history = self.histories[episode_key]
        history.append(image_array)
        if self.use_learned_condition_adapter:
            has_previous = len(history) > 1
            previous = history[-2] if has_previous else image_array
            return [self.reference_image, previous, image_array], [True, has_previous, True]

        images = list(history)
        images = [images[0]] * (self.args.num_images_in_input - len(images)) + images
        return images[-self.args.num_images_in_input :], None

    def process_file(self, file_path):
        should_remove = False
        try:
            data = load_json_when_ready(file_path)
            should_remove = True

            episode_key = data.get("episode_key", "")
            image_path = data.get("image_path", "")
            coordinates = normalize_coords(data.get("coordinates", []))

            self.load_instruction()
            if episode_key != self.current_episode:
                print(f"Skipping stale episode file: {episode_key} vs {self.current_episode}")
                return False

            if not os.path.exists(image_path):
                print(f"Image file does not exist: {image_path}")
                return False

            image_array = load_image(image_path)
            model_images, image_valid_mask = self.get_model_images(episode_key, image_array)

            obs = {
                "full_image": image_array,
                "full_image_history": model_images,
                "state": coordinates.tolist(),
            }

            if self.args.use_condition_plan:
                (
                    action_chunk,
                    selected_action,
                    plan_origin,
                    plan_step,
                    selected_branch,
                    condition_similarities,
                    topk_patch_indices,
                    replan_reason,
                    stop_probability,
                ) = self.select_from_condition_plan(
                    episode_key,
                    image_array,
                    model_images,
                    image_valid_mask,
                    coordinates,
                )
            elif self.args.use_cond_action_tokens:
                (
                    action_chunk,
                    selected_action,
                    plan_origin,
                    plan_step,
                    selected_branch,
                    condition_similarities,
                    topk_patch_indices,
                    replan_reason,
                    stop_probability,
                ) = self.select_fixed_branch_replan(
                    episode_key,
                    model_images,
                    image_valid_mask,
                    coordinates,
                )
            else:
                action_chunk = self.get_vla_action(
                    self.cfg,
                    self.vla,
                    self.processor,
                    obs,
                    self.instruction,
                    action_head=self.action_head,
                    proprio_projector=self.proprio_projector,
                    use_film=False,
                    action_branch_index=self.args.action_branch_index,
                    return_all_action_branches=False,
                )
                action_chunk = np.asarray(action_chunk, dtype=np.float32)
                selected_action = action_chunk[0]
                plan_origin = coordinates
                plan_step = 0
                selected_branch = self.args.action_branch_index
                condition_similarities = None
                topk_patch_indices = None
                replan_reason = None
                stop_probability = None
            condition_similarity = (
                None
                if condition_similarities is None
                else float(condition_similarities[selected_branch])
            )
            if self.body_delta_actions:
                new_coords = apply_body_delta(coordinates, selected_action)
            else:
                new_coords = apply_action(coordinates, selected_action, self.relative_actions, plan_origin)

            timestamp = time.time()
            output_file = os.path.join(self.model_output_dir, f"model_output_{timestamp}.json")
            with open(output_file, "w") as f:
                json.dump(
                    {
                        "episode_key": self.current_episode,
                        "coordinates": new_coords,
                        "selected_branch": selected_branch,
                        "plan_step": plan_step,
                        "condition_similarity": condition_similarity,
                        "condition_similarities": (
                            None
                            if condition_similarities is None
                            else np.asarray(condition_similarities, dtype=float).tolist()
                        ),
                        "topk_patch_indices": (
                            None
                            if topk_patch_indices is None
                            else np.asarray(topk_patch_indices, dtype=int).tolist()
                        ),
                        "condition_threshold": self.args.condition_threshold,
                        "condition_selection": self.args.condition_selection,
                        "condition_plan_steps": self.condition_plan_steps,
                        "replan_reason": replan_reason,
                        "action_chunk_shape": list(action_chunk.shape),
                        "selected_action": selected_action.astype(float).tolist(),
                        "action_representation": self.action_representation,
                        "stop_after_action_probability": stop_probability,
                        "stop_after_action": (
                            bool(stop_probability >= self.stop_threshold)
                            if stop_probability is not None
                            else False
                        ),
                        "stop_threshold": self.stop_threshold,
                        "image_valid_mask": image_valid_mask,
                        "plan_origin": (
                            None
                            if self.body_delta_actions
                            else np.asarray(plan_origin, dtype=float).tolist()
                        ),
                    },
                    f,
                )

            print(
                "OpenVLA inference complete - "
                f"plan_step={plan_step}, "
                f"branch={selected_branch}, "
                f"condition_similarity={condition_similarity}, "
                f"replan_reason={replan_reason}, "
                f"action_shape={list(action_chunk.shape)}, "
                f"coords={coordinates.astype(float).tolist()}, "
                f"selected_action={selected_action.astype(float).tolist()}, "
                f"action_representation={self.action_representation}, "
                f"next_coords={new_coords}"
            )
            return True

        except Exception as exc:
            print(f"Error processing {file_path}: {exc}")
            traceback.print_exc()
            return False
        finally:
            if should_remove and os.path.exists(file_path):
                os.remove(file_path)


def main():
    args = parse_args()
    service = OpenVLAModelService(args)
    print("OpenVLA model runner started.")

    try:
        while True:
            service.load_instruction()
            processed = False
            for file_name in os.listdir(service.model_input_dir):
                if not file_name.endswith(".json"):
                    continue
                file_path = os.path.join(service.model_input_dir, file_name)
                if service.process_file(file_path):
                    processed = True
            if not processed:
                time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("OpenVLA model runner stopped.")


if __name__ == "__main__":
    main()
