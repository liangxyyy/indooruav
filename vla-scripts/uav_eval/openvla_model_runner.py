import argparse
import json
import os
import sys
import time
import traceback
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from types import MethodType

import numpy as np
import torch
from PIL import Image


DEFAULT_OPENVLA_ROOT = "/VLM/liangxinyue_25/openvla-oft"
DEFAULT_CHECKPOINT = "/VLM/liangxinyue_25/openvla-oft/runs/uav/stage6_30k_ckpt+indoor_uav+b1+lr-0.0005+lora-r32+dropout-0.0--image_aug--stage12--30000_chkpt"


def parse_args():
    parser = argparse.ArgumentParser(description="OpenVLA-OFT model runner for IndoorUAV online VLA evaluation.")
    parser.add_argument("--openvla_root", default=DEFAULT_OPENVLA_ROOT)
    parser.add_argument("--pretrained_checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--shared_folder", default="shared_folder")
    parser.add_argument("--unnorm_key", default="indoor_uav")
    parser.add_argument("--num_action_branches", type=int, default=3)
    parser.add_argument("--action_branch_index", type=int, default=0)
    parser.add_argument("--use_condition_plan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--condition_threshold", type=float, default=0.6)
    parser.add_argument("--num_images_in_input", type=int, default=3)
    parser.add_argument("--relative_actions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--center_crop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--poll_interval", type=float, default=0.1)
    return parser.parse_args()


def build_cfg(args):
    return SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=args.pretrained_checkpoint,
        use_l1_regression=True,
        use_diffusion=False,
        num_diffusion_steps_train=50,
        num_diffusion_steps_inference=50,
        num_action_branches=args.num_action_branches,
        action_branch_index=args.action_branch_index,
        return_all_action_branches=False,
        use_cond_action_tokens=args.use_condition_plan,
        use_film=False,
        num_images_in_input=args.num_images_in_input,
        use_image_history=True,
        use_proprio=True,
        center_crop=args.center_crop,
        lora_rank=32,
        unnorm_key=args.unnorm_key,
        use_relative_actions=args.relative_actions,
        load_in_8bit=False,
        load_in_4bit=False,
        seed=7,
    )


def load_image(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def normalize_coords(coords):
    coords = list(coords or [])
    if len(coords) < 4:
        coords = coords + [0.0] * (4 - len(coords))
    return np.asarray(coords[:4], dtype=np.float32)


def apply_action(coords, action, relative_actions):
    action = np.asarray(action, dtype=np.float32)[:4]
    if relative_actions:
        next_coords = np.asarray(coords, dtype=np.float32)[:4] + action
    else:
        next_coords = action
    return next_coords.astype(float).tolist()


def normalize_vector(vector):
    return vector / (np.linalg.norm(vector) + 1e-8)


def require_shape(name, value, expected):
    actual = tuple(value.shape)
    if actual != tuple(expected):
        raise RuntimeError(f"{name} shape mismatch: expected {tuple(expected)}, got {actual}")


def patch_multimodal_attention_mask_dtype(vla):
    def _build_multimodal_attention(self, input_embeddings, projected_patch_embeddings, attention_mask):
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.long)
            projected_patch_attention_mask = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                fill_value=1,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
        else:
            projected_patch_attention_mask = None

        multimodal_embeddings = torch.cat(
            [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
        )

        multimodal_attention_mask = None
        if attention_mask is not None:
            multimodal_attention_mask = torch.cat(
                [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1
            ).to(torch.long)

        return multimodal_embeddings, multimodal_attention_mask

    vla._build_multimodal_attention = MethodType(_build_multimodal_attention, vla)


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


class OpenVLAModelService:
    def __init__(self, args):
        os.environ.setdefault("ROBOT_PLATFORM", "UAV")
        sys.path.insert(0, args.openvla_root)

        from experiments.robot.openvla_utils import (
            get_action_head,
            get_processor,
            get_proprio_projector,
            get_vla,
            get_vla_action,
            normalize_proprio,
            prepare_images_for_vla,
        )
        from prismatic.vla.constants import IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, get_act_token, get_cond_token

        self.args = args
        self.cfg = build_cfg(args)
        self.get_vla_action = get_vla_action
        self.normalize_proprio = normalize_proprio
        self.prepare_images_for_vla = prepare_images_for_vla
        self.num_actions_chunk = NUM_ACTIONS_CHUNK
        self.ignore_index = IGNORE_INDEX
        self.get_cond_token = get_cond_token
        self.get_act_token = get_act_token

        print("Loading OpenVLA base model...", flush=True)
        self.vla = get_vla(self.cfg)
        patch_multimodal_attention_mask_dtype(self.vla)
        print(f"Using checkpoint: {args.pretrained_checkpoint}", flush=True)
        print(
            "Condition plan config: "
            f"enabled={args.use_condition_plan}, "
            f"threshold={args.condition_threshold}, "
            f"branches={args.num_action_branches}, "
            f"images={args.num_images_in_input}",
            flush=True,
        )
        print("Loading OpenVLA processor...", flush=True)
        self.processor = get_processor(self.cfg)
        print("Loading proprio projector...", flush=True)
        self.proprio_projector = get_proprio_projector(self.cfg, self.vla.llm_dim, PROPRIO_DIM)
        print("Loading action head...", flush=True)
        self.action_head = get_action_head(self.cfg, self.vla.llm_dim)
        print("OpenVLA model components ready.", flush=True)

        self.current_episode = None
        self.instruction = None
        self.end_coords = None
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
            self.histories[episode_key] = deque(maxlen=self.args.num_images_in_input)
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

    def prepare_vla_inputs(self, prompt, image_history):
        all_images = self.prepare_images_for_vla(list(image_history), self.cfg)
        primary_image = all_images.pop(0)
        inputs = self.processor(prompt, primary_image).to("cuda:0", dtype=torch.bfloat16)
        inputs["attention_mask"] = inputs["attention_mask"].to(torch.long)
        if all_images:
            all_wrist_inputs = [self.processor(prompt, image).to("cuda:0", dtype=torch.bfloat16) for image in all_images]
            inputs["pixel_values"] = torch.cat(
                [inputs["pixel_values"]] + [wrist_inputs["pixel_values"] for wrist_inputs in all_wrist_inputs], dim=1
            )
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
        placeholder_ids = torch.cat([cond_ids, act_ids])
        prompt_mask = ~torch.isin(shifted_input_ids, placeholder_ids)
        prompt_lengths = prompt_mask.sum(dim=1, keepdim=True).clamp(min=1)
        instruction_hidden = (text_hidden_states.float() * prompt_mask.unsqueeze(-1)).sum(dim=1) / prompt_lengths
        return cond_hidden, act_hidden, instruction_hidden.squeeze(0)

    def normalize_proprio_for_model(self, coordinates):
        proprio_norm_stats = self.vla.norm_stats[self.cfg.unnorm_key]["proprio"]
        proprio = self.normalize_proprio(coordinates, proprio_norm_stats)
        return torch.as_tensor(proprio, device="cuda:0", dtype=torch.bfloat16)

    def create_condition_plan(self, episode_key, image_history, coordinates):
        prompt = self.build_prompt(include_condition_tokens=True)
        inputs = self.prepare_vla_inputs(prompt, image_history)
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
            )

        num_patches = self.vla.vision_backbone.get_num_patches() * self.vla.vision_backbone.get_num_images_in_input()
        if self.cfg.use_proprio:
            num_patches += 1
        text_hidden_states = output.hidden_states[-1][:, num_patches:-1]
        shifted_input_ids = inputs["input_ids"][:, 1:]
        cond_hidden, act_hidden, instruction_hidden = self.gather_plan_hidden_states(text_hidden_states, shifted_input_ids)
        require_shape("cond_hidden", cond_hidden, (1, self.num_actions_chunk, self.args.num_action_branches, self.vla.llm_dim))
        require_shape("act_hidden", act_hidden, (1, self.num_actions_chunk, self.args.num_action_branches, self.vla.llm_dim))
        with torch.inference_mode():
            normalized_actions = self.action_head.predict_action(act_hidden.to(torch.bfloat16)).squeeze(0)
        require_shape("normalized_actions", normalized_actions, (self.num_actions_chunk, self.args.num_action_branches, 4))
        actions = self.vla._unnormalize_actions(normalized_actions.float().detach().cpu().numpy(), self.cfg.unnorm_key)
        require_shape("actions", np.asarray(actions), (self.num_actions_chunk, self.args.num_action_branches, 4))

        plan = {
            "actions": np.asarray(actions, dtype=np.float32),
            "conditions": cond_hidden.squeeze(0).float().cpu().numpy(),
            "instruction_hidden": instruction_hidden.float().cpu().numpy(),
            "step_index": 0,
        }
        self.plans[episode_key] = plan
        return plan

    def encode_observed_condition(self, image_array, instruction_hidden):
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
        image_embedding = patch_embeddings.float().mean(dim=1).squeeze(0).cpu().numpy()
        return normalize_vector(image_embedding + instruction_hidden)

    def select_from_condition_plan(self, episode_key, image_array, image_history, coordinates):
        plan = self.plans.get(episode_key)
        replan_reason = None
        if plan is None or plan["step_index"] >= self.num_actions_chunk:
            replan_reason = "new_plan" if plan is None else "horizon_exhausted"
            plan = self.create_condition_plan(episode_key, image_history, coordinates)

        step_index = plan["step_index"]
        if step_index == 0:
            selected_branch = self.args.action_branch_index
            similarity = None
        else:
            observed_embedding = self.encode_observed_condition(image_array, plan["instruction_hidden"])
            cond_embeddings = np.stack([normalize_vector(v) for v in plan["conditions"][step_index]], axis=0)
            similarities = cond_embeddings @ observed_embedding
            selected_branch = int(np.argmax(similarities))
            similarity = float(similarities[selected_branch])
            if similarity < self.args.condition_threshold:
                replan_reason = f"condition_below_threshold:{similarity:.4f}"
                plan = self.create_condition_plan(episode_key, image_history, coordinates)
                step_index = 0
                selected_branch = self.args.action_branch_index
                similarity = None

        action_chunk = plan["actions"]
        require_shape("planned_actions", action_chunk, (self.num_actions_chunk, self.args.num_action_branches, 4))
        selected_action = action_chunk[step_index, selected_branch]
        plan["step_index"] = step_index + 1
        return action_chunk, selected_action, step_index, selected_branch, similarity, replan_reason

    def get_image_history(self, episode_key, image_array):
        history = self.histories.setdefault(episode_key, deque(maxlen=self.args.num_images_in_input))
        if not history:
            for _ in range(self.args.num_images_in_input - 1):
                history.append(image_array)
        history.append(image_array)

        images = list(history)
        if len(images) < self.args.num_images_in_input:
            images = [images[0]] * (self.args.num_images_in_input - len(images)) + images
        return images[-self.args.num_images_in_input :]

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
            image_history = self.get_image_history(episode_key, image_array)

            obs = {
                "full_image": image_array,
                "full_image_history": image_history,
                "state": coordinates.tolist(),
            }

            if self.args.use_condition_plan:
                action_chunk, selected_action, plan_step, selected_branch, condition_similarity, replan_reason = (
                    self.select_from_condition_plan(episode_key, image_array, image_history, coordinates)
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
                plan_step = 0
                selected_branch = self.args.action_branch_index
                condition_similarity = None
                replan_reason = None
            new_coords = apply_action(coordinates, selected_action, self.args.relative_actions)

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
                        "condition_threshold": self.args.condition_threshold,
                        "replan_reason": replan_reason,
                        "action_chunk_shape": list(action_chunk.shape),
                        "selected_action": selected_action.astype(float).tolist(),
                        "relative_actions": self.args.relative_actions,
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
                f"relative_actions={self.args.relative_actions}, "
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
