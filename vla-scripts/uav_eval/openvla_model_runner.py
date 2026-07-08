import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


DEFAULT_OPENVLA_ROOT = "/VLM/liangxinyue_25/openvla-oft"
DEFAULT_CHECKPOINT = "/VLM/liangxinyue_25/openvla-oft/runs/uav/openvla-7b-oft-finetuned-libero-spatial+indoor_uav+b1+lr-0.0005+lora-r32+dropout-0.0--image_aug--stage6_full_train_30k_3img_5act_3branch--30000_chkpt"


def parse_args():
    parser = argparse.ArgumentParser(description="OpenVLA-OFT model runner for IndoorUAV online VLA evaluation.")
    parser.add_argument("--openvla_root", default=DEFAULT_OPENVLA_ROOT)
    parser.add_argument("--pretrained_checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--shared_folder", default="shared_folder")
    parser.add_argument("--unnorm_key", default="indoor_uav")
    parser.add_argument("--num_action_branches", type=int, default=3)
    parser.add_argument("--action_branch_index", type=int, default=0)
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
        )
        from prismatic.vla.constants import PROPRIO_DIM

        self.args = args
        self.cfg = build_cfg(args)
        self.get_vla_action = get_vla_action

        print("Loading OpenVLA base model...", flush=True)
        self.vla = get_vla(self.cfg)
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
            print(f"Loaded episode instruction: {episode_key}")

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
            new_coords = apply_action(coordinates, selected_action, self.args.relative_actions)

            timestamp = time.time()
            output_file = os.path.join(self.model_output_dir, f"model_output_{timestamp}.json")
            with open(output_file, "w") as f:
                json.dump(
                    {
                        "episode_key": self.current_episode,
                        "coordinates": new_coords,
                        "selected_branch": self.args.action_branch_index,
                        "action_chunk_shape": list(action_chunk.shape),
                        "selected_action": selected_action.astype(float).tolist(),
                        "relative_actions": self.args.relative_actions,
                    },
                    f,
                )

            print(
                "OpenVLA inference complete - "
                f"branch={self.args.action_branch_index}, "
                f"action_shape={list(action_chunk.shape)}, "
                f"coords={coordinates.astype(float).tolist()}, "
                f"selected_action={selected_action.astype(float).tolist()}, "
                f"relative_actions={self.args.relative_actions}, "
                f"next_coords={new_coords}"
            )
            return True

        except Exception as exc:
            print(f"Error processing {file_path}: {exc}")
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
