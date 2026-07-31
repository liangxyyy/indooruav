"""Audit a relative-action checkpoint on fixed IndoorUAV episode starts."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from openvla_model_runner import OpenVLAModelService, load_image


AXES = ("x", "y", "z", "yaw")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--openvla_root", type=Path, default=Path("/VLM/liangxinyue_25/openvla-oft"))
    parser.add_argument("--indoor_uav_base", type=Path, default=Path("/VLM/datasets/Indoor_UAV"))
    parser.add_argument(
        "--episode_keys_file",
        type=Path,
        default=Path("/VLM/liangxinyue_25/IndoorUAV-Agent-main/online_eval/vla_eval/test_vla.json"),
    )
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_episodes", type=int, default=100)
    parser.add_argument("--num_action_branches", type=int, default=3)
    parser.add_argument("--num_images_in_input", type=int, default=3)
    parser.add_argument("--unnorm_key", default="indoor_uav")
    parser.add_argument("--scratch_dir", type=Path, default=Path("/tmp/stage18_checkpoint_audit"))
    parser.add_argument(
        "--output_file",
        type=Path,
        default=Path("runs/uav/stage18_checkpoint_audit.json"),
    )
    return parser.parse_args()


def wrap_to_pi(values):
    values = np.asarray(values)
    return np.mod(values + np.pi, 2.0 * np.pi) - np.pi


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "median": np.median(values, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def select_oracle_predictions(predictions, targets):
    """Select one of K branches independently at every time using 4D L1 error."""
    errors = np.abs(predictions - targets[:, :, None, :]).mean(axis=-1)
    winners = errors.argmin(axis=-1)
    batch_indices = np.arange(predictions.shape[0])[:, None]
    time_indices = np.arange(predictions.shape[1])[None, :]
    selected = predictions[batch_indices, time_indices, winners]
    return selected, winners


def direction_cosine(predictions, targets):
    pred_xyz = predictions[..., :3]
    target_xyz = targets[..., :3]
    pred_norm = np.linalg.norm(pred_xyz, axis=-1)
    target_norm = np.linalg.norm(target_xyz, axis=-1)
    valid = (pred_norm > 1e-6) & (target_norm > 1e-6)
    if not np.any(valid):
        return None, 0
    cosine = np.sum(pred_xyz[valid] * target_xyz[valid], axis=-1) / (pred_norm[valid] * target_norm[valid])
    return float(cosine.mean()), int(valid.sum())


def summarize_strategy(predictions, targets):
    absolute_error = np.abs(predictions - targets)
    cosine, cosine_count = direction_cosine(predictions, targets)
    pred_displacement = np.linalg.norm(predictions[..., :3], axis=-1)
    target_displacement = np.linalg.norm(targets[..., :3], axis=-1)
    return {
        "mae_by_axis": dict(zip(AXES, absolute_error.mean(axis=(0, 1)).tolist())),
        "mean_mae": float(absolute_error.mean()),
        "t1_mae_by_axis": dict(zip(AXES, absolute_error[:, 0].mean(axis=0).tolist())),
        "t1_mean_mae": float(absolute_error[:, 0].mean()),
        "position_l2_error_mean": float(np.linalg.norm(predictions[..., :3] - targets[..., :3], axis=-1).mean()),
        "yaw_abs_error_mean": float(absolute_error[..., 3].mean()),
        "direction_cosine_mean": cosine,
        "direction_cosine_count": cosine_count,
        "predicted_position_delta_norm_mean": float(pred_displacement.mean()),
        "target_position_delta_norm_mean": float(target_displacement.mean()),
        "position_delta_norm_ratio": float(pred_displacement.mean() / max(target_displacement.mean(), 1e-8)),
        "prediction_distribution": summarize(predictions.reshape(-1, 4)),
    }


def load_episode(base_dir, episode_key, num_images, horizon, stride, wrap_yaw):
    parts = episode_key.lstrip("/").split("/")
    if len(parts) != 4:
        raise ValueError(f"Invalid episode key: {episode_key}")
    group, scene, trajectory, instruction_name = parts
    instruction_path = base_dir / "vla_ins" / group / scene / trajectory / instruction_name
    posture_path = base_dir / "without_screenshot" / group / scene / trajectory / "posture.json"

    with instruction_path.open("r", encoding="gbk") as handle:
        instruction_data = json.load(handle)
    with posture_path.open("r") as handle:
        posture = np.asarray(json.load(handle), dtype=np.float32)
    posture[:, 3] = np.deg2rad(posture[:, 3])

    start_frame = int(instruction_data["source"][0])
    start_index = start_frame - 1
    final_index = start_index + horizon * stride
    if start_index < 0 or final_index >= len(posture):
        raise ValueError(
            f"Insufficient future states: start={start_index}, required={final_index}, length={len(posture)}"
        )

    screenshot_dir = base_dir / group / scene / trajectory / "screenshots"
    requested_frames = list(range(start_frame - num_images + 1, start_frame + 1))
    existing_paths = [screenshot_dir / f"{frame}.png" for frame in requested_frames if frame >= 1]
    existing_paths = [path for path in existing_paths if path.is_file()]
    current_path = screenshot_dir / f"{start_frame}.png"
    if not current_path.is_file():
        raise FileNotFoundError(current_path)
    if not existing_paths:
        existing_paths = [current_path]
    history_was_padded = len(existing_paths) < num_images
    while len(existing_paths) < num_images:
        existing_paths.insert(0, existing_paths[0])
    image_history = [load_image(path) for path in existing_paths[-num_images:]]

    origin = posture[start_index].copy()
    arrival_indices = start_index + np.arange(1, horizon + 1) * stride
    targets = posture[arrival_indices] - origin[None]
    if wrap_yaw:
        targets[:, 3] = wrap_to_pi(targets[:, 3])

    return {
        "instruction": instruction_data["instruction"],
        "origin": origin,
        "targets": targets.astype(np.float32),
        "image_history": image_history,
        "history_was_padded": history_was_padded,
    }


def build_service(args):
    service_args = SimpleNamespace(
        openvla_root=str(args.openvla_root),
        pretrained_checkpoint=str(args.checkpoint),
        shared_folder=str(args.scratch_dir),
        unnorm_key=args.unnorm_key,
        num_action_branches=args.num_action_branches,
        action_branch_index=0,
        use_condition_plan=False,
        use_cond_action_tokens=True,
        condition_threshold=0.6,
        condition_patch_topk=None,
        num_images_in_input=args.num_images_in_input,
        relative_actions=None,
        center_crop=True,
        poll_interval=0.1,
    )
    return OpenVLAModelService(service_args)


def main():
    args = parse_args()
    if args.start_index < 0 or args.max_episodes < 1:
        raise ValueError("start_index must be >= 0 and max_episodes must be >= 1")

    service = build_service(args)
    action_stats = service.vla.norm_stats[args.unnorm_key]["action"]
    representation = action_stats.get("representation")
    if representation != "relative_plan_origin":
        raise ValueError(f"Checkpoint does not contain relative plan actions: representation={representation}")
    horizon = int(action_stats.get("horizon", service.num_actions_chunk))
    stride = int(action_stats.get("stride", 1))
    wrap_yaw = bool(action_stats.get("yaw_delta_wrapped", False))
    if horizon != service.num_actions_chunk:
        raise ValueError(f"Checkpoint horizon {horizon} != model horizon {service.num_actions_chunk}")

    with args.episode_keys_file.open("r") as handle:
        episode_keys = list(json.load(handle).keys())
    episode_keys = episode_keys[args.start_index : args.start_index + args.max_episodes]

    predictions = []
    targets = []
    completed_keys = []
    skipped = []
    padded_histories = 0
    for index, episode_key in enumerate(episode_keys, start=1):
        try:
            episode = load_episode(
                args.indoor_uav_base,
                episode_key,
                args.num_images_in_input,
                horizon,
                stride,
                wrap_yaw,
            )
            service.current_episode = episode_key
            service.instruction = episode["instruction"]
            service.plans.pop(episode_key, None)
            plan = service.create_condition_plan(
                episode_key,
                episode["image_history"],
                episode["origin"],
            )
            predictions.append(plan["actions"])
            targets.append(episode["targets"])
            completed_keys.append(episode_key)
            padded_histories += int(episode["history_was_padded"])
            print(f"[{index}/{len(episode_keys)}] audited {episode_key}", flush=True)
        except Exception as exc:
            skipped.append({"episode": episode_key, "reason": str(exc)})
            print(f"[{index}/{len(episode_keys)}] skipped {episode_key}: {exc}", flush=True)

    if not predictions:
        raise RuntimeError("No episodes were successfully audited")
    predictions = np.stack(predictions)
    targets = np.stack(targets)
    oracle_predictions, winners = select_oracle_predictions(predictions, targets)
    strategy_predictions = {
        "zero_action": np.zeros_like(targets),
        "branch0": predictions[:, :, 0],
        "branch_mean": predictions.mean(axis=2),
        "oracle_best_of_k": oracle_predictions,
    }
    strategy_metrics = {
        name: summarize_strategy(values, targets) for name, values in strategy_predictions.items()
    }
    zero_mae = strategy_metrics["zero_action"]["mean_mae"]
    for metrics in strategy_metrics.values():
        metrics["relative_improvement_over_zero"] = float(
            (zero_mae - metrics["mean_mae"]) / max(zero_mae, 1e-8)
        )

    result = {
        "checkpoint": str(args.checkpoint),
        "action_metadata": {
            "representation": representation,
            "horizon": horizon,
            "stride": stride,
            "yaw_delta_wrapped": wrap_yaw,
            "normalization_q01": action_stats.get("q01"),
            "normalization_q99": action_stats.get("q99"),
        },
        "requested_episodes": len(episode_keys),
        "completed_episodes": len(completed_keys),
        "padded_image_histories": padded_histories,
        "skipped": skipped,
        "tensor_shapes": {
            "predictions": list(predictions.shape),
            "targets": list(targets.shape),
        },
        "target_distribution": summarize(targets.reshape(-1, 4)),
        "strategies": strategy_metrics,
        "oracle_winner_rate": {
            f"branch{branch}": float(np.mean(winners == branch))
            for branch in range(args.num_action_branches)
        },
        "episode_keys": completed_keys,
    }
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w") as handle:
        json.dump(result, handle, indent=2)
    print("\nStage18 checkpoint audit summary")
    print(json.dumps({
        "action_metadata": result["action_metadata"],
        "requested_episodes": result["requested_episodes"],
        "completed_episodes": result["completed_episodes"],
        "padded_image_histories": result["padded_image_histories"],
        "skipped_count": len(result["skipped"]),
        "strategies": {
            name: {
                "mean_mae": metrics["mean_mae"],
                "t1_mean_mae": metrics["t1_mean_mae"],
                "direction_cosine_mean": metrics["direction_cosine_mean"],
                "position_delta_norm_ratio": metrics["position_delta_norm_ratio"],
                "relative_improvement_over_zero": metrics["relative_improvement_over_zero"],
            }
            for name, metrics in result["strategies"].items()
        },
        "oracle_winner_rate": result["oracle_winner_rate"],
    }, indent=2))
    print(f"Saved checkpoint audit to: {args.output_file}")


if __name__ == "__main__":
    main()
