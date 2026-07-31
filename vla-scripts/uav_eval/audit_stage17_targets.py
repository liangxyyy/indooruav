"""Audit IndoorUAV plan-relative action targets without image decoding or shuffling."""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root_dir",
        type=Path,
        default=Path("/VLM/datasets/indoorUAV_rlds_data/rlds_data_all"),
    )
    parser.add_argument("--dataset_name", default="indoor_uav")
    parser.add_argument("--split", default="train")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max_trajectories", type=int, default=100)
    parser.add_argument("--max_windows", type=int, default=10_000)
    parser.add_argument(
        "--shuffle_files",
        action="store_true",
        help="Shuffle only TFRecord file order; this does not create an example shuffle buffer.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--wrap_yaw", action="store_true")
    parser.add_argument(
        "--output_file",
        type=Path,
        default=Path("runs/uav/stage17_target_audit.json"),
    )
    return parser.parse_args()


def wrap_to_pi(values):
    return np.mod(values + np.pi, 2.0 * np.pi) - np.pi


def build_relative_targets(states, actions, horizon, stride, wrap_yaw=False):
    """Returns [num_windows, horizon, 4] targets and their raw action indices."""
    if horizon < 1 or stride < 1:
        raise ValueError("horizon and stride must both be >= 1")
    if states.ndim != 2 or actions.ndim != 2 or states.shape[1] != 4 or actions.shape[1] != 4:
        raise ValueError("states and actions must both have shape [trajectory_length, 4]")
    if len(states) != len(actions):
        raise ValueError("states and actions must have the same trajectory length")

    offsets = np.arange(stride - 1, horizon * stride, stride)
    num_windows = max(len(actions) - int(offsets[-1]), 0)
    if num_windows == 0:
        return np.empty((0, horizon, 4), dtype=np.float32), np.empty((0, horizon), dtype=np.int64)

    indices = np.arange(num_windows)[:, None] + offsets[None]
    targets = actions[indices].astype(np.float32) - states[:num_windows, None].astype(np.float32)
    if wrap_yaw:
        targets[..., 3] = wrap_to_pi(targets[..., 3])
    return targets, indices


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "median": np.median(values, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
    }


def main():
    args = parse_args()
    if args.max_trajectories < 1 or args.max_windows < 1:
        raise ValueError("max_trajectories and max_windows must both be >= 1")

    builder = tfds.builder(args.dataset_name, data_dir=str(args.data_root_dir))
    skip_image_decode = tfds.decode.SkipDecoding()
    read_config = tfds.ReadConfig(shuffle_seed=args.seed)
    dataset = builder.as_dataset(
        split=args.split,
        shuffle_files=args.shuffle_files,
        read_config=read_config,
        decoders={
            "steps": {
                "observation": {
                    "image": skip_image_decode,
                    "ref_image": skip_image_decode,
                }
            }
        },
    )

    targets_by_time = [[] for _ in range(args.horizon)]
    plain_yaw_deltas = []
    action_next_state_errors = []
    trajectory_lengths = []
    windows_collected = 0
    trajectories_read = 0
    short_trajectories = 0

    for episode in dataset.take(args.max_trajectories):
        steps = episode["steps"].map(
            lambda step: (step["observation"]["state"], step["action"]),
            num_parallel_calls=1,
        )
        pairs = list(tfds.as_numpy(steps))
        if not pairs:
            continue

        states = np.stack([pair[0] for pair in pairs])
        actions = np.stack([pair[1] for pair in pairs])
        trajectory_lengths.append(len(actions))
        trajectories_read += 1

        if len(actions) > 1:
            action_next_state_errors.append(np.abs(actions[:-1] - states[1:]))

        plain_targets, _ = build_relative_targets(
            states,
            actions,
            horizon=args.horizon,
            stride=args.stride,
            wrap_yaw=False,
        )
        if len(plain_targets) == 0:
            short_trajectories += 1
            continue

        remaining = args.max_windows - windows_collected
        plain_targets = plain_targets[:remaining]
        targets = plain_targets.copy()
        if args.wrap_yaw:
            targets[..., 3] = wrap_to_pi(targets[..., 3])
        plain_yaw_deltas.append(plain_targets[..., 3].reshape(-1))
        for time_idx in range(args.horizon):
            targets_by_time[time_idx].append(targets[:, time_idx])
        windows_collected += len(targets)
        if windows_collected >= args.max_windows:
            break

    if windows_collected == 0:
        raise RuntimeError("No valid target windows were found")

    per_time = [np.concatenate(parts, axis=0) for parts in targets_by_time]
    all_targets = np.concatenate(per_time, axis=0)
    all_plain_yaw = np.concatenate(plain_yaw_deltas, axis=0)
    all_wrapped_yaw = wrap_to_pi(all_plain_yaw)
    constant_target = np.median(all_targets, axis=0)
    constant_baseline_mae = np.abs(all_targets - constant_target).mean(axis=0)
    next_state_error = np.concatenate(action_next_state_errors, axis=0)

    result = {
        "dataset": args.dataset_name,
        "split": args.split,
        "shuffle_files": args.shuffle_files,
        "seed": args.seed,
        "horizon": args.horizon,
        "stride": args.stride,
        "target_action_offsets": list(range(args.stride - 1, args.horizon * args.stride, args.stride)),
        "target_arrival_observation_offsets": list(range(args.stride, (args.horizon + 1) * args.stride, args.stride)),
        "condition_observation_offsets": list(range(0, args.horizon * args.stride, args.stride)),
        "yaw_delta_wrapped": args.wrap_yaw,
        "trajectories_read": trajectories_read,
        "short_trajectories": short_trajectories,
        "trajectory_length": summarize(np.asarray(trajectory_lengths)[:, None]),
        "windows_collected": windows_collected,
        "action_matches_next_state_abs_error": summarize(next_state_error),
        "plain_yaw_delta": summarize(all_plain_yaw[:, None]),
        "wrapped_yaw_delta": summarize(all_wrapped_yaw[:, None]),
        "yaw_wrap_changes_rate": float(np.mean(np.abs(all_plain_yaw - all_wrapped_yaw) > 1e-5)),
        "relative_target_all_times": summarize(all_targets),
        "relative_target_by_time": {
            f"t{time_idx + 1}": summarize(values) for time_idx, values in enumerate(per_time)
        },
        "constant_median_target": constant_target.tolist(),
        "constant_median_baseline_mae": constant_baseline_mae.tolist(),
        "constant_median_baseline_mean_mae": float(constant_baseline_mae.mean()),
    }

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"Saved audit report to: {args.output_file}")


if __name__ == "__main__":
    tf.config.set_visible_devices([], "GPU")
    main()
