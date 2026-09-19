"""Audit the Stage20 temporal/action contract on real IndoorUAV RLDS episodes."""

import argparse
import json
from pathlib import Path

import numpy as np

from prismatic.vla.datasets.rlds.dataset import apply_trajectory_transforms, make_dataset_from_rlds
from prismatic.vla.datasets.rlds.oxe import get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds.traj_transforms import body_delta_to_world_pose


DEFAULT_ROOT = Path("/VLM/datasets/indoorUAV_rlds_data/rlds_data_all")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--compute_statistics", action="store_true")
    parser.add_argument("--num_episodes", type=int, default=1)
    return parser.parse_args()


def audit_episode(raw, chunked):
    episode_length = int(raw["action"].shape[0])
    raw_episode_ids = np.asarray(raw["episode_id"])
    chunked_episode_ids = np.asarray(chunked["episode_id"])
    assert raw_episode_ids.shape == (episode_length,)
    assert chunked_episode_ids.shape == (episode_length,)
    assert np.all(raw_episode_ids == raw_episode_ids[0])
    assert np.all(chunked_episode_ids == raw_episode_ids[0])
    raw_actions = np.asarray(raw["action"])
    raw_states = np.asarray(raw["observation"]["proprio"])
    next_state_error = float(np.max(np.abs(raw_actions[:-1] - raw_states[1:])))
    assert next_state_error < 2e-5

    cyclic_condition_poses = np.asarray(chunked["future_observation"]["proprio"])
    assert cyclic_condition_poses.shape[-1] == 5
    np.testing.assert_allclose(
        np.linalg.norm(cyclic_condition_poses[..., 3:], axis=-1),
        1.0,
        atol=1e-5,
    )
    condition_poses = np.concatenate(
        [
            cyclic_condition_poses[..., :3],
            np.arctan2(
                cyclic_condition_poses[..., 3:4],
                cyclic_condition_poses[..., 4:5],
            ),
        ],
        axis=-1,
    )
    composed_targets = body_delta_to_world_pose(condition_poses, chunked["action"]).numpy()
    target_indices = np.minimum(
        np.arange(episode_length)[:, None] + np.arange(5)[None],
        episode_length - 1,
    )
    expected_targets = raw_actions[target_indices]
    np.testing.assert_allclose(composed_targets, expected_targets, atol=2e-5)

    history_mask = np.asarray(chunked["observation"]["pad_mask"])
    plan_mask = np.asarray(chunked["plan_valid_mask"])
    assert history_mask[0].tolist() == [False, True]
    assert plan_mask[-1].tolist() == [True, False, False, False, False]
    assert chunked["observation"]["image_primary"][0, 0] == raw["observation"]["image_primary"][0]
    assert chunked["observation"]["image_secondary"][0, 1] == raw["observation"]["image_secondary"][0]
    return episode_length, float(np.max(np.abs(composed_targets - expected_targets))), next_state_error


def main():
    args = parse_args()
    if args.num_episodes < 1:
        raise ValueError("num_episodes must be positive")
    dataset_kwargs, _ = get_oxe_dataset_kwargs_and_weights(
        args.data_root,
        [("indoor_uav", 1.0)],
        load_camera_views=("primary", "secondary"),
        load_proprio=True,
        load_language=True,
    )
    dataset, statistics = make_dataset_from_rlds(
        **dataset_kwargs[0],
        train=True,
        shuffle=False,
        dataset_statistics=None,
        body_delta_action_targets=True,
        cyclic_yaw_proprio=True,
        relative_action_horizon=5,
        relative_action_stride=1,
        num_parallel_reads=32,
        num_parallel_calls=16,
    )
    transformed = apply_trajectory_transforms(
        dataset.take(args.num_episodes),
        train=False,
        window_size=2,
        future_action_window_size=4,
        future_action_stride=1,
        body_delta_action_targets=True,
        cyclic_yaw_proprio=True,
        pad_future_horizon=True,
        skip_unlabeled=True,
        num_parallel_calls=1,
    )
    results = [
        audit_episode(raw, chunked)
        for raw, chunked in zip(
            dataset.take(args.num_episodes).iterator(),
            transformed.iterator(),
        )
    ]
    if len(results) != args.num_episodes:
        raise RuntimeError(f"Expected {args.num_episodes} episodes, audited {len(results)}")
    episode_lengths, round_trip_errors, next_state_errors = map(list, zip(*results))

    if args.compute_statistics:
        assert int(statistics["num_transitions"]) == 474_340
        assert int(statistics["num_trajectories"]) == 25_567
        yaw_min = float(statistics["action"]["min"][3])
        yaw_max = float(statistics["action"]["max"][3])
        assert -np.pi <= yaw_min < np.pi
        assert -np.pi <= yaw_max < np.pi
        assert len(statistics["proprio"]["mean"]) == 5
        assert str(statistics["proprio"]["representation"]) == "xyz_sin_yaw_cos_yaw_v1"
        assert str(statistics["action"]["normalization_representation"]) == "per_axis_symmetric_minmax_v1"
        np.testing.assert_allclose(
            statistics["action"]["normalization_low"],
            -np.asarray(statistics["action"]["normalization_high"]),
            atol=1e-7,
        )

    print(
        json.dumps(
            {
                "episodes_audited": len(results),
                "episode_length_range": [min(episode_lengths), max(episode_lengths)],
                "action_tail_shape": [5, 4],
                "proprio_tail_shape": [5],
                "first_history_mask": [False, True],
                "last_plan_valid_mask": [True, False, False, False, False],
                "episode_id_preserved": True,
                "max_world_round_trip_error": max(round_trip_errors),
                "max_raw_action_to_next_state_error": max(next_state_errors),
                "statistics_num_transitions": int(statistics["num_transitions"]),
                "statistics_num_trajectories": int(statistics["num_trajectories"]),
                "action_representation": str(statistics["action"].get("representation", "raw")),
                "status": "ok",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
