"""
traj_transforms.py

Contains trajectory transforms used in the orca data pipeline. Trajectory transforms operate on a dictionary
that represents a single trajectory, meaning each tensor has the same leading dimension (the trajectory length).
"""

import logging
from typing import Dict

import tensorflow as tf


def _wrap_to_pi(angle: tf.Tensor) -> tf.Tensor:
    pi = tf.constant(3.141592653589793, angle.dtype)
    return tf.math.floormod(angle + pi, 2.0 * pi) - pi


def _relative_pose(
    absolute_pose: tf.Tensor,
    origin_pose: tf.Tensor,
    wrap_yaw: bool = False,
) -> tf.Tensor:
    """Expresses absolute (x, y, z, yaw) poses relative to one origin pose."""
    position = absolute_pose[..., :3] - origin_pose[..., :3]
    yaw = absolute_pose[..., 3:4] - origin_pose[..., 3:4]
    if wrap_yaw:
        yaw = _wrap_to_pi(yaw)
    return tf.concat([position, yaw], axis=-1)


def relative_action_statistics_trajectory(
    traj: Dict,
    horizon: int,
    stride: int,
    wrap_yaw: bool = False,
) -> Dict:
    """Builds flattened plan-relative targets used only to compute action statistics."""
    if horizon < 1 or stride < 1:
        raise ValueError("horizon and stride must both be >= 1")

    traj_len = tf.shape(traj["action"])[0]
    effective_traj_len = tf.maximum(traj_len - (horizon * stride - 1), 0)
    target_offsets = tf.range(stride - 1, horizon * stride, stride)
    target_indices = tf.range(effective_traj_len)[:, None] + target_offsets[None]
    target_actions = tf.gather(traj["action"], target_indices)
    origins = traj["observation"]["proprio"][:effective_traj_len]
    relative_actions = _relative_pose(target_actions, origins[:, None], wrap_yaw=wrap_yaw)

    return {
        "action": tf.reshape(relative_actions, [-1, tf.shape(relative_actions)[-1]]),
        "observation": {"proprio": tf.repeat(origins, repeats=horizon, axis=0)},
    }


def convert_action_chunks_to_relative(
    traj: Dict,
    window_size: int,
    wrap_yaw: bool = False,
) -> Dict:
    """Converts each absolute action chunk to cumulative offsets from the current state."""
    origins = traj["observation"]["proprio"][:, window_size - 1]
    traj["action"] = _relative_pose(traj["action"], origins[:, None], wrap_yaw=wrap_yaw)
    traj["absolute_action_mask"] = tf.zeros_like(traj["absolute_action_mask"], dtype=tf.bool)
    return traj


def chunk_act_obs(
    traj: Dict,
    window_size: int,
    future_action_window_size: int = 0,
    future_action_stride: int = 1,
    relative_action_targets: bool = False,
) -> Dict:
    """
    Chunks actions and observations into the given window_size.

    "observation" keys are given a new axis (at index 1) of size `window_size` containing `window_size - 1`
    observations from the past and the current observation. "action" is given a new axis (at index 1) of size
    `window_size + future_action_window_size` containing `window_size - 1` actions from the past, the current
    action, and `future_action_window_size` actions from the future. "pad_mask" is added to "observation" and
    indicates whether an observation should be considered padding (i.e. if it had come from a timestep
    before the start of the trajectory).
    """
    if future_action_stride < 1:
        raise ValueError("future_action_stride must be >= 1")

    traj_len = tf.shape(traj["action"])[0]
    num_action_targets = future_action_window_size + 1
    if relative_action_targets:
        final_action_offset = num_action_targets * future_action_stride - 1
        target_action_offsets = tf.range(
            future_action_stride - 1,
            num_action_targets * future_action_stride,
            future_action_stride,
        )
        future_obs_offsets = tf.range(
            0,
            num_action_targets * future_action_stride,
            future_action_stride,
        )
    else:
        final_action_offset = future_action_window_size * future_action_stride
        target_action_offsets = tf.range(
            0,
            num_action_targets * future_action_stride,
            future_action_stride,
        )
        future_obs_offsets = target_action_offsets + 1

    effective_traj_len = tf.maximum(traj_len - final_action_offset, 0)
    chunk_indices = tf.broadcast_to(tf.range(-window_size + 1, 1), [effective_traj_len, window_size]) + tf.broadcast_to(
        tf.range(effective_traj_len)[:, None], [effective_traj_len, window_size]
    )

    action_offsets = tf.concat(
        [
            tf.range(-window_size + 1, 0),
            target_action_offsets,
        ],
        axis=0,
    )
    action_chunk_indices = tf.broadcast_to(
        action_offsets,
        [effective_traj_len, window_size + future_action_window_size],
    ) + tf.broadcast_to(
        tf.range(effective_traj_len)[:, None],
        [effective_traj_len, window_size + future_action_window_size],
    )

    floored_chunk_indices = tf.maximum(chunk_indices, 0)

    goal_timestep = tf.fill([effective_traj_len], traj_len - 1)

    floored_action_chunk_indices = tf.minimum(tf.maximum(action_chunk_indices, 0), goal_timestep[:, None])
    future_obs_indices = tf.broadcast_to(
        future_obs_offsets,
        [effective_traj_len, future_action_window_size + 1],
    ) + tf.broadcast_to(
        tf.range(effective_traj_len)[:, None],
        [effective_traj_len, future_action_window_size + 1],
    )
    floored_future_obs_indices = tf.minimum(tf.maximum(future_obs_indices, 0), goal_timestep[:, None])
    future_observation = tf.nest.map_structure(lambda x: tf.gather(x, floored_future_obs_indices), traj["observation"])

    traj["observation"] = tf.nest.map_structure(lambda x: tf.gather(x, floored_chunk_indices), traj["observation"])
    traj["action"] = tf.gather(traj["action"], floored_action_chunk_indices)
    traj["future_observation"] = future_observation

    # indicates whether an entire observation is padding
    traj["observation"]["pad_mask"] = chunk_indices >= 0

    # Truncate other elements of the trajectory dict
    traj["task"] = tf.nest.map_structure(lambda x: tf.gather(x, tf.range(effective_traj_len)), traj["task"])
    traj["dataset_name"] = tf.gather(traj["dataset_name"], tf.range(effective_traj_len))
    traj["absolute_action_mask"] = tf.gather(traj["absolute_action_mask"], tf.range(effective_traj_len))

    return traj


def subsample(traj: Dict, subsample_length: int) -> Dict:
    """Subsamples trajectories to the given length."""
    traj_len = tf.shape(traj["action"])[0]
    if traj_len > subsample_length:
        indices = tf.random.shuffle(tf.range(traj_len))[:subsample_length]
        traj = tf.nest.map_structure(lambda x: tf.gather(x, indices), traj)

    return traj


def add_pad_mask_dict(traj: Dict) -> Dict:
    """
    Adds a dictionary indicating which elements of the observation/task should be treated as padding.
        =>> traj["observation"|"task"]["pad_mask_dict"] = {k: traj["observation"|"task"][k] is not padding}
    """
    traj_len = tf.shape(traj["action"])[0]

    for key in ["observation", "task"]:
        pad_mask_dict = {}
        for subkey in traj[key]:
            # Handles "language_instruction", "image_*", and "depth_*"
            if traj[key][subkey].dtype == tf.string:
                pad_mask_dict[subkey] = tf.strings.length(traj[key][subkey]) != 0

            # All other keys should not be treated as padding
            else:
                pad_mask_dict[subkey] = tf.ones([traj_len], dtype=tf.bool)

        traj[key]["pad_mask_dict"] = pad_mask_dict

    return traj
