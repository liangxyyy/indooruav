import importlib.util
from pathlib import Path
import unittest

import numpy as np
import tensorflow as tf

from prismatic.vla.datasets.rlds.traj_transforms import (
    chunk_act_obs,
    convert_action_chunks_to_relative,
    relative_action_statistics_trajectory,
)


def _trajectory(length=12):
    state = np.zeros((length, 4), dtype=np.float32)
    state[:, 0] = np.arange(length)
    action = state.copy()
    action[:, 0] += 1.0
    return {
        "action": tf.constant(action),
        "observation": {
            "proprio": tf.constant(state),
            "image_primary": tf.range(length),
        },
        "task": {"language_instruction": tf.repeat("move", length)},
        "dataset_name": tf.repeat("indoor_uav", length),
        "absolute_action_mask": tf.ones((length, 4), dtype=tf.bool),
    }


def _load_audit_module():
    audit_path = Path(__file__).parents[1] / "vla-scripts" / "uav_eval" / "audit_stage17_targets.py"
    spec = importlib.util.spec_from_file_location("stage17_target_audit", audit_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Stage13TrajectoryTransformTest(unittest.TestCase):
    def test_stride_two_targets_and_future_observations_match(self):
        chunked = chunk_act_obs(
            _trajectory(),
            window_size=3,
            future_action_window_size=4,
            future_action_stride=2,
            relative_action_targets=True,
        )
        relative = convert_action_chunks_to_relative(chunked, window_size=3)

        self.assertEqual(tuple(relative["action"].shape), (3, 7, 4))
        np.testing.assert_allclose(relative["action"][0, 2:, 0], [2, 4, 6, 8, 10])
        np.testing.assert_array_equal(relative["future_observation"]["image_primary"][0], [0, 2, 4, 6, 8])

    def test_relative_statistics_default_matches_pai0_plain_yaw_subtraction(self):
        traj = _trajectory()
        proprio = traj["observation"]["proprio"].numpy()
        action = traj["action"].numpy()
        proprio[0, 3] = 6.2
        action[1, 3] = 0.1
        traj["observation"]["proprio"] = tf.constant(proprio)
        traj["action"] = tf.constant(action)

        stats_traj = relative_action_statistics_trajectory(traj, horizon=5, stride=2)
        self.assertEqual(tuple(stats_traj["action"].shape), (15, 4))
        self.assertAlmostEqual(float(stats_traj["action"][0, 3]), -6.1, places=5)

    def test_relative_statistics_can_wrap_yaw_explicitly(self):
        traj = _trajectory()
        proprio = traj["observation"]["proprio"].numpy()
        action = traj["action"].numpy()
        proprio[0, 3] = 6.2
        action[1, 3] = 0.1
        traj["observation"]["proprio"] = tf.constant(proprio)
        traj["action"] = tf.constant(action)

        stats_traj = relative_action_statistics_trajectory(traj, horizon=5, stride=2, wrap_yaw=True)
        self.assertAlmostEqual(float(stats_traj["action"][0, 3]), 0.1831853, places=5)

    def test_stride_two_requires_ten_action_entries(self):
        valid = chunk_act_obs(
            _trajectory(length=10),
            window_size=3,
            future_action_window_size=4,
            future_action_stride=2,
            relative_action_targets=True,
        )
        too_short = chunk_act_obs(
            _trajectory(length=9),
            window_size=3,
            future_action_window_size=4,
            future_action_stride=2,
            relative_action_targets=True,
        )

        self.assertEqual(int(valid["action"].shape[0]), 1)
        self.assertEqual(int(too_short["action"].shape[0]), 0)

    def test_audit_target_builder_uses_the_same_offsets(self):
        audit = _load_audit_module()
        traj = _trajectory(length=12)
        targets, indices = audit.build_relative_targets(
            traj["observation"]["proprio"].numpy(),
            traj["action"].numpy(),
            horizon=5,
            stride=2,
        )

        np.testing.assert_array_equal(indices[0], [1, 3, 5, 7, 9])
        np.testing.assert_allclose(targets[0, :, 0], [2, 4, 6, 8, 10])


class Stage13InferenceRestoreTest(unittest.TestCase):
    def test_cumulative_delta_uses_plan_origin_and_wraps_yaw(self):
        runner_path = Path(__file__).parents[1] / "vla-scripts" / "uav_eval" / "openvla_model_runner.py"
        spec = importlib.util.spec_from_file_location("stage13_openvla_model_runner", runner_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        result = module.apply_action(
            coords=[99.0, 99.0, 99.0, 0.0],
            action=[2.0, -1.0, 0.5, 0.3],
            relative_actions=True,
            plan_origin=[10.0, 20.0, 3.0, 6.1],
        )
        np.testing.assert_allclose(result[:3], [12.0, 19.0, 3.5])
        self.assertAlmostEqual(result[3], (6.1 + 0.3) % (2.0 * np.pi), places=6)


if __name__ == "__main__":
    unittest.main()
