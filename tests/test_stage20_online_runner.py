import importlib.util
import unittest
from pathlib import Path

import numpy as np


def _load_runner_module():
    path = Path(__file__).parents[1] / "vla-scripts" / "uav_eval" / "openvla_model_runner.py"
    spec = importlib.util.spec_from_file_location("stage20_online_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Stage20OnlineRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = _load_runner_module()

    def test_normalize_coords_accepts_numpy_arrays(self):
        coords = self.runner.normalize_coords(np.asarray([1.0, 2.0, 3.0, 0.5]))
        np.testing.assert_allclose(coords, [1.0, 2.0, 3.0, 0.5])

    def test_body_delta_uses_wrapped_yaw_and_declared_body_axes(self):
        result = self.runner.apply_body_delta(
            [0.0, 0.0, 1.0, 2.0 * np.pi - 0.1],
            [1.0, 0.0, 0.5, 0.2],
        )
        np.testing.assert_allclose(result[:3], [-np.sin(0.1), -np.cos(0.1), 1.5], atol=1e-6)
        self.assertAlmostEqual(result[3], 0.1, places=6)


if __name__ == "__main__":
    unittest.main()
