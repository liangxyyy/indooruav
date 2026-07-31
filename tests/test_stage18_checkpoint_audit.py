import importlib.util
from pathlib import Path
import unittest

import numpy as np


def _load_audit_module():
    path = Path(__file__).parents[1] / "vla-scripts" / "uav_eval" / "audit_stage18_checkpoint.py"
    spec = importlib.util.spec_from_file_location("stage18_checkpoint_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Stage18CheckpointAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = _load_audit_module()

    def test_oracle_selects_a_branch_independently_at_each_time(self):
        predictions = np.array(
            [[[[0.0], [2.0], [3.0]], [[3.0], [0.0], [2.0]]]],
            dtype=np.float32,
        )
        targets = np.zeros((1, 2, 1), dtype=np.float32)

        selected, winners = self.audit.select_oracle_predictions(predictions, targets)

        np.testing.assert_array_equal(winners, [[0, 1]])
        np.testing.assert_allclose(selected, 0.0)

    def test_strategy_summary_reports_scale_and_direction(self):
        targets = np.array([[[1.0, 0.0, 0.0, 0.2]]], dtype=np.float32)
        predictions = np.array([[[2.0, 0.0, 0.0, 0.1]]], dtype=np.float32)

        summary = self.audit.summarize_strategy(predictions, targets)

        self.assertAlmostEqual(summary["mean_mae"], 0.275, places=6)
        self.assertAlmostEqual(summary["direction_cosine_mean"], 1.0, places=6)
        self.assertAlmostEqual(summary["position_delta_norm_ratio"], 2.0, places=6)


if __name__ == "__main__":
    unittest.main()
