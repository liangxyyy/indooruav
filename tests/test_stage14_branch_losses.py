import importlib.util
from pathlib import Path
import sys
import unittest

import torch


def _load_finetune_module():
    finetune_path = Path(__file__).parents[1] / "vla-scripts" / "finetune.py"
    spec = importlib.util.spec_from_file_location("stage14_finetune", finetune_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BestOfKActionLossTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_finetune_module()

    def test_assigns_a_winner_at_each_time(self):
        predictions = torch.tensor(
            [[[[0.1], [1.0], [2.0]], [[2.0], [0.1], [1.0]]]],
            requires_grad=True,
        )
        targets = torch.zeros((1, 2, 1))

        best_loss, balance_loss, winners, metrics = self.module.compute_best_of_k_action_loss(
            predictions,
            targets,
            assignment_temperature=0.2,
        )

        self.assertAlmostEqual(best_loss.item(), 0.1, places=6)
        self.assertEqual(winners.tolist(), [[0, 1]])
        self.assertGreaterEqual(balance_loss.item(), 0.0)
        self.assertAlmostEqual(metrics["branch0_winner_rate"], 0.5)
        self.assertAlmostEqual(metrics["branch1_winner_rate"], 0.5)
        self.assertAlmostEqual(metrics["branch2_winner_rate"], 0.0)

        (best_loss + 0.1 * balance_loss).backward()
        self.assertIsNotNone(predictions.grad)

    def test_condition_contrastive_loss_uses_time_negatives(self):
        conditions = torch.eye(3).reshape(1, 3, 3)
        targets = torch.eye(3).reshape(1, 3, 3)

        loss, accuracy = self.module.compute_condition_contrastive_loss(
            conditions,
            targets,
            temperature=0.1,
        )

        self.assertLess(loss.item(), 0.01)
        self.assertEqual(accuracy.item(), 1.0)

    def test_balance_loss_penalizes_collapsed_soft_assignments(self):
        predictions = torch.tensor(
            [[[[0.0], [2.0], [3.0]], [[0.0], [2.0], [3.0]]]],
        )
        targets = torch.zeros((1, 2, 1))

        _, balance_loss, _, metrics = self.module.compute_best_of_k_action_loss(
            predictions,
            targets,
            assignment_temperature=0.1,
        )

        self.assertGreater(balance_loss.item(), 1.0)
        self.assertGreater(metrics["branch0_soft_usage"], 0.99)


if __name__ == "__main__":
    unittest.main()
