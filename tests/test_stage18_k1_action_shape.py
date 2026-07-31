import importlib.util
from pathlib import Path
import sys
import unittest

import torch

from prismatic.models.action_heads import L1RegressionActionHead


def _load_finetune_module():
    path = Path(__file__).parents[1] / "vla-scripts" / "finetune.py"
    spec = importlib.util.spec_from_file_location("stage18_finetune", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage18K1ActionShapeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.finetune = _load_finetune_module()

    def test_cond_action_k1_keeps_explicit_branch_axis(self):
        head = L1RegressionActionHead(
            input_dim=8,
            hidden_dim=16,
            action_dim=4,
            num_action_branches=1,
            use_cond_action_tokens=True,
        )
        predictions = head.predict_action(torch.randn(2, 5, 1, 8))
        targets = torch.randn(2, 5, 4)

        matched_targets = self.finetune.match_action_target_shape(predictions, targets)

        self.assertEqual(tuple(predictions.shape), (2, 5, 1, 4))
        self.assertEqual(tuple(matched_targets.shape), (2, 5, 1, 4))
        torch.nn.L1Loss()(predictions, matched_targets).backward()
        self.assertIsNotNone(head.model.fc2.weight.grad)

    def test_overfit_batch_selector_caches_then_cycles(self):
        fixed_batches = []
        selected = []
        for batch_idx in range(8):
            incoming = {"id": batch_idx}
            batch = self.finetune.select_overfit_batch(batch_idx, incoming, fixed_batches, 4)
            selected.append(batch["id"])

        self.assertEqual([batch["id"] for batch in fixed_batches], [0, 1, 2, 3])
        self.assertEqual(selected, [0, 1, 2, 3, 0, 1, 2, 3])

    def test_overfit_batch_selector_is_disabled_by_default(self):
        fixed_batches = []
        incoming = {"id": 9}
        selected = self.finetune.select_overfit_batch(5, incoming, fixed_batches, 0)

        self.assertIs(selected, incoming)
        self.assertEqual(fixed_batches, [])

    def test_overfit_can_isolate_action_head_parameters(self):
        module = torch.nn.Linear(3, 2)

        self.finetune.set_module_trainable(module, False)
        self.assertTrue(all(not parameter.requires_grad for parameter in module.parameters()))

        self.finetune.set_module_trainable(module, True)
        self.assertTrue(all(parameter.requires_grad for parameter in module.parameters()))

    def test_point_regression_loss_supports_controlled_l1_mse_ablation(self):
        predictions = torch.tensor([[[[2.0, -1.0]]]], requires_grad=True)
        targets = torch.tensor([[[0.0, 1.0]]])

        l1_loss = self.finetune.compute_action_regression_loss(predictions, targets, "l1")
        mse_loss = self.finetune.compute_action_regression_loss(predictions, targets, "mse")

        self.assertEqual(l1_loss.item(), 2.0)
        self.assertEqual(mse_loss.item(), 4.0)
        mse_loss.backward()
        self.assertIsNotNone(predictions.grad)

    def test_point_regression_loss_rejects_unknown_objective(self):
        predictions = torch.zeros(1, 5, 1, 4)
        targets = torch.zeros(1, 5, 4)

        with self.assertRaisesRegex(ValueError, "Unsupported action_regression_loss"):
            self.finetune.compute_action_regression_loss(predictions, targets, "huber")


if __name__ == "__main__":
    unittest.main()
