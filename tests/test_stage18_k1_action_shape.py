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


if __name__ == "__main__":
    unittest.main()
