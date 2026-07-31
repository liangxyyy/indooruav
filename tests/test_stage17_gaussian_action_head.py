import importlib.util
from pathlib import Path
import sys
import unittest

import torch

from prismatic.models.action_heads import GaussianActionHead


def _load_finetune_module():
    finetune_path = Path(__file__).parents[1] / "vla-scripts" / "finetune.py"
    spec = importlib.util.spec_from_file_location("stage17_finetune", finetune_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class GaussianActionHeadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.finetune = _load_finetune_module()

    def test_cond_action_head_returns_one_distribution_per_time_and_branch(self):
        head = GaussianActionHead(
            input_dim=8,
            hidden_dim=16,
            action_dim=4,
            num_action_branches=3,
            use_cond_action_tokens=True,
            log_std_min=-4.0,
            log_std_max=1.0,
            initial_log_std=-0.5,
        )
        hidden = torch.randn(2, 5, 3, 8)

        mean, log_std = head.predict_distribution(hidden)

        self.assertEqual(tuple(mean.shape), (2, 5, 3, 4))
        self.assertEqual(tuple(log_std.shape), (2, 5, 3, 4))
        self.assertTrue(torch.allclose(head.predict_action(hidden), mean))
        self.assertTrue(torch.allclose(log_std, torch.full_like(log_std, -0.5), atol=1e-5))

    def test_best_of_k_gaussian_loss_assigns_each_time_independently(self):
        mean = torch.tensor(
            [[[[0.0], [1.0], [2.0]], [[2.0], [0.0], [1.0]]]],
            requires_grad=True,
        )
        log_std = torch.zeros_like(mean, requires_grad=True)
        targets = torch.zeros((1, 2, 1))

        loss, balance_loss, winners, metrics = self.finetune.compute_best_of_k_gaussian_action_loss(
            mean,
            log_std,
            targets,
            assignment_temperature=0.2,
        )

        self.assertEqual(winners.tolist(), [[0, 1]])
        self.assertEqual(metrics["branch0_winner_rate"], 0.5)
        self.assertEqual(metrics["branch1_winner_rate"], 0.5)
        self.assertEqual(metrics["branch2_winner_rate"], 0.0)
        (loss + 0.1 * balance_loss).backward()
        self.assertIsNotNone(mean.grad)
        self.assertIsNotNone(log_std.grad)

    def test_gaussian_nll_penalizes_action_error(self):
        target = torch.zeros((1, 1, 1))
        log_std = torch.zeros_like(target)
        matching = self.finetune.diagonal_gaussian_nll(torch.zeros_like(target), log_std, target)
        wrong = self.finetune.diagonal_gaussian_nll(torch.ones_like(target), log_std, target)

        self.assertLess(matching.item(), wrong.item())

    def test_fixed_std_head_only_predicts_means_and_keeps_configured_std(self):
        head = GaussianActionHead(
            input_dim=8,
            hidden_dim=16,
            action_dim=4,
            num_action_branches=3,
            use_cond_action_tokens=True,
            initial_log_std=-0.5,
            learn_log_std=False,
        )
        hidden = torch.randn(2, 5, 3, 8)

        mean, log_std = head.predict_distribution(hidden)

        self.assertEqual(head.model.fc2.out_features, 4)
        self.assertEqual(tuple(mean.shape), (2, 5, 3, 4))
        self.assertTrue(torch.allclose(log_std, torch.full_like(log_std, -0.5)))
        self.assertFalse(head.fixed_log_std.requires_grad)


if __name__ == "__main__":
    unittest.main()
