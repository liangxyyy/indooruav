import importlib.util
from pathlib import Path
import sys
import unittest

import torch


def _load_finetune_module():
    path = Path(__file__).parents[1] / "vla-scripts" / "finetune.py"
    spec = importlib.util.spec_from_file_location("stage19_finetune", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage19StructuredPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.finetune = _load_finetune_module()

    def test_joint_assignment_is_per_time_and_fixes_initial_online_branch(self):
        mean = torch.zeros((1, 2, 3, 1), requires_grad=True)
        log_std = torch.zeros_like(mean, requires_grad=True)
        target = torch.zeros((1, 2, 1))
        similarities = torch.tensor(
            [[[0.1, 0.2, 0.9], [0.1, 0.2, 0.9]]],
            requires_grad=True,
        )

        loss, balance, winners, metrics = self.finetune.compute_best_of_k_gaussian_action_loss(
            mean,
            log_std,
            target,
            assignment_temperature=0.5,
            condition_similarities=similarities,
            condition_assignment_weight=2.0,
            condition_loss_start_time_index=1,
            initial_action_branch_index=0,
        )

        self.assertEqual(winners.tolist(), [[0, 2]])
        self.assertEqual(metrics["initial_action_branch_index"], 0.0)
        self.assertEqual(metrics["joint_condition_assignment_weight"], 2.0)
        (loss + balance).backward()
        self.assertIsNotNone(mean.grad)
        self.assertIsNotNone(log_std.grad)
        self.assertIsNotNone(similarities.grad)

    def test_joint_assignment_requires_matching_condition_scores(self):
        mean = torch.zeros((1, 5, 3, 4))
        log_std = torch.zeros_like(mean)
        target = torch.zeros((1, 5, 4))

        with self.assertRaisesRegex(ValueError, "must match Gaussian branch costs"):
            self.finetune.compute_best_of_k_gaussian_action_loss(
                mean,
                log_std,
                target,
                assignment_temperature=0.5,
                condition_similarities=torch.zeros((1, 5, 2)),
                condition_assignment_weight=1.0,
            )

    def test_gaussian_group_relative_loss_uses_exact_policy_gradients(self):
        torch.manual_seed(7)
        mean = torch.zeros((2, 3, 3, 4), requires_grad=True)
        log_std = torch.full_like(mean, -0.5, requires_grad=True)
        target = torch.tensor(
            [
                [[0.3, -0.2, 0.1, 0.2]] * 3,
                [[-0.1, 0.4, 0.0, -0.3]] * 3,
            ]
        )
        selected = torch.tensor([[0, 1, 2], [2, 1, 0]])

        loss, metrics = self.finetune.compute_gaussian_group_relative_policy_loss(
            action_mean=mean,
            action_log_std=log_std,
            ground_truth_actions=target,
            action_norm_stats=None,
            selected_branch_indices=selected,
            group_size=4,
            advantage_eps=1e-4,
            advantage_clip=5.0,
            clip_epsilon=0.2,
            safety_weight=0.2,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["grpo_group_size"], 4.0)
        self.assertAlmostEqual(metrics["grpo_advantage_mean"], 0.0, places=5)
        loss.backward()
        self.assertGreater(mean.grad.abs().sum().item(), 0.0)
        self.assertGreater(log_std.grad.abs().sum().item(), 0.0)

    def test_gaussian_group_size_is_independent_of_branch_count(self):
        mean = torch.zeros((1, 2, 3, 4))
        log_std = torch.zeros_like(mean)
        target = torch.zeros((1, 2, 4))
        selected = torch.zeros((1, 2), dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "group_size must be >= 2"):
            self.finetune.compute_gaussian_group_relative_policy_loss(
                mean,
                log_std,
                target,
                None,
                selected,
                group_size=1,
                advantage_eps=1e-4,
                advantage_clip=5.0,
                clip_epsilon=0.2,
                safety_weight=0.2,
            )


if __name__ == "__main__":
    unittest.main()
