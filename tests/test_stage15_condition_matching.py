import unittest

import torch

from prismatic.vla.condition_matching import (
    center_condition_branches,
    center_visual_patches,
    condition_branch_contrastive_loss,
    condition_to_patch_similarity,
)


class ConditionPatchMatchingTest(unittest.TestCase):
    def test_condition_scores_its_matching_patch_set(self):
        conditions = torch.tensor([[[[3.0, 1.0], [1.0, 3.0]]]])
        patches = torch.tensor([[[[3.0, 1.0], [2.0, 2.0], [1.0, 3.0]]]])

        scores = condition_to_patch_similarity(conditions, patches, topk_patches=1)

        self.assertEqual(tuple(scores.shape), (1, 1, 2))
        self.assertGreater(scores[0, 0, 0].item(), 0.99)
        self.assertGreater(scores[0, 0, 1].item(), 0.99)

    def test_contrastive_loss_identifies_action_supervised_branch(self):
        similarities = torch.tensor(
            [[
                [0.9, 0.1, 0.0],
                [0.2, 0.1, 0.8],
                [0.0, 0.7, 0.1],
            ]]
        )
        selected_branches = torch.tensor([[0, 2, 1]])

        loss, accuracy, margin = condition_branch_contrastive_loss(
            similarities,
            selected_branches,
            temperature=0.1,
            loss_start_time_index=1,
        )

        self.assertLess(loss.item(), 0.01)
        self.assertEqual(accuracy.item(), 1.0)
        self.assertGreater(margin.item(), 0.5)

    def test_centering_removes_large_shared_direction(self):
        common_condition = torch.tensor([1000.0, 1000.0, 1000.0])
        conditions = common_condition + torch.tensor(
            [[[[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]]]]
        )
        common_patch = torch.tensor([500.0, 500.0, 500.0])
        patches = common_patch + torch.tensor(
            [[[[2.0, -2.0, 0.0], [-2.0, 2.0, 0.0]]]]
        )

        centered_conditions = center_condition_branches(conditions)
        centered_patches = center_visual_patches(patches)
        scores = condition_to_patch_similarity(conditions, patches, topk_patches=1)

        self.assertTrue(torch.allclose(centered_conditions.mean(dim=2), torch.zeros(1, 1, 3)))
        self.assertTrue(torch.allclose(centered_patches.mean(dim=2), torch.zeros(1, 1, 3)))
        self.assertGreater(scores.min().item(), 0.99)


if __name__ == "__main__":
    unittest.main()
