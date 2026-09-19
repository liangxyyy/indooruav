import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import tensorflow as tf
import torch

from prismatic.models.uav_condition_adapter import IndoorUAVConditionAdapter
from prismatic.models.uav_progress_stop_head import IndoorUAVProgressStopHead
from prismatic.models.uav_stop_head import IndoorUAVStopHead
from prismatic.vla.condition_matching import CrossEpisodeImageQueue, projected_condition_to_patch_similarity
from prismatic.vla.datasets.rlds.traj_transforms import (
    body_delta_to_world_pose,
    chunk_act_obs,
    convert_action_chunks_to_body_delta,
    convert_pose_observations_to_cyclic,
    pose_to_cyclic_proprio,
    world_pose_pair_to_body_delta,
)
from prismatic.vla.datasets.rlds.utils.data_utils import normalize_action_and_proprio
from prismatic.vla.constants import NormalizationType


ROOT = Path(__file__).parents[1]


def _load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _trajectory(length=6):
    state = np.zeros((length, 4), dtype=np.float32)
    state[:, 0] = np.arange(length)
    action = state.copy()
    action[:, 0] += 1.0
    return {
        "action": tf.constant(action),
        "observation": {
            "proprio": tf.constant(state),
            "image_primary": tf.range(length),
            "image_secondary": tf.zeros(length, dtype=tf.int32),
        },
        "task": {"language_instruction": tf.repeat("move", length)},
        "dataset_name": tf.repeat("indoor_uav", length),
        "stop_after_action": tf.range(length) == length - 1,
        "absolute_action_mask": tf.ones((length, 4), dtype=tf.bool),
    }


class BodyDeltaContractTest(unittest.TestCase):
    def test_world_body_round_trip_at_cardinal_yaws(self):
        poses = tf.constant(
            [[1.0, 2.0, 3.0, yaw] for yaw in (0.0, np.pi / 2, np.pi, -np.pi / 2)],
            dtype=tf.float32,
        )
        deltas = tf.constant([[0.8, -0.3, 0.2, 0.1]] * 4)
        targets = body_delta_to_world_pose(poses, deltas)
        recovered = world_pose_pair_to_body_delta(poses, targets)
        np.testing.assert_allclose(recovered.numpy(), deltas.numpy(), atol=1e-6)

    def test_indoor_uav_zero_yaw_axes(self):
        pose = tf.constant([10.0, 20.0, 3.0, 0.0])
        target = body_delta_to_world_pose(pose, tf.constant([2.0, 1.0, 0.5, 0.2]))
        np.testing.assert_allclose(target.numpy(), [11.0, 18.0, 3.5, 0.2], atol=1e-6)
        recovered = world_pose_pair_to_body_delta(pose, target)
        np.testing.assert_allclose(recovered.numpy(), [2.0, 1.0, 0.5, 0.2], atol=1e-6)

    def test_yaw_uses_shortest_signed_turn_across_zero(self):
        position = [1.0, 2.0, 3.0]
        right_turn = world_pose_pair_to_body_delta(
            tf.constant(position + [np.deg2rad(350.0)], dtype=tf.float32),
            tf.constant(position + [np.deg2rad(10.0)], dtype=tf.float32),
        )
        left_turn = world_pose_pair_to_body_delta(
            tf.constant(position + [np.deg2rad(10.0)], dtype=tf.float32),
            tf.constant(position + [np.deg2rad(350.0)], dtype=tf.float32),
        )
        self.assertAlmostEqual(np.rad2deg(right_turn.numpy()[3]), 20.0, places=4)
        self.assertAlmostEqual(np.rad2deg(left_turn.numpy()[3]), -20.0, places=4)

        runner = _load_module("stage20_runner_yaw", "vla-scripts/uav_eval/openvla_model_runner.py")
        right_target = runner.apply_body_delta(
            position + [np.deg2rad(350.0)], [0, 0, 0, np.deg2rad(20.0)]
        )
        left_target = runner.apply_body_delta(
            position + [np.deg2rad(10.0)], [0, 0, 0, np.deg2rad(-20.0)]
        )
        self.assertAlmostEqual(np.rad2deg(right_target[3]), 10.0, places=4)
        self.assertAlmostEqual(np.rad2deg(left_target[3]), 350.0, places=4)

    def test_cyclic_proprio_is_continuous_across_yaw_wrap(self):
        poses = tf.constant(
            [[1.0, 2.0, 3.0, np.deg2rad(359.0)], [1.0, 2.0, 3.0, np.deg2rad(1.0)]],
            dtype=tf.float32,
        )
        cyclic = pose_to_cyclic_proprio(poses).numpy()
        self.assertEqual(cyclic.shape, (2, 5))
        self.assertLess(np.linalg.norm(cyclic[0, 3:] - cyclic[1, 3:]), 0.04)
        np.testing.assert_allclose(np.linalg.norm(cyclic[:, 3:], axis=-1), 1.0, atol=1e-6)

    def test_half_turn_uses_negative_pi_tie_break(self):
        pose = tf.constant([0.0, 0.0, 0.0, 0.0])
        target = tf.constant([0.0, 0.0, 0.0, np.pi])
        delta = world_pose_pair_to_body_delta(pose, target)
        self.assertAlmostEqual(float(delta[3]), -np.pi, places=6)

    def test_chunk_alignment_and_terminal_mask(self):
        chunked = chunk_act_obs(
            _trajectory(),
            window_size=2,
            future_action_window_size=4,
            future_action_stride=1,
            body_delta_action_targets=True,
            pad_future_horizon=True,
        )
        converted = convert_action_chunks_to_body_delta(chunked, window_size=2)

        self.assertEqual(tuple(converted["action"].shape), (6, 5, 4))
        np.testing.assert_array_equal(converted["observation"]["pad_mask"][0], [False, True])
        np.testing.assert_array_equal(converted["plan_valid_mask"][0], [True] * 5)
        np.testing.assert_array_equal(converted["plan_valid_mask"][-1], [True, False, False, False, False])
        np.testing.assert_array_equal(
            converted["stop_after_action"], [False, False, False, False, False, True]
        )
        np.testing.assert_array_equal(
            converted["actions_remaining_after_root"], [5, 4, 3, 2, 1, 0]
        )
        # At IndoorUAV yaw zero, +x is the body-right axis.
        np.testing.assert_allclose(converted["action"][0, :, 1], np.ones(5), atol=1e-6)

        cyclic = convert_pose_observations_to_cyclic(converted)
        self.assertEqual(tuple(cyclic["observation"]["proprio"].shape), (6, 2, 5))
        self.assertEqual(tuple(cyclic["future_observation"]["proprio"].shape), (6, 5, 5))
        # Geometry must already be complete: changing the proprio representation leaves actions untouched.
        np.testing.assert_allclose(cyclic["action"][0, :, 1], np.ones(5), atol=1e-6)

    def test_explicit_symmetric_action_bounds_preserve_rare_lateral_motion(self):
        trajectory = {
            "action": tf.constant([[0.0, 0.0, 0.0, 0.0], [0.0, 0.28, 0.0, 0.0], [0.0, 0.56, 0.0, 0.0]]),
            "observation": {"proprio": tf.zeros((3, 5), dtype=tf.float32)},
        }
        metadata = {
            "action": {
                "min": tf.constant([-0.56] * 4),
                "max": tf.constant([0.56] * 4),
                "q01": tf.constant([-1e-5] * 4),
                "q99": tf.constant([1e-5] * 4),
                "normalization_low": tf.constant([-0.56] * 4),
                "normalization_high": tf.constant([0.56] * 4),
            },
            "proprio": {
                "min": tf.constant([-1.0] * 5),
                "max": tf.constant([1.0] * 5),
                "q01": tf.constant([-1.0] * 5),
                "q99": tf.constant([1.0] * 5),
                "normalization_low": tf.constant([-1.0] * 5),
                "normalization_high": tf.constant([1.0] * 5),
            },
        }
        normalized = normalize_action_and_proprio(
            trajectory,
            metadata,
            NormalizationType.BOUNDS_Q99,
        )
        np.testing.assert_allclose(normalized["action"][:, 1], [0.0, 0.5, 1.0], atol=1e-6)

    def test_body_delta_rejects_non_unit_stride(self):
        with self.assertRaisesRegex(ValueError, "requires future_action_stride=1"):
            chunk_act_obs(
                _trajectory(),
                window_size=2,
                future_action_window_size=4,
                future_action_stride=2,
                body_delta_action_targets=True,
            )

    def test_online_body_composition_uses_current_yaw(self):
        runner = _load_module("stage20_runner", "vla-scripts/uav_eval/openvla_model_runner.py")
        pose = [10.0, 20.0, 3.0, np.pi / 2]
        delta = [2.0, 1.0, 0.5, 0.2]
        result = runner.apply_body_delta(pose, delta)
        tensorflow_result = body_delta_to_world_pose(tf.constant(pose), tf.constant(delta)).numpy()
        np.testing.assert_allclose(result[:3], [12.0, 21.0, 3.5], atol=1e-6)
        np.testing.assert_allclose(result, tensorflow_result, atol=1e-6)


class ConditionSFTTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.finetune = _load_module("stage20_finetune", "vla-scripts/finetune.py")

    def test_projector_and_topk_shapes(self):
        adapter = IndoorUAVConditionAdapter(llm_dim=16, hidden_dim=8, match_dim=4)
        conditions = adapter.project_conditions(torch.randn(2, 5, 3, 16))
        patches = adapter.project_vision(torch.randn(2, 4, 7, 16))
        scores, indices = projected_condition_to_patch_similarity(
            conditions[:, 1:], patches, topk_patches=3, return_indices=True
        )

        self.assertEqual(tuple(scores.shape), (2, 4, 3))
        self.assertEqual(tuple(indices.shape), (2, 4, 3, 3))
        torch.testing.assert_close(conditions.norm(dim=-1), torch.ones(2, 5, 3))

    def test_step_events_only_fire_at_optimizer_boundaries(self):
        observed = [
            self.finetune.completed_optimizer_step(batch_idx, 8)
            for batch_idx in range(16)
        ]
        self.assertEqual(observed[:7], [None] * 7)
        self.assertEqual(observed[7], 1)
        self.assertEqual(observed[8:15], [None] * 7)
        self.assertEqual(observed[15], 2)
        self.assertEqual(self.finetune.completed_optimizer_step(7, 8, resume_step=300), 301)
        with self.assertRaises(ValueError):
            self.finetune.completed_optimizer_step(0, 0)

    def test_validation_aggregation_ignores_fully_masked_future_windows(self):
        metrics = self.finetune.aggregate_validation_metrics(
            [
                {
                    "condition_branch_accuracy": 0.5,
                    "condition_retrieval_accuracy": 0.25,
                    "condition_queue_accuracy": 0.75,
                    "future_valid_count": 4.0,
                    "condition_temporal_query_count": 4.0,
                    "condition_queue_queries": 4.0,
                },
                {
                    "condition_branch_accuracy": 0.0,
                    "condition_retrieval_accuracy": 0.0,
                    "condition_queue_accuracy": 0.0,
                    "future_valid_count": 0.0,
                    "condition_temporal_query_count": 0.0,
                    "condition_queue_queries": 0.0,
                },
            ]
        )

        self.assertEqual(metrics["condition_branch_accuracy"], 0.5)
        self.assertEqual(metrics["condition_retrieval_accuracy"], 0.25)
        self.assertEqual(metrics["condition_queue_accuracy"], 0.75)
        self.assertEqual(metrics["future_valid_count"], 4.0)

    def test_slot_zero_and_invalid_slots_have_no_action_gradient(self):
        actions = torch.zeros((1, 5, 3, 4), requires_grad=True)
        targets = torch.ones((1, 5, 4))
        conditions = torch.randn(1, 5, 3, 8, requires_grad=True)
        patches = torch.randn(1, 4, 6, 8, requires_grad=True)
        similarities = projected_condition_to_patch_similarity(conditions[:, 1:], patches, 2)
        valid = torch.tensor([[True, True, False, False, False]])

        loss, _, _ = self.finetune.compute_indoor_uav_sft_loss(
            actions,
            targets,
            conditions[:, 1:],
            patches,
            similarities,
            valid,
            assignment_temperature=0.5,
            root_action_weight=1.0,
            future_action_weight=1.0,
            condition_alignment_weight=0.1,
            condition_contrastive_weight=1.0,
            condition_temporal_weight=0.25,
            condition_queue_weight=0.0,
            condition_contrastive_temperature=0.07,
            branch_balance_weight=0.01,
            condition_diversity_weight=0.005,
            condition_diversity_margin=0.05,
            patch_topk=2,
        )
        loss.backward()

        self.assertGreater(actions.grad[0, 0, 0].abs().sum().item(), 0.0)
        self.assertEqual(actions.grad[0, 0, 1:].abs().sum().item(), 0.0)
        self.assertEqual(actions.grad[0, 2:].abs().sum().item(), 0.0)
        self.assertGreater(patches.grad.abs().sum().item(), 0.0)

    def test_condition_logits_do_not_choose_their_own_training_label(self):
        actions = torch.ones((1, 5, 3, 4))
        actions[:, 1:, 0] = 0.0
        targets = torch.zeros((1, 5, 4))
        conditions = torch.randn(1, 4, 3, 8)
        patches = torch.randn(1, 4, 6, 8)
        # Branch 2 looks best visually, but branch 0 exactly matches the expert action.
        similarities = torch.tensor([[[0.0, 0.1, 0.9]] * 4])

        _, winners, metrics = self.finetune.compute_indoor_uav_sft_loss(
            actions,
            targets,
            conditions,
            patches,
            similarities,
            torch.ones((1, 5), dtype=torch.bool),
            assignment_temperature=0.5,
            root_action_weight=1.0,
            future_action_weight=1.0,
            condition_alignment_weight=0.0,
            condition_contrastive_weight=1.0,
            condition_temporal_weight=0.0,
            condition_queue_weight=0.0,
            condition_contrastive_temperature=0.07,
            branch_balance_weight=0.0,
            condition_diversity_weight=0.0,
            condition_diversity_margin=0.05,
            patch_topk=2,
        )

        torch.testing.assert_close(winners, torch.zeros_like(winners))
        self.assertEqual(metrics["condition_branch_accuracy"], 0.0)

    def test_action_branch_diagnostics_detect_distinct_non_tied_hypotheses(self):
        actions = torch.zeros((1, 5, 3, 4))
        actions[:, 1:, 1] = 0.2
        actions[:, 1:, 2] = 0.4
        targets = torch.zeros((1, 5, 4))
        conditions = torch.randn(1, 4, 3, 8)
        patches = torch.randn(1, 4, 6, 8)
        similarities = torch.zeros((1, 4, 3))

        _, winners, metrics = self.finetune.compute_indoor_uav_sft_loss(
            actions,
            targets,
            conditions,
            patches,
            similarities,
            torch.ones((1, 5), dtype=torch.bool),
            assignment_temperature=0.5,
            root_action_weight=1.0,
            future_action_weight=1.0,
            condition_alignment_weight=0.0,
            condition_contrastive_weight=1.0,
            condition_temporal_weight=0.0,
            condition_queue_weight=0.0,
            condition_contrastive_temperature=0.07,
            branch_balance_weight=0.0,
            condition_diversity_weight=0.0,
            condition_diversity_margin=0.05,
            patch_topk=2,
        )

        torch.testing.assert_close(winners, torch.zeros_like(winners))
        self.assertAlmostEqual(metrics["action_branch_pair_l1"], 0.8 / 3.0, places=6)
        self.assertAlmostEqual(metrics["action_branch_min_pair_l1"], 0.2, places=6)
        self.assertAlmostEqual(metrics["action_winner_gap"], 0.02, places=6)
        self.assertEqual(metrics["action_winner_near_tie_rate"], 0.0)
        self.assertEqual(metrics["future_slot1_branch0_winner_rate"], 1.0)
        self.assertEqual(metrics["future_slot4_valid_count"], 1.0)

    def test_validation_aggregation_weights_per_slot_winner_rates(self):
        metrics = self.finetune.aggregate_validation_metrics(
            [
                {
                    "future_slot1_valid_count": 1.0,
                    "future_slot1_branch0_winner_rate": 1.0,
                },
                {
                    "future_slot1_valid_count": 3.0,
                    "future_slot1_branch0_winner_rate": 0.0,
                },
            ]
        )

        self.assertEqual(metrics["future_slot1_valid_count"], 4.0)
        self.assertEqual(metrics["future_slot1_branch0_winner_rate"], 0.25)

    def test_condition_selected_action_metrics_measure_oracle_recovery(self):
        actions = torch.zeros((1, 5, 3, 4))
        actions[:, 1:, 0] = 0.4
        actions[:, 1:, 1] = 0.2
        targets = torch.zeros((1, 5, 4))
        # The condition matcher selects branch 1; branch 2 is the oracle.
        similarities = torch.tensor([[[0.0, 1.0, -1.0]] * 4])

        metrics = self.finetune.compute_condition_selected_action_metrics(
            actions,
            targets,
            similarities,
            torch.ones((1, 5), dtype=torch.bool),
            {
                "normalization_low": [-1.0] * 4,
                "normalization_high": [1.0] * 4,
                "mask": [True] * 4,
            },
        )

        self.assertAlmostEqual(metrics["oracle_future_action_loss"], 0.0, places=6)
        self.assertAlmostEqual(metrics["condition_selected_action_loss"], 0.02, places=6)
        self.assertAlmostEqual(metrics["branch0_future_action_loss"], 0.08, places=6)
        self.assertAlmostEqual(metrics["condition_selection_regret"], 0.02, places=6)
        self.assertAlmostEqual(metrics["condition_gain_vs_branch0"], 0.06, places=6)
        self.assertAlmostEqual(metrics["condition_oracle_recovery"], 0.75, places=6)
        self.assertEqual(metrics["condition_branch1_selected_rate"], 1.0)

    def test_validation_aggregation_recomputes_global_oracle_recovery(self):
        metrics = self.finetune.aggregate_validation_metrics(
            [
                {
                    "future_valid_count": 1.0,
                    "oracle_future_action_loss": 0.0,
                    "condition_selected_action_loss": 0.1,
                    "branch0_future_action_loss": 0.2,
                    "condition_oracle_recovery": 0.5,
                },
                {
                    "future_valid_count": 3.0,
                    "oracle_future_action_loss": 0.1,
                    "condition_selected_action_loss": 0.2,
                    "branch0_future_action_loss": 0.5,
                    "condition_oracle_recovery": 0.75,
                },
            ]
        )

        self.assertAlmostEqual(metrics["oracle_future_action_loss"], 0.075, places=6)
        self.assertAlmostEqual(metrics["condition_selected_action_loss"], 0.175, places=6)
        self.assertAlmostEqual(metrics["branch0_future_action_loss"], 0.425, places=6)
        self.assertAlmostEqual(metrics["condition_selection_regret"], 0.1, places=6)
        self.assertAlmostEqual(metrics["condition_gain_vs_branch0"], 0.25, places=6)
        self.assertAlmostEqual(metrics["condition_oracle_recovery"], 5.0 / 7.0, places=6)

    def test_cross_episode_queue_excludes_same_episode(self):
        queue = CrossEpisodeImageQueue(capacity=4, storage_dtype=torch.float32)
        queue.enqueue(
            torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]], [[-1.0, 0.0]]]),
            ["episode-a", "episode-b", "episode-c"],
        )
        patches, episode_ids = queue.entries()
        loss, metrics = self.finetune.compute_cross_episode_queue_loss(
            torch.tensor([[1.0, 0.0]], requires_grad=True),
            torch.tensor([[[1.0, 0.0]]], requires_grad=True),
            ["episode-a"],
            patches,
            episode_ids,
            temperature=0.1,
            patch_topk=1,
            min_negatives=2,
        )
        loss.backward()

        self.assertEqual(metrics["condition_queue_negatives"], 2.0)
        self.assertEqual(metrics["condition_queue_accuracy"], 1.0)
        self.assertGreater(metrics["condition_queue_margin"], 0.9)

    def test_k1_action_baseline_masks_invalid_tail(self):
        actions = torch.zeros((1, 5, 1, 4), requires_grad=True)
        targets = torch.ones((1, 5, 4))
        valid = torch.tensor([[True, True, False, False, False]])

        loss, metrics = self.finetune.compute_masked_single_branch_sft_loss(
            actions,
            targets,
            valid,
            root_action_weight=1.0,
            future_action_weight=1.0,
        )
        loss.backward()

        self.assertAlmostEqual(metrics["sft_root_action_loss"], 0.5)
        self.assertAlmostEqual(metrics["sft_future_action_loss"], 0.5)
        self.assertGreater(actions.grad[0, :2].abs().sum().item(), 0.0)
        self.assertEqual(actions.grad[0, 2:].abs().sum().item(), 0.0)

    def test_stop_head_shape_prior_and_gradient(self):
        head = IndoorUAVStopHead(input_dim=16, hidden_dim=8, initial_positive_rate=0.05)
        hidden = torch.zeros((4, 16), requires_grad=True)
        logits = head(hidden)

        self.assertEqual(tuple(logits.shape), (4,))
        torch.testing.assert_close(
            torch.sigmoid(logits),
            torch.full((4,), 0.05),
            atol=1e-6,
            rtol=0.0,
        )
        logits.sum().backward()
        self.assertGreater(head.network[-1].weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(head.network[-1].bias.grad.abs().sum().item(), 0.0)
        with self.assertRaisesRegex(ValueError, "shape"):
            head(torch.zeros((2, 3, 16)))

    def test_stop_loss_and_global_validation_confusion_metrics(self):
        logits = torch.tensor([4.0, -4.0, 0.2, -0.2], requires_grad=True)
        targets = torch.tensor([1.0, 0.0, 0.0, 1.0])
        loss, first = self.finetune.compute_stop_after_action_loss(
            logits, targets, positive_weight=3.0, threshold=0.5
        )
        loss.backward()

        self.assertGreater(logits.grad.abs().sum().item(), 0.0)
        self.assertEqual(first["stop_true_positive"], 1.0)
        self.assertEqual(first["stop_true_negative"], 1.0)
        self.assertEqual(first["stop_false_positive"], 1.0)
        self.assertEqual(first["stop_false_negative"], 1.0)

        _, second = self.finetune.compute_stop_after_action_loss(
            torch.tensor([5.0, -5.0]),
            torch.tensor([1.0, 0.0]),
            positive_weight=3.0,
            threshold=0.5,
        )
        metrics = self.finetune.aggregate_validation_metrics([first, second])
        self.assertAlmostEqual(metrics["stop_accuracy"], 4.0 / 6.0)
        self.assertAlmostEqual(metrics["stop_precision"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["stop_recall"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["stop_specificity"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["stop_balanced_accuracy"], 2.0 / 3.0)

    def test_stop_score_diagnostics_compute_exact_auc_and_threshold(self):
        metrics = self.finetune.compute_binary_score_diagnostics(
            [0.9, 0.8, 0.7, 0.1],
            [1, 0, 1, 0],
        )

        self.assertAlmostEqual(metrics["stop_roc_auc"], 0.75)
        self.assertAlmostEqual(metrics["stop_probability_class_gap"], 0.35)
        self.assertAlmostEqual(metrics["stop_best_threshold"], 0.9)
        self.assertAlmostEqual(metrics["stop_best_balanced_accuracy"], 0.75)
        self.assertAlmostEqual(metrics["stop_best_precision"], 1.0)
        self.assertAlmostEqual(metrics["stop_best_recall"], 0.5)
        self.assertAlmostEqual(metrics["stop_best_specificity"], 1.0)
        with self.assertRaisesRegex(ValueError, "both positive and negative"):
            self.finetune.compute_binary_score_diagnostics([0.1, 0.2], [0, 0])

    def test_progress_stop_head_and_ordinal_targets(self):
        head = IndoorUAVProgressStopHead(
            input_dim=16,
            projection_dim=8,
            hidden_dim=12,
            initial_positive_rates=(0.05, 0.10, 0.15, 0.25),
        )
        actions = torch.zeros((2, 16))
        conditions = torch.zeros((2, 16))
        logits = head(actions, conditions)
        self.assertEqual(tuple(logits.shape), (2, 4))
        torch.testing.assert_close(
            torch.sigmoid(logits[0]),
            torch.tensor([0.05, 0.10, 0.15, 0.25]),
            atol=1e-6,
            rtol=0.0,
        )

        zero_logits = torch.zeros((2, 4), requires_grad=True)
        loss, metrics = self.finetune.compute_progress_stop_loss(
            zero_logits,
            torch.tensor([0, 3]),
            positive_weights=(1.0, 1.0, 1.0, 1.0),
            auxiliary_weight=0.5,
        )
        loss.backward()
        self.assertAlmostEqual(loss.item(), 1.5 * np.log(2.0), places=6)
        self.assertEqual(metrics["_stop_targets"], [1.0, 0.0])
        self.assertEqual(metrics["stop_progress_h1_target_rate"], 0.5)
        self.assertEqual(metrics["stop_progress_h2_target_rate"], 0.5)
        self.assertEqual(metrics["stop_progress_h4_target_rate"], 1.0)
        self.assertGreater(zero_logits.grad.abs().sum().item(), 0.0)
        with self.assertRaisesRegex(ValueError, "matching shapes"):
            head(torch.zeros((2, 16)), torch.zeros((3, 16)))

    def test_root_axis_metrics_are_reported_in_physical_units(self):
        predictions = torch.zeros((2, 5, 3, 4))
        targets = torch.zeros((2, 5, 4))
        predictions[:, 0, 0] = torch.tensor(
            [[0.5, -0.5, 0.0, 1.0], [-0.5, 0.5, 0.0, -1.0]]
        )
        targets[:, 0] = torch.tensor(
            [[1.0, -1.0, 0.0, 1.0], [-1.0, 1.0, 0.0, -1.0]]
        )
        stats = {
            "normalization_low": [-2.0, -4.0, -1.0, -0.5],
            "normalization_high": [2.0, 4.0, 1.0, 0.5],
            "mask": [True] * 4,
        }
        metrics = self.finetune.compute_root_action_axis_metrics(predictions, targets, stats)

        self.assertAlmostEqual(metrics["root_forward_abs_error"], 1.0)
        self.assertAlmostEqual(metrics["root_right_abs_error"], 2.0)
        self.assertAlmostEqual(metrics["root_yaw_abs_error"], 0.0)
        self.assertEqual(metrics["root_forward_sign_accuracy"], 1.0)
        self.assertEqual(metrics["root_up_nonzero_count"], 0.0)


if __name__ == "__main__":
    unittest.main()
