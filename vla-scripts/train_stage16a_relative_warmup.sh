#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/VLM/liangxinyue_25/openvla-oft}"
PYTHON_BIN="${PYTHON_BIN:-/home/liangxinyue_25/.conda/envs/openvla-oft/bin/python}"
VLA_PATH="${VLA_PATH:-${ROOT_DIR}/runs/uav/stage12_30k_ckpt}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-${ROOT_DIR}/runs/uav}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/VLM/datasets/indoorUAV_rlds_data/rlds_data_all}"

cd "${ROOT_DIR}"

exec env CUDA_VISIBLE_DEVICES=2 ROBOT_PLATFORM=UAV \
  "${PYTHON_BIN}" vla-scripts/finetune.py \
  --vla_path "${VLA_PATH}" \
  --data_root_dir "${DATA_ROOT_DIR}" \
  --dataset_name indoor_uav \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --shuffle_buffer_size 256 \
  --relative_action_targets True \
  --future_action_stride 2 \
  --use_l1_regression True \
  --use_diffusion False \
  --use_proprio True \
  --num_images_in_input 3 \
  --use_image_history True \
  --require_full_image_history True \
  --num_action_branches 3 \
  --use_cond_action_tokens True \
  --use_best_of_k_action_loss False \
  --branch_balance_weight 0.0 \
  --branch_diversity_weight 0.0 \
  --couple_condition_to_action_branch False \
  --condition_alignment_weight 0.0 \
  --condition_contrastive_weight 0.0 \
  --condition_diversity_weight 0.0 \
  --grpo_reward_weight 0.0 \
  --batch_size 1 \
  --learning_rate 0.00005 \
  --lr_warmup_steps 100 \
  --num_steps_before_decay 800 \
  --max_steps 1000 \
  --save_freq 500 \
  --use_lora True \
  --lora_rank 32 \
  --merge_lora_during_training True \
  --wandb_entity 3244403140-jilin-university \
  --wandb_project openvla-uav \
  --run_id_override stage16a_relative_warmup_1k \
  --wandb_log_freq 10 \
  --debug_batch_shapes True \
  --debug_grad_norm True \
  --debug_num_batches 2
