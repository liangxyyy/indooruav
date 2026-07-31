#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ("$1" != "smoke" && "$1" != "1k") ]]; then
  echo "Usage: bash vla-scripts/uav_eval/run_stage17_fixed_bc.sh {smoke|1k}" >&2
  exit 2
fi

MODE="$1"
REPO_ROOT="/VLM/liangxinyue_25/openvla-oft"
RUN_ROOT="${REPO_ROOT}/runs/uav"
DATA_ROOT="/VLM/datasets/indoorUAV_rlds_data/rlds_data_all"
STAGE12_CHECKPOINT="${RUN_ROOT}/stage6_30k_ckpt+indoor_uav+b1+lr-0.0005+lora-r32+dropout-0.0--image_aug--stage12--30000_chkpt"

if [[ "$MODE" == "smoke" ]]; then
  MAX_STEPS=1
  SAVE_FREQ=1
  RUN_ID="stage17c_fixedstd_smoke"
  IMAGE_AUG=False
else
  MAX_STEPS=1000
  SAVE_FREQ=1000
  RUN_ID="stage17c_fixedstd_1k"
  IMAGE_AUG=True
fi

if [[ ! -f "${STAGE12_CHECKPOINT}/proprio_projector--30000_checkpoint.pt" ]]; then
  echo "Missing Stage12 proprio projector: ${STAGE12_CHECKPOINT}" >&2
  exit 1
fi

if [[ -e "${RUN_ROOT}/${RUN_ID}" || -e "${RUN_ROOT}/${RUN_ID}--${MAX_STEPS}_chkpt" ]]; then
  echo "Run output already exists for ${RUN_ID}; refusing to overwrite it." >&2
  exit 1
fi

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES=2
export ROBOT_PLATFORM=UAV

python -u vla-scripts/finetune.py \
  --vla_path "$STAGE12_CHECKPOINT" \
  --auxiliary_init_checkpoint_path "$STAGE12_CHECKPOINT" \
  --auxiliary_init_checkpoint_step 30000 \
  --reset_action_head True \
  --data_root_dir "$DATA_ROOT" \
  --dataset_name indoor_uav \
  --run_root_dir "$RUN_ROOT" \
  --shuffle_buffer_size 1000 \
  --relative_action_targets True \
  --relative_action_wrap_yaw True \
  --future_action_stride 2 \
  --use_l1_regression True \
  --use_diffusion False \
  --use_gaussian_action_head True \
  --gaussian_log_std_min -5.0 \
  --gaussian_log_std_max 1.0 \
  --gaussian_initial_log_std -0.5 \
  --gaussian_learn_log_std False \
  --use_proprio True \
  --num_images_in_input 3 \
  --use_image_history True \
  --require_full_image_history True \
  --use_cond_action_tokens True \
  --num_action_branches 3 \
  --use_best_of_k_action_loss True \
  --branch_assignment_temperature 0.5 \
  --branch_balance_weight 0.01 \
  --branch_diversity_weight 0.0 \
  --condition_alignment_weight 0.0 \
  --condition_contrastive_weight 0.0 \
  --condition_diversity_weight 0.0 \
  --grpo_reward_weight 0.0 \
  --image_aug "$IMAGE_AUG" \
  --batch_size 1 \
  --learning_rate 0.0005 \
  --lr_warmup_steps 100 \
  --max_grad_norm 10.0 \
  --seed 17 \
  --max_steps "$MAX_STEPS" \
  --save_freq "$SAVE_FREQ" \
  --wandb_entity 3244403140-jilin-university \
  --wandb_project openvla-uav \
  --run_id_override "$RUN_ID" \
  --debug_batch_shapes True \
  --debug_grad_norm True \
  --debug_num_batches 2 \
  2>&1 | tee "${RUN_ROOT}/${RUN_ID}.log"
