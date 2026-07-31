#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ("$1" != "stage12" && "$1" != "base") ]]; then
  echo "Usage: bash vla-scripts/uav_eval/run_stage17_ab.sh {stage12|base}" >&2
  exit 2
fi

VARIANT="$1"
REPO_ROOT="/VLM/liangxinyue_25/openvla-oft"
RUN_ROOT="${REPO_ROOT}/runs/uav"
DATA_ROOT="/VLM/datasets/indoorUAV_rlds_data/rlds_data_all"
STAGE12_CHECKPOINT="${RUN_ROOT}/stage6_30k_ckpt+indoor_uav+b1+lr-0.0005+lora-r32+dropout-0.0--image_aug--stage12--30000_chkpt"
ORIGINAL_CHECKPOINT="/VLM/liangxinyue_25/checkpoints/openvla-7b-oft-finetuned-libero-spatial"
BASE_CHECKPOINT="${RUN_ROOT}/stage17_base_openvla"

if [[ ! -f "${STAGE12_CHECKPOINT}/proprio_projector--30000_checkpoint.pt" ]]; then
  echo "Missing Stage12 proprio projector: ${STAGE12_CHECKPOINT}" >&2
  exit 1
fi

if [[ "$VARIANT" == "stage12" ]]; then
  VLA_INIT="$STAGE12_CHECKPOINT"
  RUN_ID="stage17b_ab_stage12_100step"
else
  if [[ ! -d "$BASE_CHECKPOINT" ]]; then
    echo "Preparing lightweight copy of the original OpenVLA checkpoint..."
    mkdir -p "$BASE_CHECKPOINT"
    cp -as "${ORIGINAL_CHECKPOINT}/." "${BASE_CHECKPOINT}/"
    for mutable_file in config.json modeling_prismatic.py configuration_prismatic.py; do
      cp --remove-destination \
        "${ORIGINAL_CHECKPOINT}/${mutable_file}" \
        "${BASE_CHECKPOINT}/${mutable_file}"
    done
  fi
  VLA_INIT="$BASE_CHECKPOINT"
  RUN_ID="stage17b_ab_base_100step"
fi

if [[ -e "${RUN_ROOT}/${RUN_ID}" || -e "${RUN_ROOT}/${RUN_ID}--100_chkpt" ]]; then
  echo "Run output already exists for ${RUN_ID}; refusing to overwrite it." >&2
  exit 1
fi

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES=2
export ROBOT_PLATFORM=UAV

echo "Stage17 A/B variant: ${VARIANT}"
echo "VLA initialization: ${VLA_INIT}"
echo "Auxiliary initialization: ${STAGE12_CHECKPOINT}"
echo "Run ID: ${RUN_ID}"

python -u vla-scripts/finetune.py \
  --vla_path "$VLA_INIT" \
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
  --image_aug False \
  --batch_size 1 \
  --learning_rate 0.0005 \
  --max_grad_norm 10.0 \
  --seed 17 \
  --max_steps 100 \
  --save_freq 100 \
  --wandb_entity 3244403140-jilin-university \
  --wandb_project openvla-uav \
  --run_id_override "$RUN_ID" \
  --debug_batch_shapes True \
  --debug_grad_norm True \
  --debug_num_batches 2 \
  2>&1 | tee "${RUN_ROOT}/${RUN_ID}.log"
