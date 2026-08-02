#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ("$1" != "smoke" && "$1" != "grpo-smoke" && "$1" != "sft" && "$1" != "grpo") ]]; then
  echo "Usage: bash vla-scripts/uav_eval/run_stage19_structured_policy.sh {smoke|grpo-smoke|sft|grpo}" >&2
  exit 2
fi

MODE="$1"
REPO_ROOT="/VLM/liangxinyue_25/openvla-oft"
RUN_ROOT="${REPO_ROOT}/runs/uav"
DATA_ROOT="/VLM/datasets/indoorUAV_rlds_data/rlds_data_all"
STAGE12_CHECKPOINT="${RUN_ROOT}/stage6_30k_ckpt+indoor_uav+b1+lr-0.0005+lora-r32+dropout-0.0--image_aug--stage12--30000_chkpt"
STAGE19_SFT_CHECKPOINT="${RUN_ROOT}/stage19_structured_sft_30k--30000_chkpt"

case "$MODE" in
  smoke)
    VLA_INIT="$STAGE12_CHECKPOINT"
    AUX_INIT="$STAGE12_CHECKPOINT"
    AUX_STEP=30000
    RESET_ACTION_HEAD=True
    RUN_ID="stage19_structured_smoke"
    MAX_STEPS=1
    SAVE_FREQ=100000
    SHUFFLE_BUFFER=32
    IMAGE_AUG=False
    LEARNING_RATE=0.00005
    GRPO_WEIGHT=0.0
    ;;
  grpo-smoke)
    VLA_INIT="$STAGE12_CHECKPOINT"
    AUX_INIT="$STAGE12_CHECKPOINT"
    AUX_STEP=30000
    RESET_ACTION_HEAD=True
    RUN_ID="stage19_structured_grpo_smoke"
    MAX_STEPS=1
    SAVE_FREQ=100000
    SHUFFLE_BUFFER=32
    IMAGE_AUG=False
    LEARNING_RATE=0.00001
    GRPO_WEIGHT=0.02
    ;;
  sft)
    VLA_INIT="$STAGE12_CHECKPOINT"
    AUX_INIT="$STAGE12_CHECKPOINT"
    AUX_STEP=30000
    RESET_ACTION_HEAD=True
    RUN_ID="stage19_structured_sft_30k"
    MAX_STEPS=30000
    SAVE_FREQ=10000
    SHUFFLE_BUFFER=1000
    IMAGE_AUG=True
    LEARNING_RATE=0.00005
    GRPO_WEIGHT=0.0
    ;;
  grpo)
    VLA_INIT="$STAGE19_SFT_CHECKPOINT"
    AUX_INIT="$STAGE19_SFT_CHECKPOINT"
    AUX_STEP=30000
    RESET_ACTION_HEAD=False
    RUN_ID="stage19_structured_grpo_5k"
    MAX_STEPS=5000
    SAVE_FREQ=5000
    SHUFFLE_BUFFER=1000
    IMAGE_AUG=True
    LEARNING_RATE=0.00001
    GRPO_WEIGHT=0.02
    ;;
esac

if [[ ! -f "${VLA_INIT}/model.safetensors.index.json" ]]; then
  echo "Missing VLA checkpoint: ${VLA_INIT}" >&2
  exit 1
fi
if [[ ! -f "${AUX_INIT}/proprio_projector--${AUX_STEP}_checkpoint.pt" ]]; then
  echo "Missing proprio projector: ${AUX_INIT}/proprio_projector--${AUX_STEP}_checkpoint.pt" >&2
  exit 1
fi
if [[ "$RESET_ACTION_HEAD" == "False" && ! -f "${AUX_INIT}/action_head--${AUX_STEP}_checkpoint.pt" ]]; then
  echo "Missing Stage19 action head: ${AUX_INIT}/action_head--${AUX_STEP}_checkpoint.pt" >&2
  exit 1
fi
if [[ -e "${RUN_ROOT}/${RUN_ID}" || -e "${RUN_ROOT}/${RUN_ID}--${MAX_STEPS}_chkpt" ]]; then
  echo "Run output already exists for ${RUN_ID}; refusing to overwrite it." >&2
  exit 1
fi

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES=2
export ROBOT_PLATFORM=UAV

conda run --no-capture-output -n openvla-oft python -u vla-scripts/finetune.py \
  --vla_path "$VLA_INIT" \
  --auxiliary_init_checkpoint_path "$AUX_INIT" \
  --auxiliary_init_checkpoint_step "$AUX_STEP" \
  --reset_action_head "$RESET_ACTION_HEAD" \
  --data_root_dir "$DATA_ROOT" \
  --dataset_name indoor_uav \
  --run_root_dir "$RUN_ROOT" \
  --shuffle_buffer_size "$SHUFFLE_BUFFER" \
  --relative_action_targets True \
  --relative_action_wrap_yaw True \
  --future_action_stride 2 \
  --use_l1_regression True \
  --use_diffusion False \
  --use_gaussian_action_head True \
  --gaussian_log_std_min -5.0 \
  --gaussian_log_std_max 1.0 \
  --gaussian_initial_log_std -0.5 \
  --gaussian_learn_log_std True \
  --use_proprio True \
  --num_images_in_input 3 \
  --use_image_history True \
  --require_full_image_history True \
  --use_cond_action_tokens True \
  --num_action_branches 3 \
  --use_best_of_k_action_loss True \
  --initial_action_branch_index 0 \
  --branch_assignment_temperature 0.5 \
  --condition_assignment_weight 0.25 \
  --branch_balance_weight 0.01 \
  --branch_diversity_weight 0.01 \
  --branch_diversity_margin 0.05 \
  --couple_condition_to_action_branch True \
  --condition_similarity_threshold 0.6 \
  --condition_alignment_weight 0.2 \
  --condition_contrastive_weight 0.05 \
  --condition_contrastive_temperature 0.07 \
  --condition_loss_start_time_index 1 \
  --condition_patch_topk 8 \
  --condition_diversity_weight 0.01 \
  --condition_diversity_margin 0.05 \
  --grpo_reward_weight "$GRPO_WEIGHT" \
  --grpo_group_size 4 \
  --grpo_clip_epsilon 0.2 \
  --grpo_safety_weight 0.2 \
  --grpo_advantage_eps 0.0001 \
  --grpo_advantage_clip 5.0 \
  --freeze_vla False \
  --freeze_proprio_projector False \
  --image_aug "$IMAGE_AUG" \
  --batch_size 1 \
  --grad_accumulation_steps 8 \
  --learning_rate "$LEARNING_RATE" \
  --lr_warmup_steps 200 \
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
