#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ("$1" != "smoke" && "$1" != "overfit_action_k1" && "$1" != "overfit_match_k1" && "$1" != "overfit" && "$1" != "audit_k3" && "$1" != "audit_pilot" && "$1" != "pilot" && "$1" != "train") ]]; then
  echo "Usage: bash vla-scripts/uav_eval/run_stage20_bodydelta_sft.sh {smoke|overfit_action_k1|overfit_match_k1|overfit|audit_k3|audit_pilot|pilot|train}" >&2
  exit 2
fi

MODE="$1"
REPO_ROOT="/VLM/liangxinyue_25/openvla-oft"
RUN_ROOT="${REPO_ROOT}/runs/uav"
DATA_ROOT="/VLM/datasets/indoorUAV_rlds_data/rlds_data_all"
SOURCE_BASE_CHECKPOINT="/VLM/base-model/openvla-7b"
BASE_CHECKPOINT="${RUN_ROOT}/stage20_openvla_base_runtime"
K3_OVERFIT_CHECKPOINT="${RUN_ROOT}/stage20_base_bodydelta_sft_overfit_k3_v2--300_chkpt"
PILOT_AUDIT_STEP="${PILOT_AUDIT_STEP:-1000}"
PILOT_CHECKPOINT="${RUN_ROOT}/stage20_base_bodydelta_sft_pilot_1k_v3--${PILOT_AUDIT_STEP}_chkpt"

NUM_ACTION_BRANCHES=3
USE_BEST_OF_K=True
USE_CONDITION_ADAPTER=True
COUPLE_CONDITION_ACTION=True
CONDITION_ASSIGNMENT_WEIGHT=0.0
BRANCH_BALANCE_WEIGHT=0.01
CONDITION_ALIGNMENT_WEIGHT=0.0
CONDITION_CONTRASTIVE_WEIGHT=1.0
CONDITION_TEMPORAL_WEIGHT=0.25
CONDITION_QUEUE_WEIGHT=0.05
CONDITION_QUEUE_SIZE=256
CONDITION_QUEUE_MIN_NEGATIVES=32
CONDITION_DIVERSITY_WEIGHT=0.005
TRAIN_REPORT_FREQ=50
VALIDATION_ARGS=()

if [[ "$MODE" == "smoke" ]]; then
  RUN_ID="stage20_base_bodydelta_sft_matchv2_smoke"
  MAX_STEPS=2
  SAVE_FREQ=999999
  SHUFFLE_BUFFER=32
  IMAGE_AUG=False
  GRAD_ACCUMULATION_STEPS=8
  LEARNING_RATE=0.00005
  LR_WARMUP_STEPS=200
  OVERFIT_FIXED_BATCH_COUNT=0
  OVERFIT_REPORT_FREQ=25
  TRAIN_REPORT_FREQ=1
elif [[ "$MODE" == "overfit_action_k1" || "$MODE" == "overfit_match_k1" ]]; then
  if [[ "$MODE" == "overfit_action_k1" ]]; then
    RUN_ID="stage20_base_bodydelta_sft_overfit_action_k1_v2"
    USE_CONDITION_ADAPTER=False
    CONDITION_CONTRASTIVE_WEIGHT=0.0
    CONDITION_TEMPORAL_WEIGHT=0.0
    CONDITION_QUEUE_WEIGHT=0.0
  else
    RUN_ID="stage20_base_bodydelta_sft_overfit_match_k1_v2"
    CONDITION_CONTRASTIVE_WEIGHT=0.0
    CONDITION_TEMPORAL_WEIGHT=1.0
    CONDITION_QUEUE_WEIGHT=0.0
  fi
  NUM_ACTION_BRANCHES=1
  USE_BEST_OF_K=False
  COUPLE_CONDITION_ACTION=False
  CONDITION_ASSIGNMENT_WEIGHT=0.0
  BRANCH_BALANCE_WEIGHT=0.0
  CONDITION_DIVERSITY_WEIGHT=0.0
  MAX_STEPS=300
  SAVE_FREQ=300
  SHUFFLE_BUFFER=32
  IMAGE_AUG=False
  GRAD_ACCUMULATION_STEPS=1
  LEARNING_RATE=0.0001
  LR_WARMUP_STEPS=20
  OVERFIT_FIXED_BATCH_COUNT=1
  OVERFIT_REPORT_FREQ=25
elif [[ "$MODE" == "overfit" ]]; then
  RUN_ID="stage20_base_bodydelta_sft_overfit_k3_v2"
  MAX_STEPS=300
  SAVE_FREQ=300
  SHUFFLE_BUFFER=32
  IMAGE_AUG=False
  GRAD_ACCUMULATION_STEPS=1
  LEARNING_RATE=0.0001
  LR_WARMUP_STEPS=20
  OVERFIT_FIXED_BATCH_COUNT=1
  OVERFIT_REPORT_FREQ=25
elif [[ "$MODE" == "audit_k3" ]]; then
  RUN_ID="stage20_base_bodydelta_sft_overfit_k3_v2_audit"
  MAX_STEPS=301
  SAVE_FREQ=999999
  SHUFFLE_BUFFER=32
  IMAGE_AUG=False
  GRAD_ACCUMULATION_STEPS=1
  LEARNING_RATE=0.0
  LR_WARMUP_STEPS=0
  OVERFIT_FIXED_BATCH_COUNT=1
  OVERFIT_REPORT_FREQ=1
elif [[ "$MODE" == "audit_pilot" ]]; then
  if ! [[ "$PILOT_AUDIT_STEP" =~ ^[0-9]+$ ]]; then
    echo "PILOT_AUDIT_STEP must be a non-negative integer" >&2
    exit 2
  fi
  RUN_ID="stage20_base_bodydelta_sft_pilot_1k_v3_condition_action_audit_step${PILOT_AUDIT_STEP}"
  MAX_STEPS=$((PILOT_AUDIT_STEP + 1))
  SAVE_FREQ=999999
  SHUFFLE_BUFFER=1000
  IMAGE_AUG=False
  GRAD_ACCUMULATION_STEPS=1
  LEARNING_RATE=0.0
  LR_WARMUP_STEPS=0
  OVERFIT_FIXED_BATCH_COUNT=0
  OVERFIT_REPORT_FREQ=25
  TRAIN_REPORT_FREQ=1
  VALIDATION_ARGS=(
    --use_val_set True
    --train_tfds_split 'train[:95%]'
    --val_tfds_split 'train[95%:]'
    --val_freq 1
    --val_time_limit 120
  )
elif [[ "$MODE" == "pilot" ]]; then
  RUN_ID="stage20_base_bodydelta_sft_pilot_1k_v3"
  MAX_STEPS=1000
  SAVE_FREQ=250
  SHUFFLE_BUFFER=1000
  IMAGE_AUG=True
  GRAD_ACCUMULATION_STEPS=8
  LEARNING_RATE=0.00005
  LR_WARMUP_STEPS=200
  OVERFIT_FIXED_BATCH_COUNT=0
  OVERFIT_REPORT_FREQ=25
  VALIDATION_ARGS=(
    --use_val_set True
    --train_tfds_split 'train[:95%]'
    --val_tfds_split 'train[95%:]'
    --val_freq 250
    --val_time_limit 120
  )
else
  RUN_ID="stage20_base_bodydelta_sft_30k_v2"
  MAX_STEPS=30000
  SAVE_FREQ=5000
  SHUFFLE_BUFFER=1000
  IMAGE_AUG=True
  GRAD_ACCUMULATION_STEPS=8
  LEARNING_RATE=0.00005
  LR_WARMUP_STEPS=200
  OVERFIT_FIXED_BATCH_COUNT=0
  OVERFIT_REPORT_FREQ=25
fi

if [[ ! -f "${SOURCE_BASE_CHECKPOINT}/model.safetensors.index.json" ]]; then
  echo "Missing pristine OpenVLA checkpoint: ${SOURCE_BASE_CHECKPOINT}" >&2
  exit 1
fi
if [[ "$MODE" == "audit_k3" && ! -f "${K3_OVERFIT_CHECKPOINT}/model.safetensors.index.json" ]]; then
  echo "Missing K=3 overfit checkpoint: ${K3_OVERFIT_CHECKPOINT}" >&2
  exit 1
fi
if [[ "$MODE" == "audit_pilot" && ! -f "${PILOT_CHECKPOINT}/model.safetensors.index.json" ]]; then
  echo "Missing pilot checkpoint: ${PILOT_CHECKPOINT}" >&2
  exit 1
fi
if [[ -e "${RUN_ROOT}/${RUN_ID}" || -e "${RUN_ROOT}/${RUN_ID}--${MAX_STEPS}_chkpt" ]]; then
  echo "Run output already exists for ${RUN_ID}; refusing to overwrite it." >&2
  exit 1
fi

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export ROBOT_PLATFORM=UAV

# The trainer synchronizes local modeling/config files into the checkpoint directory.
# Use a symlink-based runtime mirror so the pristine base model is never modified.
if [[ ! -d "$BASE_CHECKPOINT" ]]; then
  mkdir -p "$BASE_CHECKPOINT"
  cp -as "${SOURCE_BASE_CHECKPOINT}/." "${BASE_CHECKPOINT}/"
  for mutable_file in config.json modeling_prismatic.py configuration_prismatic.py; do
    cp --remove-destination \
      "${SOURCE_BASE_CHECKPOINT}/${mutable_file}" \
      "${BASE_CHECKPOINT}/${mutable_file}"
  done
fi

MODEL_ARGS=(
  --vla_path "$BASE_CHECKPOINT"
  --reset_action_head True
  --reset_proprio_projector True
)
if [[ "$MODE" == "audit_k3" ]]; then
  MODEL_ARGS=(
    --vla_path "$K3_OVERFIT_CHECKPOINT"
    --resume True
    --resume_step 300
  )
elif [[ "$MODE" == "audit_pilot" ]]; then
  MODEL_ARGS=(
    --vla_path "$PILOT_CHECKPOINT"
    --resume True
    --resume_step "$PILOT_AUDIT_STEP"
  )
fi

conda run --no-capture-output -n openvla-oft python -u vla-scripts/finetune.py \
  "${MODEL_ARGS[@]}" \
  --data_root_dir "$DATA_ROOT" \
  --dataset_name indoor_uav \
  "${VALIDATION_ARGS[@]}" \
  --run_root_dir "$RUN_ROOT" \
  --shuffle_buffer_size "$SHUFFLE_BUFFER" \
  --relative_action_targets False \
  --body_delta_action_targets True \
  --cyclic_yaw_proprio True \
  --future_action_stride 1 \
  --use_l1_regression True \
  --use_diffusion False \
  --use_gaussian_action_head False \
  --action_regression_loss l1 \
  --use_proprio True \
  --num_images_in_input 3 \
  --use_image_history True \
  --require_full_image_history False \
  --use_reference_previous_current True \
  --use_cond_action_tokens True \
  --use_indoor_uav_condition_adapter "$USE_CONDITION_ADAPTER" \
  --condition_match_hidden_dim 1024 \
  --condition_match_dim 512 \
  --num_action_branches "$NUM_ACTION_BRANCHES" \
  --use_best_of_k_action_loss "$USE_BEST_OF_K" \
  --initial_action_branch_index 0 \
  --root_action_weight 1.0 \
  --future_action_weight 1.0 \
  --branch_assignment_temperature 0.5 \
  --condition_assignment_weight "$CONDITION_ASSIGNMENT_WEIGHT" \
  --branch_balance_weight "$BRANCH_BALANCE_WEIGHT" \
  --branch_diversity_weight 0.0 \
  --couple_condition_to_action_branch "$COUPLE_CONDITION_ACTION" \
  --condition_alignment_weight "$CONDITION_ALIGNMENT_WEIGHT" \
  --condition_contrastive_weight "$CONDITION_CONTRASTIVE_WEIGHT" \
  --condition_temporal_weight "$CONDITION_TEMPORAL_WEIGHT" \
  --condition_queue_weight "$CONDITION_QUEUE_WEIGHT" \
  --condition_queue_size "$CONDITION_QUEUE_SIZE" \
  --condition_queue_min_negatives "$CONDITION_QUEUE_MIN_NEGATIVES" \
  --condition_contrastive_temperature 0.07 \
  --condition_loss_start_time_index 1 \
  --condition_patch_topk 8 \
  --condition_diversity_weight "$CONDITION_DIVERSITY_WEIGHT" \
  --condition_diversity_margin 0.05 \
  --grpo_reward_weight 0.0 \
  --freeze_vla False \
  --freeze_proprio_projector False \
  --image_aug "$IMAGE_AUG" \
  --batch_size 1 \
  --grad_accumulation_steps "$GRAD_ACCUMULATION_STEPS" \
  --learning_rate "$LEARNING_RATE" \
  --lr_warmup_steps "$LR_WARMUP_STEPS" \
  --max_grad_norm 10.0 \
  --seed 17 \
  --max_steps "$MAX_STEPS" \
  --save_freq "$SAVE_FREQ" \
  --train_report_freq "$TRAIN_REPORT_FREQ" \
  --wandb_entity 3244403140-jilin-university \
  --wandb_project openvla-uav \
  --run_id_override "$RUN_ID" \
  --debug_batch_shapes True \
  --debug_grad_norm True \
  --debug_num_batches 2 \
  --overfit_fixed_batch_count "$OVERFIT_FIXED_BATCH_COUNT" \
  --overfit_report_freq "$OVERFIT_REPORT_FREQ" \
  2>&1 | tee "${RUN_ROOT}/${RUN_ID}.log"
