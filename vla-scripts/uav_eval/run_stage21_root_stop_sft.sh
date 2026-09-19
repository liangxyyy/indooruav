#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ("$1" != "smoke" && "$1" != "pilot" && "$1" != "audit" && "$1" != "progress_smoke" && "$1" != "progress_pilot" && "$1" != "progress_audit") ]]; then
  echo "Usage: bash vla-scripts/uav_eval/run_stage21_root_stop_sft.sh {smoke|pilot|audit|progress_smoke|progress_pilot|progress_audit}" >&2
  exit 2
fi

MODE="$1"
REPO_ROOT="/VLM/liangxinyue_25/openvla-oft"
RUN_ROOT="${REPO_ROOT}/runs/uav"
DATA_ROOT="/VLM/datasets/indoorUAV_rlds_data/rlds_data_all"
SOURCE_CHECKPOINT="${RUN_ROOT}/stage20_base_bodydelta_sft_pilot_1k_v3--750_chkpt"
SOURCE_STEP=750
USE_PROGRESS_STOP=False
STOP_PROGRESS_LOSS_WEIGHT=0.0

if [[ "$MODE" == "smoke" || "$MODE" == "progress_smoke" ]]; then
  if [[ "$MODE" == "progress_smoke" ]]; then
    RUN_ID="stage22_step750_progress_stop_sft_smoke"
    USE_PROGRESS_STOP=True
    STOP_PROGRESS_LOSS_WEIGHT=0.25
  else
    RUN_ID="stage21_step750_root_stop_sft_smoke"
  fi
  MAX_STEPS=2
  SAVE_FREQ=999999
  SHUFFLE_BUFFER=32
  IMAGE_AUG=False
  GRAD_ACCUMULATION_STEPS=1
  LEARNING_RATE=0.0001
  LR_WARMUP_STEPS=0
  TRAIN_REPORT_FREQ=1
  VALIDATION_ARGS=()
elif [[ "$MODE" == "pilot" || "$MODE" == "progress_pilot" ]]; then
  if [[ "$MODE" == "progress_pilot" ]]; then
    RUN_ID="stage22_step750_progress_stop_sft_pilot_1k"
    USE_PROGRESS_STOP=True
    STOP_PROGRESS_LOSS_WEIGHT=0.25
  else
    RUN_ID="stage21_step750_root_stop_sft_pilot_1k"
  fi
  MAX_STEPS=1000
  SAVE_FREQ=250
  SHUFFLE_BUFFER=1000
  IMAGE_AUG=True
  GRAD_ACCUMULATION_STEPS=8
  LEARNING_RATE=0.0001
  LR_WARMUP_STEPS=100
  TRAIN_REPORT_FREQ=25
  VALIDATION_ARGS=(
    --use_val_set True
    --train_tfds_split 'train[:95%]'
    --val_tfds_split 'train[95%:]'
    --val_freq 250
    --val_time_limit 120
  )
else
  AUDIT_STEP="${PILOT_AUDIT_STEP:-500}"
  if ! [[ "$AUDIT_STEP" =~ ^(250|500|750|1000)$ ]]; then
    echo "PILOT_AUDIT_STEP must be one of 250, 500, 750, or 1000" >&2
    exit 2
  fi
  SOURCE_STEP="$AUDIT_STEP"
  if [[ "$MODE" == "progress_audit" ]]; then
    SOURCE_CHECKPOINT="${RUN_ROOT}/stage22_step750_progress_stop_sft_pilot_1k--${AUDIT_STEP}_chkpt"
    RUN_ID="stage22_progress_stop_score_audit_step${AUDIT_STEP}_v1_n1000"
    USE_PROGRESS_STOP=True
    STOP_PROGRESS_LOSS_WEIGHT=0.25
  else
    SOURCE_CHECKPOINT="${RUN_ROOT}/stage21_step750_root_stop_sft_pilot_1k--${AUDIT_STEP}_chkpt"
    RUN_ID="stage21_stop_score_audit_step${AUDIT_STEP}_v2_n1000"
  fi
  MAX_STEPS=$((AUDIT_STEP + 1))
  SAVE_FREQ=999999
  # Validation caches shuffle_buffer/10 frames; 10k gives a stable audit pool
  # of up to 1,000 windows instead of the 100-window training-time quick check.
  SHUFFLE_BUFFER=10000
  IMAGE_AUG=False
  GRAD_ACCUMULATION_STEPS=1
  LEARNING_RATE=0.0
  LR_WARMUP_STEPS=0
  TRAIN_REPORT_FREQ=1
  VALIDATION_ARGS=(
    --use_val_set True
    --train_tfds_split 'train[:95%]'
    --val_tfds_split 'train[95%:]'
    --val_freq "$MAX_STEPS"
    --val_time_limit "${STOP_AUDIT_SECONDS:-600}"
  )
fi

RUNTIME_CHECKPOINT="${RUN_ROOT}/${RUN_ID}_runtime"

for required in \
  model.safetensors.index.json \
  policy_contract.json \
  "action_head--${SOURCE_STEP}_checkpoint.pt" \
  "condition_adapter--${SOURCE_STEP}_checkpoint.pt" \
  "proprio_projector--${SOURCE_STEP}_checkpoint.pt"; do
  if [[ ! -f "${SOURCE_CHECKPOINT}/${required}" ]]; then
    echo "Missing source component: ${SOURCE_CHECKPOINT}/${required}" >&2
    exit 1
  fi
done
if [[ ("$MODE" == "audit" || "$MODE" == "progress_audit") && ! -f "${SOURCE_CHECKPOINT}/stop_head--${SOURCE_STEP}_checkpoint.pt" ]]; then
  echo "Missing Stage21 STOP head: ${SOURCE_CHECKPOINT}/stop_head--${SOURCE_STEP}_checkpoint.pt" >&2
  exit 1
fi

if [[ -e "${RUN_ROOT}/${RUN_ID}" || -e "${RUN_ROOT}/${RUN_ID}--${MAX_STEPS}_chkpt" ]]; then
  echo "Run output already exists for ${RUN_ID}; refusing to overwrite it." >&2
  exit 1
fi

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export ROBOT_PLATFORM=UAV

# Avoid modifying the selected Stage20 checkpoint when Transformers synchronizes
# local modeling/config files.  Large immutable weights are hard-linked; files
# that the loader may update are private copies in this runtime mirror.
if [[ ! -d "$RUNTIME_CHECKPOINT" ]]; then
  mkdir -p "$RUNTIME_CHECKPOINT"
  cp -al "${SOURCE_CHECKPOINT}/." "$RUNTIME_CHECKPOINT/"
  for mutable_file in config.json modeling_prismatic.py configuration_prismatic.py; do
    cp --remove-destination \
      "${SOURCE_CHECKPOINT}/${mutable_file}" \
      "${RUNTIME_CHECKPOINT}/${mutable_file}"
  done
fi

if [[ "$MODE" == "audit" || "$MODE" == "progress_audit" ]]; then
  MODEL_ARGS=(
    --vla_path "$RUNTIME_CHECKPOINT"
    --resume True
    --resume_step "$SOURCE_STEP"
  )
else
  MODEL_ARGS=(
    --vla_path "$RUNTIME_CHECKPOINT"
    --auxiliary_init_checkpoint_path "$SOURCE_CHECKPOINT"
    --auxiliary_init_checkpoint_step "$SOURCE_STEP"
    --reset_action_head False
    --reset_proprio_projector False
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
  --use_indoor_uav_condition_adapter True \
  --condition_match_hidden_dim 1024 \
  --condition_match_dim 512 \
  --num_action_branches 3 \
  --use_best_of_k_action_loss True \
  --initial_action_branch_index 0 \
  --root_action_weight 0.0 \
  --future_action_weight 0.0 \
  --branch_assignment_temperature 0.5 \
  --condition_assignment_weight 0.0 \
  --branch_balance_weight 0.0 \
  --branch_diversity_weight 0.0 \
  --couple_condition_to_action_branch True \
  --condition_alignment_weight 0.0 \
  --condition_contrastive_weight 0.0 \
  --condition_temporal_weight 0.0 \
  --condition_queue_weight 0.0 \
  --condition_queue_size 256 \
  --condition_queue_min_negatives 32 \
  --condition_contrastive_temperature 0.07 \
  --condition_loss_start_time_index 1 \
  --condition_patch_topk 8 \
  --condition_diversity_weight 0.0 \
  --condition_diversity_margin 0.05 \
  --grpo_reward_weight 0.0 \
  --use_indoor_uav_stop_head True \
  --use_indoor_uav_progress_stop_head "$USE_PROGRESS_STOP" \
  --stop_head_hidden_dim 1024 \
  --stop_progress_projection_dim 512 \
  --stop_progress_loss_weight "$STOP_PROGRESS_LOSS_WEIGHT" \
  --stop_initial_positive_rate 0.053900156 \
  --stop_loss_weight 1.0 \
  --stop_positive_weight 0.0 \
  --stop_threshold 0.5 \
  --freeze_vla True \
  --freeze_proprio_projector True \
  --freeze_action_head True \
  --freeze_condition_adapter True \
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
  2>&1 | tee "${RUN_ROOT}/${RUN_ID}.log"
