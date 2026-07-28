#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/VLM/liangxinyue_25/IndoorUAV-Agent-main}"
CHECKPOINT="${CHECKPOINT:-/VLM/liangxinyue_25/openvla-oft/runs/uav/openvla-7b-oft-finetuned-libero-spatial+indoor_uav+b1+lr-0.0005+lora-r32+dropout-0.0--image_aug--stage10b_cond_act_token_smoke--1_chkpt}"
CONDITION_THRESHOLD="${CONDITION_THRESHOLD:-0.2}"
USE_CONDITION_PLAN="${USE_CONDITION_PLAN:-true}"

CONDITION_PLAN_FLAG="--use_condition_plan"
if [[ "${USE_CONDITION_PLAN}" == "false" || "${USE_CONDITION_PLAN}" == "0" ]]; then
  CONDITION_PLAN_FLAG="--no-use_condition_plan"
fi

cd "${ROOT_DIR}"

exec conda run --no-capture-output -n openvla-oft \
  env CUDA_VISIBLE_DEVICES=2 ROBOT_PLATFORM=UAV \
  python -u online_eval/vla_eval/openvla_model_runner.py \
    --pretrained_checkpoint "${CHECKPOINT}" \
    --num_action_branches 3 \
    --action_branch_index 0 \
    --num_images_in_input 3 \
    --condition_threshold "${CONDITION_THRESHOLD}" \
    "${CONDITION_PLAN_FLAG}" \
    --no-relative_actions
