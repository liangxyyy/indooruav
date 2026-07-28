#!/bin/bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-openvla_uav_eval}"
ROOT_DIR="/VLM/liangxinyue_25/IndoorUAV-Agent-main"
LOG_DIR="${ROOT_DIR}/shared_folder/logs"
MODEL_LOG="${LOG_DIR}/openvla_model_runner.log"
SIM_LOG="${LOG_DIR}/sim_runner.log"
CONTROLLER_LOG="${LOG_DIR}/vla_controller.log"
CHECKPOINT="${CHECKPOINT:-/VLM/liangxinyue_25/openvla-oft/runs/uav/openvla-7b-oft-finetuned-libero-spatial+indoor_uav+b1+lr-0.0005+lora-r32+dropout-0.0--image_aug--stage10b_cond_act_token_smoke--1_chkpt}"
CONDITION_THRESHOLD="${CONDITION_THRESHOLD:-0.2}"
USE_CONDITION_PLAN="${USE_CONDITION_PLAN:-true}"

mkdir -p "${LOG_DIR}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session '${SESSION_NAME}' already exists. Attach with:"
  echo "  tmux attach -t ${SESSION_NAME}"
  exit 1
fi

tmux new-session -d -s "${SESSION_NAME}" -n model
tmux set-environment -t "${SESSION_NAME}" CHECKPOINT "${CHECKPOINT}"
tmux set-environment -t "${SESSION_NAME}" CONDITION_THRESHOLD "${CONDITION_THRESHOLD}"
tmux set-environment -t "${SESSION_NAME}" USE_CONDITION_PLAN "${USE_CONDITION_PLAN}"
tmux send-keys -t "${SESSION_NAME}:model" "cd ${ROOT_DIR} && bash online_eval/vla_eval/run_openvla_model_runner.sh 2>&1 | tee ${MODEL_LOG}" C-m

echo "Waiting for OpenVLA model runner to become ready..."
for _ in $(seq 1 7200); do
  if grep -q "OpenVLA model runner started." "${MODEL_LOG}" 2>/dev/null; then
    break
  fi
  if grep -q "Traceback" "${MODEL_LOG}" 2>/dev/null; then
    echo "OpenVLA model runner failed during startup. Check:"
    echo "  ${MODEL_LOG}"
    tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
    exit 1
  fi
  sleep 1
done

if ! grep -q "OpenVLA model runner started." "${MODEL_LOG}" 2>/dev/null; then
  echo "Timed out waiting for OpenVLA model runner. Check:"
  echo "  ${MODEL_LOG}"
  echo "Model session is still available for debugging: tmux attach -t ${SESSION_NAME}"
  exit 1
fi

tmux new-window -t "${SESSION_NAME}" -n sim
tmux send-keys -t "${SESSION_NAME}:sim" "cd ${ROOT_DIR} && conda run --no-capture-output -n habitat python -u online_eval/vla_eval/sim_runner.py 2>&1 | tee ${SIM_LOG}" C-m

tmux new-window -t "${SESSION_NAME}" -n controller
tmux send-keys -t "${SESSION_NAME}:controller" "cd ${ROOT_DIR} && conda run --no-capture-output -n habitat python -u online_eval/vla_eval/vla_controller.py 2>&1 | tee ${CONTROLLER_LOG}" C-m

echo "Started tmux session '${SESSION_NAME}'."
echo "Attach:"
echo "  tmux attach -t ${SESSION_NAME}"
echo "Logs:"
echo "  ${MODEL_LOG}"
echo "  ${SIM_LOG}"
echo "  ${CONTROLLER_LOG}"
