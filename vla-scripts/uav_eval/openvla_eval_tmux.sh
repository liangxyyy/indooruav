#!/bin/bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-openvla_uav_eval}"
ROOT_DIR="/VLM/liangxinyue_25/IndoorUAV-Agent-main"
LOG_DIR="${ROOT_DIR}/shared_folder/logs"
MODEL_LOG="${LOG_DIR}/openvla_model_runner.log"
SIM_LOG="${LOG_DIR}/sim_runner.log"
CONTROLLER_LOG="${LOG_DIR}/vla_controller.log"

mkdir -p "${LOG_DIR}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session '${SESSION_NAME}' already exists. Attach with:"
  echo "  tmux attach -t ${SESSION_NAME}"
  exit 1
fi

tmux new-session -d -s "${SESSION_NAME}" -n model
tmux send-keys -t "${SESSION_NAME}:model" "cd ${ROOT_DIR} && conda run --no-capture-output -n openvla-oft env CUDA_VISIBLE_DEVICES=2 ROBOT_PLATFORM=UAV python -u online_eval/vla_eval/openvla_model_runner.py --pretrained_checkpoint /VLM/liangxinyue_25/openvla-oft/runs/uav/openvla-7b-oft-finetuned-libero-spatial+indoor_uav+b1+lr-0.0005+lora-r32+dropout-0.0--image_aug--stage6_full_train_30k_3img_5act_3branch--30000_chkpt --num_action_branches 3 --action_branch_index 0 --num_images_in_input 3 --no-relative_actions 2>&1 | tee ${MODEL_LOG}" C-m

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
