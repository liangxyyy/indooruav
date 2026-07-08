#!/bin/bash

gnome-terminal -- bash -c "conda activate openvla-oft && cd /VLM/liangxinyue_25/IndoorUAV-Agent-main && CUDA_VISIBLE_DEVICES=2 ROBOT_PLATFORM=UAV python online_eval/vla_eval/openvla_model_runner.py --pretrained_checkpoint /VLM/liangxinyue_25/openvla-oft/runs/uav/stage6_deploy_smoke --num_action_branches 3 --action_branch_index 0 --num_images_in_input 3 --relative_actions; exec bash"

sleep 12

gnome-terminal -- bash -c "conda activate habitat && cd /VLM/liangxinyue_25/IndoorUAV-Agent-main && python online_eval/vla_eval/sim_runner.py; exec bash"

sleep 1

gnome-terminal -- bash -c "conda activate habitat && cd /VLM/liangxinyue_25/IndoorUAV-Agent-main && python online_eval/vla_eval/vla_controller.py; exec bash"
