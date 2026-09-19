# IndoorUAV 多分支视觉先决条件 VLA：设计、进度与复现交接说明

> 快照时间：2026-08-02（Asia/Shanghai）
>
> OpenVLA：`/VLM/liangxinyue_25/openvla-oft`
>
> 仿真评估：`/VLM/liangxinyue_25/IndoorUAV-Agent-main`

本文用于代码交接和精确复现。“已完成”“正在运行”“尚未验证”均按上述时间点定义。

## 1. 当前结论与最高优先级风险

本项目在 OpenVLA-OFT 上实现面向 IndoorUAV 的多分支视觉先决条件策略。输入是 3 帧历史图像、自然语言指令和无人机状态 `[x,y,z,yaw]`；一次前向输出未来 `T=5` 个时刻、每个时刻 `K=3` 个 `(condition, action)` 对：

```text
time 1: (cond[1,1], action[1,1]), ..., (cond[1,3], action[1,3])
...
time 5: (cond[5,1], action[5,1]), ..., (cond[5,3], action[5,3])
```

输出语义是“每个未来时刻有 K 个候选”，不是“K 条完整的 T 步轨迹”：

```text
condition hidden [B,T,K,4096] = [B,5,3,4096]
action mean      [B,T,K,4]    = [B,5,3,4]
action log std   [B,T,K,4]    = [B,5,3,4]
ground truth     [B,T,4]      = [B,5,4]
```

Stage19 是当前最新设计：对角高斯动作头、逐时刻联合 condition-action 分支分配、未来视觉 condition 监督，以及与 K 正交的 `G=4` 组相对策略采样。当前先训练 30k SFT，通过离线和仿真门槛后才允许进入 GRPO 微调。

**历史状态说明：Stage19 在该快照时间尚未提交。** 远端 `origin/main` 和本地 `HEAD` 当时都停在 Stage18：

```text
branch: main
commit: 50f1ba63e0c069d931c62418c9a83c9c57853ad2
origin: https://github.com/liangxyyy/indooruav.git
```

当前工作区：

```text
 M vla-scripts/finetune.py
?? tests/test_stage19_structured_policy.py
?? vla-scripts/uav_eval/run_stage19_structured_policy.sh
?? Liu 等 - 2025 - IndoorUAV ...pdf   # 论文，不应提交
```

以上仅记录 2026-08-02 当时的仓库状态；后续提交状态应以 `git log` 和当前主实施计划为准。不要把论文和 checkpoint 加入 Git。

## 2. 目录、数据与环境

### 2.1 路径

| 内容 | 路径 | 注意 |
|---|---|---|
| OpenVLA 主仓库 | `/VLM/liangxinyue_25/openvla-oft` | Git 仓库，Stage19 未提交 |
| IndoorUAV 仿真/评估 | `/VLM/liangxinyue_25/IndoorUAV-Agent-main` | 当前不是 Git 仓库 |
| Transformers fork | `/VLM/liangxinyue_25/transformers-openvla-oft-main` | 环境实际导入路径 |
| RLDS 训练数据 | `/VLM/datasets/indoorUAV_rlds_data/rlds_data_all` | dataset name 为 `indoor_uav` |
| 原始 IndoorUAV | `/VLM/datasets/Indoor_UAV` | 场景、姿态、指令、截图 |
| 算法构思 | `/VLM/liangxinyue_25/新算法构思.pdf` | 设计依据 |

Stage12 初始化 checkpoint：

```text
/VLM/liangxinyue_25/openvla-oft/runs/uav/stage6_30k_ckpt+indoor_uav+b1+lr-0.0005+lora-r32+dropout-0.0--image_aug--stage12--30000_chkpt
```

数据统计：

```text
num_trajectories = 25,567
num_transitions  = 474,340
action_dim       = 4
proprio_dim      = 4
```

迁移服务器时首先修改训练脚本中的 `REPO_ROOT`、`DATA_ROOT`、checkpoint、GPU 和 WandB entity，并修改仿真启动脚本中的根目录。当前这些参数存在硬编码。

### 2.2 环境快照

训练环境 `openvla-oft`：

```text
Python 3.10.20
PyTorch 2.2.0+cu121
TensorFlow 2.15.0
Transformers 4.40.1（本地 fork）
PEFT 0.11.1
```

仿真环境 `habitat`：

```text
Python 3.9.23
NumPy 1.26.4
habitat-sim 0.3.3
```

服务器为 4 张约 48 GB 的 RTX 4090，训练和 VLA 推理使用物理 GPU 2：

```bash
export CUDA_VISIBLE_DEVICES=2
export ROBOT_PLATFORM=UAV
```

`ROBOT_PLATFORM=UAV` 必须在 Python 启动前设置。预期常量：

```text
NUM_ACTIONS_CHUNK=5
ACTION_DIM=4
PROPRIO_DIM=4
ACTION_PROPRIO_NORMALIZATION_TYPE=bounds_q99
```

新机器先按仓库 `SETUP.md` 安装，再核对版本。正式交接还应导出：

```bash
conda env export -n openvla-oft --no-builds > openvla-oft.environment.yml
conda env export -n habitat --no-builds > habitat.environment.yml
```

## 3. 数据语义与时序

### 3.1 原始 action 是绝对下一状态

IndoorUAV 论文中的 VLA 基线预测未来 UAV 状态 `(x,y,z,yaw)`。RLDS 审计显示 `action[i]` 与 `observation[i+1].proprio` 完全一致，四维绝对误差均为 0。因此原始 action 不是天然 delta，而是下一状态绝对位姿。

Stage13 以后转换为相对计划原点的累计位姿：

```text
a_t = [x_arrival-x_origin,
       y_arrival-y_origin,
       z_arrival-z_origin,
       wrap_to_pi(yaw_arrival-yaw_origin)]
```

它不是相对上一步的单步增量。在线执行必须使用：

```text
next_pose = plan_origin + action[t,selected_branch]
```

不能使用 `current_pose + action[t,...]`，否则累计增量会被重复累加。

### 3.2 T=5、stride=2

以当前观测为 offset 0：

```text
target action offsets              [1,3,5,7,9]
target arrival observation offsets [2,4,6,8,10]
condition observation offsets      [0,2,4,6,8]
```

原因是 `action[i]` 到达 `observation[i+1]`：

```text
pair t=0: condition@obs0 -> action@raw1 -> arrival@obs2
pair t=1: condition@obs2 -> action@raw3 -> arrival@obs4
...
pair t=4: condition@obs8 -> action@raw9 -> arrival@obs10
```

第一步 condition 没有选择价值，因为当前图像已用于生成计划。因此：

- `initial_action_branch_index=0`：第一步固定 branch 0。
- `condition_loss_start_time_index=1`：有效 condition 监督和在线匹配从第二个计划时刻开始。

### 3.3 checkpoint 必须携带的 action metadata

```json
{
  "representation": "relative_plan_origin",
  "horizon": 5,
  "stride": 2,
  "yaw_delta_wrapped": true,
  "q01": [-2.508798599243164, -2.2689709377288825, -0.1400003433227539, -0.9581856727600098],
  "q99": [2.4487852191925055, 2.2756099700927734, 0.9800000190734863, 0.9581854343414307]
}
```

推理程序依据 `representation` 判断动作语义，不要只靠命令行猜测。

## 4. 当前模型结构

### 4.1 输入与维度

逻辑输入为 3 帧 RGB、指令和四维状态。实际 debug：

```text
pixel_values        [B,18,224,224]
future_pixel_values [B,5,6,224,224]  # condition 训练使用
proprio             [4]              # batch=1 时当前 collator 表现
ground_truth_actions[B,5,4]
```

每帧源图仍是 RGB，但 OpenVLA 双视觉骨干的 processor 使每帧模型输入表现为 6 通道，3 帧为 18，不是简单 RGB 拼接得到的 9。视觉 token 数为：

```text
3 * 256 visual patches + 1 proprio token = 769
```

### 4.2 COND/ACT token

Tokenizer 新增 `T*K*2=30` 个带时刻/分支编号的特殊 token：

```text
<COND_t1_k1><ACT_t1_k1> ... <COND_t1_k3><ACT_t1_k3>
...
<COND_t5_k1><ACT_t5_k1> ... <COND_t5_k3><ACT_t5_k3>
```

从 LLM 最后一层提取：

```text
cond_hidden    [B,5,3,4096]
actions_hidden [B,5,3,4096]
```

当前实现没有新建构思 PDF 中的 `MultiBranchVLA` 子类，而是集成进现有 OpenVLA forward、tokenizer、数据转换和外部 action head。这是代码实际行为。

`format_cond_token_count=15`、`format_act_token_count=15`、`format_complete_rate=1.0` 是结构断言/诊断，不是可学习格式奖励。token 序列由程序构造，不是模型自由生成字符串。

### 4.3 高斯动作头

`GaussianActionHead` 对每个 `actions_hidden[b,t,k]` 输出 4 维均值和 log-std：

```text
mean/log_std shape [B,5,3,4]
log_std range [-5,1]
initial log_std -0.5
learn_log_std true
```

使用高斯头而未直接切换 diffusion，是因为 Stage19 组相对目标需要精确策略 `log_prob`。对角高斯可直接计算；当前 diffusion head 没有可直接用于 PPO/GRPO ratio 的精确轨迹概率。Diffusion 可作为独立基线，但不能无修改替换当前高斯 GRPO。

### 4.4 实际训练参数

这是 LoRA 微调，不是 7B 全参数训练：

```text
use_lora=true, rank=32, dropout=0
freeze_vla=false
freeze_proprio_projector=false
reset_action_head=true（Stage19 SFT）
```

`freeze_vla=false` 表示 LoRA 和新增 embedding 可训练；PEFT 仍冻结基础权重。Smoke 中可训练参数：

```text
VLA/LoRA             110,828,288
Proprio projector     16,801,792
Gaussian action head  50,409,480
Total                178,039,560
```

保存时 LoRA 合并进 7B 主模型，同时单独保存 action head 和 proprio projector；一个 merged checkpoint 约 15 GB。

## 5. Stage19 损失

### 5.1 联合 condition-action 分配

归一化动作的对角高斯 NLL：

```text
NLL[b,t,k] = mean_d(0.5*((a*-mu)/sigma)^2 + log(sigma) + const)
```

未来图像只编码一次。condition 与未来视觉 patch 都去除共享方向并 L2 normalize；每个 condition 对所有 patch 求余弦相似度，平均 top-8：

```text
sim[b,t,k] = mean(topk_8(cos(center(cond),center(patches))))
```

联合分配：

```text
cost = NLL + 0.25*(1-sim), t>=1
cost = NLL, t=0；但 winner[b,0] 强制为 branch 0
winner[b,t] = argmin_k cost[b,t,k]
```

同一 winner 定义正 action 和正 condition，避免训练动作分支与在线 condition 分支脱节。

SFT 总损失启用：

```text
Gaussian best-of-K NLL
+ 0.01 * branch balance
+ 0.01 * action diversity
+ 0.20 * selected condition alignment
+ 0.05 * K-way condition contrastive classification
+ 0.01 * condition diversity
```

关键参数：

```text
assignment temperature=0.5
condition assignment weight=0.25
condition contrastive temperature=0.07
condition patch topk=8
condition threshold=0.6（训练诊断/在线初值）
```

### 5.2 GRPO-like 实现的边界

`K=3` 是候选数；`G=4` 是从已选高斯分支采样的策略组大小，两者正交。对每个已选 `(b,t,k*)` 采样 G 个动作，离线奖励：

```text
reward = -position_error
         -0.25*wrapped_yaw_error
         -0.20*normalized_action_out_of_envelope_penalty
```

组内标准化 advantage，使用精确高斯 log-prob 和 PPO-style clipped ratio。`grpo_loss` 数值可能接近 0，因为组内 advantage 被中心化，但梯度不为 0；`grpo_logprob_objective_proxy` 用于日志解释。

准确边界：

- 已实现：demonstration target 奖励、G 组采样、组相对 advantage、精确 log-prob、clipped surrogate。
- 未实现：Habitat rollout 后更新、持久 old/reference policy、reference KL、真实碰撞和最终成功奖励。
- 因而它是离线 on-policy group-relative 辅助目标，不是完整 simulator-online GRPO，也不是 DeepSeek-R1 语言推理训练的直接复刻。

## 6. 在线 condition plan

代码：`IndoorUAV-Agent-main/online_eval/vla_eval/openvla_model_runner.py`。

1. 用当前 3 帧、指令和状态生成 `[5,3,4]` actions、`[5,3,4096]` conditions，保存 `plan_origin`。
2. `plan_step=0` 不匹配 condition，执行 branch 0。
3. 执行后采集新图像。
4. `plan_step>=1` 时用相同 centered top-k patch matching 比较当前图像和该时刻 K 个 condition。
5. 选最高相似度 condition 配对的 action。
6. 若最高分低于阈值（默认 0.6），丢弃旧计划并重新推理，执行新 plan step0/branch0。
7. 5 步 horizon 用尽也重新规划。

日志验收：

```text
action_shape=[5,3,4]
relative_actions=True
plan_step=0..4
condition_similarity=None at step0
condition_similarity=<float> later
replan_reason=new_plan|horizon_exhausted|condition_below_threshold:...
```

阈值 0.6 尚未校准。若所有分数都高于阈值，重规划失效；若都低于，则每步重规划。正式实验必须报告相似度分布、pass rate 和 replan rate。

## 7. 阶段进度

| 阶段 | 工作 | 结论 |
|---|---|---|
| 1 | 设备/形状/梯度诊断 | 修复 CPU/CUDA mask 冲突 |
| 2 | 严格 3 帧历史 | 完成多帧输入 |
| 3 | T=5 action chunk | 输出 `[B,5,4]` |
| 4-5 | 多分支/多样性 | 初期为 `[B,3,5,4]`，语义不符合最终设想 |
| 6-7 | 30k baseline、部署和绝对坐标修复 | 建立完整链路 |
| 8-9 | 离线 reward 和旧 GRPO | 旧实现把 K 当轨迹分支，已废弃 |
| 10-12 | COND/ACT token、未来图像、在线计划 | 修正输出为 `[B,5,3,4]`，完成 Stage12 30k |
| 13 | relative plan-origin、stride2、yaw wrap | 修复绝对坐标发散 |
| 14-15 | per-time best-of-K、patch matching | 逐时刻 K 分支配对 |
| 16-18 | 单分支/L1/MSE/head-only/overfit/audit 对照 | 管线可拟合，但泛化弱于 zero baseline |
| 19 | 高斯结构策略、联合分配、G/K 解耦 | smoke 已通过；30k SFT 正在运行，无最终仿真结果 |

Stage18 audit：

```text
zero action MAE       0.32289
Stage18E L1 MAE       0.38224, direction cosine 0.32773, scale ratio 0.13744
Stage18F MSE MAE      0.39756, direction cosine 0.56789, scale ratio 0.44053
```

因此继续给单分支 L1/MSE 盲目加步数没有依据。Stage19 用于同时修复逐时刻 K 候选、condition-action 配对和随机策略概率，但仍必须由 audit/仿真决定是否有效。

## 8. 当前训练状态

2026-08-01 启动：

```text
tmux: stage19_sft_30k
WandB run ID: 989qy6ns
log: /VLM/liangxinyue_25/openvla-oft/runs/uav/stage19_structured_sft_30k.log
```

快照时约 `14,210/30,000`（47%）。已有：

```text
/VLM/liangxinyue_25/openvla-oft/runs/uav/stage19_structured_sft_30k--10000_chkpt
size about 15 GB
```

它包含 4 个 merged shard、action head、proprio projector、processor/tokenizer、config 和 statistics。

监控：

```bash
tmux attach -t stage19_sft_30k
tail -f /VLM/liangxinyue_25/openvla-oft/runs/uav/stage19_structured_sft_30k.log
tr '\r' '\n' < /VLM/liangxinyue_25/openvla-oft/runs/uav/stage19_structured_sft_30k.log | tail -n 20
```

当前训练运行时不要再次执行 SFT。停止自己的会话：

```bash
tmux kill-session -t stage19_sft_30k
```

## 9. 精确复现命令

### 9.1 先固化源码

交接人提交时排除论文：

```bash
cd /VLM/liangxinyue_25/openvla-oft
git status --short
git add vla-scripts/finetune.py \
        tests/test_stage19_structured_policy.py \
        vla-scripts/uav_eval/run_stage19_structured_policy.sh \
        docs/INDOORUAV_STAGE19_REPRODUCTION_HANDOFF.md
git commit -m "stage19: add structured Gaussian condition-action policy"
git push origin main
git rev-parse HEAD
```

复现者固定新的 `<STAGE19_COMMIT>`，不要写“最新 main”：

```bash
git clone https://github.com/liangxyyy/indooruav.git openvla-oft
cd openvla-oft
git checkout <STAGE19_COMMIT>
git status --short
```

`IndoorUAV-Agent-main` 当前无 Git 元数据。至少归档并校验：

```text
openvla_eval_tmux.sh
online_eval/vla_eval/openvla_model_runner.py
online_eval/vla_eval/run_openvla_model_runner.sh
online_eval/vla_eval/sim_runner.py
online_eval/vla_eval/vla_controller.py
online_eval/vla_eval/test_vla.json
eval_metric/vla_metric.py
```

### 9.2 单元测试

```bash
cd /VLM/liangxinyue_25/openvla-oft
conda activate openvla-oft
PYTHONPATH=. python -m unittest discover -s tests -p 'test_stage*.py' -v
```

当前 Stage13-19 共 28 个测试通过。若直接导入专项测试失败，使用 `discover` 并确认 Stage19 测试已提交。

### 9.3 GPU smoke

```bash
cd /VLM/liangxinyue_25/openvla-oft
conda activate openvla-oft
bash vla-scripts/uav_eval/run_stage19_structured_policy.sh smoke
bash vla-scripts/uav_eval/run_stage19_structured_policy.sh grpo-smoke
```

SFT smoke 必须出现正确形状、`format_complete_rate=1`，且 VLA/action head/proprio 梯度均有限且非 None。GRPO smoke 还应有 `grpo_group_size=4`、advantage std 约 1、有限 log-prob/proxy。

### 9.4 后台正式 SFT

```bash
cd /VLM/liangxinyue_25/openvla-oft
tmux new-session -d -s stage19_sft_30k \
  "cd /VLM/liangxinyue_25/openvla-oft && \
   bash vla-scripts/uav_eval/run_stage19_structured_policy.sh sft"
```

脚本参数：

```text
init Stage12 30k
reset action head true
max steps 30000, save every 10000
shuffle 1000, image aug true
batch 1, grad accumulation 8
lr 5e-5, warmup 200, max grad norm 10, seed 17
```

软件/seed 一致也不保证 TensorFlow、CUDA 和多线程 bitwise 一致；目标是统计结果可重复。

### 9.5 GRPO-like（仅在 SFT 通过门槛后）

```bash
cd /VLM/liangxinyue_25/openvla-oft
tmux new-session -d -s stage19_grpo_5k \
  "cd /VLM/liangxinyue_25/openvla-oft && \
   bash vla-scripts/uav_eval/run_stage19_structured_policy.sh grpo"
```

从 Stage19 SFT 30k 初始化，不重置 head，lr `1e-5`，5k 步，GRPO 权重 `0.02`。

## 10. 训练后验证顺序

### 10.1 离线 audit

分别审计 10k、20k、30k：

```bash
cd /VLM/liangxinyue_25/openvla-oft
CUDA_VISIBLE_DEVICES=2 ROBOT_PLATFORM=UAV \
python vla-scripts/uav_eval/audit_stage18_checkpoint.py \
  --checkpoint runs/uav/stage19_structured_sft_30k--30000_chkpt \
  --max_episodes 100 \
  --num_action_branches 3 \
  --output_file runs/uav/stage19_structured_sft_30k_audit100.json
```

门槛：

- oracle best-of-K MAE 应优于 zero-action MAE。
- branch0 t1 MAE 应显著改善。
- direction cosine 应正且稳定。
- displacement ratio 不应接近 0 或严重大于 1。
- winner rate 不应长期塌缩。

若 30k 仍弱于 zero baseline，不进入 GRPO。

### 10.2 固定 100 episode 仿真

```bash
cd /VLM/liangxinyue_25/IndoorUAV-Agent-main
SESSION_NAME=stage19_sft_eval100 \
CHECKPOINT=/VLM/liangxinyue_25/openvla-oft/runs/uav/stage19_structured_sft_30k--30000_chkpt \
USE_CONDITION_PLAN=true \
USE_COND_ACTION_TOKENS=true \
NUM_ACTION_BRANCHES=3 \
ACTION_BRANCH_INDEX=0 \
CONDITION_THRESHOLD=0.6 \
EVAL_START_INDEX=0 \
MAX_EVAL_EPISODES=100 \
LOG_DIR=/VLM/liangxinyue_25/IndoorUAV-Agent-main/shared_folder/evals/stage19_sft_eval100/logs \
TRAJECTORY_OUTPUT=/VLM/liangxinyue_25/IndoorUAV-Agent-main/shared_folder/evals/stage19_sft_eval100/trajectories \
bash openvla_eval_tmux.sh
```

监控 model/sim/controller：

```bash
tail -f shared_folder/evals/stage19_sft_eval100/logs/openvla_model_runner.log
tail -f shared_folder/evals/stage19_sft_eval100/logs/sim_runner.log
tail -f shared_folder/evals/stage19_sft_eval100/logs/vla_controller.log
find shared_folder/evals/stage19_sft_eval100/trajectories -maxdepth 1 -name '*.json' | wc -l
```

### 10.3 论文指标

```bash
cd /VLM/liangxinyue_25/IndoorUAV-Agent-main
conda run -n habitat python eval_metric/vla_metric.py \
  --trajectories_dir shared_folder/evals/stage19_sft_eval100/trajectories \
  --indoor_uav_base /VLM/datasets/Indoor_UAV \
  --episode_keys_file online_eval/vla_eval/test_vla.json \
  --start_index 0 \
  --max_episodes 100 \
  --output_file shared_folder/evals/stage19_sft_eval100/evaluation_results.json
```

成功条件与论文一致：最终位置误差 `<0.5m` 且 wrapped yaw 误差 `<pi/4`。evaluator 分别计算 position/yaw nDTW，再按 `position_path_length/2.2` 与累计旋转长度加权。它还实现停止启发式：相邻预测点距离 `<0.15m` 且 yaw 变化 `<pi/12` 时截断。该实现细节必须在所有对照中保持一致。

脚本输出为 0-1 比例，论文表格通常为百分数，比较时乘 100。

## 11. 已有对照

| 设置 | episodes | SR | nDTW | pos nDTW | yaw nDTW |
|---|---:|---:|---:|---:|---:|
| Stage12 全量 | 9352 | 0.01315 | 0.00536 | 0.00445 | 0.00113 |
| Stage12 前100 | 100 | 0.01000 | 0.00632 | 0.00593 | 0.00066 |
| Stage15b 前100 | 100 | 0 | 0.00011 | 0.00002 | 0.00009 |
| Stage16a 前100 | 100 | 0 | 0.00023 | 0.00005 | 0.00020 |
| Stage17c 前100 | 100 | 0 | 0.03920 | 约0 | 0.05149 |

Stage12 全量旧目录中 6 个 episode 没执行动作，处理 9352/9358。修订 evaluator 会把 zero-action episode保留在分母；比较时必须确认 evaluator 版本相同。

论文参考：fine-tuned OpenVLA SR 7.81%、nDTW 2.42%；fine-tuned pi0 SR 27.16%、nDTW 9.44%。本地 Stage12 SR 约 1.315%，明显低于论文基线，当前仍是方法和实现验证阶段。

## 12. 已知问题

1. **Stage19 未提交**：当前训练使用未提交源码，必须立即固化 commit。
2. **重复保存**：保存条件未检查梯度累积边界。当前 `grad_accumulation_steps=8` 会在同一 10k step 重复 merge/save 约 8 次，覆盖同一目录并浪费 I/O。应在当前 run 完成后改为：

```python
if gradient_step_boundary and gradient_step_idx > 0 and log_step % cfg.save_freq == 0:
```

当前进程已加载旧代码，不建议为此中断 30k；修复需记录为下一 commit。

3. **仿真目录无 Git**：只提交 OpenVLA 不足以复现。必须建仓或提供带 SHA256 的不可变归档。
4. **硬编码路径/账号/GPU**：迁移时必须修改，私人凭证不可入库。
5. **加载会修改 checkpoint config**：工具会生成 `config.json.back.<timestamp>` 并更新 Auto 映射；备份不是训练产物。
6. **threshold 未校准**：必须做阈值扫描和 replan 统计。
7. **安全奖励不是真实碰撞**：当前仅惩罚归一化动作超出 `[-1,1]`；未来在线 RL 才加入碰撞、非法高度、距离变化、成功和轨迹效率。
8. **Stage19 暂无最终指标**：训练 loss、非零梯度和 smoke 只能证明管线运行，不能证明导航性能。

## 13. 后续门槛

按顺序执行，未通过就不要叠加模块：

1. 提交 Stage19 和文档；版本化仿真代码；导出环境。
2. 完成 30k SFT，保存 10k/20k/30k。
3. 同一固定 100 episode 做 checkpoint audit，与 Stage12 和 zero baseline 比较。
4. 仿真先做固定 branch/replan-every-step 对照，再开 condition plan。
5. 扫描 condition threshold（如 0.5/0.6/0.7/0.8）。
6. 只有 SFT 明显改善才运行 GRPO 5k。
7. 固定配置后跑全部 9358 episode，报告 SR/nDTW，并按 easy/medium/hard 分组。
8. 若离线 GRPO 有效，再实现 Habitat rollout、old/reference policy、KL 和真实安全/成功 reward；Diffusion 作为独立基线。

## 14. 关键源码

```text
openvla-oft/vla-scripts/finetune.py
  compute_best_of_k_gaussian_action_loss
  compute_condition_similarity_tensors
  compute_gaussian_group_relative_policy_loss
  FinetuneConfig / run_forward_pass / save_training_checkpoint

openvla-oft/prismatic/models/action_heads.py
  GaussianActionHead

openvla-oft/prismatic/vla/condition_matching.py
  centered top-k patch matching / contrastive loss

openvla-oft/tests/test_stage19_structured_policy.py
openvla-oft/vla-scripts/uav_eval/run_stage19_structured_policy.sh
openvla-oft/vla-scripts/uav_eval/audit_stage18_checkpoint.py

IndoorUAV-Agent-main/openvla_eval_tmux.sh
IndoorUAV-Agent-main/online_eval/vla_eval/openvla_model_runner.py
IndoorUAV-Agent-main/online_eval/vla_eval/run_openvla_model_runner.sh
IndoorUAV-Agent-main/online_eval/vla_eval/sim_runner.py
IndoorUAV-Agent-main/online_eval/vla_eval/vla_controller.py
IndoorUAV-Agent-main/eval_metric/vla_metric.py
```

## 15. 实验必须附带的信息

```text
两个代码库的 commit SHA（或仿真归档 SHA256）
conda environment files
GPU/driver/CUDA
训练命令、完整日志、WandB run ID
初始化 checkpoint 和 SHA256
dataset_statistics.json
seed、shuffle buffer、episode subset
最终 config/action head/proprio
固定100和全量 evaluation_results.json
condition threshold、replan rate、branch selection rate
```

checkpoint 校验：

```bash
find <CHECKPOINT_DIR> -maxdepth 1 -type f -print0 \
  | sort -z | xargs -0 sha256sum > <CHECKPOINT_DIR>.sha256
```

## 16. 最终交接检查表

- [ ] Stage19 三个源码文件和本文档已提交并推送。
- [ ] 论文 PDF 未进入 Git。
- [ ] 新 commit SHA 已记录。
- [ ] `IndoorUAV-Agent-main` 已版本化或归档并生成 SHA256。
- [ ] 两个 conda 环境已导出。
- [ ] 初始化和最终 checkpoint 已校验。
- [ ] 28 个单元测试通过。
- [ ] SFT smoke 和 GRPO smoke 通过。
- [ ] 30k SFT 完成，10k/20k/30k audit 已保存。
- [ ] 固定 100 使用同一 test keys 和 evaluator。
- [ ] threshold 扫描和重规划统计完成。
- [ ] 仅在 SFT 过门槛后执行 GRPO。
- [ ] 最终全量 SR/nDTW 与论文口径一致。

本文应随代码更新。Stage19 commit、30k 终点、audit、固定 100 和全量评估完成后，必须把“正在运行/尚未验证”更新成实际结果。
