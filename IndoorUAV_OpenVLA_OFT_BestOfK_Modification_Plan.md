# IndoorUAV × OpenVLA-OFT Best-of-K 修改计划

> 生成日期：2026-09-15；最近更新：2026-09-16
> 对照规范：`IndoorUAV_OpenVLA_OFT_BestOfK_Implementation_Spec.md`
> 审计代码：`dc835c35efdeb9e7386ead750399c6858a95caf8`
> 训练数据：`/VLM/datasets/indoorUAV_rlds_data/rlds_data_all`，TFDS 名称 `indoor_uav`
> 训练方式：**只使用 SFT，不再使用 GRPO**

## 实现状态（2026-09-16）

Phase 1～4 的代码主链已经实现，包括 RLDS 时序窗口、首帧/尾部 mask、IndoorUAV
机体系单步动作、独立 condition/vision matching projector、image role embedding、
masked deterministic Best-of-K SFT、checkpoint contract 和 Habitat 在线 condition plan。

真实数据验收结果：

```text
TFDS episodes                 25,567
raw transitions              474,340
raw action == next state     max error 0.0（抽查 20 episodes）
body -> world round trip     max error 4.77e-7（抽查 20 episodes）
Stage20 statistics cache     ~/.cache/orca/dataset_statistics_ca3405cf...777763.json
pytest                       39 passed
Python/Shell/diff checks     passed
GPU2 7B one-step smoke       passed
```

全量新动作统计也验证了 Habitat 坐标语义：`forward mean=+0.103885`、
`forward q99=0.490002`，而 `right mean=-3.42e-5`。2026-09-16 已在 GPU 2
（RTX 4090 48GB）完成真实 7B 一步训练与 checkpoint 合并：

```bash
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=2 \
  bash vla-scripts/uav_eval/run_stage20_bodydelta_sft.sh smoke
```

smoke 的输入/输出形状分别为 `proprio [1,5]`、`actions [1,5,3,4]`、
`conditions [1,4,3,512]`、`future patches [1,4,256,512]`；VLA、action head、
condition adapter、proprio projector 梯度均非零。生成的
`runs/uav/stage20_bodydelta_sft_smoke--1_chkpt` 已通过 config/stats/contract/hash 和
`proprio fc1=[4096,5]` 校验。

正式 `train` 前仍建议执行 K=1 小样本过拟合门槛。threshold 标定、消融实验和 Habitat
固定集评测属于后续实验门槛，不能由静态/CPU 测试替代。

### 2026-09-16 已关闭的两个 P0 数值问题

1. body action 不再使用全局 q01/q99 范围。每轴使用
   `s=max(abs(min),abs(max))` 构造 `[-s,+s]`，metadata 名称为
   `per_axis_symmetric_minmax_v1`。因此零动作严格映射到 0，right 轴约 `±0.56` 的稀有
   横移也保留幅度；训练归一化、reward 反归一化、模型输出反归一化和在线 runner
   都优先读取 `normalization_low/high`。
2. 模型 proprio 已固定为 `[x,y,z,sin(yaw),cos(yaw)]`，metadata 名称为
   `xyz_sin_yaw_cos_yaw_v1`。原始 RLDS/Habitat 位姿仍保持 `[x,y,z,yaw]`；所有 body
   action 几何计算结束后才转 5D，避免把 sin/cos 错当 yaw。Stage20 启动脚本强制
   `--reset_proprio_projector True`，不再复用 Stage12 的 4D projector。

## 1. 结论先行

当前 Stage19 并不是实现规范的小偏差版本，而是在输入语义、动作坐标系、时间间隔、视觉匹配空间、mask 和在线执行方式上同时存在结构性差异。继续沿用 Stage19 checkpoint 调 loss 权重或 threshold，不能修复这些问题。

新实现必须固定以下契约：

```text
主输入      = [ref_image, I_(t-1), I_t] + state_t + instruction
起始帧输入  = [ref_image, I_0, I_0]，image_valid_mask=[1,0,1]
T           = 5
K           = 3
stride      = 1
condition   = 执行动作之前应看到的图像 I_(t+j)
action      = p_(t+j) -> p_(t+j+1) 的单步机体系局部位姿增量
slot 0      = 只监督/执行 branch 0，不计算 condition、balance、diversity loss
训练        = SFT：确定性动作回归 + condition 图像对齐 + Best-of-K 联合分配
在线执行    = 每一步都以真实 current_pose/current_yaw 合成 target_pose
```

`rlds_data_all` 继续作为唯一训练数据源，不需要切换到 `train_history_part*`。统一数据集本身已有完整 episode、当前图、参考图、绝对状态和绝对下一位姿；previous image、future image、局部动作和 mask 应在同一 episode 内动态构造。

旧 Stage19 的以下产物不得继续用于新训练：

- `action_head`：学习的是 `relative_plan_origin` 累计位移，而且是 Gaussian 头；
- action normalization statistics：统计对象和新动作定义不同；
- condition 表征和 threshold：使用 4096 维 centered matching，而不是新的 512 维共享匹配空间；
- Stage19 在线执行缓存：依赖 `plan_origin + cumulative_action`。

可以复用 OpenVLA/OFT 主干；proprio projector、action head、matching projectors、role embedding 必须重新初始化。是否复用 Stage12/Stage19 的 VLA LoRA，应通过短程 A/B 实验决定，不能默认旧 LoRA 一定有利。

## 2. 已确认的 RLDS 数据语义

### 2.1 数据集事实

`rlds_data_all/indoor_uav/1.0.0` 是一个 TFDS `train` split：

```text
episodes / trajectories = 25,567
raw transitions         = 474,340
shards                  = 2,304
```

实际 `features.json` 中每个 step 使用：

```text
observation.image       uint8 [720,1280,3]
observation.ref_image   uint8 [720,1280,3]
observation.state       float32 [4] = [x,y,z,yaw(rad)]
action                  float32 [4]
language_instruction    string
is_first/is_last/is_terminal
```

统一数据集实际 schema 不依赖 `hist_image_0/1/2`；历史帧必须从 episode 的 `observation.image` 动态生成。这正是继续使用 `rlds_data_all` 而不是 history part 的原因。

### 2.2 一步的真实含义

RLDS builder 使用：

```python
state  = postures[frame_num - 1]
action = postures[frame_num]
image  = screenshots[frame_num]
```

在训练管线中应把该 step 解释为：

```text
observation.state = p_t
observation.image = I_t
action            = p_(t+1)       # 世界系绝对下一位姿
ref_image         = 子轨迹起始图像 I_0
```

因此原始 action 不能直接做局部动作监督，也不能仅减去整个 plan 的起点。

### 2.3 新样本的时间对齐

对 episode 中的训练起点 `t`：

| slot `j` | condition label | action label | 有效条件 |
|---:|---|---|---|
| 0 | `I_t`，但不计算 condition loss | `p_t -> p_(t+1)` | `t < episode_len` |
| 1 | `I_(t+1)` | `p_(t+1) -> p_(t+2)` | `t+1 < episode_len` |
| 2 | `I_(t+2)` | `p_(t+2) -> p_(t+3)` | `t+2 < episode_len` |
| 3 | `I_(t+3)` | `p_(t+3) -> p_(t+4)` | `t+3 < episode_len` |
| 4 | `I_(t+4)` | `p_(t+4) -> p_(t+5)` | `t+4 < episode_len` |

每个 RLDS step 都有自己的绝对下一位姿 action，所以 episode 最后一个 RLDS step 的 slot 0 仍可作为有效动作样本；超出 episode 的后续 slot 才是 padding。

### 2.4 目标 batch 契约

processor 前的逻辑数据：

```text
input_images             [B,3,H,W,3]       # ref, previous, current
image_valid_mask         [B,3]
proprio                  [B,5]           # x,y,z,sin(yaw),cos(yaw)
future_condition_images  [B,T,H,W,3]
ground_truth_actions     [B,T,4]
plan_valid_mask          [B,T]
instruction              B strings
```

processor/collator 后，以当前 fused SigLIP+DINO 处理器为准：

```text
pixel_values             [B,18,224,224]    # 每图 6 channel × 3 图
future_pixel_values      [B,T,6,224,224]
image_valid_mask         [B,3]
proprio                  [B,5]
actions                  [B,T,4]
plan_valid_mask          [B,T]
input_ids/attention_mask [B,L]
```

18 channel 只是 processor 的容器形式。当前视觉 backbone 已经把它按三组 6 channel 分开，并将每张图分别送入共享 SigLIP/DINO；这一部分应保留。

未来图像只允许进入训练标签编码支路，绝不能进入生成 plan 的主 Transformer。

## 3. 绝对位姿到单步机体系动作

### 3.1 转换位置

转换必须发生在：

```text
读取原始 float32 绝对 state/action
-> 构造时间窗口和有效 mask
-> 绝对世界位姿转单步机体系 delta
-> proprio yaw 从标量转为 sin/cos（此步不能提前）
-> 用新动作统计量归一化到模型训练范围
```

不能先用绝对坐标的 q01/q99 做归一化，再相减或旋转。

### 3.2 数学定义

令 condition 时刻位姿为 `p_j=[x_j,y_j,z_j,yaw_j]`，下一位姿为 `p_next`。先计算：

```text
dx = x_(j+1) - x_j
dy = y_(j+1) - y_j
dz = z_(j+1) - z_j
dyaw = wrap_to_pi(yaw_(j+1) - yaw_j)
```

按规范候选约定：

```text
[delta_forward, delta_right]^T = R_indoor_uav(-yaw_j) [dx,dy]^T

delta_forward =  sin(yaw_j)*dx - cos(yaw_j)*dy
delta_right   =  cos(yaw_j)*dx + sin(yaw_j)*dy
delta_up      = dz
```

最终监督为 `[delta_forward,delta_right,delta_up,delta_yaw]`。在线逆变换：

```text
dx_world =  sin(yaw_current)*df + cos(yaw_current)*dr
dy_world = -cos(yaw_current)*df + sin(yaw_current)*dr
target = [x+dx_world, y+dy_world, z+du, wrap(yaw+dyaw)]
```

这里使用的是已由原始 IndoorUAV 轨迹和 Habitat 渲染转换共同确认的项目坐标约定：
`yaw=0` 时机体前方对应世界 `-y`，机体右方对应世界 `+x`。例如真实轨迹在
`yaw=11.700093°` 时主要沿世界 `[+x,-y]` 移动，转换后应主要落在正 forward 轴。

IndoorUAV 中 yaw 增大表示右转、减小表示左转。训练标签必须使用圆周上的最短有符号角差：

```text
dyaw = ((yaw_next - yaw_current + pi) mod (2*pi)) - pi
350° -> 10° = +20°（右转），不是 -340°
10° -> 350° = -20°（左转），不是 +340°
```

该符号约定由论文数据中的纯转向片段交叉验证：`Turn right to face the staircase`
对应 yaw 从约 `99°` 增至 `182.7°`；`Turn left to the washer and dryer` 对应 yaw
从 `2.7°` 递减并跨过 `0°` 到 `275.4°`。

### 3.3 坐标约定是阻断训练的验收项

IndoorUAV 文件中的前两个坐标会映射到 Habitat 的水平 `x/z`，第三维映射到高度；`test_sim.py` 又给 sensor 设置了 `yaw=pi` 并做图像翻转。因此上面“forward/right”的名字和符号不能只凭矩阵猜测。

正式训练前必须同时通过：

1. yaw 为 `0, ±pi/2, pi` 的合成 round-trip 单测；
2. 在 Habitat 空场景把 `[+forward,+right,+up,+yaw]` 分别施加一次，确认渲染画面和世界坐标变化；
3. 从真实 expert trajectory 抽样，验证正向局部化再反向合成能恢复 raw `action`，误差 `<1e-5`；
4. 可视化 expert body delta 分布，确认主要前进方向的符号与相机朝向一致。

如果 Habitat 实测显示轴顺序或符号相反，只修改一个有版本号的坐标转换函数及 metadata，训练与在线端共同调用，禁止两处各写一套公式。

### 3.4 新 normalization statistics

统计量应基于 474,340 个原始 step 各自的一步 body delta，每个 raw transition 只计一次；不要把同一个动作因为不同滑窗重复统计，也不要因 horizon 截掉 episode 尾部。

checkpoint 中至少保存：

```json
{
  "representation": "body_delta_one_step_v1",
  "source_action": "absolute_next_world_pose",
  "axes": ["forward", "right", "up", "yaw"],
  "yaw_delta_wrapped": true,
  "stride": 1,
  "horizon": 5,
  "normalization": "bounds_q99",
  "coordinate_contract_version": 1
}
```

loader 遇到 `relative_plan_origin` 或 metadata 缺失时必须拒绝按新策略执行，不能自动猜测。

## 4. 当前实现与规范的逐项差异

| 模块 | 当前 Stage19 | 规范要求 | 影响与处理 |
|---|---|---|---|
| RLDS 根目录 | `rlds_data_all` | `rlds_data_all` | 正确，保留 |
| RLDS image mapping | 只映射 `image -> image_primary` | 同时保留 `image` 和 `ref_image` | 当前 `ref_image` 在 `restructure()` 时被丢弃 |
| 三图输入 | `[I_(t-2),I_(t-1),I_t]` | `[ref,I_(t-1),I_t]` | 主输入语义错误 |
| episode 开头 | `require_full_image_history=True`，丢弃前两步 | 重复首帧并 mask | 浪费开头数据，训练/在线首步分布不一致 |
| image mask | collator 返回并只用于 debug | 扩到 patch、置零并进入 attention mask | 当前 padding 图仍完整参与注意力 |
| image role | 无 | reference/history/current role embedding | 模型只能靠序列位置猜角色 |
| 多图视觉 backbone | 逐图共享编码，patch 沿序列拼接 | 同左 | 已正确，不重写成视频模型 |
| horizon/stride | `T=5, stride=2` | `T=5, stride=1` | condition 与环境步不一致 |
| episode 尾部 | 为满足完整 horizon 直接截掉 | padding + `plan_valid_mask` | 尾部短 horizon 样本全部损失 |
| action target | 相对 plan 起点的累计世界位移 | condition 时刻到下一时刻的机体系单步 delta | 最严重的监督语义差异 |
| action stats | 滑窗化累计位移统计 | raw step 唯一的一步 body delta 统计 | q01/q99 完全不兼容 |
| action head | 对角 Gaussian mean/log-std | SFT 确定性连续回归 | 新主线关闭 Gaussian 与 GRPO |
| condition hidden | `[B,5,3,4096]` | 同形状 | token 结构可以保留 |
| matching space | 对 4096 hidden/patch 直接中心化并 cosine | 两个独立 `4096->1024->512` MLP | 缺少可学习共享空间 |
| future visual target | detach 的 4096 patch | shared vision 后进入可训练 matching projector | 当前图像侧无法适配匹配空间 |
| patch aggregation | centered cosine + Top-8 | 512 维 cosine + Top-8 | 保留 Top-8，移除默认 centering |
| contrastive loss | 同一图像的 K-way branch 分类 | winner condition 对正确图像，其他图像为 negatives | 当前没有学图像身份匹配 |
| winner cost | Gaussian NLL + condition cost | Smooth L1 + condition cost | 改成确定性 SFT 联合分配 |
| slot 0 | 固定 winner0，但仍混入 balance/diversity | branch1/2 和 condition 完全 mask | 当前 root 污染正则统计 |
| future mask | 无 | 所有 future loss 乘 `plan_valid_mask` | 短 horizon 无法训练 |
| online 输入 | 三帧 deque，不读取 `start_image_path` | 固定 ref + previous/current | 训练与部署目标不一致 |
| online condition | 缓存 4096 condition | 缓存 512 condition | 旧 threshold 不可复用 |
| online action | `plan_origin + cumulative_action` | `current_pose ⊕ body_delta` | 执行公式完全不同 |
| checkpoint | 未保存 matching/role 模块 | 保存所有新模块及 contract | 需要新 manifest |

## 5. 当前效果差的主要原因排序

### P0：动作监督和部署语义不符合最终算法

当前 `convert_action_chunks_to_relative()` 把所有未来绝对位姿减去同一个 current origin，得到累计世界位移。它没有用每个 condition 时刻自己的 yaw 做局部化。新设想要求 action 与 condition 绑定，并从 condition 时刻的真实位置执行单步增量；这是不同的学习问题。

### P0：模型没有看到设计要求的 reference image

RLDS 有 `ref_image`，controller 也把 `start_image_path` 写入 instruction JSON，但 OXE mapping 和 OpenVLA runner 都没有把它接入 Stage19。现模型训练/推理看到的是三个时间历史帧。

### P0：视觉 condition 没有规范中的共享 512 空间

当前直接比较 LLM 4096 hidden 和原始视觉 projector 4096 patch，并通过 K-branch/patch centering 消除公共方向。该规则强依赖 `K>=2`，与“预测一个可和未来图像检索的 condition embedding”不同。

future backbone 使用固定 target encoder 可以作为稳定策略，但新 `VisionMatchingProjector` 必须正常获得梯度，不能也包进 `no_grad()`。

### P1：contrastive objective 不是图像检索 objective

当前 K 个 logits 都来自同一张未来图像，标签是 action winner branch。它训练“哪个 branch 应被选”，没有要求 condition 区分正确未来图像与其他未来图像。

### P1：mask 是假接入，且尾部被截断

`image_history_pad_mask` 到了 collator 和 debug 日志，但模型为所有视觉/proprio token 创建全 1 mask。trajectory chunking 也只产生完整 horizon 起点，所以不存在 `plan_valid_mask`。

### P1：当前结果说明不只是 threshold 问题

Stage19 30k 固定 100 episode：

```text
condition plan，threshold=0.6：SR=0.01，nDTW=0.01339
近似强制每步重规划：          SR=0.00，nDTW=0.03711
```

因此 root action 与数据表征也没有形成可靠闭环。

### 方法本身的数据上限

RLDS 每个 `(state,instruction)` 只有一条 expert continuation，并没有 K 条反事实未来。Best-of-K 可以把不同样本分配给不同 latent branches，但无法从单条 demonstration 证明“若看见另一场景就执行另一动作”。因此 balance/diversity 权重必须小；先证明 K=1 SFT 和视觉检索成立，再启用 K=3。真正反事实监督需要同状态多 rollout、扰动恢复轨迹或 Habitat 数据采集，不属于本轮第一版范围。

## 6. 建议的代码结构

### 6.1 配置不要继续复用模糊 boolean

在 `FinetuneConfig` 增加明确字段：

```text
input_image_layout="reference_previous_current"
temporal_history_window=2
num_images_in_input=3
action_target_representation="body_delta_one_step"
plan_horizon=5
future_action_stride=1
pad_future_horizon=true
match_dim=512
condition_patch_topk=8
use_gaussian_action_head=false
grpo_reward_weight=0
training_objective="sft"
```

保留 `relative_plan_origin` 只用于复现旧实验，并在新脚本中禁用。若 `training_objective=sft`，配置校验应拒绝非零 GRPO 参数。

### 6.2 新增组合辅助模块

建议新增 `prismatic/models/uav_condition_action_policy.py`，集中定义：

```text
ImageRoleEmbedding       nn.Embedding(3,4096)
ConditionProjector       LN(4096)->Linear(1024)->GELU->Linear(512)->normalize
VisionMatchingProjector  LN(4096)->Linear(1024)->GELU->Linear(512)->normalize
DeterministicActionHead  ACT hidden [B,T,K,4096] -> [B,T,K,4]
```

可以包装为一个 `IndoorUAVPolicyAuxiliaries`，避免四个模块分别遗漏保存、加载或加入 optimizer。proprio projector 暂时保持现有独立文件以降低改动面。

该组合模块必须：

- 一次性加入 optimizer/DDP；
- checkpoint 保存和恢复为一个 state dict；
- 训练与在线共用相同类；
- 在 manifest 中记录维度、T/K、Top-8、action contract；
- 做 save/load round-trip 测试。

## 7. 文件级修改计划

### 7.1 `prismatic/vla/datasets/rlds/oxe/configs.py`

修改 `indoor_uav`：

```python
"image_obs_keys": {
    "primary": "image",
    "secondary": "ref_image",
    "wrist": None,
}
```

同时调整 camera view materialization，使 IndoorUAV 加载 `secondary`，否则配置会在 `materialize.py` 中被 `load_camera_views` 过滤。

### 7.2 `prismatic/vla/datasets/rlds/dataset.py`

- 保证 `image_secondary` 在 restructure 后保留为 reference；
- 保留稳定 episode/trajectory id，供边界测试、验证集划分和日志使用；
- 新 action representation 使用新的 statistics cache hash；hash 中包含转换源码、坐标版本、stride 和 yaw wrap；
- metadata representation 改为 `body_delta_one_step_v1`；
- action 保存显式、逐轴对称的 `normalization_low/high`，不再用 q99 裁剪稀有横移；
- proprio 保存 `xyz_sin_yaw_cos_yaw_v1` 及其显式归一化范围；
- normalization 只能发生在 body delta 构造之后；
- 保留 legacy 分支，避免破坏其他 OXE 数据集。

`rlds_data_all` 只有 train split。不要假设存在 TFDS `val` split。应从 train episodes 按稳定的 scene/trajectory key 建验证集，不能按 step 随机切分，也不能让同一原始 trajectory 的不同 instruction 同时落入 train/val。保存一份版本化 split manifest，所有实验复用。

### 7.3 `prismatic/vla/datasets/rlds/traj_transforms.py`

新增并单测：

```text
wrap_to_pi()
world_pose_pair_to_body_delta(current_pose, next_pose)
body_delta_to_world_pose(current_pose, body_delta)
```

重写新模式的 temporal chunk：

- primary history 只取 `[I_(t-1),I_t]`；
- 负历史 index clamp 到 0，并产生 history valid mask；
- future condition index 使用 `[t,t+1,...,t+4]`；
- action 取各 raw step 的绝对下一位姿，并分别以该 step 的 state 转为 body delta；
- future index 超界时 clamp 到最后有效元素，但对应 `plan_valid_mask=False`；
- `effective_traj_len=episode_len`，不再为了完整 T 丢掉尾部起点；
- 输出 action 直接是 `[T,4]` target，不再夹带无用的历史 action 前缀。

不要直接改变所有机器人数据的 `chunk_act_obs()` 语义。通过 representation 参数或 IndoorUAV 专用 wrapper 隔离新逻辑。

### 7.4 `prismatic/vla/datasets/datasets.py`：`RLDSBatchTransform`

新输入布局显式构造：

```text
ref      = observation.image_secondary[current]
previous = observation.image_primary[0]
current  = observation.image_primary[1]
images   = [ref, previous, current]
mask     = [True, history_pad_mask[0], True]
```

episode 起点自然得到 `[1,0,1]`。不要再用 `require_full_image_history=True` 返回 `None`。

返回键统一为：

```text
pixel_values
image_valid_mask
proprio
actions
plan_valid_mask
future_pixel_values
input_ids / labels / dataset_name / episode_id
```

slot 0 future image 可留在数据契约中便于时间对齐审计；训练时只编码 slots 1..4，节约约 20% future visual cost。

图像 augmentation 必须检查语义：同一物理帧作为 current 和 future label 时，应明确是否共享随机 augmentation。第一版建议主输入与 label 使用独立但温和的 photometric augmentation，不做会改变导航几何的强 crop；另做无 augmentation 对照。

### 7.5 `prismatic/util/data_utils.py`：collator

- stack `image_valid_mask -> bool [B,3]`；
- stack `plan_valid_mask -> bool [B,T]`；
- stack `future_pixel_values` 并断言 T；
- `proprio` 不要无条件 `squeeze`，保证 batch=1 仍为 `[1,5]`；
- 对每个 batch 做 shape/dtype 断言；
- 保留 episode id 供 debug，不送 GPU。

### 7.6 `prismatic/extern/hf/modeling_prismatic.py`

扩展 `forward()`：

```text
image_valid_mask=None
image_role_embedding=None
```

主视觉路径：

```text
pixel_values [B,18,224,224]
-> shared dual backbone [B,768,D_dual]
-> original projector   [B,768,4096]
-> reshape              [B,3,256,4096]
-> add role embeddings
-> zero invalid patches
-> flatten              [B,768,4096]
```

将 `[B,3]` repeat-interleave 到 `[B,768]`。proprio token 追加后，为它追加一个有效 mask。`_build_multimodal_attention()` 不再自行创建全 1 prefix mask，而是接收真实 prefix mask并插入文本 attention mask。

必须兼容非 IndoorUAV/单图调用：`image_valid_mask=None` 时保持旧行为；启用 role 时严格断言图像数为 3、每图 patch 数一致。

### 7.7 `prismatic/vla/condition_matching.py`

替换默认算法为 512 维直接 normalized Top-8：

```text
conditions [B,T-1,K,512]
patches    [B,T-1,256,512]
einsum     -> [B,T-1,K,256]
topk(8)    -> scores [B,T-1,K] + indices [B,T-1,K,8]
```

旧 centering 函数可保留为 legacy 实验函数，但新 SFT 配置禁止默认启用。

新增跨图像 InfoNCE logit：每个 winner condition 与所有有效 future image 的 patch 集分别做 Top-8，得到 `[N_query,N_image]`。只对 `plan_valid_mask=True` 的 `(b,j)` 建正负样本。

当前常用 batch size 为 1，梯度累积不会自动产生跨 microbatch negatives。第一版至少使用同一 batch 的有效 time negatives；正式版建议增加固定长度、detach 的 image embedding queue 或多卡 all-gather negatives。相邻时间图像可能高度相似，日志中要区分跨 episode negatives 与同 episode time negatives。

### 7.8 `vla-scripts/finetune.py`

主训练路径只保留 SFT：

1. VLA forward 得到 `COND/ACT hidden [B,T,K,4096]`；
2. deterministic action head 得到 `[B,T,K,4]`；
3. condition projector 得到 `[B,T,K,512]`；
4. future image 经 shared vision encoder/original projector；
5. vision matching projector 得到 `[B,T-1,256,512]`；
6. Top-8 得到 `[B,T-1,K]`；
7. Smooth L1 + similarity 做联合 winner；
8. 应用 root/future/mask loss；
9. 反向传播 SFT 总 loss。

移除新主线中的：

- `GaussianActionHead`；
- `diagonal_gaussian_nll`；
- Gaussian sampling/log-prob；
- `compute_gaussian_group_relative_policy_loss`；
- `grpo_reward_weight` 和 GRPO phase。

这些函数可标注 legacy，而不是立即删除，避免破坏旧 checkpoint audit；新脚本完全不调用它们。

修复 checkpoint 保存条件，只在 `gradient_step_boundary` 保存，避免 grad accumulation 的同一步重复 merge。

### 7.9 `experiments/robot/openvla_utils.py`

- 加载 `IndoorUAVPolicyAuxiliaries`；
- 严格验证 policy contract、T/K、match dim、action representation；
- 加载新 body-delta statistics；
- 暴露与训练相同的单图 vision matching encoder；
- 保证图像预处理和训练 processor 一致；
- 禁止找不到新辅助模块时静默回退到 Stage19 action head。

### 7.10 在线 runner

同时修改并保持一致：

```text
openvla-oft/vla-scripts/uav_eval/openvla_model_runner.py
IndoorUAV-Agent-main/online_eval/vla_eval/openvla_model_runner.py
```

具体修改：

- `load_instruction()` 读取并缓存 `start_image_path` 作为固定 `ref_image`；
- history deque 只管理 previous/current，不再把三个时间帧当模型三图；
- episode 首步构造 `[ref,I0,I0]` 和 `[1,0,1]`；
- 后续构造 `[ref,I_(now-1),I_now]` 和 `[1,1,1]`；
- 规划时缓存 projected conditions `[T,K,512]`，不是 4096 hidden；
- 新观测编码成 `[256,512]` patch；
- 执行 slot 0/branch 0 后，下一观测匹配 slot 1；
- action 反归一化后用真实 current pose/current yaw 做 body-to-world 合成；
- 删除 `plan_origin` 对新模式的执行依赖；
- 输出所有 K similarity、selected branch、Top-8 patch index、body action、current pose、target pose、replan reason；
- threshold 从 checkpoint calibration 文件读取，CLI override 时打印警告。

Habitat 当前通过 `agent.set_state()` 直接设置绝对 target pose，physics 关闭，不会像离散导航 action 那样自动处理碰撞。第一版为保证与旧评估可比，仍输出 target pose；同时记录 navmesh/collision/高度异常。任何 snap-to-navmesh 或碰撞裁剪应作为单独评估协议，不能悄悄改变基线。

### 7.11 启动脚本

新增 `vla-scripts/uav_eval/run_stage20_bodydelta_sft.sh`，不要覆盖 Stage19 脚本。新脚本固定：

```text
DATA_ROOT=rlds_data_all
training_objective=sft
future_action_stride=1
action_target_representation=body_delta_one_step
num_images_in_input=3
temporal_history_window=2
require_full_image_history=false
T=5 K=3
use_gaussian_action_head=false
grpo_reward_weight=0
```

## 8. SFT 损失的精确定义

### 8.1 Root action

```text
L_root = masked_smooth_l1(A[:,0,0], GT[:,0], plan_valid_mask[:,0])
```

slot 0 的 branch 1/2 不进入任何 action、condition、balance 或 diversity 项。

### 8.2 Future 联合 winner

对 `j=1..T-1`：

```text
E_act[b,j,k] = mean_d SmoothL1(A[b,j,k,d], GT[b,j,d])
S[b,j,k]     = Top8Cosine(C[b,j,k], ImagePatches[b,j])
Cost         = E_act + lambda_assignment_similarity * (1-S)
k*           = argmin_k stop_gradient(Cost)
```

同一个 `k*` gather action 和 condition。无效 slot 的 cost 不参与 argmin、统计或 loss。

### 8.3 Future action

```text
L_future_action = masked_mean(E_act[b,j,k*], plan_valid_mask[b,j])
```

### 8.4 Condition

最小正样本项：

```text
L_positive = masked_mean(1-S[b,j,k*])
```

主项使用 image-level InfoNCE：

```text
query    = winner condition
positive = 同一 (b,j) 的未来图像 patch set
negative = 其他有效 batch/time 图像，必要时加 queue
logit    = Top8Cosine(query, candidate_image_patches) / temperature
```

当前 Stage19 的 K-way branch CE 不再作为主 condition contrastive loss。

### 8.5 Balance/diversity

balance 只统计 future valid slots 的 winner，目标为 K 均匀，但权重保持很小。diversity 只对 future valid condition branches 计算；动作 diversity 第一版默认关闭，因为单 expert target 不足以证明三个动作都应不同。

推荐初始总损失：

```text
L = 1.0 * L_root
  + 1.0 * L_future_action
  + 1.0 * L_condition_infonce
  + 0.1 * L_positive
  + 0.01 * L_balance
  + 0.001~0.01 * L_condition_diversity
```

这些权重只是起始值，必须先检查每项未加权梯度范数，避免 condition loss 淹没动作回归。assignment similarity 权重也应根据归一化 Smooth L1 与 `1-S` 的实际量级标定。

## 9. 实施顺序与门槛

### Phase 0：冻结旧基线

- 记录代码 SHA、Stage19 30k checkpoint、100 episode 结果；
- 不修改旧 Stage19 脚本和旧 runner 行为；
- 新实验使用新 run id 和新 contract version。

完成条件：旧结果可复现，新 loader 能明确拒绝把旧 action stats 当 body delta。

### Phase 1：只完成数据与坐标层

- 接入 `ref_image`；
- 构造 `[ref,prev,current]`、两个 mask、stride1；
- 构造 body delta 和新 statistics；
- 写 dataset audit，打印随机 episode 的 raw index、图像 hash、pose、action、mask。

完成条件：第 10 节数据/坐标测试全部通过；人工查看至少 20 个跨 episode、开头、结尾窗口无错位。

### Phase 2：K=1 确定性 SFT 基线

先不启用多分支和 condition 执行，只验证：

```text
[ref,prev,current]+state+instruction -> 5 个单步 body action
```

训练顺序：固定小 batch overfit -> 小规模 train/held-out validation -> 全量 SFT。离线指标使用反归一化 body delta，并报告 per-axis MAE、方向 cosine、world pose round-trip error、zero-action baseline。

完成条件：

- 小 batch 能明显过拟合；
- held-out root/future action 均优于零动作；
- 每步重规划的 100-episode Habitat 结果不低于旧 Stage19，且轨迹方向合理。

未通过时禁止进入 K=3。

### Phase 3：K=1 condition matching

- 加入两个 512 projector；
- 先训练正确未来图像检索，不做 branch assignment；
- 记录 positive/negative similarity、Recall@1/Recall@5、same-time/cross-time confusion；
- 可视化 Top-8 patch 落点。

完成条件：验证集检索显著高于随机，正负 margin 稳定，projector 梯度非零且无 collapse。

### Phase 4：K=3 Best-of-K SFT

- 启用 `COND/ACT [T,K]`；
- root 固定 branch0；
- future 使用统一 winner；
- 逐步加入 balance，再以极小权重加入 condition diversity；
- 不加入 Gaussian，不加入 GRPO。

完成条件：winner 不完全塌缩；oracle best-of-K 动作误差优于 K=1；selected condition 图像检索仍有效；branch0 root 不退化。

### Phase 5：在线 condition plan

对同一 checkpoint 依次评估：

1. 每步完整重规划，只执行 root branch0；
2. 无 threshold 的 oracle/offline matching 诊断；
3. condition plan + threshold；
4. 固定 branch/no condition 的缓存计划对照。

只有对照 1 先可靠，才能归因 condition plan。threshold 用 held-out rollout 的“继续是否正确”标签做 PR/ROC 或按任务指标扫描，禁止复用 0.6。

### Phase 6：完整评估

- 固定 100 episode 做所有 ablation；
- 再跑全量测试；
- 报告 SR、nDTW、position/yaw nDTW、碰撞/非法 pose、replan rate、模型调用数；
- 按 easy/medium/hard 分组；
- 保存逐步分支、similarity、Top-8 patch 和 body/world action 日志。

## 10. 必须新增/更新的测试

### 数据与时序

1. `ref_image` 从 RLDS 到 batch 像素一致；
2. window 不跨 episode；
3. 起点 `[ref,I0,I0]` mask 精确为 `[1,0,1]`；
4. 普通步为 `[ref,I_(t-1),I_t]` 和 `[1,1,1]`；
5. slot j condition/action 严格满足时间表；
6. episode 尾部 plan mask 正确，padding 内容改变不影响 loss；
7. 统一数据集所有 25,567 episode 可迭代，无需 history part。

### 坐标

8. yaw `0, ±pi/2, pi` 的轴向测试；
9. `pi-eps -> -pi+eps` yaw wrap；
10. world->body->world round trip；
11. 真实 RLDS `compose(state,body_delta) == raw absolute action`；
12. statistics transition 数等于 474,340，而不是滑窗重复后的数量；
13. train/online 使用同一转换函数或共享 golden vectors。

### 模型与 mask

14. 三图各自产生 256 patch，总计 768；
15. role 顺序交换会改变有效 token；
16. 中间图 mask 后的 256 patch attention 为 0；
17. 改变 masked placeholder 像素不改变输出（浮点容差内）；
18. proprio 在 batch=1 保持 `[1,5]`，且 359°/1° 编码连续；
19. 非 IndoorUAV 单图 forward 不回归。

### matching/loss

20. projector 形状 `4096->512`、输出 unit norm；
21. Top-8 同时返回 scores 和 indices；
22. joint winner 对 action/condition gather 同一 index；
23. slot0 branch1/2 梯度为 0；
24. invalid future slot 对所有 loss 和梯度为 0；
25. InfoNCE positive image index 正确，负样本不含 padding；
26. vision matching projector 梯度非零；
27. 保存/加载后 action、condition、similarity 数值一致。

### 在线

28. episode 首步和后续三图/mask 与训练一致；
29. 执行 slot0 后匹配 slot1；
30. body action 总是由真实 current yaw 转世界系；
31. replan 后 plan_step 重置且 previous/current 更新正确；
32. contract 不匹配时 fail fast；
33. offline/online 对同一图像产生的 512 patch embedding 一致。

## 11. 监控与可解释性

每个训练 interval 至少记录：

```text
root/future normalized SmoothL1
root/future unnormalized per-axis MAE
world round-trip target error
zero-action MAE 与 improvement
positive/negative condition similarity 和 margin
condition retrieval Recall@1
winner rate/entropy（只含 future valid）
每个 loss 的未加权值和加权值
VLA/action/condition projector/vision projector/role/proprio 梯度范数
plan_valid ratio、start-sample ratio、各 horizon 有效样本数
```

在线至少记录：

```text
plan_step, branch, all K similarities, threshold
Top-8 patch indices
selected_action_body
current_pose, target_pose
replan_reason
navmesh/collision/height validity
```

Top-8 indices 应映射回 `16×16` patch 网格并叠到图像上。它是调试匹配是否在门、拐角、障碍物等区域的必要工具，但不进入控制逻辑。

## 12. Checkpoint 与兼容性

新 checkpoint 至少包含：

```text
merged/base VLA or LoRA adapter
indoor_uav_policy_auxiliaries--STEP_checkpoint.pt
proprio_projector--STEP_checkpoint.pt
dataset_statistics.json
policy_contract.json
processor/tokenizer/config
train/validation split manifest
source commit SHA 与完整启动参数
```

`policy_contract.json` 应记录：

```text
schema_version
input roles/order
image mask semantics
T/K/action_dim/match_dim/topk
condition/action time alignment
action frame/axes/yaw convention
stride
source/model proprio dim 与 `xyz_sin_yaw_cos_yaw_v1`
action/proprio 显式 normalization 名称与联合 hash
training_objective=sft
threshold calibration id/value
```

恢复训练和在线推理都按 contract 验证。旧 Stage19 checkpoint 可以用于历史对照，但不得通过手工 CLI 参数伪装成新 checkpoint。

## 13. 建议的 ablation 矩阵

| ID | 输入 | action | K | condition | 执行 |
|---|---|---|---:|---|---|
| A | ref/prev/current + mask/role | body one-step | 1 | 无 | 每步重规划 |
| B | 同 A | body one-step | 1 | 512 retrieval | 每步重规划 |
| C | 同 A | body one-step | 3 | 联合 winner | 每步重规划 root |
| D | 同 A | body one-step | 3 | 联合 winner | condition cache plan |
| E | 去掉 ref | body one-step | 3 | 联合 winner | 同 D |
| F | 去掉 role 或 mask | body one-step | 3 | 联合 winner | 同 D |

所有实验只做 SFT。Gaussian、diffusion、GRPO、在线 RL 均不进入本轮主线。

## 14. 最终验收清单

- [x] 唯一数据源是 `rlds_data_all`，25,567 episode 可读取；
- [x] 输入真实为 `[ref,previous,current]`；
- [x] episode 首帧被保留，mask 真实进入 attention；
- [x] stride=1，尾部由 plan mask 保留；
- [x] action 是 condition 时刻到下一时刻的单步 body delta；
- [x] 新 stats 基于唯一 raw transitions 重新计算并版本化；
- [x] action 使用逐轴对称 min/max，稀有 right 横移不再被 q99 饱和；
- [x] 训练/在线 proprio 均为 5D 周期 yaw，旧 projector 被强制重置；
- [x] 两个独立 4096->512 projector 已接入训练、保存和加载；
- [x] future image 不泄漏到主 Transformer；
- [x] slot0 只有 branch0 参与监督；
- [x] future condition/action 使用同一 winner；
- [x] InfoNCE 使用其他图像作为 negatives；
- [x] 新主线只有 SFT，没有 GRPO/Gaussian 依赖；
- [x] 在线读取 reference image，并用 current yaw 执行 body delta；
- [ ] threshold 由验证集标定；
- [ ] K=1、matching、K=3、online plan 按阶段分别通过门槛；
- [ ] 固定 100 和全量 Habitat 指标、日志和代码版本均可复现。

## 15. 推荐的第一批实际提交

```text
1. data: preserve IndoorUAV reference image and add body-delta temporal contract
2. model: apply image roles and real visual attention masks
3. policy: add 512-d condition/vision matching projectors
4. train: implement masked deterministic Best-of-K SFT
5. infer: execute current-pose body deltas and calibrated condition plans
6. eval: add data audits, retrieval metrics, patch visualization and ablations
```

第一批实现完成后先运行 Phase 1/2，不直接启动 30k K=3。只有数据 golden test、坐标 round-trip 和 K=1 小样本过拟合全部通过，才值得投入完整训练。

## 16. 2026-09-16 实施记录

本次按“周期状态 + 对称动作范围”方案修改，保留全局 UAV `PROPRIO_DIM=4` 作为
RLDS/Habitat 原始位姿维度，避免破坏旧 Stage；Stage20 单独从配置派生
`model_proprio_dim=5`。

| 文件 | 本次变更 |
|---|---|
| `prismatic/vla/datasets/rlds/traj_transforms.py` | 增加 4D pose→5D cyclic proprio；明确在 body action 转换后执行 |
| `prismatic/vla/datasets/rlds/dataset.py` | 计算 5D proprio stats；生成 action 对称边界与 proprio 混合边界；新 cache 依赖版本 |
| `prismatic/vla/datasets/rlds/utils/data_utils.py` | 训练归一化优先读取显式 `normalization_low/high` |
| `experiments/robot/openvla_utils.py` | 在线 proprio 归一化读取同一显式边界 |
| `prismatic/extern/hf/modeling_prismatic.py` | 动作反归一化读取同一显式边界 |
| `vla-scripts/finetune.py` | 新 5D projector、强制重置、schema v2 contract、联合 normalization hash |
| `vla-scripts/uav_eval/openvla_model_runner.py` | 原始 4D Habitat pose 在线编码为 5D；校验 config/stats/contract/hash |
| `vla-scripts/uav_eval/run_stage20_bodydelta_sft.sh` | 固定启用 cyclic proprio 并重置 projector |
| `tests/test_stage20_bodydelta_sft.py` | 增加跨角度边界、转换顺序、稀有横移不饱和测试 |
| `vla-scripts/uav_eval/audit_stage20_data.py` | 对真实 RLDS 验证 5D 单位圆、对称范围和 world/body round trip |

兼容性边界：旧 Stage12 的 4D proprio projector 不能加载进 Stage20；旧 Stage20
草稿 checkpoint 因缺少 schema v2 contract/5D stats 会在 runner 中 fail fast。恢复由本版
Stage20 产生的 checkpoint 时使用 `resume=True`，直接加载已训练的 5D projector，不再次
重置。

CPU 回归命令：

```bash
PYTHONPYCACHEPREFIX=/tmp/stage20_pycache \
  conda run --no-capture-output -n openvla-oft \
  pytest -q -p no:cacheprovider tests

PYTHONPYCACHEPREFIX=/tmp/stage20_pycache \
  conda run --no-capture-output -n openvla-oft python -u \
  vla-scripts/uav_eval/audit_stage20_data.py --compute_statistics --num_episodes 3
```

早期 GPU smoke/overfit 使用过 Stage12 merged VLA，只能作为代码链路诊断，不能作为
Stage20 的训练起点或收敛证据。最终主线固定使用 `/VLM/base-model/openvla-7b` 的原始
权重和 OpenVLA-OFT 训练代码；action head、5D proprio projector、condition/vision
projector、image role embedding 全部随机初始化，不加载 Stage12/Stage19 的任何权重。
为避免 trainer 同步本地 model/config 文件时修改原始基座，启动脚本先建立只复制可变
文件、其余模型 shard 使用符号链接的 runtime mirror。

部署侧也已用 GPU 2 完成加载 smoke：Habitat runner 成功加载 merged 7B、5D proprio
projector、action head 与 condition adapter，通过 schema/config/stats/hash 校验，并把
4D Habitat pose 编码、归一化为 shape `[5]`；没有通过 CLI 覆盖或兼容分支绕过 contract。

### 后台运行约定

服务器已确认安装 `tmux 3.2a`，GPU 2 为 RTX 4090 48GB。长实验统一使用
`openvla-oft` conda 环境、GPU 2、独立 tmux session 和落盘日志；本地电脑关闭或 SSH
断开不影响服务器任务。正式训练的启动形式为：

```bash
tmux new-session -d -s stage20_train \
  "cd /VLM/liangxinyue_25/openvla-oft && \
   CUDA_VISIBLE_DEVICES=2 \
   bash vla-scripts/uav_eval/run_stage20_bodydelta_sft.sh train"

tmux attach -t stage20_train
tail -f /VLM/liangxinyue_25/openvla-oft/runs/uav/stage20_base_bodydelta_sft_30k.log
```

不要在 smoke 之后自动盲跑 30k：先完成 K=1 小样本过拟合/短程稳定性门槛；通过后再用
上述 session 启动正式训练。若服务器本身关机，tmux 不能保存进程，需从 checkpoint
恢复。

### 固定批次诊断结果与目标修正

在切换回原始 OpenVLA 基座前，旧 Stage12 初始化仅用于隔离故障，得到两条有用结论：

1. K=1 纯动作固定批次可以把 masked root+future Smooth-L1 从 `0.1771` 降到约
   `1e-5`，说明 body-delta、yaw wrap、5D proprio、归一化和动作 mask 的数据流可微；
2. K=1/K=3 同时使用 `0.1 * (1-positive_similarity)` 和跨图像 InfoNCE 时，所有
   condition/image embedding 会被推向同一方向：positive similarity 接近 1，但四图
   CE 停在 `ln(4)=1.3863`。这是表示坍缩，不是动作或 Best-of-K 的问题。

因此正式配置把 `condition_alignment_weight` 固定为 0，只使用跨图像 InfoNCE；
positive similarity 继续作为诊断指标但不直接优化。日志新增 positive、hardest-negative、
margin 和 Recall@1。所有门槛必须在原始 OpenVLA 初始化下重新运行，旧 Stage12 诊断
checkpoint 不进入后续训练。

原始 OpenVLA 初始化已完成 GPU 2 验证：一步 K=3 smoke 成功保存
`stage20_base_bodydelta_sft_smoke--1_chkpt`，contract 明确记录
`uav_modules_initialized_fresh=true`；随后 K=1 纯动作固定批次在不加载任何旧 UAV
checkpoint 的情况下，把 root/future 总损失从 `0.186059` 降到 step 100 的
`0.000110`，step 300 为 `0.000027`，并保存
`stage20_base_bodydelta_sft_overfit_action_k1--300_chkpt`。因此 Phase 2 的“全新动作头可学习”
门槛通过；该 checkpoint 仅是诊断产物，后续隔离实验仍可直接从原始 base 独立启动。

Phase 3 的 K=1 condition 固定批次实验也已从原始 OpenVLA base 独立启动并通过：关闭
positive-only alignment，只优化跨图像 InfoNCE 后，检索 CE 从 step 1 的 `1.388718`
降到 step 50 的 `0.002001`、step 300 的 `0.000035`；Recall@1 从随机水平升至
`1.0` 并保持，positive-hardest-negative margin 从 `-0.006547` 增长到 `0.767870`。
同时 root/future action loss 在 step 300 分别为 `0.000003/0.000002`。checkpoint 已保存为
`stage20_base_bodydelta_sft_overfit_match_k1--300_chkpt`。这证明 K=1 图像检索和动作能够在
同一 SFT 中共同过拟合；下一门槛是 K=3 联合 winner，而不是直接启动 30k。

K=3 联合 winner 固定批次也已从原始 base 独立通过。step 300 的总损失为
`0.000274`，其中 root action `0.000136`、future action `0.000120`、InfoNCE
`0.000014`；检索 Recall@1 为 `1.0`，positive-hardest-negative margin 为
`0.822612`。零学习率恢复审计确认 checkpoint 各模块均正确加载，四个有效 future slot 的
hard winner 为 branch 1/2 各 50%，soft usage 为 branch 0/1/2 =
`32.17%/34.10%/33.73%`。固定批次样本太少，不能据此判断全数据 branch 利用率，但可确认
三分支没有发生 soft collapse。诊断 checkpoint
`stage20_base_bodydelta_sft_overfit_k3--300_chkpt` 仅用于门槛审计，不作为 pilot 初始化。

### 多轨迹 pilot 与验证留出

`rlds_data_all` 只有一个名为 `train` 的 TFDS split，共 25,567 episode；原有
`use_val_set=True` 会错误请求不存在的 `val` split。Stage20 现支持显式 TFDS split
instruction，pilot 按 episode 做确定性留出：

```text
训练：train[:95%]  = episode [0, 24289)，24,289 条
验证：train[95%:]  = episode [24289, 25567)，1,278 条
```

这不是在单条轨迹中切帧，因此同一 episode 不会同时出现在训练和验证中。训练保留图像
增强，验证强制关闭增强；验证每 250 optimizer steps 运行一次并把 root/future action、
InfoNCE、Recall@1 和 margin 写入 stdout 日志。pilot 为 1,000 optimizer steps，保存
step 500/1000 checkpoint，仍从原始 OpenVLA base 全新初始化全部 UAV 模块：

```bash
CUDA_VISIBLE_DEVICES=2 \
  bash vla-scripts/uav_eval/run_stage20_bodydelta_sft.sh pilot
```

只有 pilot 的训练/留出验证曲线稳定、branch 使用未坍缩后，才启动 30k 正式训练。

第一次 pilot 在 step 250 暴露了训练循环的边界错误：`log_step` 由 microbatch 下标整除得到，
但 W&B、validation、checkpoint 没有限定在 gradient-accumulation boundary；因此同一
optimizer step 会触发 8 次验证，step 500 还会重复合并保存 7B checkpoint。该 run 已在
step 259 停止并完整保留为 `stage20_base_bodydelta_sft_pilot_1k`。其第一次有效留出验证为：

```text
root action loss       0.018567
future action loss     0.014375
condition InfoNCE      1.200534
retrieval Recall@1     0.239167
retrieval margin      -0.003830
```

动作已能泛化，但 condition 检索仍接近四图随机水平，不能判定 matching 通过。修复后以
`completed_optimizer_step()` 作为唯一事件边界，optimizer、warmup、W&B、stdout、保存、
验证和停止条件共享同一绝对 step；增加了 accumulation=8 及 resume offset 的回归测试。
干净复现实验命名为 `stage20_base_bodydelta_sft_pilot_1k_v2`，每 250 step 保存一次，仍从
原始 OpenVLA base 全新初始化，不从失败 run 恢复。

2026-09-16 当晚尝试启动 v2 时，GPU 2 已被其他用户的 `worldvln` 进程占用约 39 GiB，
模型加载阶段按预期因显存不足退出，尚未执行任何训练 step。该启动日志已保留为
`stage20_base_bodydelta_sft_pilot_1k_v2_oom_20260916.log`，空 run 目录也使用同一 `_oom`
后缀留档；正式 v2 run ID 仍可用。按用户要求不创建 GPU watcher、不自动重启，待人工确认
GPU 2 空闲后再执行：

```bash
tmux new-session -d -s stage20_pilot_v2 \
  "cd /VLM/liangxinyue_25/openvla-oft && \
   CUDA_VISIBLE_DEVICES=2 \
   bash vla-scripts/uav_eval/run_stage20_bodydelta_sft.sh pilot"
```

2026-09-18 09:14，确认 GPU 3 尚有约 45.9 GiB 可用显存后，v2 已改用
`CUDA_VISIBLE_DEVICES=3` 在 tmux `stage20_pilot_v2` 中启动。GPU 3 上另有其他用户约
3.1 GiB 的 UnrealZoo/UE4 进程，本实验不操作该进程；两者合计显存约 27.3 GiB，仍有
约 20 GiB 余量。首个 optimizer step 已完成，新边界日志只输出一次：

```text
step=1, lr=5.23e-6
loss=1.095581
root/future action=0.099266/0.076664
InfoNCE=0.919463, Recall@1=0.260417, margin=-0.002743
soft usage=0.333201/0.333391/0.333408
```

这确认修复后的 accumulation 事件边界、全新模块初始化、数据流和 GPU 运行均正常。

### 1k pilot v2 最终结果（2026-09-18）

`stage20_base_bodydelta_sft_pilot_1k_v2` 已在 GPU 3 正常完成 1,000 个 optimizer step，
日志以 `Max step 1000 reached! Stopping training...` 结束，没有 traceback 或 OOM。step
250/500/750/1000 均只触发一次合并保存和一次固定 100-window 留出验证；四个 checkpoint
目录均约 15 GiB，最终 checkpoint 包含 merged VLA、LoRA、action head、5D proprio
projector、condition adapter、policy contract 和 dataset statistics。

| step | root action | future action | condition InfoNCE | Recall@1 | margin |
|---:|---:|---:|---:|---:|---:|
| 250 | 0.071715 | 0.058609 | 1.199930 | 0.244167 | -0.003167 |
| 500 | 0.021410 | 0.015142 | 1.198880 | 0.246667 | -0.001452 |
| 750 | **0.010940** | **0.005123** | 1.198897 | 0.239167 | -0.001352 |
| 1000 | 0.017914 | 0.007920 | 1.198930 | 0.241667 | -0.001408 |

结论是“训练工程链路通过，但算法 pilot 未通过”。动作监督在不同 episode 的留出窗口上
明显下降，最佳动作 checkpoint 是 step 750；step 1000 已有轻微回退。这里的 loss 是
归一化 body-delta 上的训练损失，不能直接解释为米或弧度误差，仍需反归一化评估和
Habitat 闭环验证。condition 检索则没有泛化：四个 future image 候选下随机 Recall@1
约为 25%，四次验证均只有 23.92%--24.67%，且 positive-hardest-negative margin 始终为
负。固定批次能过拟合而跨 episode 失败，说明问题不是链路不可微，而是当前检索监督和
负样本构造不足。step 1000 单个训练报告的 soft branch usage 为
32.77%/33.82%/33.42%，只能排除当批次的明显分支占用坍缩；由于 action diversity 权重为
0，不能证明 K 个动作假设具有语义差异。

因此禁止直接以当前配置启动 30k。下一轮应优先改 condition 学习：引入跨 microbatch/
跨 episode 的负样本队列或等价的大检索 batch，避免每个样本只有同一轨迹中四张相邻未来
图作为负样本；同时增加跨 episode 检索评估。修改后先重新跑短 pilot，以 Recall@1 明显
高于 25%、margin 转正且动作验证不退化作为放行条件。当前应保留 step 750 和 step 1000：
前者用于动作侧对照，后者用于复现最终状态，但两者都不应被称为可部署的完整策略。

### Condition matching v2：目标对齐与跨 episode 队列（2026-09-18）

复盘 v2 后确认不应只扩大负样本。旧 `compute_indoor_uav_sft_loss()` 先用
`action_error + 0.25 * (1-condition_similarity)` 产生 winner，随后又要求同一组 condition
logit 预测该 winner；这会让 condition 相似度参与生成自己的监督标签，matching accuracy
具有循环自我标注成分。与此同时，旧 InfoNCE 在 batch size 1 下只有四个有效 future
time image，gradient accumulation=8 仍是八个独立 `4×4` loss，不会形成 `32×32` 候选集。

当前实现改为三层明确目标：

1. `winner = argmin(stop_gradient(action_error))`，condition 相似度不再参与 branch 分配；
   同一 winner 仍绑定 `(condition, action)`，因此图像被直接监督去找回动作最优分支；
2. 主 condition loss 是每个未来时刻的 K-way CE，K=3 随机准确率为 33.3%；
3. 保留低权重的同窗口时间 InfoNCE，并增加更低权重的跨 episode image-patch FIFO 队列。

队列容量为 256 张 future image，每张保留 `[256,512]` patch embedding，并以 bfloat16、
detach 形式存储，不跨 batch 保留计算图。TFDS 的 `episode_metadata.file_path` 经 dlimp 的
`traj_metadata` 显式传到 PyTorch batch；计算 queue loss 时严格排除 query 所属 episode，
至少有 32 张其他 episode 图像后才启用。权重固定为：

```text
per-time K-way       1.00
temporal InfoNCE     0.25
cross-episode queue  0.05
positive-only        0.00
branch balance       0.01
condition diversity  0.005
condition assignment 0.00  # 禁止 condition 自己参与标签生成
```

训练、验证和 checkpoint contract 新增独立的 branch/temporal/queue loss、accuracy、margin、
queue query count 与 eligible negative count。真实 RLDS 审计确认 episode ID 从源文件一路保留，
25,567 episode 和 474,340 transitions 不变，动作 world/body round trip 最大误差为
`1.19e-7`。完整 CPU 测试为 43/43 通过。

GPU 3 的 `stage20_base_bodydelta_sft_matchv2_smoke` 已从原始 OpenVLA base 全新初始化并完成
2 个 optimizer step，无 OOM/traceback，也未生成不必要的 15 GiB checkpoint。step 2 的
queue loss 为 `1.843707`，证明 warm-up 后队列实际参与反向传播；condition adapter 梯度非零。
smoke 数值不用于判断收敛。下一门槛是
`stage20_base_bodydelta_sft_overfit_k3_v2` 固定批次过拟合，随后才允许运行
`stage20_base_bodydelta_sft_pilot_1k_v3` 多轨迹留出验证；v3 必须分别超过 K-way 33.3%
和 temporal 25% 随机基线、margin 转正且动作验证不退化，才允许启动 30k。

K=3 新目标固定批次门槛随后已从原始 `/VLM/base-model/openvla-7b` 独立完成，run 为
`stage20_base_bodydelta_sft_overfit_k3_v2`。step 300 指标如下：

```text
total loss                    0.000501
root/future action loss       0.000048 / 0.000046
per-time K-way loss           0.000072
K-way accuracy / margin       1.000000 / 0.783421
temporal loss                 0.001260
temporal Recall@1 / margin    1.000000 / 0.499325
soft branch usage             32.78% / 32.48% / 34.75%
```

该固定 window 的 hard winner 为 25%/0%/75%；只有四个未来槽，不能用 hard count 判断全数据
分支利用率。队列达到 256 张但全部来自同一 episode，因此 eligible query、queue loss 均为
0，符合严格过滤预期。merged checkpoint、action head、proprio projector、condition adapter
均已保存到 `stage20_base_bodydelta_sft_overfit_k3_v2--300_chkpt`。随后零学习率 resume audit
重新加载所有组件，得到 K-way accuracy 1.0/margin 0.779943、temporal Recall@1 1.0/margin
0.500601，说明保存/重载无误。该 checkpoint 仍只用于诊断，pilot v3 不从它初始化。

第一次启动 v3 后在 step 14 前主动中止：真实数据包含只有 root 有效、future mask 全 0 的
终点窗口，旧验证汇总会把这些窗口的 condition 指标按 0 参与普通 batch 平均，从而人为
压低 accuracy/margin。该无 checkpoint run 已无损归档为
`stage20_base_bodydelta_sft_pilot_1k_v3_unweighted_val_20260918`。训练 loss 的 mask 原本正确，
本次修复只改变指标汇总：root、future、K-way、temporal、queue 指标分别按实际有效 query
数加权，完全无 future 的窗口不再污染 condition 指标；日志同时给出每种检索任务在实际
候选数下的随机 accuracy，而不是只假定固定 33.3%/25%。新增回归测试后 Stage20 测试为
16/16 通过。

2026-09-18 12:29，正式 `stage20_base_bodydelta_sft_pilot_1k_v3` 已再次从原始 OpenVLA
base 干净初始化并在 GPU 3/tmux `stage20_pilot_v3` 启动。step 1 正常完成，root/future
action loss 为 `0.099607/0.076811`，K-way/temporal accuracy 均为 `0.229167`，符合尚未
学习的初始状态；无 OOM、NaN 或残留旧 checkpoint 初始化。

### 1k pilot v3 最终结果与放行结论（2026-09-18）

`stage20_base_bodydelta_sft_pilot_1k_v3` 已正常完成 1,000 个 optimizer step。日志以
`Max step 1000 reached! Stopping training...` 结束，无 traceback、OOM 或残留训练进程。
step 250/500/750/1000 四个 checkpoint 均约 15 GiB；最终目录包含 merged VLA、LoRA、
action head、5D proprio projector、condition adapter、dataset statistics 和完整
`policy_contract.json`。四次固定 100-window 留出验证结果如下：

| step | root action | future action | K-way acc / random | K-way margin | temporal R@1 / random | temporal margin | queue R@1 / random | queue margin |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 250 | 0.011852 | 0.009195 | 0.502857 / 0.333333 | 0.000042 | 0.257971 / 0.260870 | -0.002367 | 0.173633 / 0.009086 | -0.008015 |
| 500 | 0.030709 | 0.024266 | 0.531429 / 0.333333 | 0.006235 | 0.240580 / 0.260870 | -0.001395 | 0.015974 / 0.009084 | -0.005741 |
| 750 | **0.010845** | **0.006437** | 0.528571 / 0.333333 | 0.000454 | 0.272464 / 0.260870 | -0.001023 | 0.044728 / 0.008882 | -0.002741 |
| 1000 | 0.022280 | 0.011641 | **0.682857 / 0.333333** | **0.010484** | 0.260870 / 0.260870 | -0.001130 | 0.009868 / 0.008635 | -0.004998 |

结论为“动作侧和 inference-aligned K-way 目标部分通过，但完整 condition matching 门槛未
通过”。K-way 分类在 step 1000 达到 68.29%，明显超过 K=3 的 33.33% 随机基线，且
margin 转正，说明 future image 确实能够在同一时刻的三个 condition 中找回由 action error
确定的 winner；这一指标不再含 condition 参与生成标签的循环自标注。动作侧最佳 checkpoint
为 step 750，step 1000 的 root/future loss 均已回退。二者的 loss 仍位于归一化 body-delta
空间，不能直接当作米或弧度误差。

跨时间检索没有学会：step 1000 的 temporal Recall@1 恰好等于实际随机基线 26.09%，四次
margin 全为负。跨 episode queue 检索也不稳定：step 250 虽有较高 top-1，但 margin 为负，
随后持续退化，step 1000 已接近随机且 margin 仍为负。这说明当前 condition 表示学到的是
“在同一 future slot 的 K 个分支中选择 action winner”，尚未形成可跨图像候选稳定比较的
共享视觉匹配空间。另一个必须排查的问题是 K 个动作分支是否真正产生有意义的行为差异；
K-way 高准确率本身不能排除动作接近、winner tie/bias 或分支语义不稳定。

因此当前配置禁止启动 30k，也不把任何 v3 checkpoint 称为可部署策略。step 750 仅作为
动作质量最佳对照，step 1000 仅作为 K-way 匹配最佳对照；没有单一 checkpoint 同时通过
动作、K-way、temporal 和 queue 四项。下一步先做只读诊断：在固定留出集统计每个时刻的
hard winner 分布、K 分支动作两两距离、winner 与 runner-up action-error gap，并按 episode/
scene 分层检查 queue 检索。根据诊断结果再决定是修正分支多样性/标签稳定性，还是重构
跨图像对比目标，然后重新跑短 pilot；在 temporal 与 queue 的 accuracy 稳定超过各自随机
基线、margin 转正且动作不退化前，不进入长训练。

### step 1000 分支可辨识性诊断（2026-09-18）

为排除“K-way 准确率只是由三个近乎相同动作之间的任意 tie 产生”，验证路径新增了纯指标，
不改变 loss：归一化动作的平均/最近分支 L1、winner 与 runner-up action-error gap、
`gap < 1e-3` 的 near-tie rate、反归一化后的分支位置/yaw 间距，以及每个 future slot 的
hard winner 分布。`tests/test_stage20_bodydelta_sft.py` 已覆盖数值和按 slot 有效样本数加权聚合，
Stage20 测试为 18/18 通过，完整测试集为 46/46 通过。可复现入口为：

```bash
CUDA_VISIBLE_DEVICES=3 bash vla-scripts/uav_eval/run_stage20_bodydelta_sft.sh audit_pilot
```

该模式从 `stage20_base_bodydelta_sft_pilot_1k_v3--1000_chkpt` 重载全部组件，learning rate=0，
只执行一次 100-window 留出验证，不保存新模型。结果与原 step 1000 验证一致，且新增指标为：

```text
action branch pair/min-pair L1     0.083190 / 0.049637  (归一化动作空间)
branch position separation         0.137721 m
branch yaw separation              0.025806 rad (约 1.48 deg)
winner-runner-up error gap          0.010680
near-tie rate (gap < 1e-3)         11.7143%
hard winner branch 0/1/2           26.57% / 21.43% / 52.00%
soft usage branch 0/1/2            32.96% / 33.47% / 33.57%
```

按 future slot 的 hard winner 分布为：

| future slot | valid | branch 0 | branch 1 | branch 2 |
|---:|---:|---:|---:|---:|
| 1 | 95 | 18.95% | 28.42% | 52.63% |
| 2 | 90 | 27.78% | 24.44% | 47.78% |
| 3 | 85 | 35.29% | 9.41% | 55.29% |
| 4 | 80 | 25.00% | 22.50% | 52.50% |

诊断结论：三个动作分支不是完全重合，平均物理位置差已达到约 13.8 cm，且 88.3% 的有效
future slot 不属于阈值内 near-tie；所以 step 1000 的 68.29% K-way accuracy 不能简单归因于
全体动作并列。然而分支 2 在所有 future slot 都占约 48%--55%，branch 1 在 slot 3 只有
9.41%，说明仍存在稳定的 hard-winner 偏置，且尚无证据表明 branch index 具有跨样本一致的
语义。重载诊断的 temporal R@1 为 26.38%（随机 26.09%）、margin `-0.001147`；queue R@1
为 2.57%（随机 0.91%）、margin `-0.004938`。二者即使偶有 top-1 高于随机，hardest-negative
margin 仍为负，不能放行。

由此下一轮的优先级应是重构跨图像 condition 监督，而不是先增大动作 branch diversity。
现有动作多样性已经非零；直接强推分支更远可能损害动作精度，却无法修复 temporal/queue
检索。任何新方案仍应保留 action-error-only winner 与 paired condition-action 绑定，并先用
短 pilot 同时检查 action、K-way、temporal 和 queue 四类指标。

### Inference-aligned Condition 动作选择审计（2026-09-18）

在决定是否重构 temporal/queue 之前，先补齐真正对应推理的数据指标。此前的
`sft_future_action_loss` 是由专家动作真值选择分支的 Oracle Best-of-K 上限，
`condition_branch_accuracy` 只统计 condition 是否选中相同 branch；两者都没有直接回答
“由 condition 选出的动作误差是多少”。新增的每个有效 future slot 指标为：

```text
k_oracle = argmin_k SmoothL1(A_k, A*)
k_cond   = argmax_k similarity(C_k, I_future)

oracle_future_action_loss    = error[k_oracle]
condition_selected_action_loss = error[k_cond]
branch0_future_action_loss   = error[0]
selection_regret             = condition_selected - oracle
gain_vs_branch0              = branch0 - condition_selected
oracle_recovery              = gain_vs_branch0 / (branch0 - oracle)
```

同时在反归一化 body-delta 空间报告三种选择的 position error（米）和 wrapped yaw error
（弧度），并统计 condition 自身选择 branch 0/1/2 的比例。验证聚合严格按有效 future slot
数量加权；regret、gain 和 recovery 在全局聚合后的三个基础 loss 上重新计算，避免平均每个
window 的比率造成偏差。该变更只增加 detached 诊断，不进入总 loss，不改变 checkpoint。

可复现审计入口仍为 `audit_pilot`，并支持通过环境变量选择 checkpoint：

```bash
PILOT_AUDIT_STEP=750  CUDA_VISIBLE_DEVICES=3 bash vla-scripts/uav_eval/run_stage20_bodydelta_sft.sh audit_pilot
PILOT_AUDIT_STEP=1000 CUDA_VISIBLE_DEVICES=3 bash vla-scripts/uav_eval/run_stage20_bodydelta_sft.sh audit_pilot
```

放行逻辑调整为：先比较 step 750/1000 的 `condition_selected_action_loss` 与 branch-0、Oracle；
如果 recovery 高且物理误差接近 Oracle，temporal/queue 仅作为辅助检索任务，不应因其未通过
就盲目增权。如果 condition-selected 接近或劣于 branch 0，才进入位姿感知 multi-positive
InfoNCE、稳定视觉 key encoder 和反事实 Habitat 数据的结构修改。

#### step 750/1000 审计结果

新增测试全部通过：Stage20 为 20/20，完整测试集为 48/48。两个 checkpoint 均使用同一
`train[95%:]` 固定留出数据、100 windows、learning rate=0 重载审计；没有修改参数或保存
新模型。核心结果为：

| checkpoint | Oracle loss | Condition-selected loss | branch-0 loss | regret | gain vs branch 0 | Oracle recovery |
|---:|---:|---:|---:|---:|---:|---:|
| 750 | **0.006425** | **0.011641** | 0.016903 | 0.005216 | 0.005262 | 50.22% |
| 1000 | 0.011662 | 0.015608 | 0.032018 | **0.003946** | **0.016410** | **80.61%** |

反归一化的一步 body-delta 物理误差为：

| checkpoint | position Oracle / Condition / branch 0 | yaw Oracle / Condition / branch 0 |
|---:|---:|---:|
| 750 | **0.08575 / 0.12185 / 0.16134 m** | 0.07672 / 0.07591 / 0.07762 rad |
| 1000 | 0.13077 / 0.15946 / 0.24566 m | 0.06993 / 0.07217 / 0.06108 rad |

这证明Condition不是无效装饰：两个 checkpoint 上，由未来图像选择分支都明显优于固定
branch 0。step 1000 的 K-way accuracy 68.29% 将可获得的 Oracle 改善回收了80.61%，说明
当前同一时刻三选一目标已经具有较强的推理价值；temporal/queue 的实例检索失败不能再作为
单独否决整个策略的理由。另一方面，step 1000 的动作候选整体退化，导致即使选择效率更高，
最终 Condition-selected loss `0.015608` 和位置误差15.95 cm仍劣于step 750的`0.011641`
和12.18 cm。

因此当前离线最优 checkpoint 定为step 750：它提供最低的实际Condition-selected动作误差，
不是只看Oracle或只看K-way准确率得出的结论。step 1000保留为“Condition选择更强但动作
过拟合”的对照。下一步不增强temporal/queue，也不修改动作头；先使用step 750进行Habitat
短程闭环，直接比较Condition选择与固定branch 0。若后续要继续训练，合理方向是从step 750
锁定动作链路后单独细化Condition，而不是继续联合训练让动作质量回退。

### Habitat 闭环执行审计（2026-09-18）

#### 在线数据流与对照定义

step 750 在线 runner 现在实际执行两步 receding-horizon plan：

```text
observation(ref, previous, current, proprio, instruction)
  -> 一次预测 actions [T=5,K=3,4] 和 conditions [T=5,K=3,512]
  -> plan step 0: 执行 root action A[0,0]
  -> Habitat 渲染新观测
  -> plan step 1: 将新图像 patch 投影到512维，与 C[1,0:3] 比较
  -> Condition 组执行 argmax similarity 对应的 A[1,k]
  -> 重新规划
```

fixed-branch0 对照使用完全相同的 checkpoint、历史图像、重规划频率和位姿合成，
唯一区别是 plan step 1 始终执行 `A[1,0]`。root-only 隔离实验则设
`condition_plan_steps=1`，每帧重规划并只执行 `A[0,0]`。因此三者可以分别回答
“Condition 选择是否有用”和“当前 root action 是否本身可用”。

在这一轮中修复了四个评估链路问题，都不改变模型权重：

1. `coords or []` 对 NumPy array 会触发布尔值歧义，改为显式 `None` 判断；
2. 旧 controller 在最后一个动作送给 Habitat 后立即结束，没有渲染和评估该动作，
   现在等待最后一帧 simulator output 后再结束；
3. episode 结束时的 `terminate.json` 会与下一个 episode 初始化竞态，导致新场景刚创建
   就被旧信号关闭。现在 episode 切换由下一个 `is_new_scene` 请求管理，并且 simulator
   会忽略 episode key 不匹配的过期终止信号；
4. `MAX_INFERENCE_STEPS=12` 是旧的大步绝对位姿策略遗留值，与当前每次只执行一个
   body delta 不匹配。该值现在可通过环境变量配置且默认为60；同一 HM3D 场景
   在 episode 间复用 simulator，只更新绝对位姿。

controller 现在为每一步保存 branch、condition similarities、selected action 和 plan step，
并在轨迹 JSON 中保存 start/end/final pose、wrapped yaw error 和 position error。新增
`vla-scripts/uav_eval/summarize_stage20_closed_loop.py` 可对两个 matched 轨迹目录做可复现汇总。

#### 12步 matched pilot

前10条测试轨迹来自同一 HM3D 场景，难度为 easy/medium/hard = 3/6/1。成功标准是
位置误差 `<=0.1 m` 且 wrapped yaw 误差 `<=pi/6`。

| 执行方式 | success | final pos. | min pos. | final yaw | position progress | path length |
|---|---:|---:|---:|---:|---:|---:|
| Condition, 2-step plan | 0/10 | 1.5138 m | 1.1580 m | 0.8649 rad | 0.1180 m | 1.5914 m |
| fixed branch 0, 2-step plan | 0/10 | **1.3594 m** | **0.9662 m** | **0.8725 rad** | **0.2724 m** | 1.7509 m |
| root-only, every-step replan | 0/10 | 1.5240 m | 1.1230 m | 0.8978 rad | 0.1078 m | 1.6143 m |

Condition 相对 fixed 的终点位置胜/平/负为3/1/6，平均终点 gain 为 `-0.1544 m`。
60次 future 选择的 branch 0/1/2 计数为17/29/14，top1-top2 similarity margin 仅
`0.002515`。root-only 与 Condition 几乎相同，但 fixed future branch 0 明显更好：这表明
future 动作候选不是完全失效，而 Condition 在线选择没有把离线收益转化成闭环收益。

这10条专家片段的原始间隔为6--25步，其中6/10超过12步，所以12步的0成功不能
单独用来判定策略。它的配对误差差异仍然有效，但必须再以60步预算检查“是否只是步数
不足”。

#### 60步 matched pilot

| 执行方式 | success | final pos. | min pos. | final yaw | position progress | path length |
|---|---:|---:|---:|---:|---:|---:|
| Condition, 2-step plan | 0/10 | **4.9091 m** | 1.1551 m | **1.2759 rad** | -3.2774 m | **6.3374 m** |
| fixed branch 0, 2-step plan | 0/10 | 5.4526 m | **0.8891 m** | 1.2961 rad | -3.8208 m | 7.0784 m |

Condition 的最终误差平均小 `0.5435 m`，但它的平均最小误差反而大 `0.2660 m`。结合
路径长度可知，前者不是导航更好，只是 Condition 漂移得稍慢；两者在前12步后都没有
继续靠近目标，而是持续累积正 forward/up 动作后离开场景有效区域。增加动作预算因此
排除了“仅仅是12步不够”的解释。

300次在线 future 选择的 branch 0/1/2 计数为75/151/74，branch 1 占50.33%，平均
top1-top2 margin 仍只有 `0.002556`。这与离线 action-oracle winner 中 branch 2 约52%的偏置不一致，
是明确的 expert-image 到 model-induced-image 分布偏移信号。绝对 cosine similarity 约为0.99，
所以现在的 `condition_threshold=0.6` 在线上实际不会触发不确定重规划。

两轮的完整轨迹和日志分别保存在：

```text
shared_folder/trajectories_stage20_step750_cond10_v2
shared_folder/trajectories_stage20_step750_fixed10_v2
shared_folder/trajectories_stage20_step750_root10_12
shared_folder/trajectories_stage20_step750_cond10_60
shared_folder/trajectories_stage20_step750_fixed10_60
shared_folder/logs_stage20_step750_*
```

#### 当前决策与下一阶段门槛

这一轮是工程链路通过、闭环策略失败的诊断，不是算法放行。前10条只有一个场景，
不足以给出泛化指标；但两种预算都是0成功、且长时间漂移，已足以禁止当前配置进入30k训练。

下一阶段不应先增加 temporal/queue 权重，也不应仅靠增大 rollout 步数，而应按以下顺序进行：

1. 从 RLDS episode 边界生成 terminal/remaining-step 标签，增加独立 root STOP 头；先做每帧重规划的
   root-action+STOP SFT 基线，必须在 Habitat 中学会到达后停止，再谈 future plan；
2. 将 action 验证按 forward/right/up/yaw 分解，增加每类指令的符号准确率、均值偏置和
   goal-progress，优先查明持续正 up/forward 偏置；
3. 当前 Condition 用专家轨迹的 `I_(t+1)` 监督，在线却对模型自己的 `A_t` 产生的图像做匹配。
   应使用 Habitat 收集模型偏移状态，再投影到专家轨迹/局部 expert action 生成 SFT recovery 数据；
   这仍是 SFT/DAgger-style 数据增强，不需要 GRPO；
4. Condition 在线门控应使用 top1-top2 margin 或校准后概率，不再使用对当前0.99级 cosine
   无效的绝对 `0.6` 阈值；低置信时放弃 future branch 并重规划 root action；
5. 新 pilot 的放行条件不再只看 teacher-forced loss：在至少3个场景的 matched 闭环子集上，
   root+STOP 基线必须先产生非零成功率且不持续漂移；然后 Condition 版必须在 success/minimum
   position error 上稳定优于 fixed branch 0，才允许扩大训练。

截至本节记录，Stage20 相关测试为22/22通过，项目完整测试为50/50通过。

### Stage21：Root Action + Post-action STOP SFT（2026-09-18）

#### 目标与监督语义

Stage20 闭环失败后，先不修改已经表现较好的动作头，也不从 stage12/stage19 恢复训练。
Stage21 以离线和闭环审计中动作误差最低的
`stage20_base_bodydelta_sft_pilot_1k_v3--750_chkpt` 为唯一初始化来源，并冻结 VLA、5D
proprio projector、K=3 动作头和 condition adapter，仅训练独立 STOP 分类头。这样本阶段只回答
“现有感知特征能否判断当前动作执行后应该结束”，不会让 STOP 学习破坏已有动作。

RLDS builder 的最后一个 step 并不是无动作终点：该 step 的 state/image 来自
`postures[frame_num-1]`，action 是 `postures[frame_num]`，所以最后一条仍是从当前状态到轨迹
终点的有效动作。因此标签定义为：

```text
stop_after_action(t) = is_terminal(t)

target=1 的含义：执行 A[t,0,0]，让 Habitat 产生下一帧，然后结束当前 instruction。
target=0 的含义：执行 A[t,0,0] 后继续闭环重规划。
```

不把 terminal 样本的动作置零，也不把它解释成“当前状态立刻 STOP”。controller 中 success
判定优先于 learned STOP；二者都发生在模拟器确认动作执行并渲染新观测之后。

全数据有25,567个 episode、474,340条 transition，因此正样本率为
`25567/474340 = 5.3900%`。训练自动从 dataset statistics 得到 BCE
`pos_weight=(474340-25567)/25567=17.552822`，而不是手写一个可能与数据版本不一致的值。
对真实 TFDS 前32个 episode 的抽查结果是：每个 episode 恰有一个 `is_terminal=True`，且都在
最后一个 step。

#### 新数据流与模块

```text
RLDS is_terminal
  -> restructure: stop_after_action [episode_length]
  -> chunk_act_obs: 与有效 root transition 同索引 gather
  -> RLDSBatchTransform / collator: stop_after_action [B]

ref/previous/current + 5D proprio + instruction
  -> 冻结的 OpenVLA-OFT
  -> root ACT hidden h_act[:,0,0] [B,4096]
  -> STOP head: LayerNorm(4096) -> Linear(4096,1024)
                -> GELU -> Linear(1024,1)
  -> sigmoid -> P(stop after root action)
```

最后一层 weight 初始化为0、bias初始化为数据先验的 logit，因此未训练时严格输出约5.39%，
不会因随机 hidden projection 产生大量假 STOP。训练 loss 当前只有加权二元交叉熵：

```text
L = BCEWithLogits(stop_logit, stop_after_action, pos_weight=17.552822)
```

Stage20 的动作、K-way condition、temporal 与 queue loss 权重在本 pilot 中均为0，但保留其
前向输出作为不变性诊断。验证新增 STOP confusion matrix、accuracy、precision、recall、
specificity、balanced accuracy；动作诊断新增反归一化后的 forward/right/up/yaw prediction
mean、target mean、bias、absolute error 和非零目标 sign accuracy。验证完成后会恢复每个模块
原本的 train/eval 状态，尤其保证冻结 VLA 始终为 eval，避免 dropout 使 STOP 特征源漂移。

checkpoint contract 升级为schema 3，并记录 post-action 语义、正类权重和推理阈值。在线
runner 只在新 plan 的 root slot 计算 STOP 概率；future slot 不复用旧的 STOP 判断。Habitat
controller 执行动作、收到渲染结果后才以 `termination_reason=model_stop` 结束，并把概率、阈值
和终止原因写入轨迹。

#### 工程验证与训练入口

新增入口：

```bash
CUDA_VISIBLE_DEVICES=3 bash vla-scripts/uav_eval/run_stage21_root_stop_sft.sh smoke
CUDA_VISIBLE_DEVICES=3 bash vla-scripts/uav_eval/run_stage21_root_stop_sft.sh pilot
```

启动器为step750建立硬链接 runtime mirror，只复制可能被 Transformers 更新的配置/建模文件，
避免修改原 checkpoint；auxiliary component 加载前还会核对 policy contract 中的5D cyclic
proprio、one-step body delta、T=5和K=3。pilot为1,000 optimizer steps、batch size 1、梯度累积8、
每250步保存与验证，W&B 当前使用offline模式。

回归测试结果为完整项目 **53/53 passed**。2-step GPU3 smoke 已完成：输入和输出形状均符合
contract，两个 batch 的 loss 分别为`0.055407`和`0.054220`；只有 STOP head 有梯度
（norm `0.5566/0.5538`），VLA、动作头、condition adapter和proprio projector梯度均为None。
两个 smoke 样本恰好都是负类，预测概率从初始化先验约0.0539开始，没有误触发STOP。

1k pilot 已在 tmux session `stage21_stop_pilot` 启动，日志为：

```text
runs/uav/stage21_step750_root_stop_sft_pilot_1k.log
```

pilot 完成后不能只看普通 accuracy（全预测 continue 就有94.61%）；首先比较 validation
balanced accuracy、terminal recall、specificity和 predicted stop rate，再做阈值校准。只有正负类
都可分且不会提前大量停止，才把新 checkpoint 接到 Habitat root-only matched 多场景评估。
如果冻结特征上的 STOP 无法分离，再解冻最小范围的 VLA LoRA；在得到证据前不改动作头，
也不开始 recovery SFT。

#### Stage21 1k pilot 完成结果（2026-09-19）

训练在step 1000正常结束，无 traceback；step 250/500/750/1000均保存了完整的merged VLA、
action head、condition adapter、5D proprio projector、STOP head、dataset statistics和schema-3
policy contract。四个contract都明确记录`stop_after_action=True`、post-action终止语义、
`pos_weight=17.552822`和默认阈值0.5。

固定`train[95%:]`验证在每个checkpoint上运行100个window，其中5个STOP正样本。结果为：

| checkpoint | STOP loss | mean P(STOP) | accuracy | recall | specificity | balanced accuracy | predicted STOP rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 250 | 2.039025 | 0.109006 | 0.95 | 0.00 | 1.00 | 0.50 | 0.00 |
| 500 | **1.917365** | **0.125294** | 0.95 | 0.00 | 1.00 | 0.50 | 0.00 |
| 750 | 2.071088 | 0.102108 | 0.95 | 0.00 | 1.00 | 0.50 | 0.00 |
| 1000 | 2.250638 | 0.080606 | 0.95 | 0.00 | 1.00 | 0.50 | 0.00 |

因此普通95% accuracy是类别不平衡造成的假象：阈值0.5下四个模型都退化成“永远continue”，
没有识别出任何一个terminal。step500虽然有最低验证loss和最高平均STOP概率，也仍然不能直接
部署。当前结论是工程链路及标签对齐通过，但“冻结VLA特征+独立MLP STOP头”的策略未通过
放行门槛；暂不进行Habitat STOP闭环评估，因为它必然不会触发learned STOP。

冻结正确性也由验证结果侧面确认：四次验证的全部动作指标完全相同，仍是原step750的
`root action loss=0.010912`、`future action loss=0.006425`、condition-selected loss
`0.011641`；这轮训练没有改变动作策略。

下一步先在同一固定验证集导出正/负样本各自的STOP概率分布并扫描阈值，判断模型是“有排序
能力但0.5未校准”还是“正负完全不可分”。若AUC/最佳balanced accuracy明显高于随机，则选
step500做阈值校准并扩大验证样本；若仍接近随机，则不靠降低阈值掩盖问题，改为最小范围解冻
最后若干LLM LoRA层或让STOP读取融合后的current-vision/proprio token，再做短SFT pilot。

#### STOP 概率与阈值扩大审计（2026-09-19）

验证链路新增逐样本STOP probability/target导出、精确ROC-AUC、正负类概率分布和全候选阈值
balanced-accuracy扫描；balanced accuracy并列时优先更高specificity，避免用大量提前停止换取
表面recall。原先validation的`shuffle_buffer_size=1000`会在`train=False`路径只cache
`shuffle_buffer/10=100`条，因此最初只有5个正样本。扩大审计将shuffle buffer设为10,000，
四个checkpoint都在同一固定1000条池上运行，其中58个terminal、942个non-terminal；
learning rate为0且不保存新权重。完整测试更新为**54/54 passed**。

| checkpoint | ROC-AUC | positive-negative mean gap | best threshold | best balanced acc. | precision | recall | specificity | FP / 942 | FN / 58 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 250 | 0.6305 | 0.00100 | 0.10812 | 0.6419 | 0.0866 | 0.8103 | 0.4735 | 496 | 11 |
| 500 | 0.6235 | 0.00229 | 0.12201 | 0.6439 | 0.0846 | 0.8621 | 0.4257 | 541 | 8 |
| 750 | 0.6322 | 0.00201 | 0.10014 | 0.6492 | 0.0861 | 0.8621 | 0.4363 | 531 | 8 |
| 1000 | **0.6520** | 0.00184 | 0.08049 | **0.6753** | **0.0979** | 0.8103 | **0.5403** | **433** | 11 |

100条小样本曾让step500呈现AUC 0.7558，但扩大后降为0.6235，证明不能据此校准。step1000是
四者排序最强的checkpoint，仍需把433/942个正常继续状态误判为STOP，precision不足10%。
这会让Habitat中的无人机在远未完成指令时频繁停下，所以0.08049等低阈值均不允许部署。

该结果排除了“只需把0.5阈值调低”的解释，也说明旧root ACT hidden中存在少量terminal信号但
远不足以控制。下一实验不解冻VLA、不改变动作输出，先采用更低风险的结构修复：STOP同时读取
root ACT与root COND的融合hidden，并利用已经进入batch的`actions_remaining_after_root`构造
`remaining<=0/1/2/4`多阈值密集辅助监督。`remaining<=0`仍是唯一部署STOP；其他输出只帮助共享
trunk学习轨迹进度。如果这种冻结特征的ordinal pilot仍不能显著提升AUC/specificity，再进入
带动作保持约束的最小LoRA解冻，而不是直接破坏当前step750动作策略。

逐样本审计文件位于：

```text
runs/uav/stage21_stop_score_audit_step250_v2_n1000/stop_validation_step251.json
runs/uav/stage21_stop_score_audit_step500_v2_n1000/stop_validation_step501.json
runs/uav/stage21_stop_score_audit_step750_v2_n1000/stop_validation_step751.json
runs/uav/stage21_stop_score_audit_step1000_v2_n1000/stop_validation_step1001.json
```

### Stage22：ACT+COND Progress STOP SFT（2026-09-19）

Stage21扩大审计否决阈值校准后，先采用不影响动作策略的结构修复，而不是立即解冻VLA。
新`IndoorUAVProgressStopHead`的数据流为：

```text
root ACT hidden [B,4096]  -> LN -> Linear(4096,512) -> GELU --+
                                                               +-> concat [B,1024]
root COND hidden [B,4096] -> LN -> Linear(4096,512) -> GELU --+
  -> Linear(1024,1024) -> GELU -> Linear(1024,4)
  -> logits for actions_remaining_after_root <= [0,1,2,4]
```

最后一层仍以零weight和各任务数据先验logit初始化。第0个输出与Stage21语义完全相同：执行root
动作后是否终止；`<=1/2/4`只作为共享trunk的密集轨迹进度监督，不进入Habitat终止判断。
总loss为：

```text
L = BCE(remaining<=0, pos_weight=17.5528)
  + 0.25 * mean(
      BCE(remaining<=1, pos_weight=8.2764),
      BCE(remaining<=2, pos_weight=5.1843),
      BCE(remaining<=4, pos_weight=2.7106))
```

VLA、动作头、condition adapter和proprio projector继续全部冻结；新头共5,265,412个可训练
参数。checkpoint contract升级为schema 4，记录`act_cond_ordinal_progress_v1`、四个horizon和
辅助loss权重；在线runner只读取第0个logit，并兼容旧schema-3 binary STOP checkpoint。

完整测试为**55/55 passed**。2-step GPU3 smoke通过：真实样本的remaining=4/3正确产生ordinal
targets，只有progress STOP head有梯度，动作与condition链路无梯度；loss组合和四个自动类别
权重均与上述公式一致。

1k pilot已在GPU3后台启动：

```text
tmux socket: /tmp/stage22pilot.sock
session: progress_pilot
log: runs/uav/stage22_step750_progress_stop_sft_pilot_1k.log
```

仍在step250/500/750/1000保存并做100-window快速验证；训练结束后，对候选checkpoint使用与
Stage21完全相同的固定1000-window、58-positive扩大审计。放行不能只看0.5阈值accuracy：要求
ROC-AUC和最佳balanced accuracy明显超过Stage21上限`0.6520/0.6753`，同时优先降低false
positive；若仍需误停数百个non-terminal，则冻结特征路线失败，下一步才进入带动作保持约束的
最小VLA LoRA解冻。
