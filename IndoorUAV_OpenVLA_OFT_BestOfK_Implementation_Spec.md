# Indoor-UAV × OpenVLA-OFT：Best-of-K + Visual Alignment 完整实现规格

> 版本：v1.1（已固定 `stride = 1`；2026-09-16 固定周期 yaw proprio 与对称动作归一化）
> 目标：在尽量保持 OpenVLA-OFT 原始多图像编码路径的前提下，将 Indoor-UAV-VLA 适配为“视觉条件 - 局部动作”多分支闭环规划。
> 推荐输入：`[ref_image, I_(t-1), I_t] + state_t + instruction`。

---

## 0. 最终设计结论

模型在时刻 `t` 生成长度为 `T=5` 的计划。第 0 槽位是确定性的当前动作，未来第 1～4 槽位各有 `K=3` 个绑定的 condition-action 分支：

```text
root:
  A[0,0]

future slot j = 1..T-1:
  (C[j,0], A[j,0])
  (C[j,1], A[j,1])
  (C[j,2], A[j,2])
```

严格时间对齐：

```text
C*_j = I_(t+j)
A*_j = body_delta(p_(t+j), p_(t+j+1))
```

在线执行：

```text
t 时刻固定执行 A[0,0]
到 t+1 后，用真实 I_(t+1) 匹配 C[1,0:K]
选中分支 k* 后执行与其绑定的 A[1,k*]
匹配分数低于阈值或计划耗尽时重新规划
```

未来图像只作为训练标签进入匹配支路，禁止进入主模型输入，防止未来信息泄漏。

---

## 1. 常量、符号与核心张量

| 符号 | 含义 | 默认值 |
|---|---|---:|
| `B` | batch size | 由显存决定 |
| `T` | plan horizon | 5 |
| `K` | 每个未来槽位的候选分支数 | 3 |
| `P` | 每张图像 patch 数 | 256 |
| `D_llm` | LLM hidden size | 4096 |
| `D_match` | condition/vision 匹配维度 | 512 |
| `D_action` | UAV 有效动作维度 | 4 |
| `L_text` | 文本 token 数 | 动态或 padding 后长度 |
| `N_img` | 主模型图像数 | 3 |
| `stride` | 相邻计划槽位间隔 | 固定为 1 |

动作语义：

```text
[delta_forward, delta_right, delta_up, delta_yaw]
```

如果底层 π₀/OpenVLA-OFT 仍保留 `action_dim=32`，则算法层只使用前 4 维：

```text
effective action: [..., 0:4]
padding dims:     [..., 4:32]
```

应对后 28 维显式使用 loss mask，避免 padding 维主导动作损失；如果当前工程必须维持 32 维输出以兼容 checkpoint，不能仅凭零标签假定其影响可以忽略。

---

## 2. Indoor-UAV-VLA 数据适配

### 2.1 已确认的 RLDS step 字段

当前 `rlds_data_all` 是统一的 IndoorUAV-VLA train，约包含：

```text
25,567 episodes
474,340 steps
2,304 shards
```

每个 step 的核心字段：

| 字段 | 语义 |
|---|---|
| `observation.image` | 当前观测 `I_t` |
| `observation.ref_image` | 当前短子轨迹参考图/首帧 `I_ref` |
| `observation.state` | 当前 4-DoF 世界位姿 `[x,y,z,yaw]` |
| `action` | 下一绝对位姿，而非最终所需局部动作 |
| `instruction` | 当前 VLA 短时程导航指令 |

RLDS 单步没有 `I_(t-1)` 字段。必须在同一个 episode 的 step sequence 内构造历史窗口，禁止跨 episode 取帧。

### 2.2 推荐主模型输入

为了保留 Indoor-UAV 的 `ref_image`，同时利用 OpenVLA-OFT 原本的三图槽：

```text
image slot 0 = ref_image
image slot 1 = I_(t-1)
image slot 2 = I_t
```

形状：

```text
model_images:     [B,3,3,H,W]
image_valid_mask: [B,3]
raw_state:        [B,4] = [x,y,z,yaw]
model_state:      [B,5] = [x,y,z,sin(yaw),cos(yaw)]
instruction:      [B,L_text]
```

episode 开头：

| 当前 step | 三个图像槽 | `image_valid_mask` |
|---:|---|---|
| 0 | `[ref_image, I_0, I_0]` | `[1,0,1]` |
| ≥1 | `[ref_image, I_(t-1), I_t]` | `[1,1,1]` |

第二个 `I_0` 只是占位，必须被 attention mask 屏蔽。配置需要：

```python
require_full_image_history = False
```

### 2.3 训练监督窗口

`T=5` 时，一个起点 `t` 需要：

```text
condition images: I_t ... I_(t+4)        [B,T,3,H,W]
poses/states:     p_t ... p_(t+5)        [B,T+1,4]
```

时间表：

| 槽位 `j` | condition target | action target |
|---:|---|---|
| 0 | `I_t`（不计算匹配 loss） | `p_t -> p_(t+1)` |
| 1 | `I_(t+1)` | `p_(t+1) -> p_(t+2)` |
| 2 | `I_(t+2)` | `p_(t+2) -> p_(t+3)` |
| 3 | `I_(t+3)` | `p_(t+3) -> p_(t+4)` |
| 4 | `I_(t+4)` | `p_(t+4) -> p_(t+5)` |

IndoorUAV-VLA 是短子轨迹，末尾可能不足 5 个动作。Dataset 必须返回：

```text
plan_valid_mask: [B,T]
```

不存在的 future slot 可以做数值占位，但其 action、condition、balance、diversity loss 必须全部屏蔽；不能复制末帧后仍作为有效监督。

---

## 3. 阶段 A：Dataset window 与 batch 输出

### 输入

同一 RLDS episode 中以 `t` 为起点的 step sequence。

### 操作

1. 读取 `ref_image`、`I_t`、`state_t`、instruction。
2. 读取同 episode 的 `I_(t-1)`；若不存在则用 `I_0` 占位并置 mask=0。
3. 读取 `I_t...I_(t+4)` 作为 condition label。
4. 读取 `p_t...p_(t+5)` 或使用当前 `state` 与逐步 `action` 恢复位姿序列。
5. 构造 `plan_valid_mask`。

### 输出

```python
{
    "images":                  [B,3,3,H,W],
    "image_valid_mask":        [B,3],
    "raw_state":               [B,4],
    "state":                   [B,5],
    "input_ids":               [B,L_text],
    "text_attention_mask":     [B,L_text],
    "future_condition_images": [B,T,3,H,W],
    "future_poses":            [B,T+1,4],
    "plan_valid_mask":         [B,T],
}
```

说明：`future_condition_images` 和 `future_poses` 是训练标签，不属于在线模型输入。

---

## 4. 阶段 B：世界位姿转机体坐标一步增量

### 输入

```text
future_poses: [B,T+1,4]
```

其中：

```text
p_j = [x_j, y_j, z_j, yaw_j]
```

### 操作

对 `j=0...T-1`：

```text
dx_world = x_(j+1) - x_j
dy_world = y_(j+1) - y_j
delta_up = z_(j+1) - z_j
delta_yaw = wrap_to_pi(yaw_(j+1) - yaw_j)
```

采用与项目坐标定义一致的 `R(-yaw_j)` 把水平世界位移旋转到 UAV 机体坐标：

```text
[delta_forward, delta_right]^T
    = R(-yaw_j) [dx_world, dy_world]^T
```

必须先用 simulator/数据集的 yaw 正方向和坐标轴定义做单元测试；不要在未验证时默认某个右手系公式。

> 2026-09-15 实现验证：原始 IndoorUAV 轨迹与 `online_eval/vla_eval/utils.py`
> 的 Habitat 位姿转换共同确认，`yaw=0` 时 forward 对应世界 `-y`，right 对应世界
> `+x`。实现公式为 `df=sin(yaw)*dx-cos(yaw)*dy`、
> `dr=cos(yaw)*dx+sin(yaw)*dy`，checkpoint contract 必须保存并校验该约定。
> yaw 增大表示右转、减小表示左转；角增量必须用
> `((yaw_next-yaw_current+pi) mod 2pi)-pi` 得到 `[-pi,pi)` 内的最短有符号转角。

### 输出

```text
ground_truth_actions: [B,T,4]
```

归一化固定为每轴对称真实范围：令
`s_d=max(abs(min_d),abs(max_d))`，使用 `[-s_d,+s_d] -> [-1,+1]`。
不能对 body action 沿用 q01/q99：IndoorUAV 的 right 轴横移动作不足 1%，其
q01/q99 约为 `±1.9e-5`，而真实范围约为 `±0.56`，使用 q99 会把有效横移幅度
几乎全部裁成 `-1/+1`。训练统计、loss 空间与在线反归一化必须读取同一 contract。

proprio 必须在完成上述几何转换后才由原始 `[x,y,z,yaw]` 编码为
`[x,y,z,sin(yaw),cos(yaw)]`。前三轴使用 q01/q99，sin/cos 使用精确 `[-1,1]`；
旧 4D proprio projector 与该 5D 表示不兼容，必须重新初始化。

---

## 5. 阶段 C：保留 OpenVLA-OFT 的逐图双视觉编码

三张图像不能作为 18 通道单图送入 backbone。每张图像都独立走原始共享路径。

### 输入

```text
images: [B,3,3,H,W]
```

拆成：

```text
I_ref: [B,3,H,W]
I_prev:[B,3,H,W]
I_cur: [B,3,H,W]
```

### 每张图的共享编码过程

```text
I_n
├─ SigLIP → [B,256,D_siglip]
└─ DINOv2 → [B,256,D_dino]
                ↓ concatenate on feature dim
dual_features_n → [B,256,D_siglip+D_dino]
                ↓ shared original Vision Projector
tokens_n → [B,256,4096]
```

注意两个 concatenate：

1. 同一张图的 SigLIP 与 DINOv2 patch 沿**特征维**拼接；
2. 不同图像投影后的 token 沿**序列维**拼接。

---

## 6. 阶段 D：图像角色编码与历史 mask

原 OpenVLA-OFT 多图通常表达同一时刻的不同相机/视角。现在三个槽位的语义变成 reference、previous、current，因此添加轻量的可学习 role embedding，不引入视频编码器或 Temporal Transformer。

### 输入

```text
ref_tokens:  [B,256,4096]
prev_tokens: [B,256,4096]
cur_tokens:  [B,256,4096]
```

### 参数

```text
E_ref:     [4096]
E_history: [4096]
E_current: [4096]
```

### 操作

```text
ref_tokens  += E_ref
prev_tokens += E_history
cur_tokens  += E_current
```

然后沿 sequence 维拼接：

```text
visual_tokens = concat([ref_tokens, prev_tokens, cur_tokens], dim=1)
visual_tokens: [B,768,4096]
```

将图像 mask 扩展到 patch：

```python
visual_attention_mask = image_valid_mask.repeat_interleave(256, dim=1)
```

```text
image_valid_mask:      [B,3]
visual_attention_mask: [B,768]
```

建议双重保护：

```python
visual_tokens = visual_tokens * visual_attention_mask.unsqueeze(-1)
```

同时把 `visual_attention_mask` 真正拼入 Transformer attention mask。仅返回或打印 mask 不会产生屏蔽效果。

---

## 7. 阶段 E：多模态 Prefix 与 Transformer

### 输入

```text
visual_tokens:         [B,768,4096]
visual_attention_mask: [B,768]
state:                 [B,5]
input_ids:             [B,L_text]
text_attention_mask:   [B,L_text]
```

### Proprio 投影

沿用 OpenVLA-OFT 现有 proprio 路径。若形成 `N_state` 个 token：

```text
state_tokens: [B,N_state,4096]
state_mask:   [B,N_state]
```

### 文本 embedding

```text
text_tokens: [B,L_text,4096]
```

### 拼接

实际顺序必须沿用当前 `modeling_prismatic.py` 的原有约定，不任意交换：

```text
multimodal_prefix: [B,L_total,4096]
prefix_mask:       [B,L_total]

L_total = 768 + N_state + L_text (+ 当前实现中的其他特殊 token)
```

### Transformer 输出

```text
transformer_hidden: [B,L_total,4096]
```

---

## 8. 阶段 F：Condition-Action Planner

计划头从 Transformer 表征生成 condition-action 分支。为兼容现有代码，保持统一 `T×K` 输出；slot 0 的无意义分支通过 mask 忽略。

### 输入

```text
transformer_hidden: [B,L_total,4096]
```

### 输出

```text
condition_hidden:  [B,T,K,4096]
predicted_actions: [B,T,K,4]
```

默认：

```text
condition_hidden:  [B,5,3,4096]
predicted_actions: [B,5,3,4]
```

语义绑定：

```text
(condition_hidden[b,j,k], predicted_actions[b,j,k])
```

必须始终作为同一个 branch 处理。

### Slot 0 规则

```text
只监督并执行 predicted_actions[:,0,0]
不计算 condition[:,0,:] 的匹配 loss
不让 slot 0 branch 1/2 参与 balance/diversity
```

---

## 9. 阶段 G：Condition Projector

condition 来自 LLM hidden state。使用独立投影器映射到 512 维匹配空间：

```text
LayerNorm(4096)
→ Linear(4096,1024)
→ GELU
→ Linear(1024,512)
→ L2 Normalize
```

### 输入/输出

```text
condition_hidden:     [B,T,K,4096]
predicted_conditions: [B,T,K,512]
```

---

## 10. 阶段 H：未来图像标签编码

此分支仅在训练时存在。未来图像不进入主 Transformer。

### 输入

```text
future_condition_images: [B,T,3,H,W]
```

每张未来图像使用与主模型相同的双视觉 backbone 和原始 Vision Projector，可通过 reshape 合并 `B×T` 以提高效率：

```text
[B,T,3,H,W]
→ [B*T,3,H,W]
→ shared SigLIP + DINOv2 + original Vision Projector
→ [B*T,256,4096]
→ [B,T,256,4096]
```

随后使用独立的 Vision Matching Projector：

```text
LayerNorm(4096)
→ Linear(4096,1024)
→ GELU
→ Linear(1024,512)
→ L2 Normalize
```

### 输出

```text
future_patch_embeddings: [B,T,256,512]
```

Condition Projector 和 Vision Matching Projector 结构相同但参数不共享；共享的是可比较的 512 维输出空间。

工程优化：同一个 `I_t` 已在主输入中编码，slot 0 又不计算 condition loss，因此没有必要为 slot 0 重复运行未来图像匹配编码；实现上可以只编码 `I_(t+1)...I_(t+4)`，得到 `[B,T-1,256,512]`。

---

## 11. 阶段 I：Patch-level Top-8 Similarity

### 11.1 输入

未来有效槽位 `j=1...T-1`：

```text
predicted_conditions[:,1:]:    [B,T-1,K,512]
future_patch_embeddings[:,1:]: [B,T-1,256,512]
```

### 11.2 L2 归一化

投影器末端已经归一化；为数值稳健，也可以在计算前再次调用 `F.normalize(..., dim=-1)`。

### 11.3 每个 condition 对 256 个 patch 做点积

```python
patch_similarity = torch.einsum(
    "btkd,btpd->btkp",
    predicted_conditions[:, 1:],
    future_patch_embeddings[:, 1:],
)
```

形状流动：

```text
[B,T-1,K,512] × [B,T-1,256,512]
→ [B,T-1,K,256]
```

`patch_similarity[b,j,k,p]` 表示第 `k` 个预测 condition 与对应真实图像第 `p` 个空间 patch 的余弦相似度。

### 11.4 图内 Top-8 聚合

```python
top8_values, top8_indices = torch.topk(
    patch_similarity, k=8, dim=-1
)
condition_similarity = top8_values.mean(dim=-1)
```

形状：

```text
top8_values:          [B,T-1,K,8]
top8_indices:         [B,T-1,K,8]
condition_similarity: [B,T-1,K]
```

Top-8 的含义：一张室内图像中与导航条件最相关的通常只是门、拐角、障碍物或走廊边缘等局部区域。全 256 patch 平均会被墙面、地板等无关区域稀释；Top-8 用最相关的 8 个局部区域形成该 branch 对整张图的匹配分数。

### 11.5 两级选择不可混淆

```text
第一级：一个 condition 对图像 256 patch，取 Top-8 后得到一个标量分数
第二级：同一时刻的 K 个 branch 分数做 argmax/联合分配
```

Top-8 不是在 3 个 branch 中选 8 个，而是在每个 branch 对应的 256 个图像 patch 中选 8 个。

### 11.6 训练稳定性选项

第一版可固定 `top_patches=8`。如果早期训练不稳定，可在不改变接口的情况下采用：

```text
warm-up: Top-32 或 temperature log-sum-exp pooling
stable:  收缩到 Top-8
```

应记录 `top8_indices` 并可视化到 `16×16` patch 网格，检查高分区域是否落在门、走廊、转角或障碍物等合理位置。

---

## 12. 阶段 J：动作监督的 Condition-Action Winner 分配

### 输入

```text
predicted_actions[:,1:]:     [B,T-1,K,4]
ground_truth_actions[:,1:]:  [B,T-1,4]
condition_similarity:        [B,T-1,K]
future_valid_mask:           [B,T-1]
```

### 动作误差

```python
action_error = smooth_l1(
    predicted_actions[:, 1:],
    ground_truth_actions[:, 1:, None, :],
    reduction="none",
).mean(dim=-1)
```

```text
action_error: [B,T-1,K]
```

### 动作监督 winner

```text
winner_branch = argmin(stop_gradient(action_error), dim=-1)
```

```text
winner_branch: [B,T-1]
```

不要让 `condition_similarity` 参与其自身 K-way 监督标签的产生。否则初始时哪个 condition
偶然更相似，哪个分支就更容易成为 winner，随后再用同一组相似度预测这个 winner，会形成
循环自我标注，matching accuracy 不能反映图像能否找回动作最优分支。旧版
`action_error + lambda_s * (1-condition_similarity)` 联合代价已在 1k pilot v2 后废弃。

同一个 `winner_branch[b,j]` 同时决定：

1. 哪个 condition 分支与真实未来图像对齐；
2. 哪个 action 分支拟合真实局部动作。

不要分别为 condition 和 action 选 winner，否则会破坏 `(C[j,k] -> A[j,k])` 的分支语义。

winner 分配使用 detach 后的动作误差。同一个动作监督 winner 仍同时决定 condition-action
配对，因此不会破坏 `(C[j,k] -> A[j,k])` 的分支语义。

---

## 13. 阶段 K：训练损失

推荐总损失：

```text
L_total =
    lambda_root * L_root_action
  + lambda_act  * L_future_action
  + lambda_K    * L_branch_Kway
  + lambda_T    * L_temporal
  + lambda_Q    * L_cross_episode_queue
  + lambda_bal  * L_balance
  + lambda_div  * L_diversity
```

### 13.1 Root action loss

```text
input:
  predicted_actions[:,0,0] [B,4]
  ground_truth_actions[:,0] [B,4]
  plan_valid_mask[:,0]      [B]

output:
  L_root_action             scalar
```

### 13.2 Future winner action loss

按 `winner_branch` gather：

```text
winner_actions: [B,T-1,4]
```

只在 `plan_valid_mask[:,1:] == 1` 的位置计算 Smooth L1/MSE。

### 13.3 与部署一致的 per-time K-way loss

对每个有效未来时刻，用真实未来图像分别给 K 个 condition 打分，并要求图像选择由动作误差
确定的 `winner_branch`：

```text
scores[b,j,k] = Top8PatchSimilarity(C[b,j,k], I[b,t+j])
L_branch_Kway = CE(scores / temperature, winner_branch)
```

它直接监督部署时“当前观察图像应该执行 K 个配对动作中的哪一个”。K=3 时随机准确率为
`1/3=33.3%`，必须作为独立的 `condition_branch_accuracy/margin` 记录。

### 13.4 时间和跨 episode 对比损失

最小版本可对 winner 使用：

```text
L_positive = 1 - winner_similarity
```

但只有正样本容易发生 embedding collapse。固定批次实验已实际观察到该坍缩：相似度接近
1，而四图交叉熵停在 `ln(4)`。因此正式主线令 `lambda_positive=0`，改用有负样本的
K-way、时间和跨 episode 对比目标。首先保留同一 window 内的时间 InfoNCE：

```text
positive: C[b,j,k*] ↔ I[b,t+j]
negative: 其他 batch 或其他有效时间槽图像
```

Top-8 分数作为 condition-image logit，再除以 temperature。所有负样本也必须服从 `plan_valid_mask`。

由于 batch size 为 1，梯度累积不会把 8 个独立的 `4×4` InfoNCE 变成 `32×32`。因此另外
维护最多 256 张 future image patch embedding 的 device-local FIFO 队列；embedding 入队前
detach，RLDS `episode_metadata.file_path` 作为稳定 episode ID，同一 episode 的图像明确排除。
至少有 32 张其他 episode 图像后才启用队列损失。队列是低权重辅助项，防止不同场景实例
识别压过真正的局部视觉匹配。

### 13.5 Balance loss

只统计 `j=1...T-1` 且有效的 winner，避免所有样本长期坍缩到同一 branch。slot 0 不参与。

### 13.6 Diversity loss

只对未来有效槽位的不同 condition branch 计算。权重应小，避免在确定性场景中强迫模型制造不合理分支。

---

## 14. 完整训练维度流

```text
Indoor-UAV episode window
│
├─ 主输入图像 [B,3,3,H,W]
│  ├─ 每图 SigLIP             → 3 × [B,256,D_siglip]
│  ├─ 每图 DINOv2             → 3 × [B,256,D_dino]
│  ├─ 图内 feature concat     → 3 × [B,256,D_dual]
│  ├─ shared vision projector → 3 × [B,256,4096]
│  ├─ role embeddings         → 3 × [B,256,4096]
│  └─ 图间 sequence concat    → [B,768,4096]
│
├─ image_valid_mask [B,3]
│  └─ repeat 256              → [B,768]
│
├─ state [B,5]
│  └─ proprio projector       → [B,N_state,4096]
│
├─ instruction [B,L_text]
│  └─ text embedding          → [B,L_text,4096]
│
├─ multimodal concat          → [B,L_total,4096]
│  └─ Transformer             → [B,L_total,4096]
│     └─ planner
│        ├─ condition hidden  → [B,T,K,4096]
│        └─ actions           → [B,T,K,4]
│
├─ condition projector        → [B,T,K,512]
│
└─ 训练标签 future images [B,T,3,H,W]
   └─ shared dual vision      → [B,T,256,4096]
      └─ vision match proj.   → [B,T,256,512]
         └─ patch similarity  → [B,T-1,K,256]
            └─ Top-8 mean     → [B,T-1,K]
               └─ detached action error → [B,T-1,K]
                  └─ winner   → [B,T-1]
                     └─ losses→ scalar
```

---

## 15. 在线执行状态机

### 15.1 重新规划

#### 输入

```text
ref_image:        [1,3,H,W]
previous_image:   [1,3,H,W]
current_image:    [1,3,H,W]
image_valid_mask: [1,3]
raw_state:        [1,4]
model_state:      [1,5]
instruction:      [1,L_text]
```

#### 输出

```text
cached_conditions: [1,T,K,512]
cached_actions:    [1,T,K,4]
plan_step:         0
```

固定选择：

```text
selected_action = cached_actions[0,0,0]   # [4]
```

### 15.2 动作反归一化与执行

反归一化得到：

```text
[delta_forward, delta_right, delta_up, delta_yaw]
```

用**真实当前** `yaw_current` 转回世界坐标，而不是使用规划起点旧 yaw：

```text
[dx_world, dy_world]^T = R(yaw_current)[df, dr]^T
target_pose = current_pose ⊕ local_action
```

输出：

```text
target_pose: [4]
```

`stride=1`，因此一个计划动作对应数据/仿真器的一个环境步。

### 15.3 下一时刻在线图像匹配

执行 slot `j-1` 后获得真实 `I_(t+j)`：

```text
current_observation: [1,3,H,W]
```

只需编码这一张当前图：

```text
shared dual vision + vision matching projector
→ current_patches: [1,256,512]
```

取缓存计划的：

```text
slot_conditions = cached_conditions[:,j,:,:] [1,K,512]
```

相似度：

```python
patch_similarity = torch.einsum(
    "bkd,bpd->bkp", slot_conditions, current_patches
)
similarity = patch_similarity.topk(8, dim=-1).values.mean(dim=-1)
```

```text
patch_similarity: [1,K,256]
similarity:       [1,K]
selected_branch: scalar
max_similarity:  scalar
```

### 15.4 继续或重规划

```text
if max_similarity >= threshold:
    selected_action = cached_actions[0,j,selected_branch]
    plan_step += 1
else:
    replan_reason = "condition_mismatch"
    使用最新 [ref_image, I_(now-1), I_now] 完整重规划
```

当 `plan_step >= T`：

```text
replan_reason = "horizon_exhausted"
```

在线输出日志建议：

```python
{
    "plan_step": int,
    "branch": int,
    "condition_similarity": float | None,
    "branch_similarities": [K] | None,
    "top8_patch_indices": [8] | None,
    "replan_reason": str | None,
    "action_shape": [T,K,4],
    "selected_action_body": [4],
    "current_pose": [4],
    "target_pose": [4],
}
```

---

## 16. 在线时序示例（T=5, K=3）

| 时刻 | plan slot | condition 匹配 | 动作 |
|---:|---:|---|---|
| `t` | 0 | 不匹配，固定 branch 0 | `A[0,0]` |
| `t+1` | 1 | `I_(t+1)` 对 `C[1,0:3]` | `A[1,k*]` |
| `t+2` | 2 | `I_(t+2)` 对 `C[2,0:3]` | `A[2,k*]` |
| `t+3` | 3 | `I_(t+3)` 对 `C[3,0:3]` | `A[3,k*]` |
| `t+4` | 4 | `I_(t+4)` 对 `C[4,0:3]` | `A[4,k*]` |
| `t+5` | - | horizon exhausted | 新计划 `A_new[0,0]` |

任意中间时刻低于阈值时立即丢弃剩余计划。

---

## 17. 推荐模型接口

### Dataset/Collator 输出

```python
batch = {
    "pixel_values":             ...,  # 保持当前 OpenVLA-OFT 多图容器格式
    "image_valid_mask":         Tensor[B, 3],
    "proprio":                  Tensor[B, 5],
    "input_ids":                Tensor[B, L_text],
    "text_attention_mask":      Tensor[B, L_text],
    "future_condition_images":  Tensor[B, T, 3, H, W],
    "ground_truth_actions":     Tensor[B, T, 4],
    "plan_valid_mask":          Tensor[B, T],
}
```

不要根据文档假设 `pixel_values` 的内部维度；以当前 processor/collator 的真实多图容器为准。必须保证 forward 中最终是三张图逐张送入共享视觉编码路径。

### 训练 forward 输出

```python
outputs = {
    "condition_hidden":          Tensor[B, T, K, 4096],
    "predicted_conditions":      Tensor[B, T, K, 512],
    "predicted_actions":         Tensor[B, T, K, 4],
    "condition_similarity":      Tensor[B, T-1, K],
    "winner_branch":             LongTensor[B, T-1],
    "top8_patch_indices":        LongTensor[B, T-1, K, 8],
    "loss":                      scalar,
}
```

### 在线 plan cache

```python
plan_cache = {
    "conditions":   Tensor[1, T, K, 512],
    "actions":      Tensor[1, T, K, 4],
    "plan_step":    int,
    "generated_at": int,
}
```

---

## 18. 推荐配置项

```yaml
# data
num_images_in_input: 3
image_roles: [reference, history_1, current]
require_full_image_history: false
plan_horizon: 5
stride: 1
action_dim_effective: 4
action_frame: body
proprio_representation: xyz_sin_yaw_cos_yaw_v1
action_normalization: per_axis_symmetric_minmax_v1

# branches
num_branches: 3
root_branch: 0
root_condition_loss: false

# matching
match_dim: 512
top_patches: 8
match_temperature: 0.07
condition_threshold: null  # 训练后在验证集标定，不能先拍脑袋固定

# losses
lambda_root: 1.0
lambda_action: 1.0
lambda_branch_Kway: 1.0
lambda_temporal: 0.25
lambda_cross_episode_queue: 0.05
lambda_positive: 0.0
lambda_balance: 0.01
lambda_diversity: 0.005
lambda_assignment_similarity: 0.0
queue_size_images: 256
queue_min_other_episode_negatives: 32
```

阈值应在验证集上根据“正确分支匹配分数”和“应重规划样本分数”的分布标定。现有用 `1.1` 的设置只适用于强制每步重规划的对照实验，不是部署阈值。

验证汇总不能按 batch 直接平均带 mask 的零值。root action 按有效 root 数、future action 与
K-way 指标按有效 future slot 数、temporal 指标按实际参与时间检索的 query 数、queue 指标
按满足最小跨 episode 负样本条件的 query 数分别加权。完全没有 future label 的终点窗口不
参与 condition accuracy/margin；同时记录实际候选数对应的随机 accuracy 作为基线。

---

## 19. 代码修改清单

### 数据层

- 在同一 RLDS episode 内构造 `[ref_image, I_(t-1), I_t]`。
- 设置 `require_full_image_history=False`。
- 从绝对下一位姿生成相邻一步机体坐标动作。
- 生成 `future_condition_images`、`ground_truth_actions`、`plan_valid_mask`。
- 确保 window 不跨 episode。

### Collator

- 保留三图原始容器形式。
- 保留 `image_valid_mask [B,3]`。
- 对可变长度 future window 正确 padding，并同步生成 `plan_valid_mask`。

### `modeling_prismatic.py`

- 保留“逐图双视觉编码 -> 图内特征拼接 -> shared projector -> 图间 token 拼接”。
- 添加 `E_ref/E_history/E_current`。
- 把 `[B,3]` 扩展成 `[B,768]` 并真正并入 attention mask。
- 无效视觉 token 同时置零。

### Planner/Action Head

- 输出 `[B,T,K,4]` 局部动作。
- 输出 `[B,T,K,4096]` condition hidden。
- slot 0 只启用 branch 0。

### Matching Head

- 添加独立 Condition Projector。
- 添加独立 Vision Matching Projector。
- 实现 `[B,T-1,K,256] -> Top-8 -> [B,T-1,K]`。
- 保存 Top-8 patch index 用于调试可视化。

### Loss

- 实现动作误差 winner，并用同一索引绑定 condition-action。
- 所有 future loss 使用 `plan_valid_mask`。
- slot 0 不进入 condition/balance/diversity。
- 若底层仍为 32 维 action，增加有效维度 mask。

### 在线推理

- 缓存 condition-action plan。
- 每步只对最新当前图运行匹配视觉支路。
- 使用当前真实 yaw 执行机体动作。
- 实现 `condition_mismatch` 与 `horizon_exhausted` 重规划。

---

## 20. 必须先通过的单元测试

1. **Episode 边界测试**：`I_(t-1)` 和 future window 不跨 episode。
2. **历史 mask 测试**：`[1,0,1]` 精确扩展为中间 256 个 token 为 0。
3. **Attention 测试**：修改被 mask 占位图像的像素，不应改变模型有效输出（允许浮点微差）。
4. **图像顺序测试**：交换 previous/current 应改变 role embedding 后的 token。
5. **坐标测试**：在 yaw 为 0、±π/2 的合成轨迹上验证 forward/right 正负号。
6. **yaw wrap 测试**：从 `π-ε` 到 `-π+ε` 应得到小角度增量。
7. **时间对齐测试**：slot `j` 的 condition 是 `I_(t+j)`，action 是 `p_(t+j)->p_(t+j+1)`。
8. **Top-8 形状测试**：`[B,T-1,K,256] -> [B,T-1,K,8] -> [B,T-1,K]`。
9. **Winner 绑定测试**：condition 与 action 必须 gather 同一个 branch index。
10. **短轨迹 mask 测试**：无效 future slot 对总 loss 和梯度贡献为 0。
11. **未来泄漏测试**：删除 future label 张量后，在线 plan forward 仍可独立运行。
12. **在线步序测试**：执行 slot 0 后必须匹配 slot 1，不得再次匹配 slot 0。

---

## 21. 推荐训练顺序

### Phase 1：数据与确定性 root 验证

```text
目标：确认历史窗口、role embedding、attention mask、机体动作全部正确。
训练：先重点观察 slot 0 action loss。
```

### Phase 2：加入未来 K 分支与动作监督 winner

```text
目标：未来 action 能学习，winner 分布不过早坍缩。
训练：先只加入 future action，不启用 positive-only condition loss。
```

### Phase 3：加入完整对比匹配

```text
目标：condition 能区分对应未来图像与负样本。
训练：per-time K-way 为主，同窗口时间 InfoNCE 和跨 episode 队列为辅，再加 balance 与小权重 diversity。
```

### Phase 4：在线闭环与阈值标定

```text
记录正确继续、错误继续、正确重规划、错误重规划的 similarity 分布；
在 validation episodes 上选择 threshold；
最后与“每一步完整 VLA 重规划”对照实验比较 SR、SPL/nDTW、模型调用次数和碰撞情况。
```

---

## 22. 一句话实现定义

```text
使用 Indoor-UAV 的 ref_image、前一帧、当前帧，沿用 OpenVLA-OFT 的逐图共享双视觉编码和 token 拼接；通过 role embedding 与真实 attention mask 表达图像身份；模型输出多步多分支 condition-action plan；每个 condition 与真实当前图像的 256 个 patch 做余弦匹配并以 Top-8 聚合；动作误差确定配对 winner，per-time K-way、时间负样本和跨 episode 队列共同训练视觉匹配；匹配失败或计划耗尽时重规划。
```
