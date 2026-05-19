# GRPO Implementation Technical Details

## 核心架构设计

### 1. 参考模型 (Reference Model) 处理

**问题**: 需要保留原始模型 π_ref 来计算 KL 散度，但不能占用双倍显存。

**解决方案**: 使用 PEFT 的 `disable_adapter()` 上下文管理器

```python
# 当前策略 (with LoRA)
self.model.llm.set_adapter('t2m')
outputs_policy = self.model.llm(input_ids, ...)

# 参考策略 (without LoRA)
with self.model.llm.disable_adapter():
    outputs_ref = self.model.llm(input_ids, ...)
```

**优势**:
- ✓ 不需要额外内存加载第二个模型
- ✓ 自动切换：`disable_adapter()` 临时禁用 LoRA，使用基座权重
- ✓ 退出上下文后自动恢复 LoRA

**显存成本**: ~0 额外成本（只是计算时禁用 adapter）

---

### 2. 组采样 (Group Sampling) 实现

**流程**: 对每个 caption 生成 G=4 个不同的 motion 序列

```python
def group_sample(self, captions, num_samples=4):
    all_sampled_motions = []  # [B][G]
    
    with torch.no_grad():  # 采样阶段不计算梯度
        for caption in captions:
            caption_samples = []
            for _ in range(num_samples):
                motion_tokens = self.generate_with_sampling(
                    caption,
                    temperature=1.0,  # 控制多样性
                    do_sample=True,   # 启用采样
                    top_p=0.9         # Nucleus sampling
                )
                caption_samples.append(motion_tokens)
            all_sampled_motions.append(caption_samples)
    
    return all_sampled_motions
```

**关键参数**:
- `temperature=1.0`: 较高 → 更多样化，较低 → 更确定性
- `do_sample=True`: 使用概率采样而非贪婪/beam search
- `top_p=0.9`: 只从累积概率前 90% 的 tokens 中采样

---

### 3. 优势值 (Advantage) 计算

**标准化策略**: 组内标准化（group-wise normalization）

```python
# 对每个 caption 的 G 个样本
rewards = reward_model.compute_reward([caption]*G, motions)  # [G]

# 组内归一化
mean_reward = rewards.mean()
std_reward = rewards.std() + 1e-8  # 防止除零
advantages = (rewards - mean_reward) / std_reward  # [G]
```

**效果**:
- 将 rewards 转换为相对优势（relative advantage）
- 高于平均的样本 → 正 advantage (鼓励)
- 低于平均的样本 → 负 advantage (抑制)
- 组内标准化消除绝对 reward scale 的影响

---

### 4. GRPO 损失函数详解

**核心公式**:
```
L_GRPO = -E[min(ratio * A, clip(ratio) * A)] + β * KL(π_ref || π_θ)

where:
  ratio = exp(log π_θ(a|s) - log π_ref(a|s))
  A = advantage (normalized reward)
  clip(ratio) = clamp(ratio, 1/c, c)  # c = 10.0
  KL ≈ ratio * log(ratio) - (ratio - 1)
```

**代码实现**:
```python
def compute_grpo_loss(self, caption, sampled_motions, advantages):
    total_loss = 0.0
    
    for i, motion_tokens in enumerate(sampled_motions):
        # 1. 获取当前策略的 log-probs
        log_probs_policy, _ = self.compute_log_probs(
            caption, motion_tokens, use_ref_model=False
        )
        
        # 2. 获取参考策略的 log-probs (冻结)
        with torch.no_grad():
            log_probs_ref, _ = self.compute_log_probs(
                caption, motion_tokens, use_ref_model=True
            )
        
        # 3. 计算重要性比率 (importance ratio)
        log_ratio = log_probs_policy - log_probs_ref
        ratio = torch.exp(log_ratio)
        
        # 4. Clip ratio 防止过大更新
        ratio_clipped = torch.clamp(
            ratio, 
            1.0 / self.args.grpo_clip_ratio,  # 下界 0.1
            self.args.grpo_clip_ratio         # 上界 10.0
        )
        
        # 5. KL 散度 (近似)
        kl_div = (ratio * log_ratio - (ratio - 1)).mean()
        
        # 6. 策略损失 + KL 惩罚
        policy_loss = -(ratio_clipped * advantages[i]).mean()
        kl_penalty = self.args.grpo_beta * kl_div  # β = 0.01
        
        sample_loss = policy_loss + kl_penalty
        total_loss += sample_loss / len(sampled_motions)  # 平均
    
    return total_loss
```

---

### 5. 显存优化策略

**问题**: 
- Batch size B=4
- 每个 caption 生成 G=4 个样本
- 总共 B×G=16 个序列需要反向传播
- 显存不足以同时处理 16 个序列

**解决方案**: 三层梯度累积

```python
def train_step(self, batch):
    # Layer 1: Batch-level accumulation (over B captions)
    self.optimizer.zero_grad()
    
    for caption_idx in range(batch_size):  # B=4
        caption = captions[caption_idx]
        sampled_motions = sampled_motions_all[caption_idx]  # [G=4]
        
        # Layer 2: Group-level accumulation (over G samples)
        loss = self.compute_grpo_loss(caption, sampled_motions, advantages)
        
        # Layer 3: Sequence-level (inside compute_grpo_loss)
        # Process one motion sequence at a time
        
        # Scale loss for gradient accumulation
        loss = loss / batch_size
        loss.backward()  # 累积梯度
    
    # 一次性更新所有累积的梯度
    self.optimizer.step()
```

**显存占用分析**:
```
传统做法 (全部 forward): 16 sequences × seq_len × hidden_dim
我们的做法 (逐个处理):  1 sequence  × seq_len × hidden_dim

显存节省: ~16x (仅需处理 1/16 的 activations)
```

**代价**: 
- 训练速度变慢（16 次串行 forward/backward vs 1 次并行）
- 但可以在有限显存下训练，这是必要的权衡

---

### 6. LoRA 切换机制详解

**PEFT 库的 adapter 管理**:

```python
# 初始化时添加两个 adapters
self.llm = get_peft_model(self.llm, lora_config_t2m, adapter_name='t2m')
self.llm.add_adapter('m2t', lora_config_m2t)

# 训练 t2m 任务
self.llm.set_adapter('t2m')  # 激活 t2m adapter
outputs = self.llm(...)       # 使用 base + t2m LoRA

# 获取参考模型输出
with self.llm.disable_adapter():  # 临时禁用所有 adapters
    outputs_ref = self.llm(...)   # 仅使用 base model

# 自动恢复
outputs = self.llm(...)  # 又回到 base + t2m LoRA
```

**底层原理**:
- LoRA: `h = W_base @ x + (W_down @ W_up) @ x`
- `disable_adapter()`: 临时让第二项为 0
- 不改变参数，只改变计算图

---

### 7. 学习率调度

**Cosine with Warmup**:

```python
def get_lr(self):
    if step < warmup_steps:  # 线性 warmup
        return lr_max * (step / warmup_steps)
    else:  # Cosine decay
        progress = (step - warmup_steps) / max_steps
        return lr_max * 0.5 * (1.0 + cos(π * progress))
```

**为什么需要 warmup**:
- GRPO 在训练初期不稳定（reward signal 噪声大）
- Warmup 让模型逐渐适应 RL 训练
- 避免初期大步长破坏 SFT 预训练的知识

---

## 关键超参数建议

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_samples_per_prompt` | 4 | G 值，越大越稳定但越慢 |
| `grpo_beta` | 0.01 | KL 惩罚系数，越大越保守 |
| `grpo_clip_ratio` | 10.0 | Ratio clipping，防止崩溃 |
| `learning_rate` | 1e-5 | RL 通常比 SFT 更小 |
| `temperature` | 1.0 | 采样温度，控制多样性 |
| `batch_size` | 4 | 显存限制下尽量小 |
| `reward_scale` | 1.0 | Reward 缩放 |
| `max_grad_norm` | 1.0 | 梯度裁剪，防止爆炸 |

---

## 与标准 PPO 的区别

| | PPO | GRPO (本实现) |
|---|-----|---------------|
| 优势估计 | GAE (Generalized Advantage Estimation) | 组内标准化 |
| Clipping | Ratio clipping | Ratio clipping |
| Value network | 需要单独的 critic | 不需要 (用组内平均代替 baseline) |
| On-policy | 是 | 是 |
| KL 惩罚 | 可选 | 必须 (β * KL) |
| 显存效率 | 需要 2 个模型 | 仅需 1 个模型 + adapter 切换 |

---

## 潜在问题与解决方案

### 问题 1: Reward Hacking
**现象**: 模型学会欺骗 reward model（生成高分但质量差的动作）

**解决**:
- 定期在真实评估集上验证 (FID, R-Precision)
- 监控 KL 散度（如果 KL 太大说明偏离太多）
- 增加 reward 的多样性（加入平滑度、物理合理性等）

### 问题 2: 训练不稳定
**现象**: Loss 震荡，reward 不增反降

**解决**:
- 减小 learning rate
- 增加 warmup steps
- 增大 KL penalty (beta)
- 检查 advantage 计算是否正确

### 问题 3: 显存溢出
**现象**: OOM (Out of Memory)

**解决**:
- 减小 batch_size
- 减小 num_samples_per_prompt (G)
- 使用更小的 LLM backbone
- 启用 gradient checkpointing

---

## 训练流程总结

```
1. Load SFT checkpoint (warmstart)
   ↓
2. For each epoch:
   ↓
3. For each batch (B captions):
   ↓
4. Sample G motions per caption (no grad)
   ↓
5. Compute rewards for all B×G samples
   ↓
6. Compute advantages (group-wise normalization)
   ↓
7. For each caption:
      For each of G samples:
         - Compute log_probs (policy)
         - Compute log_probs (ref, no grad)
         - Compute ratio, KL, loss
         - Backward (accumulate gradients)
   ↓
8. Clip gradients & optimizer step
   ↓
9. Validate periodically
```

---

## 使用方法

```bash
# 基础训练 (从 SFT checkpoint 开始)
python train_grpo.py \
  --sft-checkpoint ckpt/motionllm_t2m_best.pth \
  --exp-name grpo_baseline \
  --num-samples-per-prompt 4 \
  --batch-size 4 \
  --learning-rate 1e-5 \
  --epochs 100

# 高级配置 (调整超参数)
python train_grpo.py \
  --sft-checkpoint ckpt/motionllm_t2m_best.pth \
  --exp-name grpo_tuned \
  --num-samples-per-prompt 8 \
  --grpo-beta 0.02 \
  --temperature 1.2 \
  --reward-scale 2.0 \
  --batch-size 2
```
