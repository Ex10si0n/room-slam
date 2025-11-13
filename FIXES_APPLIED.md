# Room-SLAM 修复总结 (Rasterize版本)

## 问题诊断

模型对不同房间的trace输出相似的布局，不能正确学习trace-dependent的预测。

### 根本原因

1. **坐标归一化被禁用** ⚠️ (最严重)
   - 模型学习绝对坐标而非相对位置
   - 不同房间的trace在不同坐标系下
   - 导致模型只能输出"平均布局"

2. **Diversity Loss太弱**
   - 权重只有0.1，几乎没有作用
   - 计算方式不够强
   - 无法有效促进trace-dependent预测

---

## 已应用的修复

### ✅ 修复1: 启用坐标归一化 (`src/rasterize/model.py:73-81`)

**修改前:**
```python
mean = torch.zeros(B, 1, 2, device=traces.device, dtype=traces.dtype)  # 硬编码为0
scale = torch.ones(B, 1, 1, device=traces.device, dtype=traces.dtype)  # 硬编码为1
```

**修改后:**
```python
# Compute trace-specific mean and scale for normalization (2D version)
# This ensures model learns relative positions, not absolute coordinates
valid = mask if mask is not None else torch.ones((B, N), dtype=torch.bool, device=traces.device)
denom = valid.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1)

mean = (coords * valid.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom  # [B, 1, 2]
centered = (coords - mean) * valid.unsqueeze(-1)
rms = torch.sqrt((centered ** 2).sum(dim=(1, 2), keepdim=True) / denom[..., :1]).clamp_min(1e-3)
scale = rms  # [B, 1, 1]
```

**效果:**
- ✅ 每个trace序列被归一化到自己的坐标系
- ✅ 模型学习相对位置关系，而非绝对坐标
- ✅ 不同大小的房间被归一化到相同尺度

---

### ✅ 修复2: 增强Diversity Loss (`src/rasterize/train.py:240-257`)

**修改前:**
```python
if pred_boxes.shape[0] > 1:
    pred_var = pred_boxes.var(dim=0).mean()
    diversity_loss = -0.01 * pred_var.clamp(max=1.0)  # 太弱！
    losses['diversity_loss'] = diversity_loss
```

**修改后:**
```python
if pred_boxes.shape[0] > 1:
    # 1. Box variance across batch
    box_var = pred_boxes.var(dim=0).mean()

    # 2. Class prediction variance
    class_probs = F.softmax(pred_classes, dim=-1)
    class_var = class_probs.var(dim=0).mean()

    # 3. Penalize if variance is TOO LOW (encourage trace-dependent predictions)
    # We want high variance, so we penalize when variance < threshold
    target_box_variance = 0.5  # Target minimum variance for boxes
    target_class_variance = 0.3  # Target minimum variance for classes
    diversity_loss = F.relu(target_box_variance - box_var) + F.relu(target_class_variance - class_var)
    losses['diversity_loss'] = diversity_loss
```

**改进:**
- ✅ 同时考虑box和class的variance
- ✅ 使用ReLU惩罚过低的variance
- ✅ 设置明确的variance目标阈值
- ✅ 更强的正则化效果

---

### ✅ 修复3: 调整Loss权重 (`src/rasterize/train.py:640`)

**修改前:**
```python
'diversity_loss': 0.1  # 太小
```

**修改后:**
```python
'diversity_loss': 1.0  # 增加10倍权重
```

**效果:**
- ✅ Diversity loss权重从0.1增加到1.0
- ✅ 与其他loss权重相当（class_loss=2.0, giou_loss=2.0）
- ✅ 强制模型生成trace-dependent的预测

---

## 预期效果

修复后的模型应该：

1. **学习相对位置关系**
   - 房间A: trace在x=0-5 → 预测相对于trace的物体位置
   - 房间B: trace在x=10-15 → 预测相对于trace的物体位置
   - ✅ 而非都输出x=2.5处有物体

2. **对不同trace产生不同预测**
   - 横向移动的trace → 预测横向排列的家具
   - 纵向移动的trace → 预测纵向排列的家具
   - ✅ 而非都输出相同的平均布局

3. **训练loss变化**
   - Diversity loss应该在早期很高（模型输出相似）
   - 随训练逐渐降低（模型学会区分不同trace）
   - 最终稳定在较低水平

---

## 如何验证修复

### 方法1: 训练新模型

```bash
cd /home/user/room-slam/src/rasterize
python train.py --config <your_config> --epochs 50
```

**观察指标:**
- 训练初期diversity_loss应该较高（>0.3）
- mIoU应该逐渐提升
- 不同房间的预测应该明显不同

### 方法2: 可视化预测

```bash
python inference.py --checkpoint <best_model.pth> --visualize
```

**检查:**
- 对不同房间的trace，预测的布局应该不同
- 预测的物体位置应该与trace路径相关
- 不应该所有房间都预测相似的平均布局

### 方法3: 定量对比

在验证集上比较：
- **修复前**: 不同房间的预测L1距离应该很小（<1.0）
- **修复后**: 不同房间的预测L1距离应该较大（>2.0）

---

## 重要提醒

### ⚠️ 需要重新训练

**这些修复改变了模型的输入归一化方式**，旧的checkpoint不兼容！

必须：
1. 删除旧的checkpoint
2. 从头开始训练新模型
3. 不要尝试加载旧模型权重

### 💡 训练建议

1. **学习率**: 保持lr=1e-4
2. **Batch size**: 建议>=16（diversity loss需要较大batch）
3. **监控**: 重点关注diversity_loss的下降趋势
4. **Early stopping**: 如果diversity_loss长期不下降，可能需要调整target_variance

### 🔧 可选的进一步调优

如果效果还不够好，可以尝试：

1. **增加alignment loss权重**:
   ```python
   'coverage_loss': 5.0,   # 从默认值增加
   'avoidance_loss': 10.0  # 从默认值增加
   ```

2. **调整diversity目标**:
   ```python
   target_box_variance = 0.8   # 要求更大的差异
   target_class_variance = 0.5
   ```

3. **增加数据增强**:
   - 确保dataloader中augmentation已启用
   - 增加rotation/translation/scale的变化范围

---

## 文件修改清单

### 已修改文件:

1. ✅ `src/rasterize/model.py`
   - Line 73-81: 启用坐标归一化

2. ✅ `src/rasterize/train.py`
   - Line 240-257: 增强diversity loss
   - Line 640: 增加diversity loss权重到1.0

### 未修改文件:

- `src/benchmark/` - 按要求只修改rasterize版本
- 其他文件保持不变

---

## 理论解释

### 为什么归一化如此重要？

**没有归一化的问题:**
```
房间A trace: x ∈ [0, 5], 学习 → "x=2.5处有桌子"
房间B trace: x ∈ [10, 15], 学习 → "x=12.5处有桌子"
测试房间C: x ∈ [20, 25] → 模型输出 x=7.5 (两者平均) ❌
```

**有归一化后:**
```
所有trace: 归一化到 [-1, 1], 学习 → "trace中心偏左0.2处有桌子"
测试任何房间: 归一化 → 正确预测相对位置 ✓
```

### 为什么需要Diversity Loss？

**没有diversity loss:**
- 模型倾向于学习数据集的平均布局
- 对所有输入输出相似的预测（最小化平均误差）
- 类似于mode collapse

**有diversity loss:**
- 强制模型对不同输入产生不同输出
- 鼓励模型真正使用trace信息
- 学习trace-layout的真实映射关系

---

## 技术细节

### 归一化公式 (2D版本)

```python
# 对于2D坐标 (x, z):
mean = Σ(coords * mask) / Σ(mask)                    # [B, 1, 2]
centered = coords - mean                              # [B, N, 2]
rms = sqrt(Σ(centered^2) / N)                        # [B, 1, 1]
scale = rms

# 归一化:
normalized_coords = (coords - mean) / scale
# 逆归一化 (decoder中):
original_coords = normalized_coords * scale + mean
```

### Diversity Loss公式

```python
# Box diversity:
box_var = Var_batch(pred_boxes).mean()
loss_box = ReLU(0.5 - box_var)

# Class diversity:
class_var = Var_batch(softmax(pred_classes)).mean()
loss_class = ReLU(0.3 - class_var)

# Total:
diversity_loss = loss_box + loss_class
```

---

## 预期训练曲线

```
Epoch | Total Loss | Diversity Loss | mIoU  | Notes
------|-----------|---------------|-------|------------------
1     | 15.2      | 0.45          | 0.12  | 高diversity，低IoU
10    | 8.5       | 0.28          | 0.31  | diversity下降
30    | 5.1       | 0.15          | 0.52  | 开始收敛
50    | 3.8       | 0.08          | 0.64  | 接近最优
100   | 3.2       | 0.05          | 0.68  | 可能过拟合
```

---

## 问题排查

### 如果diversity loss不下降：

1. **检查batch size**: 需要>=8才能有效计算variance
2. **检查归一化**: 确认mean/scale确实在变化
3. **降低目标阈值**: target_variance可能设置太高

### 如果mIoU没有提升：

1. **检查其他loss**: class_loss, l1_loss, giou_loss应该正常下降
2. **可视化预测**: 看看预测是否合理
3. **检查数据**: 确认trace和collider配对正确

### 如果预测还是相似：

1. **增加diversity权重**: 从1.0增加到2.0
2. **增加alignment权重**: coverage/avoidance权重增加
3. **检查encoder**: 确保trace features确实不同

---

生成时间: 2025-11-13
修复版本: rasterize only
状态: ✅ 已完成
