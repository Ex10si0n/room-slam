# Room-SLAM Fix Summary (Rasterize Version)

## Problem Diagnosis

The model outputs similar layouts for different room traces and cannot learn trace-dependent predictions correctly.

### Root Causes

1. **Coordinate Normalization Disabled** ⚠️ (Most Critical)
   - Model learns absolute coordinates instead of relative positions
   - Different room traces exist in different coordinate systems
   - Results in model only outputting "average layouts"

2. **Diversity Loss Too Weak**
   - Weight is only 0.1, almost no effect
   - Computation method not strong enough
   - Cannot effectively promote trace-dependent predictions

---

## Applied Fixes

### ✅ Fix 1: Enable Coordinate Normalization (`src/rasterize/model.py:73-81`)

**Before:**
```python
mean = torch.zeros(B, 1, 2, device=traces.device, dtype=traces.dtype)  # Hardcoded to 0
scale = torch.ones(B, 1, 1, device=traces.device, dtype=traces.dtype)  # Hardcoded to 1
```

**After:**
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

**Effect:**
- ✅ Each trace sequence is normalized to its own coordinate system
- ✅ Model learns relative position relationships instead of absolute coordinates
- ✅ Different sized rooms are normalized to the same scale

---

### ✅ Fix 2: Enhanced Diversity Loss (`src/rasterize/train.py:240-257`)

**Before:**
```python
if pred_boxes.shape[0] > 1:
    pred_var = pred_boxes.var(dim=0).mean()
    diversity_loss = -0.01 * pred_var.clamp(max=1.0)  # Too weak!
    losses['diversity_loss'] = diversity_loss
```

**After:**
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

**Improvements:**
- ✅ Considers both box and class variance
- ✅ Uses ReLU penalty for variance below target thresholds
- ✅ Sets explicit variance target thresholds
- ✅ Stronger regularization effect

---

### ✅ Fix 3: Adjust Loss Weights (`src/rasterize/train.py:640`)

**Before:**
```python
'diversity_loss': 0.1  # Too small
```

**After:**
```python
'diversity_loss': 1.0  # Increase 10x
```

**Effect:**
- ✅ Diversity loss weight increased from 0.1 to 1.0
- ✅ Comparable to other loss weights (class_loss=2.0, giou_loss=2.0)
- ✅ Forces model to generate trace-dependent predictions

---

## Expected Effects

After the fixes, the model should:

1. **Learn Relative Position Relationships**
   - Room A: trace at x=0-5 → predicts object positions relative to trace
   - Room B: trace at x=10-15 → predicts object positions relative to trace
   - ✅ Instead of always outputting objects at x=2.5

2. **Generate Different Predictions for Different Traces**
   - Horizontal movement trace → predicts horizontally arranged furniture
   - Vertical movement trace → predicts vertically arranged furniture
   - ✅ Instead of always outputting the same average layout

3. **Training Loss Changes**
   - Diversity loss should be high early (model outputs similar predictions)
   - Gradually decreases during training (model learns to distinguish different traces)
   - Eventually stabilizes at lower level

---

## Verification Methods

### Method 1: Train New Model

```bash
cd /home/user/room-slam/src/rasterize
python train.py --config <your_config> --epochs 50
```

**Observe Metrics:**
- Diversity_loss should be high early in training (>0.3)
- mIoU should gradually improve
- Predictions for different rooms should be significantly different

### Method 2: Visualize Predictions

```bash
python inference.py --checkpoint <best_model.pth> --visualize
```

**Check:**
- Predictions for different room traces should differ
- Predicted object positions should correlate with trace paths
- Not all rooms should predict similar average layouts

### Method 3: Quantitative Comparison

Compare on validation set:
- **Before fix**: L1 distance between predictions for different rooms should be small (<1.0)
- **After fix**: L1 distance between predictions for different rooms should be large (>2.0)

---

## Important Reminders

### ⚠️ Requires Retraining

**These fixes change the model's input normalization method**, old checkpoints are incompatible!

Must:
1. Delete old checkpoints
2. Train new model from scratch
3. Do not try to load old model weights

### 💡 Training Recommendations

1. **Learning rate**: Keep lr=1e-4
2. **Batch size**: Recommend >=16 (diversity loss needs larger batch)
3. **Monitoring**: Focus on diversity_loss decrease trend
4. **Early stopping**: If diversity_loss doesn't decrease for long time, may need to adjust target_variance

### 🔧 Optional Further Tuning

If results are still insufficient, try:

1. **Increase alignment loss weights**:
   ```python
   'coverage_loss': 5.0,   # Increase from default
   'avoidance_loss': 10.0  # Increase from default
   ```

2. **Adjust diversity targets**:
   ```python
   target_box_variance = 0.8   # Require larger differences
   target_class_variance = 0.5
   ```

3. **Increase data augmentation**:
   - Ensure dataloader augmentation is enabled
   - Increase range of rotation/translation/scale variations

---

## Modified Files List

### Modified Files:

1. ✅ `src/rasterize/model.py`
   - Line 73-81: Enable coordinate normalization

2. ✅ `src/rasterize/train.py`
   - Line 240-257: Enhanced diversity loss
   - Line 640: Increase diversity loss weight to 1.0

### Unmodified Files:

- `src/benchmark/` - Only rasterize version modified as requested
- Other files remain unchanged

---

## Theoretical Explanation

### Why is Normalization So Important?

**Problem Without Normalization:**
```
Room A trace: x ∈ [0, 5], learns → "desk at x=2.5"
Room B trace: x ∈ [10, 15], learns → "desk at x=12.5"
Test Room C: x ∈ [20, 25] → model outputs x=7.5 (average of both) ❌
```

**With Normalization:**
```
All traces: normalized to [-1, 1], learns → "desk at 0.2 left of trace center"
Test any room: normalized → correctly predicts relative position ✓
```

### Why Diversity Loss is Needed?

**Without Diversity Loss:**
- Model tends to learn dataset's average layout
- Outputs similar predictions for all inputs (minimize average error)
- Similar to mode collapse

**With Diversity Loss:**
- Forces model to produce different outputs for different inputs
- Encourages model to truly use trace information
- Learns real trace-layout mapping relationship

---

## Technical Details

### Normalization Formula (2D Version)

```python
# For 2D coordinates (x, z):
mean = Σ(coords * mask) / Σ(mask)                    # [B, 1, 2]
centered = coords - mean                              # [B, N, 2]
rms = sqrt(Σ(centered^2) / N)                        # [B, 1, 1]
scale = rms

# Normalization:
normalized_coords = (coords - mean) / scale
# Denormalization (in decoder):
original_coords = normalized_coords * scale + mean
```

### Diversity Loss Formula

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

## Expected Training Curves

```
Epoch | Total Loss | Diversity Loss | mIoU  | Notes
------|-----------|---------------|-------|------------------
1     | 15.2      | 0.45          | 0.12  | High diversity, low IoU
10    | 8.5       | 0.28          | 0.31  | Diversity decreasing
30    | 5.1       | 0.15          | 0.52  | Starting to converge
50    | 3.8       | 0.08          | 0.64  | Near optimal
100   | 3.2       | 0.05          | 0.68  | Possible overfitting
```

---

## Troubleshooting

### If diversity loss doesn't decrease:

1. **Check batch size**: Need >=8 to effectively compute variance
2. **Check normalization**: Confirm mean/scale are actually changing
3. **Lower target threshold**: target_variance may be set too high

### If mIoU doesn't improve:

1. **Check other losses**: class_loss, l1_loss, giou_loss should decrease normally
2. **Visualize predictions**: See if predictions are reasonable
3. **Check data**: Confirm trace and collider pairing is correct

### If predictions are still similar:

1. **Increase diversity weight**: From 1.0 to 2.0
2. **Increase alignment weights**: Increase coverage/avoidance weights
3. **Check encoder**: Ensure trace features are actually different

---

## Summary

This fix addresses the critical issue where the model outputs similar layouts for different rooms by:

1. **Enabling proper coordinate normalization** - Forces model to learn relative positions
2. **Strengthening diversity loss** - Prevents mode collapse and encourages trace-dependent predictions
3. **Adjusting loss weights** - Gives diversity loss sufficient influence in training

**Breaking Change:** Models must be retrained from scratch due to normalization changes.

**Expected Result:** Model will produce significantly different layouts for different trace patterns, learning the true relationship between human movement and room layout.

---

Generated: 2025-11-13
Fix Version: Rasterize only
Status: ✅ Complete
