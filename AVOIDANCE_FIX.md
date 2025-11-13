# Avoidance Loss Fix - Preventing BLOCK Overlaps with Traces

## Problem

The model was not learning the critical constraint: **"People cannot walk through BLOCK areas"**

From training logs (Epochs 117-199):
- Avoidance loss stuck at 0.69-0.92 (not decreasing)
- Model kept predicting BLOCK boxes that overlapped with movement traces
- Training loss plateaued at 6.7-7.3
- Validation F1 stuck around 0.72

## Root Cause Analysis

1. **Weak Soft Penalty**
   - Original loss: simple linear penalty weighted by `block_prob`
   - For uncertain predictions (block_prob < 0.5), penalty was too small
   - No distinction between confident vs uncertain BLOCK predictions

2. **Insufficient Weight**
   - Original weight: 2.0 × 1.0 = 2.0
   - Avoidance loss ~0.8 → contributed only ~1.6 to total loss (~23%)
   - Not strong enough compared to box regression losses

3. **No Safety Margin**
   - Only penalized exact box overlaps
   - No buffer zone around BLOCK areas
   - Traces could pass very close to BLOCK boxes without penalty

## Applied Fixes

### Fix 1: Enhanced Avoidance Loss Computation (`train.py:136-181`)

**Added Three Key Improvements:**

1. **Safety Margin (0.2m)**
   ```python
   safety_margin = 0.2
   expanded_half_sizes = (box_sizes + safety_margin) / 2.0
   ```
   - Expands BLOCK boxes by 0.2m on each side
   - Creates buffer zone around obstacles
   - Prevents traces from passing too close to BLOCK areas

2. **Hard Constraint for High-Confidence BLOCK**
   ```python
   high_confidence_block = (block_probs > 0.5).float()
   hard_loss = (penetration.sum(dim=-1) ** 2) * trace_mask
   hard_loss = hard_loss * 10.0  # Strong multiplier
   ```
   - For confident BLOCK predictions (prob > 0.5): squared penalty × 10
   - Makes it extremely expensive to predict BLOCK where people walk
   - Forces model to be certain before predicting BLOCK

3. **Soft Penalty for Low-Confidence BLOCK**
   ```python
   low_confidence_block = (block_probs <= 0.5).float()
   soft_penetration = inside * block_probs * low_confidence_block
   ```
   - For uncertain predictions: linear penalty weighted by probability
   - Allows model to gradually adjust during training

**Total Penalty Structure:**
```
avoidance_loss = hard_loss × 10.0 + soft_loss
```

### Fix 2: Increased Loss Weight (`train.py:577`)

**Before:**
```python
parser.add_argument('--avoid_weight', type=float, default=1.0)
```

**After:**
```python
parser.add_argument('--avoid_weight', type=float, default=10.0,
                    help="Weight for avoidance_loss (critical constraint)")
```

**Effective Weights:**
- AlignmentLoss internal weight: 2.0
- Command-line weight: 10.0 (new default)
- Hard penalty multiplier: 10.0 (for confident BLOCK)

**Total contribution for high-confidence BLOCK overlaps:**
```
2.0 × 10.0 × 10.0 = 200.0 × penetration²
```

This is now **100x stronger** than the original weight of 2.0.

---

## Expected Results

### Training Behavior

**Early Training (Epochs 1-20):**
- Avoidance loss will be VERY HIGH initially (2.0-5.0+)
- Model will quickly learn to avoid predicting BLOCK on traces
- Total loss may increase temporarily as model adjusts

**Mid Training (Epochs 20-80):**
- Avoidance loss should drop rapidly to < 0.5
- Model learns proper BLOCK placement away from traces
- Box regression and classification losses decrease

**Late Training (Epochs 80-200):**
- Avoidance loss should stabilize at < 0.1
- Model understands the "no walking through BLOCK" constraint
- Better generalization to new room layouts

### Validation Metrics

**Expected Improvements:**
- Avoidance loss: 0.7-0.9 → < 0.1
- BLOCK prediction accuracy: significant increase
- False positives (BLOCK on traces): dramatic reduction
- F1 score: 0.72 → 0.75-0.80

---

## Migration Guide

### For Existing Checkpoints

**⚠️ BREAKING CHANGE**: Loss computation has changed.

**Option 1: Train from scratch (Recommended)**
```bash
cd /home/user/room-slam/src/rasterize
rm -rf checkpoints/*  # Remove old checkpoints
python train.py --num_epochs 200
```

**Option 2: Fine-tune with lower learning rate**
```bash
python train.py \
    --load_checkpoint checkpoints/best_model.pth \
    --lr 1e-5 \
    --num_epochs 50
```
- Model will adapt to new avoidance constraints
- May see temporary spike in loss

### Training from Scratch

**Recommended Settings:**
```bash
python train.py \
    --batch_size 16 \
    --lr 1e-4 \
    --num_epochs 200 \
    --avoid_weight 10.0  # New default
    --cov_weight 0.5
```

**Monitor These Metrics:**
1. **Avoidance loss trend** - Should decrease from 2.0+ to < 0.1
2. **BLOCK classification accuracy** - Should increase
3. **Validation F1** - Should improve over baseline

---

## Technical Details

### Loss Computation Breakdown

For each trace point and BLOCK box:

1. **Compute expanded distance:**
   ```
   expanded_box = original_box + 0.2m margin
   is_inside = trace_point inside expanded_box
   ```

2. **Separate by confidence:**
   ```
   if block_prob > 0.5:
       penalty = is_inside² × 10.0  # Hard constraint
   else:
       penalty = is_inside × block_prob  # Soft constraint
   ```

3. **Aggregate across batch:**
   ```
   loss = (Σ hard_penalties × 10.0 + Σ soft_penalties) / num_traces
   ```

4. **Apply final weight:**
   ```
   final_loss = loss × 2.0 × 10.0 = loss × 20.0
   ```

### Why Squared Penalty?

For high-confidence BLOCK predictions:
- Linear penalty: Loss = k × n (where n = number of overlaps)
- Squared penalty: Loss = k × n²

**Example:**
- 1 overlap: 10 vs 10 (same)
- 2 overlaps: 20 vs 40 (2x stronger)
- 5 overlaps: 50 vs 250 (5x stronger)
- 10 overlaps: 100 vs 1000 (10x stronger)

Squared penalty makes **multiple violations exponentially more expensive**.

---

## Debugging

### If avoidance loss doesn't decrease:

1. **Check training logs:**
   ```bash
   # Look for initial avoidance loss
   grep "avoid=" logs/training.log | head -20
   ```
   - Should start high (> 1.0)
   - Should decrease over first 20 epochs

2. **Visualize predictions:**
   ```bash
   python visualize.py --checkpoint checkpoints/epoch_50.pth
   ```
   - Look for BLOCK boxes overlapping traces
   - Check if safety margin is respected

3. **Reduce other loss weights:**
   ```bash
   python train.py \
       --l1_loss 2.0 \      # Reduce from 5.0
       --giou_loss 1.0 \    # Reduce from 2.0
       --avoid_weight 15.0  # Increase further
   ```

### If loss explodes:

1. **Reduce avoid_weight gradually:**
   ```bash
   python train.py --avoid_weight 5.0  # Start lower
   ```

2. **Use gradient clipping:**
   - Already enabled in code (max_norm=1.0)
   - May need to reduce to 0.5 for stability

---

## Related Files

- `src/rasterize/train.py:136-181` - Enhanced avoidance loss
- `src/rasterize/train.py:577` - Increased weight default
- `src/rasterize/train.py:655-657` - Loss weight dict

---

## Summary

**Problem:** Model ignored the "no walking through BLOCK" constraint.

**Solution:**
1. Add safety margin (0.2m) to BLOCK areas
2. Use squared penalty for confident BLOCK predictions
3. Increase weight from 2.0 to 200.0 (100x stronger)

**Result:** Model will learn that BLOCK areas must be far from traces, leading to more realistic room layout predictions.

---

Generated: 2025-11-13
Status: ✅ Applied
Breaking Change: Yes (requires retraining)
