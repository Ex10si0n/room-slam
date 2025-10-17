import torch
import torch.nn as nn
from torch.optim import AdamW
from scipy.optimize import linear_sum_assignment
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

from dataloader import create_dataloader
from model import build_model


class HungarianMatcher:
    """Hungarian matching between predictions and ground truth"""

    def __init__(self, cost_class: float = 1.0, cost_box: float = 5.0):
        self.cost_class = cost_class
        self.cost_box = cost_box

    @torch.no_grad()
    def forward(self, pred_boxes, pred_classes, gt_boxes, gt_labels, gt_valid_mask):
        """
        pred_boxes: [B, Q, 6]
        pred_classes: [B, Q, 4]
        gt_boxes: [B, M, 6]
        gt_labels: [B, M]
        gt_valid_mask: [B, M] - True for valid colliders
        """
        B, Q = pred_boxes.shape[:2]

        indices = []

        for b in range(B):
            # Get valid ground truth using valid_mask
            valid_mask = gt_valid_mask[b]
            num_valid = valid_mask.sum().item()

            if num_valid == 0:
                indices.append(([], []))
                continue

            # Classification cost
            prob = pred_classes[b].softmax(-1)  # [Q, 4]
            cost_class = -prob[:, gt_labels[b, valid_mask]]  # [Q, num_valid]

            # Box L1 cost
            pred_box = pred_boxes[b]  # [Q, 6]
            gt_box = gt_boxes[b, valid_mask]  # [num_valid, 6]
            cost_box = torch.cdist(pred_box, gt_box, p=1)  # [Q, num_valid]

            # Total cost
            cost = self.cost_class * cost_class + self.cost_box * cost_box

            # Hungarian matching
            cost = cost.cpu().numpy()
            pred_idx, gt_idx = linear_sum_assignment(cost)

            indices.append((pred_idx, gt_idx))

        return indices


class SetCriterion(nn.Module):
    """Loss computation with GIoU"""

    def __init__(self, weight_dict):
        super().__init__()
        self.weight_dict = weight_dict
        self.matcher = HungarianMatcher()

        self.class_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss(reduction='none')

    def box_iou_3d(self, boxes1, boxes2):
        """Compute 3D IoU between boxes"""
        # boxes: [N, 6] (cx, cy, cz, sx, sy, sz)

        # Convert to corner format
        boxes1_min = boxes1[:, :3] - boxes1[:, 3:] / 2
        boxes1_max = boxes1[:, :3] + boxes1[:, 3:] / 2
        boxes2_min = boxes2[:, :3] - boxes2[:, 3:] / 2
        boxes2_max = boxes2[:, :3] + boxes2[:, 3:] / 2

        # Intersection
        inter_min = torch.maximum(boxes1_min, boxes2_min)
        inter_max = torch.minimum(boxes1_max, boxes2_max)
        inter_size = torch.clamp(inter_max - inter_min, min=0)
        inter_volume = inter_size.prod(dim=1)

        # Union
        boxes1_volume = boxes1[:, 3:].prod(dim=1)
        boxes2_volume = boxes2[:, 3:].prod(dim=1)
        union_volume = boxes1_volume + boxes2_volume - inter_volume

        # IoU
        iou = inter_volume / (union_volume + 1e-6)

        # GIoU: need enclosing box
        enclosing_min = torch.minimum(boxes1_min, boxes2_min)
        enclosing_max = torch.maximum(boxes1_max, boxes2_max)
        enclosing_size = torch.clamp(enclosing_max - enclosing_min, min=0)
        enclosing_volume = enclosing_size.prod(dim=1)

        giou = iou - (enclosing_volume - union_volume) / (enclosing_volume + 1e-6)

        return iou, giou

    def forward(self, outputs, targets):
        pred_boxes = outputs['pred_boxes']
        pred_classes = outputs['pred_classes']
        gt_boxes = targets['boxes']
        gt_labels = targets['labels']
        gt_valid_mask = targets['valid_mask']  # Use valid_mask from dataloader

        # Hungarian matching
        indices = self.matcher.forward(pred_boxes, pred_classes, gt_boxes, gt_labels, gt_valid_mask)

        # Compute losses
        losses = {}

        # Classification loss
        class_loss = self._compute_class_loss(pred_classes, gt_labels, gt_valid_mask, indices)
        losses['class_loss'] = class_loss

        # Box regression loss (L1 + GIoU)
        l1_loss, giou_loss = self._compute_box_loss(pred_boxes, gt_boxes, gt_valid_mask, indices)
        losses['l1_loss'] = l1_loss
        losses['giou_loss'] = giou_loss

        # Total loss
        total_loss = sum(losses[k] * self.weight_dict.get(k, 1.0) for k in losses.keys())
        losses['total_loss'] = total_loss

        return losses

    def _compute_class_loss(self, pred_classes, gt_labels, gt_valid_mask, indices):
        device = pred_classes.device

        # Gather matched predictions and targets
        pred_list = []
        target_list = []

        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                pred_list.append(pred_classes[b, pred_idx])

                # Get valid labels using valid_mask
                valid_labels = gt_labels[b, gt_valid_mask[b]]
                target_list.append(valid_labels[gt_idx])

        if len(pred_list) == 0:
            return torch.tensor(0.0, device=device)

        pred_cat = torch.cat(pred_list, dim=0)
        target_cat = torch.cat(target_list, dim=0)

        return self.class_loss(pred_cat, target_cat)

    def _compute_box_loss(self, pred_boxes, gt_boxes, gt_valid_mask, indices):
        device = pred_boxes.device

        pred_list = []
        target_list = []

        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                pred_list.append(pred_boxes[b, pred_idx])

                # Get valid boxes using valid_mask
                valid_boxes = gt_boxes[b, gt_valid_mask[b]]
                target_list.append(valid_boxes[gt_idx])

        if len(pred_list) == 0:
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        pred_cat = torch.cat(pred_list, dim=0)
        target_cat = torch.cat(target_list, dim=0)

        # L1 loss
        l1_loss = self.l1_loss(pred_cat, target_cat).mean()

        # GIoU loss
        _, giou = self.box_iou_3d(pred_cat, target_cat)
        giou_loss = (1 - giou).mean()

        return l1_loss, giou_loss


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()

    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        # Move to device
        traces = batch['traces'].to(device)
        mask = batch['trace_mask'].to(device)
        boxes = batch['boxes'].to(device)
        labels = batch['labels'].to(device)
        valid_mask = batch['valid_mask'].to(device)  # Add valid_mask

        # Forward
        outputs = model(traces, mask)

        # Compute loss
        targets = {
            'boxes': boxes,
            'labels': labels,
            'valid_mask': valid_mask  # Pass valid_mask to criterion
        }
        losses = criterion(outputs, targets)

        loss = losses['total_loss']

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Log
        total_loss += loss.item()
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'cls': f"{losses['class_loss'].item():.4f}",
            'l1': f"{losses['l1_loss'].item():.4f}",
            'giou': f"{losses['giou_loss'].item():.4f}"
        })

    return total_loss / len(dataloader)

@torch.no_grad()
def evaluate_metrics(model, dataloader, device, iou_thresh: float = 0.5):
    model.eval()
    matcher = HungarianMatcher()
    total_iou_sum = 0.0
    total_iou_cnt = 0

    tp = 0
    fp = 0
    fn = 0

    cls_correct = 0
    cls_total = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        traces = batch['traces'].to(device)
        mask = batch['trace_mask'].to(device)
        gt_boxes = batch['boxes'].to(device)
        gt_labels = batch['labels'].to(device)
        gt_valid_mask = batch['valid_mask'].to(device)

        outputs = model(traces, mask)  # expects {'pred_boxes':[B,Q,6], 'pred_classes':[B,Q,4]}
        pred_boxes = outputs['pred_boxes']
        pred_logits = outputs['pred_classes']  # [B,Q,4]
        pred_probs = pred_logits.softmax(-1)
        pred_labels = pred_probs.argmax(-1)    # [B,Q]

        # Hungarian matching for alignment
        indices = matcher.forward(pred_boxes, pred_logits, gt_boxes, gt_labels, gt_valid_mask)

        # Accumulate metrics per batch
        B, Q = pred_boxes.shape[:2]
        for b, (p_idx, g_idx) in enumerate(indices):
            valid_mask = gt_valid_mask[b]
            num_valid = int(valid_mask.sum().item())

            # Count FN as ground-truth that got no match (if Hungarian returns < num_valid)
            fn += max(0, num_valid - len(g_idx))

            if len(p_idx) == 0:
                continue

            # Matched preds and gts
            pb = pred_boxes[b, p_idx]                # [K,6]
            gb = gt_boxes[b, valid_mask][g_idx]      # [K,6]
            pi = pred_labels[b, p_idx]               # [K]
            gi = gt_labels[b, valid_mask][g_idx]     # [K]

            # IoU / TP/FP
            # (reuse your SetCriterion box_iou_3d quickly)
            # quick inline IoU (axis-aligned 3D)
            pb_min = pb[:, :3] - pb[:, 3:] / 2
            pb_max = pb[:, :3] + pb[:, 3:] / 2
            gb_min = gb[:, :3] - gb[:, 3:] / 2
            gb_max = gb[:, :3] + gb[:, 3:] / 2

            inter_min = torch.maximum(pb_min, gb_min)
            inter_max = torch.minimum(pb_max, gb_max)
            inter = torch.clamp(inter_max - inter_min, min=0)
            inter_v = inter.prod(dim=1)

            pv = pb[:, 3:].prod(dim=1)
            gv = gb[:, 3:].prod(dim=1)
            union_v = pv + gv - inter_v + 1e-6
            ious = inter_v / union_v

            total_iou_sum += ious.sum().item()
            total_iou_cnt += ious.numel()

            # Classification accuracy over matched pairs
            cls_correct += (pi == gi).sum().item()
            cls_total += pi.numel()

            # For detection PR: a matched pred counts as TP if IoU>=thr, else FP
            tp_k = (ious >= iou_thresh).sum().item()
            fp_k = (ious < iou_thresh).sum().item()
            tp += tp_k
            fp += fp_k

            # (We already added FN above for unmatched gts)

    miou = (total_iou_sum / total_iou_cnt) if total_iou_cnt > 0 else 0.0
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    cls_acc = (cls_correct / cls_total) if cls_total > 0 else 0.0

    return {
        'mIoU': miou,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cls_acc': cls_acc,
        'tp': tp, 'fp': fp, 'fn': fn
    }

def validate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            traces = batch['traces'].to(device)
            mask = batch['trace_mask'].to(device)
            boxes = batch['boxes'].to(device)
            labels = batch['labels'].to(device)
            valid_mask = batch['valid_mask'].to(device)  # Add valid_mask

            outputs = model(traces, mask)
            targets = {
                'boxes': boxes,
                'labels': labels,
                'valid_mask': valid_mask  # Pass valid_mask
            }
            losses = criterion(outputs, targets)

            total_loss += losses['total_loss'].item()

    return total_loss / len(dataloader)


def main():
    # Setup device - use CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print(f"CUDA not available, using CPU")

    # Hyperparameters (optimized for training)
    config = {
        'batch_size': 5,
        'num_epochs': 200,
        'lr': 2e-4,
        'weight_decay': 1e-4,
        'd_model': 128,
        'num_queries': 30,
        'data_dir': '../../dataset/train',
        'val_dir': '../../dataset/val',
        'save_dir': './checkpoints',
        'warmup_epochs': 10,
        'val_every': 1,
        'iou_thresh': 0.5
    }

    # Create save directory
    Path(config['save_dir']).mkdir(exist_ok=True)

    # Save config
    with open(Path(config['save_dir']) / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Create dataloaders with AGGRESSIVE augmentation
    print("\n=== Data Augmentation Settings ===")
    print("Rotation: [0°, 90°, 180°, 270°]")
    print("Translation: ±1.0 meters")
    print("Scale: 0.8x to 1.2x")
    print("Collider Dropout: 20% probability")
    print("=" * 40 + "\n")

    train_loader = create_dataloader(
        config['data_dir'],
        batch_size=config['batch_size'],
        shuffle=True,
        augment_rotation=True,
        augment_translation=True,
        augment_scale=True,
        augment_collider_dropout=True,
        rotation_angles=[0, 90, 180, 270],
        scale_range=(0.8, 1.2),
        translation_range=1.0,
        collider_dropout_prob=0.2
    )

    val_loader = create_dataloader(
        config['val_dir'],
        batch_size=config['batch_size'],
        shuffle=False,
        augment_rotation=False,
        augment_translation=False,
        augment_scale=False,
        augment_collider_dropout=False
    )

    # Build model
    model = build_model(
        num_queries=config['num_queries'],
        d_model=config['d_model']
    ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Loss and optimizer
    weight_dict = {
        'class_loss': 2.0,  # Increased class loss weight
        'l1_loss': 5.0,  # L1 box loss
        'giou_loss': 2.0  # GIoU loss for better localization
    }
    criterion = SetCriterion(weight_dict)

    optimizer = AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < config['warmup_epochs']:
            return (epoch + 1) / config['warmup_epochs']
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - config['warmup_epochs']) /
                                     (config['num_epochs'] - config['warmup_epochs'])))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min', factor=0.5, patience=5,
        threshold=1e-3, cooldown=1, min_lr=1e-6
    )

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(config['num_epochs']):
        # === Train ===
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # === Validate ===
        if (epoch + 1) % config['val_every'] == 0:
            val_loss = validate(model, val_loader, criterion, device)
            metrics = evaluate_metrics(model, val_loader, device, iou_thresh=config['iou_thresh'])

            scheduler.step(val_loss)

            print(f"Epoch {epoch}: Train {train_loss:.4f} | Val {val_loss:.4f} | "
                  f"mIoU={metrics['mIoU']:.3f} "
                  f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f} "
                  f"ClsAcc={metrics['cls_acc']:.3f} | LR={optimizer.param_groups[0]['lr']:.6f}")

            # Save best on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'metrics': metrics,
                    'config': config
                }, Path(config['save_dir']) / 'best_model.pth')
                print(f"✓ Saved BEST model (val_loss={best_val_loss:.4f})")

        else:
            print(f"Epoch {epoch}: Train {train_loss:.4f} | "
                  f"LR={optimizer.param_groups[0]['lr']:.6f} (no val this epoch)")

        # Regular checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }, Path(config['save_dir']) / f'checkpoint_epoch_{epoch}.pth')

    print("Training completed!")

if __name__ == "__main__":
    main()