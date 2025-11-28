import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        pred_classes: [B, Q, 3]
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
            prob = pred_classes[b].softmax(-1)  # [Q, 3]
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


class TraceColliderAlignmentLoss(nn.Module):
    def __init__(
            self,
            coverage_weight: float = 1.0,
            avoidance_weight: float = 2.0,
    ):
        super().__init__()
        self.coverage_weight = coverage_weight
        self.avoidance_weight = avoidance_weight

    def forward(
            self,
            pred_boxes: torch.Tensor,  # [B, Q, 6]
            pred_classes: torch.Tensor,  # [B, Q, 3] logits
            traces: torch.Tensor,  # [B, N, 11]
            trace_mask: torch.Tensor,  # [B, N]
    ) -> dict:
        """Compute alignment losses"""
        losses = {}

        # Coverage Loss
        coverage_loss = self.compute_coverage_loss(
            pred_boxes, pred_classes, traces, trace_mask
        )
        losses['coverage'] = coverage_loss * self.coverage_weight

        # Avoidance Loss
        avoidance_loss = self.compute_avoidance_loss(
            pred_boxes, pred_classes, traces, trace_mask
        )
        losses['avoidance'] = avoidance_loss * self.avoidance_weight

        return losses

    def compute_coverage_loss(
            self,
            pred_boxes: torch.Tensor,
            pred_classes: torch.Tensor,
            traces: torch.Tensor,
            trace_mask: torch.Tensor
    ) -> torch.Tensor:
        B, Q, _ = pred_boxes.shape
        B, N, _ = traces.shape

        trace_coords = traces[..., :3]  # [B, N, 3]
        box_centers = pred_boxes[..., :3]  # [B, Q, 3]
        box_sizes = pred_boxes[..., 3:]  # [B, Q, 3]

        # Get class probabilities
        class_probs = F.softmax(pred_classes, dim=-1)  # [B, Q, 3]

        # Only consider non-BLOCK boxes (LOW/MID are passable)
        # class 0=BLOCK, 1=LOW, 2=MID
        passable_probs = class_probs[..., 1:].sum(dim=-1)  # [B, Q]

        # Compute distance from each trace point to each box
        # [B, N, Q, 3]
        diffs = trace_coords.unsqueeze(2) - box_centers.unsqueeze(1)
        dists = diffs.abs()  # L1 distance per dimension

        # Check if point is inside box (using soft margin)
        half_sizes = box_sizes.unsqueeze(1) / 2  # [B, 1, Q, 3]
        inside_margin = F.relu(dists - half_sizes)  # [B, N, Q, 3]
        point_to_box_dist = inside_margin.sum(dim=-1)  # [B, N, Q]

        # Weight by passability
        weighted_dist_t2b = point_to_box_dist * passable_probs.unsqueeze(1)  # [B, N, Q]
        min_dist_t2b, _ = weighted_dist_t2b.min(dim=-1)  # [B, N]
        valid_dists_t2b = min_dist_t2b * trace_mask
        loss_t2b = valid_dists_t2b.sum() / (trace_mask.sum() + 1e-6)

        # For each trace point, find minimum weighted distance
        min_dist_b2t, _ = point_to_box_dist.min(dim=1)  # [B, Q]

        # Apply mask and normalize
        weighted_dist_b2t = min_dist_b2t * passable_probs # [B, Q]
        loss_b2t = weighted_dist_b2t.sum() / (passable_probs.sum() + 1e-6)

        coverage_loss = loss_t2b + loss_b2t
        return coverage_loss

    def compute_avoidance_loss(
            self,
            pred_boxes: torch.Tensor,
            pred_classes: torch.Tensor,
            traces: torch.Tensor,
            trace_mask: torch.Tensor
    ) -> torch.Tensor:
        B, Q, _ = pred_boxes.shape
        B, N, _ = traces.shape

        trace_coords = traces[..., :3]  # [B, N, 3]
        box_centers = pred_boxes[..., :3]  # [B, Q, 3]
        box_sizes = pred_boxes[..., 3:]  # [B, Q, 3]

        # Get BLOCK probabilities (class 0)
        class_probs = F.softmax(pred_classes, dim=-1)  # [B, Q, 3]
        block_probs = class_probs[..., 0]  # [B, Q]

        # Compute if trace points are inside boxes
        diffs = (trace_coords.unsqueeze(2) - box_centers.unsqueeze(1)).abs()  # [B, N, Q, 3]
        half_sizes = box_sizes.unsqueeze(1) / 2  # [B, 1, Q, 3]
        inside = (diffs < half_sizes).all(dim=-1).float()  # [B, N, Q]

        # Penalize being inside BLOCK boxes
        penetration = inside * block_probs.unsqueeze(1)  # [B, N, Q]

        # Sum over boxes and points
        total_penetration = penetration.sum(dim=-1)  # [B, N]

        # Apply mask and normalize
        valid_penetration = total_penetration * trace_mask
        avoidance_loss = valid_penetration.sum() / (trace_mask.sum() + 1e-6)

        return avoidance_loss


class SetCriterion(nn.Module):
    """Loss computation with GIoU + Alignment"""

    def __init__(self, weight_dict):
        super().__init__()
        self.weight_dict = weight_dict
        self.matcher = HungarianMatcher()

        self.class_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss(reduction='none')

        self.alignment_loss = TraceColliderAlignmentLoss(
            coverage_weight=1.0,
            avoidance_weight=2.0
        )

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


    def compute_collision_loss(self, pred_boxes):
        """Compute pairwise 3D IoU collision loss to penalize overlaps"""
        B, Q = pred_boxes.shape[:2]
        if Q <= 1:
            return torch.tensor(0.0, device=pred_boxes.device)
            
        loss = 0.0
        for b in range(B):
            boxes = pred_boxes[b] # [Q, 6]
            
            # [Q, 1, 3]
            min_b = (boxes[:, :3] - boxes[:, 3:] / 2).unsqueeze(1)
            max_b = (boxes[:, :3] + boxes[:, 3:] / 2).unsqueeze(1)
            
            # [1, Q, 3]
            min_b_t = (boxes[:, :3] - boxes[:, 3:] / 2).unsqueeze(0)
            max_b_t = (boxes[:, :3] + boxes[:, 3:] / 2).unsqueeze(0)
            
            inter_min = torch.max(min_b, min_b_t)
            inter_max = torch.min(max_b, max_b_t)
            inter_dims = torch.clamp(inter_max - inter_min, min=0)
            inter_vol = inter_dims.prod(dim=-1) # [Q, Q]
            
            vol = boxes[:, 3:].prod(dim=-1) # [Q]
            union_vol = vol.unsqueeze(1) + vol.unsqueeze(0) - inter_vol
            
            iou = inter_vol / (union_vol + 1e-6)
            
            # Zero out diagonal (self-overlap)
            mask = torch.eye(Q, device=boxes.device).bool()
            iou = iou.masked_fill(mask, 0.0)
            
            # Average overlapping IoU
            if Q > 1:
                loss += iou.sum() / (Q * (Q - 1))
            
        return loss / B

    def forward(self, outputs, targets, traces, trace_mask):
        pred_boxes = outputs['pred_boxes']
        pred_classes = outputs['pred_classes']
        gt_boxes = targets['boxes']
        gt_labels = targets['labels']
        gt_valid_mask = targets['valid_mask']

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

        alignment_losses = self.alignment_loss(pred_boxes, pred_classes, traces, trace_mask)
        losses['coverage_loss'] = alignment_losses['coverage']
        losses['avoidance_loss'] = alignment_losses['avoidance']
        losses['collision_loss'] = self.compute_collision_loss(pred_boxes)

        # Diversity regularization: encourage predictions to be trace-dependent
        # Penalize predictions that are too uniform across batch
        if pred_boxes.shape[0] > 1:
            # Compute variance of predictions across batch (should be high for diverse predictions)
            pred_var = pred_boxes.var(dim=0).mean()  # Average variance across queries
            # We want this to be high, so we minimize the negative (or use a small penalty)
            # Small penalty to avoid over-penalizing (only if variance is very low)
            diversity_loss = -0.01 * pred_var.clamp(max=1.0)  # Small penalty for low variance
            losses['diversity_loss'] = diversity_loss
        else:
            losses['diversity_loss'] = torch.tensor(0.0, device=pred_boxes.device)

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
        valid_mask = batch['valid_mask'].to(device)

        # Forward with baseline colliders if available
        baseline_boxes = batch.get('baseline_boxes', None)
        baseline_valid_mask = batch.get('baseline_valid_mask', None)
        
        if baseline_boxes is not None and baseline_valid_mask is not None:
            baseline_boxes = baseline_boxes.to(device)
            baseline_valid_mask = baseline_valid_mask.to(device)
            outputs = model(traces, mask, baseline_boxes, baseline_valid_mask)
        else:
            outputs = model(traces, mask)

        targets = {
            'boxes': boxes,
            'labels': labels,
            'valid_mask': valid_mask
        }
        losses = criterion(outputs, targets, traces, mask)

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
            'giou': f"{losses['giou_loss'].item():.4f}",
            'cov': f"{losses['coverage_loss'].item():.4f}",
            'avoid': f"{losses['avoidance_loss'].item():.4f}",
            'col': f"{losses['collision_loss'].item():.4f}",
            'div': f"{losses['diversity_loss'].item():.4f}"
        })

    return total_loss / len(dataloader)


def compute_iou_matrix_3d(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise 3D IoU between two sets of boxes"""
    N = boxes1.shape[0]
    M = boxes2.shape[0]

    boxes1_min = boxes1[:, :3] - boxes1[:, 3:] / 2
    boxes1_max = boxes1[:, :3] + boxes1[:, 3:] / 2
    boxes2_min = boxes2[:, :3] - boxes2[:, 3:] / 2
    boxes2_max = boxes2[:, :3] + boxes2[:, 3:] / 2

    boxes1_min = boxes1_min.unsqueeze(1)
    boxes1_max = boxes1_max.unsqueeze(1)
    boxes2_min = boxes2_min.unsqueeze(0)
    boxes2_max = boxes2_max.unsqueeze(0)

    inter_min = torch.maximum(boxes1_min, boxes2_min)
    inter_max = torch.minimum(boxes1_max, boxes2_max)
    inter_size = torch.clamp(inter_max - inter_min, min=0)
    inter_volume = inter_size.prod(dim=2)

    boxes1_volume = boxes1[:, 3:].prod(dim=1, keepdim=True)
    boxes2_volume = boxes2[:, 3:].prod(dim=1, keepdim=True)
    union_volume = boxes1_volume + boxes2_volume.T - inter_volume

    iou = inter_volume / (union_volume + 1e-6)
    return iou


@torch.no_grad()
def evaluate_metrics(model, dataloader, device, iou_thresh: float = 0.5):
    model.eval()

    total_iou_sum = 0.0
    total_iou_cnt = 0
    tp = 0
    fp = 0
    fn = 0

    cls_correct = 0
    cls_total = 0

    # Per-class metrics
    class_names = ['BLOCK', 'LOW', 'MID']
    per_class_stats = {
        cls_name: {'correct': 0, 'total': 0, 'tp': 0, 'fp': 0, 'fn': 0}
        for cls_name in class_names
    }

    # Confusion matrix
    confusion = np.zeros((3, 3), dtype=int)

    for batch in tqdm(dataloader, desc="Evaluating"):
        traces = batch['traces'].to(device)
        mask = batch['trace_mask'].to(device)
        gt_boxes = batch['boxes'].to(device)
        gt_labels = batch['labels'].to(device)
        gt_valid_mask = batch['valid_mask'].to(device)

        # Forward with baseline colliders if available
        baseline_boxes = batch.get('baseline_boxes', None)
        baseline_valid_mask = batch.get('baseline_valid_mask', None)
        
        if baseline_boxes is not None and baseline_valid_mask is not None:
            baseline_boxes = baseline_boxes.to(device)
            baseline_valid_mask = baseline_valid_mask.to(device)
            outputs = model(traces, mask, baseline_boxes, baseline_valid_mask)
        else:
            outputs = model(traces, mask)
            
        pred_boxes = outputs['pred_boxes']
        pred_classes = outputs['pred_classes']
        pred_labels = pred_classes.argmax(dim=-1)
        pred_probs = torch.softmax(pred_classes, dim=-1)
        pred_conf = pred_probs.max(dim=-1)[0]

        B, Q = pred_boxes.shape[:2]

        for b in range(B):
            valid_mask = gt_valid_mask[b]
            num_valid = int(valid_mask.sum().item())

            if num_valid == 0:
                continue

            gt_b = gt_boxes[b, valid_mask]
            gt_l = gt_labels[b, valid_mask]

            # Filter predictions by confidence
            conf_threshold = 0.1
            valid_preds = pred_conf[b] > conf_threshold

            pred_b = pred_boxes[b, valid_preds]
            pred_l = pred_labels[b, valid_preds]

            if pred_b.shape[0] == 0:
                for gt_cls in gt_l:
                    cls_name = class_names[gt_cls.item()]
                    per_class_stats[cls_name]['fn'] += 1
                fn += num_valid
                continue

            # Compute IoU matrix
            iou_matrix = compute_iou_matrix_3d(pred_b, gt_b)

            # Hungarian matching
            cost_matrix = -iou_matrix.cpu().numpy()
            pred_idx, gt_idx = linear_sum_assignment(cost_matrix)

            matched_ious = iou_matrix[pred_idx, gt_idx]

            # Statistics for matched pairs
            for i, (p_i, g_i) in enumerate(zip(pred_idx, gt_idx)):
                iou = matched_ious[i].item()

                total_iou_sum += iou
                total_iou_cnt += 1

                pred_cls = pred_l[p_i].item()
                gt_cls = gt_l[g_i].item()

                confusion[gt_cls, pred_cls] += 1

                if iou >= iou_thresh:
                    tp += 1
                    cls_total += 1
                    if pred_cls == gt_cls:
                        cls_correct += 1
                        cls_name = class_names[gt_cls]
                        per_class_stats[cls_name]['correct'] += 1

                    per_class_stats[class_names[gt_cls]]['total'] += 1
                    per_class_stats[class_names[gt_cls]]['tp'] += 1
                else:
                    fp += 1
                    per_class_stats[class_names[pred_cls]]['fp'] += 1

            # Unmatched GT boxes are FN
            unmatched_gt = set(range(len(gt_l))) - set(gt_idx)
            for g_i in unmatched_gt:
                fn += 1
                cls_name = class_names[gt_l[g_i].item()]
                per_class_stats[cls_name]['fn'] += 1

            # Unmatched predictions are FP
            unmatched_pred = set(range(len(pred_l))) - set(pred_idx)
            for p_i in unmatched_pred:
                fp += 1
                cls_name = class_names[pred_l[p_i].item()]
                per_class_stats[cls_name]['fp'] += 1

    miou = (total_iou_sum / total_iou_cnt) if total_iou_cnt > 0 else 0.0
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    cls_acc = (cls_correct / cls_total) if cls_total > 0 else 0.0

    # Per-class metrics
    per_class_metrics = {}
    for cls_name in class_names:
        stats = per_class_stats[cls_name]
        cls_precision = stats['tp'] / (stats['tp'] + stats['fp'] + 1e-8)
        cls_recall = stats['tp'] / (stats['tp'] + stats['fn'] + 1e-8)
        cls_f1 = 2 * cls_precision * cls_recall / (cls_precision + cls_recall + 1e-8)
        cls_accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0

        per_class_metrics[cls_name] = {
            'precision': cls_precision,
            'recall': cls_recall,
            'f1': cls_f1,
            'accuracy': cls_accuracy,
            'tp': stats['tp'],
            'fp': stats['fp'],
            'fn': stats['fn']
        }

    return {
        'mIoU': miou,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cls_acc': cls_acc,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'per_class': per_class_metrics,
        'confusion_matrix': confusion,
        'cls_correct': cls_correct,
        'cls_total': cls_total
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
            valid_mask = batch['valid_mask'].to(device)

            # Forward with baseline colliders if available
            baseline_boxes = batch.get('baseline_boxes', None)
            baseline_valid_mask = batch.get('baseline_valid_mask', None)
            
            if baseline_boxes is not None and baseline_valid_mask is not None:
                baseline_boxes = baseline_boxes.to(device)
                baseline_valid_mask = baseline_valid_mask.to(device)
                outputs = model(traces, mask, baseline_boxes, baseline_valid_mask)
            else:
                outputs = model(traces, mask)
            targets = {
                'boxes': boxes,
                'labels': labels,
                'valid_mask': valid_mask
            }
            losses = criterion(outputs, targets, traces, mask)

            total_loss += losses['total_loss'].item()

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Train Room-SLAM model")
    parser.add_argument('--stage_name', type=str, default="stage1_detect",
                        help="Name for this training stage (used for save_dir)")
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help="Path to checkpoint to load for fine-tuning (e.g., ./checkpoints_stage1/best_model.pth)")
    parser.add_argument('--dropout_prob', type=float, default=0.1,
                        help="Collider dropout probability")
    parser.add_argument('--cov_weight', type=float, default=0.5,
                        help="Weight for coverage_loss")
    parser.add_argument('--avoid_weight', type=float, default=5.0,
                        help="Weight for avoidance_loss")
    parser.add_argument('--collision_weight', type=float, default=10.0,
                        help="Weight for collision_loss")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument('--num_epochs', type=int, default=200,
                        help="Number of epochs to train")

    args = parser.parse_args()

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print(f"CUDA not available, using CPU")

    # Hyperparameters
    config = {
        'model_type': 'transformer',
        'batch_size': 20,
        'num_epochs': args.num_epochs,
        'lr': args.lr,
        'weight_decay': 1e-4,
        'd_model': 256,
        'num_queries': 30,
        'data_dir': '../../dataset/train',
        'val_dir': '../../dataset/val',
        'save_dir': f'./checkpoints_{args.stage_name}',
        'val_every': 1,
        'iou_thresh': 0.5,
        'use_baseline_colliders': False,
        'baseline_encoder_layers': 4
    }

    # Create save directory
    save_path = Path(config['save_dir'])
    save_path.mkdir(exist_ok=True)

    # Save config
    with open(save_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Create dataloaders
    print(f"\n=== Training Stage: {args.stage_name} ===")
    print(f"Collider Dropout: {args.dropout_prob}")

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
        collider_dropout_prob=args.dropout_prob,
        use_baseline_colliders=False
    )

    val_loader = create_dataloader(
        config['val_dir'],
        batch_size=config['batch_size'],
        shuffle=False,
        augment_rotation=False,
        augment_translation=False,
        augment_scale=False,
        augment_collider_dropout=False,
        use_baseline_colliders=False
    )

    # Build model
    model = build_model(
        num_queries=config['num_queries'],
        d_model=config['d_model'],
        model_type=config.get('model_type', 'transformer'),
        use_baseline_colliders=False,
        baseline_encoder_layers=4
    ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    weight_dict = {
        'class_loss': 5.0,
        'l1_loss': 10.0,
        'giou_loss': 5.0,
        'coverage_loss': args.cov_weight,
        'avoidance_loss': args.avoid_weight,
        'collision_loss': args.collision_weight,
        'diversity_loss': 0.1  # Small weight for diversity regularization
    }
    print(f"Using weights: {weight_dict}")
    criterion = SetCriterion(weight_dict)

    optimizer = AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    start_epoch = 0
    if args.load_checkpoint:
        print(f"\nLoading weights from: {args.load_checkpoint}\n")
        checkpoint = torch.load(args.load_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # start_epoch = checkpoint['epoch'] + 1

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min', factor=0.5, patience=5,
        threshold=1e-3, cooldown=1, min_lr=1e-6
    )

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(start_epoch, config['num_epochs']):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        if (epoch + 1) % config['val_every'] == 0:
            val_loss = validate(model, val_loader, criterion, device)
            metrics = evaluate_metrics(model, val_loader, device, iou_thresh=config['iou_thresh'])

            scheduler.step(val_loss)

            print(f"\nEpoch {epoch}: Train {train_loss:.4f} | Val {val_loss:.4f}")
            print(f"  Overall: mIoU={metrics['mIoU']:.4f} P={metrics['precision']:.4f} "
                  f"R={metrics['recall']:.4f} F1={metrics['f1']:.4f}")
            print(f"  ClsAcc: {metrics['cls_acc']:.4f} ({metrics['cls_correct']}/{metrics['cls_total']})")

            # Per-class accuracy
            print(f"  Per-Class Acc:", end=" ")
            for cls_name in ['BLOCK', 'LOW', 'MID']:
                cls_acc = metrics['per_class'][cls_name]['accuracy']
                print(f"{cls_name}={cls_acc:.4f}", end=" ")
            print()

            # Confusion matrix (compact)
            confusion = metrics['confusion_matrix']
            print(f"  Confusion: [B:{confusion[0, 0]}/{confusion[0].sum()} "
                  f"L:{confusion[1, 1]}/{confusion[1].sum()} "
                  f"M:{confusion[2, 2]}/{confusion[2].sum()}]")

            print(f"  LR={optimizer.param_groups[0]['lr']:.6f}")

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
                }, save_path / 'best_model.pth')
                print(f"  ✓ Saved BEST model (val_loss={best_val_loss:.4f})")

        else:
            print(f"Epoch {epoch}: Train {train_loss:.4f} | "
                  f"LR={optimizer.param_groups[0]['lr']:.6f}")

        # Regular checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }, save_path / f'checkpoint_epoch_{epoch}.pth')

    print("Training completed!")


if __name__ == "__main__":
    main()