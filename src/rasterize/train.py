import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.optim import AdamW
from tqdm import tqdm

from dataloader import create_dataloader
from model import build_model


class HungarianMatcher:
    """Hungarian matching between predictions and ground truth boxes."""

    def __init__(self, cost_class: float = 1.0, cost_box: float = 5.0):
        self.cost_class = cost_class
        self.cost_box = cost_box

    @torch.no_grad()
    def forward(self, pred_boxes, pred_classes, gt_boxes, gt_labels, gt_valid_mask):
        """
        Args:
            pred_boxes: [B, Q, 6]
            pred_classes: [B, Q, 3]
            gt_boxes: [B, M, 6]
            gt_labels: [B, M]
            gt_valid_mask: [B, M] - True for valid colliders

        Returns:
            indices: list of (pred_idx, gt_idx) for each batch element
        """
        B, Q = pred_boxes.shape[:2]
        indices = []

        for b in range(B):
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
    """
    Alignment loss between predicted colliders and trace trajectories.

    - Coverage: encourage passable (LOW/MID) boxes to cover the trajectory.
    - Avoidance: penalize trajectory going through BLOCK boxes.
    """

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
        pred_boxes: torch.Tensor,    # [B, Q, 6]
        pred_classes: torch.Tensor,  # [B, Q, 3] logits
        traces: torch.Tensor,        # [B, N, 11]
        trace_mask: torch.Tensor,    # [B, N]
    ) -> dict:
        losses = {}

        coverage_loss = self.compute_coverage_loss(pred_boxes, pred_classes, traces, trace_mask)
        losses['coverage'] = coverage_loss * self.coverage_weight

        avoidance_loss = self.compute_avoidance_loss(pred_boxes, pred_classes, traces, trace_mask)
        losses['avoidance'] = avoidance_loss * self.avoidance_weight

        return losses

    def compute_coverage_loss(
        self,
        pred_boxes: torch.Tensor,
        pred_classes: torch.Tensor,
        traces: torch.Tensor,
        trace_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encourage passable colliders to cover the trace."""
        B, Q, _ = pred_boxes.shape
        B, N, _ = traces.shape

        trace_coords = traces[..., :3]      # [B, N, 3]
        box_centers = pred_boxes[..., :3]   # [B, Q, 3]
        box_sizes = pred_boxes[..., 3:]     # [B, Q, 3]

        class_probs = F.softmax(pred_classes, dim=-1)  # [B, Q, 3]
        # class 0=BLOCK, 1=LOW, 2=MID -> passable = LOW + MID
        passable_probs = class_probs[..., 1:].sum(dim=-1)  # [B, Q]

        # Distance from points to boxes (soft outside distance)
        diffs = trace_coords.unsqueeze(2) - box_centers.unsqueeze(1)  # [B, N, Q, 3]
        dists = diffs.abs()
        half_sizes = box_sizes.unsqueeze(1) / 2.0
        inside_margin = F.relu(dists - half_sizes)        # [B, N, Q, 3]
        point_to_box_dist = inside_margin.sum(dim=-1)     # [B, N, Q]

        # Trace -> box coverage
        weighted_dist_t2b = point_to_box_dist * passable_probs.unsqueeze(1)  # [B, N, Q]
        min_dist_t2b, _ = weighted_dist_t2b.min(dim=-1)                      # [B, N]
        valid_dists_t2b = min_dist_t2b * trace_mask
        loss_t2b = valid_dists_t2b.sum() / (trace_mask.sum() + 1e-6)

        # Box -> trace coverage
        min_dist_b2t, _ = point_to_box_dist.min(dim=1)  # [B, Q]
        weighted_dist_b2t = min_dist_b2t * passable_probs
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
        """Penalize the trajectory going through BLOCK boxes."""
        B, Q, _ = pred_boxes.shape
        B, N, _ = traces.shape

        trace_coords = traces[..., :3]      # [B, N, 3]
        box_centers = pred_boxes[..., :3]   # [B, Q, 3]
        box_sizes = pred_boxes[..., 3:]     # [B, Q, 3]

        class_probs = F.softmax(pred_classes, dim=-1)  # [B, Q, 3]
        block_probs = class_probs[..., 0]              # [B, Q] probability of BLOCK

        diffs = (trace_coords.unsqueeze(2) - box_centers.unsqueeze(1)).abs()  # [B, N, Q, 3]
        half_sizes = box_sizes.unsqueeze(1) / 2.0
        inside = (diffs < half_sizes).all(dim=-1).float()                     # [B, N, Q]

        # Trace inside BLOCK boxes
        penetration = inside * block_probs.unsqueeze(1)                       # [B, N, Q]
        total_penetration = penetration.sum(dim=-1)                           # [B, N]

        valid_penetration = total_penetration * trace_mask
        avoidance_loss = valid_penetration.sum() / (trace_mask.sum() + 1e-6)

        return avoidance_loss


class SetCriterion(nn.Module):
    """Combined loss: classification + box regression + alignment."""

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
        """Compute 3D GIoU between boxes."""
        # boxes: [N, 6] (cx, cy, cz, sx, sy, sz)

        boxes1_min = boxes1[:, :3] - boxes1[:, 3:] / 2
        boxes1_max = boxes1[:, :3] + boxes1[:, 3:] / 2
        boxes2_min = boxes2[:, :3] - boxes2[:, 3:] / 2
        boxes2_max = boxes2[:, :3] + boxes2[:, 3:] / 2

        inter_min = torch.maximum(boxes1_min, boxes2_min)
        inter_max = torch.minimum(boxes1_max, boxes2_max)
        inter_size = torch.clamp(inter_max - inter_min, min=0)
        inter_volume = inter_size.prod(dim=1)

        boxes1_volume = boxes1[:, 3:].prod(dim=1)
        boxes2_volume = boxes2[:, 3:].prod(dim=1)
        union_volume = boxes1_volume + boxes2_volume - inter_volume
        iou = inter_volume / (union_volume + 1e-6)

        enclosing_min = torch.minimum(boxes1_min, boxes2_min)
        enclosing_max = torch.maximum(boxes1_max, boxes2_max)
        enclosing_size = torch.clamp(enclosing_max - enclosing_min, min=0)
        enclosing_volume = enclosing_size.prod(dim=1)

        giou = iou - (enclosing_volume - union_volume) / (enclosing_volume + 1e-6)
        return iou, giou

    def forward(self, outputs, targets, traces, trace_mask):
        """
        Args:
            outputs: dict with 'pred_boxes' [B,Q,6], 'pred_classes' [B,Q,3]
            targets: dict with 'boxes' [B,M,6], 'labels' [B,M], 'valid_mask' [B,M]
        """
        pred_boxes = outputs['pred_boxes']
        pred_classes = outputs['pred_classes']
        gt_boxes = targets['boxes']
        gt_labels = targets['labels']
        gt_valid_mask = targets['valid_mask']

        # Hungarian matching
        indices = self.matcher.forward(pred_boxes, pred_classes, gt_boxes, gt_labels, gt_valid_mask)

        losses = {}

        # Classification loss
        class_loss = self._compute_class_loss(pred_classes, gt_labels, gt_valid_mask, indices)
        losses['class_loss'] = class_loss

        # Box regression loss (L1 + GIoU)
        l1_loss, giou_loss = self._compute_box_loss(pred_boxes, gt_boxes, gt_valid_mask, indices)
        losses['l1_loss'] = l1_loss
        losses['giou_loss'] = giou_loss

        # Alignment loss
        alignment_losses = self.alignment_loss(pred_boxes, pred_classes, traces, trace_mask)
        losses['coverage_loss'] = alignment_losses['coverage']
        losses['avoidance_loss'] = alignment_losses['avoidance']

        # Diversity regularization: encourage predictions to be trace-dependent
        if pred_boxes.shape[0] > 1:
            pred_var = pred_boxes.var(dim=0).mean()
            diversity_loss = -0.01 * pred_var.clamp(max=1.0)
            losses['diversity_loss'] = diversity_loss
        else:
            losses['diversity_loss'] = torch.tensor(0.0, device=pred_boxes.device)

        # Total loss (weighted sum)
        total_loss = sum(losses[k] * self.weight_dict.get(k, 1.0) for k in losses.keys())
        losses['total_loss'] = total_loss

        return losses

    def _compute_class_loss(self, pred_classes, gt_labels, gt_valid_mask, indices):
        device = pred_classes.device

        pred_list = []
        target_list = []

        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                pred_list.append(pred_classes[b, pred_idx])

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

                valid_boxes = gt_boxes[b, gt_valid_mask[b]]
                target_list.append(valid_boxes[gt_idx])

        if len(pred_list) == 0:
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        pred_cat = torch.cat(pred_list, dim=0)
        target_cat = torch.cat(target_list, dim=0)

        l1_loss = self.l1_loss(pred_cat, target_cat).mean()

        _, giou = self.box_iou_3d(pred_cat, target_cat)
        giou_loss = (1 - giou).mean()

        return l1_loss, giou_loss


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Single training epoch."""
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        traces = batch['traces'].to(device)        # [B,N,11]
        mask = batch['trace_mask'].to(device)      # [B,N]
        boxes = batch['boxes'].to(device)          # [B,M,6]
        labels = batch['labels'].to(device)        # [B,M]
        valid_mask = batch['valid_mask'].to(device)  # [B,M]

        # Grid map from dataloader (Walk2Map-style inverse-distance map)
        grid_maps = batch.get('grid_map', None)
        if grid_maps is not None:
            grid_maps = grid_maps.to(device)

        # Forward pass with trace + grid
        outputs = model(traces, mask, grid_maps=grid_maps)

        targets = {
            'boxes': boxes,
            'labels': labels,
            'valid_mask': valid_mask
        }
        losses = criterion(outputs, targets, traces, mask)
        loss = losses['total_loss']

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'cls': f"{losses['class_loss'].item():.4f}",
            'l1': f"{losses['l1_loss'].item():.4f}",
            'giou': f"{losses['giou_loss'].item():.4f}",
            'cov': f"{losses['coverage_loss'].item():.4f}",
            'avoid': f"{losses['avoidance_loss'].item():.4f}",
            'div': f"{losses['diversity_loss'].item():.4f}"
        })

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Compute validation loss (for debugging/monitoring)."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            traces = batch['traces'].to(device)
            mask = batch['trace_mask'].to(device)
            boxes = batch['boxes'].to(device)
            labels = batch['labels'].to(device)
            valid_mask = batch['valid_mask'].to(device)

            grid_maps = batch.get('grid_map', None)
            if grid_maps is not None:
                grid_maps = grid_maps.to(device)

            outputs = model(traces, mask, grid_maps=grid_maps)

            targets = {
                'boxes': boxes,
                'labels': labels,
                'valid_mask': valid_mask
            }
            losses = criterion(outputs, targets, traces, mask)
            total_loss += losses['total_loss'].item()

    return total_loss / len(dataloader)


def boxes_to_occupancy_grid(
    boxes: torch.Tensor,
    grid_size: int,
    x_min: torch.Tensor,
    x_max: torch.Tensor,
    z_min: torch.Tensor,
    z_max: torch.Tensor,
) -> torch.Tensor:
    """
    Project a set of 3D boxes (cx, cy, cz, sx, sy, sz) onto a top-down occupancy grid.

    We only use x/z dimensions to fill a [H, W] bool grid, where True = occupied.

    Args:
        boxes: [K, 6]
        grid_size: grid side length, e.g., 64
        x_min, x_max, z_min, z_max: scene bounds in X/Z plane

    Returns:
        grid: [H, W] bool tensor
    """
    H = W = grid_size
    device = boxes.device
    grid = torch.zeros((H, W), dtype=torch.bool, device=device)

    if boxes.numel() == 0:
        return grid

    cx = boxes[:, 0]
    cz = boxes[:, 2]
    sx = boxes[:, 3].clamp_min(1e-6)
    sz = boxes[:, 5].clamp_min(1e-6)

    x0 = cx - sx / 2.0
    x1 = cx + sx / 2.0
    z0 = cz - sz / 2.0
    z1 = cz + sz / 2.0

    span_x = (x_max - x_min).clamp_min(1e-6)
    span_z = (z_max - z_min).clamp_min(1e-6)

    gx0 = ((x0 - x_min) / span_x * (W - 1)).clamp(0, W - 1).long()
    gx1 = ((x1 - x_min) / span_x * (W - 1)).clamp(0, W - 1).long()
    gz0 = ((z0 - z_min) / span_z * (H - 1)).clamp(0, H - 1).long()
    gz1 = ((z1 - z_min) / span_z * (H - 1)).clamp(0, H - 1).long()

    for i in range(boxes.shape[0]):
        x_start = min(gx0[i].item(), gx1[i].item())
        x_end   = max(gx0[i].item(), gx1[i].item())
        z_start = min(gz0[i].item(), gz1[i].item())
        z_end   = max(gz0[i].item(), gz1[i].item())
        grid[z_start:z_end + 1, x_start:x_end + 1] = True

    return grid


@torch.no_grad()
def evaluate_grid_metrics(
    model,
    dataloader,
    device,
    grid_size: int = 64,
    conf_threshold: float = 0.1,
):
    """
    Walk2Map-style grid evaluation:

    - Project predicted & GT 3D boxes to top-down occupancy grids.
    - Do cell-wise binary classification (occupied vs free).
    - Compute Precision / Recall / F1 on grid cells.

    Returns:
        dict with 'precision', 'recall', 'f1', 'tp', 'fp', 'fn'.
    """
    model.eval()

    grid_tp = 0
    grid_fp = 0
    grid_fn = 0

    for batch in tqdm(dataloader, desc="Evaluating (grid)"):
        traces = batch['traces'].to(device)            # [B,N,11]
        mask = batch['trace_mask'].to(device)          # [B,N]
        gt_boxes = batch['boxes'].to(device)           # [B,M,6]
        gt_valid_mask = batch['valid_mask'].to(device) # [B,M]

        grid_maps = batch.get('grid_map', None)
        if grid_maps is not None:
            grid_maps = grid_maps.to(device)

        # Forward pass (trace + grid)
        outputs = model(traces, mask, grid_maps=grid_maps)

        pred_boxes = outputs['pred_boxes']             # [B,Q,6]
        pred_classes = outputs['pred_classes']         # [B,Q,3]
        pred_probs = torch.softmax(pred_classes, dim=-1)
        pred_conf = pred_probs.max(dim=-1)[0]          # [B,Q]

        B = pred_boxes.shape[0]

        for b in range(B):
            valid_gt = gt_valid_mask[b]                # [M]
            gt_b = gt_boxes[b, valid_gt]               # [M_valid,6]

            valid_pred = pred_conf[b] > conf_threshold
            pred_b = pred_boxes[b, valid_pred]         # [K,6]

            if gt_b.numel() == 0 and pred_b.numel() == 0:
                continue

            # Scene bounds from traces + boxes
            trace_b = traces[b][mask[b]]
            xs = []
            zs = []
            if trace_b.shape[0] > 0:
                xs.append(trace_b[:, 0])
                zs.append(trace_b[:, 2])
            if gt_b.shape[0] > 0:
                xs.append(gt_b[:, 0] - gt_b[:, 3] / 2.0)
                xs.append(gt_b[:, 0] + gt_b[:, 3] / 2.0)
                zs.append(gt_b[:, 2] - gt_b[:, 5] / 2.0)
                zs.append(gt_b[:, 2] + gt_b[:, 5] / 2.0)
            if pred_b.shape[0] > 0:
                xs.append(pred_b[:, 0] - pred_b[:, 3] / 2.0)
                xs.append(pred_b[:, 0] + pred_b[:, 3] / 2.0)
                zs.append(pred_b[:, 2] - pred_b[:, 5] / 2.0)
                zs.append(pred_b[:, 2] + pred_b[:, 5] / 2.0)

            if len(xs) == 0:
                continue

            xs_cat = torch.cat(xs)
            zs_cat = torch.cat(zs)
            x_min, x_max = xs_cat.min(), xs_cat.max()
            z_min, z_max = zs_cat.min(), zs_cat.max()

            gt_grid = boxes_to_occupancy_grid(gt_b, grid_size, x_min, x_max, z_min, z_max)
            pred_grid = boxes_to_occupancy_grid(pred_b, grid_size, x_min, x_max, z_min, z_max)

            gt_flat = gt_grid.view(-1)
            pred_flat = pred_grid.view(-1)

            tp_cells = (gt_flat & pred_flat).sum().item()
            fp_cells = ((~gt_flat) & pred_flat).sum().item()
            fn_cells = (gt_flat & (~pred_flat)).sum().item()

            grid_tp += tp_cells
            grid_fp += fp_cells
            grid_fn += fn_cells

    if grid_tp + grid_fp > 0:
        precision = grid_tp / (grid_tp + grid_fp + 1e-8)
    else:
        precision = 0.0

    if grid_tp + grid_fn > 0:
        recall = grid_tp / (grid_tp + grid_fn + 1e-8)
    else:
        recall = 0.0

    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    else:
        f1 = 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': grid_tp,
        'fp': grid_fp,
        'fn': grid_fn,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Room-SLAM model (Trace+Grid, Transformer)")
    parser.add_argument('--stage_name', type=str, default="stage1_detect",
                        help="Name for this training stage (used for save_dir)")
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help="Path to checkpoint to load for fine-tuning")
    parser.add_argument('--dropout_prob', type=float, default=0.1,
                        help="Collider dropout probability in data augmentation")
    parser.add_argument('--cov_weight', type=float, default=0.5,
                        help="Weight for coverage_loss")
    parser.add_argument('--avoid_weight', type=float, default=1.0,
                        help="Weight for avoidance_loss")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument('--num_epochs', type=int, default=200,
                        help="Number of epochs to train")
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")

    # Config
    config = {
        'batch_size': 20,
        'num_epochs': args.num_epochs,
        'lr': args.lr,
        'weight_decay': 1e-4,
        'd_model': 128,
        'num_queries': 30,
        'data_dir': '../../dataset/train',
        'val_dir': '../../dataset/val',
        'save_dir': f'./checkpoints_{args.stage_name}',
        'val_every': 1,
        'iou_thresh': 0.5,
    }

    save_path = Path(config['save_dir'])
    save_path.mkdir(exist_ok=True)
    with open(save_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Dataloaders
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
        use_baseline_colliders=False  # no baseline colliders in the simplified setup
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

    # Model: transformer-only trace+grid
    model = build_model(
        num_queries=config['num_queries'],
        d_model=config['d_model'],
        # other args use defaults from build_model
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Loss weights
    weight_dict = {
        'class_loss': 2.0,
        'l1_loss': 5.0,
        'giou_loss': 2.0,
        'coverage_loss': args.cov_weight,
        'avoidance_loss': args.avoid_weight,
        'diversity_loss': 0.1
    }
    print(f"Using loss weights: {weight_dict}")
    criterion = SetCriterion(weight_dict)

    optimizer = AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    # Scheduler: we want to maximize F1 -> scheduler mode = 'max'
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        threshold=1e-3,
        cooldown=1,
        min_lr=1e-6
    )

    start_epoch = 0
    if args.load_checkpoint:
        print(f"\nLoading weights from: {args.load_checkpoint}\n")
        checkpoint = torch.load(args.load_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Training loop
    best_f1 = 0.0

    for epoch in range(start_epoch, config['num_epochs']):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        if (epoch + 1) % config['val_every'] == 0:
            val_loss = validate(model, val_loader, criterion, device)
            grid_metrics = evaluate_grid_metrics(model, val_loader, device, grid_size=64)

            # Optimize F1 directly: scheduler and best-checkpoint selection
            scheduler.step(grid_metrics['f1'])

            print(f"\nEpoch {epoch}: Train {train_loss:.4f} | ValLoss {val_loss:.4f}")
            print(f"  Grid (Walk2Map-style): "
                  f"P={grid_metrics['precision']:.4f} "
                  f"R={grid_metrics['recall']:.4f} "
                  f"F1={grid_metrics['f1']:.4f} "
                  f"(TP={grid_metrics['tp']} FP={grid_metrics['fp']} FN={grid_metrics['fn']})")

            # Save best model on F1
            if grid_metrics['f1'] > best_f1:
                best_f1 = grid_metrics['f1']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'grid_metrics': grid_metrics,
                    'best_f1': best_f1,
                    'config': config
                }, save_path / 'best_model.pth')
                print(f"  ✓ Saved BEST model (F1={best_f1:.4f})")

        else:
            print(f"Epoch {epoch}: Train {train_loss:.4f} | "
                  f"LR={optimizer.param_groups[0]['lr']:.6f}")

        # Regular checkpoint every 10 epochs
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