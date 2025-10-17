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
        'batch_size': 4,
        'num_epochs': 200,
        'lr': 2e-4,
        'weight_decay': 1e-4,
        'd_model': 128,
        'num_queries': 30,
        'data_dir': '../../dataset/train',
        'save_dir': './checkpoints',
        'warmup_epochs': 10
    }

    # Create save directory
    Path(config['save_dir']).mkdir(exist_ok=True)

    # Save config
    with open(Path(config['save_dir']) / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Create dataloaders
    train_loader = create_dataloader(
        config['data_dir'],
        batch_size=config['batch_size'],
        shuffle=True
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

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    best_loss = float('inf')

    for epoch in range(config['num_epochs']):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Update scheduler
        scheduler.step()

        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.6f}")

        # Save checkpoint
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'config': config
            }, Path(config['save_dir']) / 'best_model.pth')
            print(f"Saved best model with loss {best_loss:.4f}")

        # Regular checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, Path(config['save_dir']) / f'checkpoint_epoch_{epoch}.pth')

    print("Training completed!")


if __name__ == "__main__":
    main()