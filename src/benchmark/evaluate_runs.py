import argparse
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import os
from dataloader import create_dataloader
from model import build_model

def compute_iou_matrix_3d(boxes1, boxes2):
    if len(boxes1) == 0 or len(boxes2) == 0:
        return torch.zeros((len(boxes1), len(boxes2)), device=boxes1.device)
    b1_min = boxes1[:, :3] - boxes1[:, 3:] / 2
    b1_max = boxes1[:, :3] + boxes1[:, 3:] / 2
    b2_min = boxes2[:, :3] - boxes2[:, 3:] / 2
    b2_max = boxes2[:, :3] + boxes2[:, 3:] / 2
    inter_min = torch.maximum(b1_min.unsqueeze(1), b2_min.unsqueeze(0))
    inter_max = torch.minimum(b1_max.unsqueeze(1), b2_max.unsqueeze(0))
    inter_vol = torch.prod(torch.clamp(inter_max - inter_min, min=0), dim=2)
    vol1 = boxes1[:, 3:].prod(dim=1)
    vol2 = boxes2[:, 3:].prod(dim=1)
    union = vol1.unsqueeze(1) + vol2.unsqueeze(0) - inter_vol
    return inter_vol / (union + 1e-6)

@torch.no_grad()
def evaluate(checkpoint, data_dir, iou_thresh, device):
    if not os.path.exists(checkpoint): return
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = ckpt.get('config', {})
    model = build_model(num_queries=cfg.get('num_queries', 30), d_model=cfg.get('d_model', 256), use_baseline_colliders=False).to(device)
    try: model.load_state_dict(ckpt['model_state_dict'], strict=True)
    except: model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    loader = create_dataloader(data_dir, batch_size=1, shuffle=False, use_baseline_colliders=False)
    tp, fp, fn = 0, 0, 0
    for batch in tqdm(loader):
        out = model(batch['traces'].to(device), batch['trace_mask'].to(device))
        probs = torch.softmax(out['pred_classes'], -1)
        scores, _ = probs.max(-1)
        pred = out['pred_boxes'][0][scores[0] > 0.7]
        gt = batch['boxes'][0][batch['valid_mask'][0]].to(device)
        if len(pred) == 0: fn += len(gt); continue
        if len(gt) == 0: fp += len(pred); continue
        iou = compute_iou_matrix_3d(pred, gt)
        r, c = linear_sum_assignment(-iou.cpu().numpy())
        matches = (iou[r, c] >= iou_thresh).sum().item()
        tp += matches; fp += len(pred) - matches; fn += len(gt) - matches
    p = tp / (tp + fp + 1e-6); r = tp / (tp + fn + 1e-6); f1 = 2 * p * r / (p + r + 1e-6)
    print(f"Results: P={p:.4f} R={r:.4f} F1={f1:.4f}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint'); p.add_argument('--data_dir'); p.add_argument('--iou', type=float, default=0.5)
    a = p.parse_args()
    evaluate(a.checkpoint, a.data_dir, a.iou, 'cuda' if torch.cuda.is_available() else 'cpu')
