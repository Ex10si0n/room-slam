import argparse, json, os, subprocess, numpy as np
def compute_3d_iou(box_a, box_b):
    c_a, s_a = np.array(list(box_a['center'].values())), np.array(list(box_a['size'].values()))
    c_b, s_b = np.array(list(box_b['center'].values())), np.array(list(box_b['size'].values()))
    min_a, max_a = c_a - s_a/2, c_a + s_a/2
    min_b, max_b = c_b - s_b/2, c_b + s_b/2
    inter = np.prod(np.maximum(0, np.minimum(max_a, max_b) - np.maximum(min_a, min_b)))
    union = np.prod(s_a) + np.prod(s_b) - inter
    return 0.0 if union == 0 else inter / union

def nms(preds, thresh=0.15):
    if not preds: return []
    preds = sorted(preds, key=lambda x: x['confidence'], reverse=True)
    keep = []
    while preds:
        cur = preds.pop(0); keep.append(cur)
        preds = [p for p in preds if compute_3d_iou(cur, p) < thresh]
    return keep

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input'); p.add_argument('--mode', choices=['coverage', 'fidelity'])
    a = p.parse_args()
    ckpt = './checkpoints_stage11_enhanced/best_model.pth' if a.mode == 'coverage' else './checkpoints_stage12_high_fidelity/best_model.pth'
    if not os.path.exists(ckpt): print(f"Missing {ckpt}"); exit(1)
    subprocess.run(['python', 'inference.py', '--checkpoint', ckpt, '--input', a.input, '--output', f'temp_{a.mode}.json', '--threshold', '0.7'])
    with open(f'temp_{a.mode}.json') as f: raw = json.load(f)
    clean = nms(raw.get('colliders', raw))
    final = f'demo_{a.mode}_predictions.json'
    with open(final, 'w') as f: json.dump({'colliders': clean}, f, indent=2)
    subprocess.run(['python', 'visualize.py', '--input', a.input, '--colliders', os.path.join(os.path.dirname(a.input), 'colliders.json'), '--predictions', final, '--output', f'demo_{a.mode}.png'])
