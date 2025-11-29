
import json
import numpy as np
import os

def analyze_alignment(pred_file):
    print(f"Analyzing alignment for: {pred_file}")
    if not os.path.exists(pred_file):
        print(f"File not found: {pred_file}")
        return

    with open(pred_file, 'r') as f:
        data = json.load(f)
        colliders = data.get('colliders', data)

    if not colliders:
        print("No colliders found.")
        return

    # Extract centers [x, z]
    centers = np.array([[c['center']['x'], c['center']['z']] for c in colliders])
    
    clusters = []
    visited = np.zeros(len(colliders), dtype=bool)
    threshold = 0.5 # meters proximity to consider potential wall alignment

    # 1. Identify X-aligned walls (Vertical in Top-Down view)
    # These have similar X coordinates but spread out Z coordinates
    indices_x = np.argsort(centers[:, 0])
    
    current_cluster = [indices_x[0]]
    for i in range(1, len(indices_x)):
        idx = indices_x[i]
        prev_idx = current_cluster[-1]
        
        # If X is close
        if abs(centers[idx, 0] - centers[prev_idx, 0]) < threshold:
            current_cluster.append(idx)
        else:
            # End of candidate cluster
            if len(current_cluster) > 1:
                pts = centers[current_cluster]
                std_x = np.std(pts[:, 0])
                std_z = np.std(pts[:, 1])
                # Criteria: Spread in Z > Spread in X (elongated in Z) AND Spread in Z > 1.0m (not just a clutter)
                if std_z > std_x and std_z > 0.5:
                    clusters.append({'axis': 'X (Vertical Wall)', 'std': std_x, 'count': len(current_cluster), 'indices': current_cluster})
                    visited[current_cluster] = True
            current_cluster = [idx]
    
    # Check last
    if len(current_cluster) > 1:
        pts = centers[current_cluster]
        std_x = np.std(pts[:, 0])
        std_z = np.std(pts[:, 1])
        if std_z > std_x and std_z > 0.5:
            clusters.append({'axis': 'X (Vertical Wall)', 'std': std_x, 'count': len(current_cluster), 'indices': current_cluster})
            visited[current_cluster] = True

    # 2. Identify Z-aligned walls (Horizontal in Top-Down view) from remaining
    # These have similar Z coordinates but spread out X coordinates
    remaining_indices = np.where(~visited)[0]
    if len(remaining_indices) > 0:
        sorted_rem = remaining_indices[np.argsort(centers[remaining_indices, 1])]
        
        current_cluster = [sorted_rem[0]]
        for i in range(1, len(sorted_rem)):
            idx = sorted_rem[i]
            prev_idx = current_cluster[-1]
            
            if abs(centers[idx, 1] - centers[prev_idx, 1]) < threshold:
                current_cluster.append(idx)
            else:
                if len(current_cluster) > 1:
                    pts = centers[current_cluster]
                    std_x = np.std(pts[:, 0])
                    std_z = np.std(pts[:, 1])
                    if std_x > std_z and std_x > 0.5:
                        clusters.append({'axis': 'Z (Horizontal Wall)', 'std': std_z, 'count': len(current_cluster), 'indices': current_cluster})
                current_cluster = [idx]
        
        if len(current_cluster) > 1:
            pts = centers[current_cluster]
            std_x = np.std(pts[:, 0])
            std_z = np.std(pts[:, 1])
            if std_x > std_z and std_x > 0.5:
                clusters.append({'axis': 'Z (Horizontal Wall)', 'std': std_z, 'count': len(current_cluster), 'indices': current_cluster})

    # Report Results
    print(f"\nIdentified {len(clusters)} structural clusters (walls).")
    total_std = 0.0
    
    for i, c in enumerate(clusters):
        print(f"  Cluster {i+1} [{c['axis']}]: {c['count']} colliders, Alignment Std Dev: {c['std']:.4f} m")
        total_std += c['std']
        
    if clusters:
        avg_error = total_std / len(clusters)
        print(f"\n==> Average Alignment Error: {avg_error:.4f} m")
        print("    (Lower is better. < 0.10m typically indicates high fidelity)")
    else:
        print("\nNo distinct wall structures identified (possibly sparse or scattered predictions).")

if __name__ == "__main__":
    analyze_alignment('val_predictions_stage12_nms.json')
