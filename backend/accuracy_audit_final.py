import os
import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from roboflow import Roboflow

# Direct logic to avoid any package/caching issues
load_dotenv()
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "P3xb3D2tKzmT0XEOIJS4")
MODEL_B_ID = os.getenv("MODEL_B_ID", "acne-project-2auvb/acne-detection-v2/1")

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def run_audit(image_dir, label_dir, limit=20):
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    parts = MODEL_B_ID.split("/")
    # Format: runner-e0dmy/acne-ijcab/2 or acne-project-2auvb/acne-detection-v2/1
    # Handle both formats
    if len(parts) == 3:
        ws, proj, ver = parts
    else:
        ws, proj, ver = "acne-project-2auvb", parts[0], parts[1]
        
    model = rf.workspace(ws).project(proj).version(int(ver)).model
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))][:limit]
    
    stats = {"tp": 0, "fp": 0, "fn": 0, "ious": []}
    
    print("\n[AUDIT] Starting Fresh Clinical Audit on " + str(len(image_files)) + " samples...")
    
    for img_name in tqdm(image_files):
        img_path = os.path.join(image_dir, img_name)
        lbl_name = os.path.splitext(img_name)[0] + ".txt"
        lbl_path = os.path.join(label_dir, lbl_name)
        
        if not os.path.exists(lbl_path): continue
        
        image = cv2.imread(img_path)
        if image is None: continue
        H, W = image.shape[:2]
        
        # 1. API Detections
        preds = model.predict(img_path, confidence=40).json().get("predictions", [])
        
        all_preds = []
        for p in preds:
            x, y, w, h = p["x"], p["y"], p["width"], p["height"]
            x1, y1, x2, y2 = int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)
            
            # COLOR FILTER (REDNESS CHECK)
            patch = image[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
            if patch.size > 0:
                mean_g = np.mean(patch[:, :, 1])
                mean_r = np.mean(patch[:, :, 2])
                redness = (mean_r - mean_g) / (mean_r + mean_g + 1e-6)
                if redness < 0.05: continue # Reject non-red
            
            all_preds.append([x1, y1, x2, y2])
            
        # 2. Load GT
        all_gt = []
        with open(lbl_path, 'r') as f:
            for line in f.readlines():
                pts = list(map(float, line.strip().split()))
                if len(pts) < 5: continue
                xc, yc, gw, gh = pts[1:5]
                gx1 = (xc - gw/2) * W; gy1 = (yc - gh/2) * H
                gx2 = (xc + gw/2) * W; gy2 = (yc + gh/2) * H
                all_gt.append([gx1, gy1, gx2, gy2])
                
        # 3. Matching
        matched_gt = set()
        for p in all_preds:
            best_iou = 0
            best_idx = -1
            for i, g in enumerate(all_gt):
                if i in matched_gt: continue
                iou = calculate_iou(p, g)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            
            if best_iou > 0.45:
                stats["tp"] += 1
                matched_gt.add(best_idx)
                stats["ious"].append(best_iou)
            else:
                stats["fp"] += 1
        stats["fn"] += (len(all_gt) - len(matched_gt))

    # Summary
    p_val = stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) > 0 else 0
    r_val = stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) > 0 else 0
    f1_val = 2 * p_val * r_val / (p_val + r_val) if (p_val + r_val) > 0 else 0
    
    print("\n=== FINAL AUDIT RESULTS ===")
    print("  Precision: " + "{:.4f}".format(p_val))
    print("  Recall:    " + "{:.4f}".format(r_val))
    print("  F1-Score:  " + "{:.4f}".format(f1_val))
    print("  Mean IoU:  " + "{:.4f}".format(np.mean(stats['ious']) if stats['ious'] else 0))
    print("  TP: " + str(stats['tp']) + ", FP: " + str(stats['fp']) + ", FN: " + str(stats['fn']))

if __name__ == "__main__":
    run_audit(r"E:\data_generation\dataset\original\images\val", r"E:\data_generation\dataset\original\labels\val")
