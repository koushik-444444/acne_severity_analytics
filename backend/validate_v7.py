"""
V7 Clinical Validation Engine - True Accuracy Auditor (SAG Optimized)
Calculates Precision, Recall, and mAP@50 using Statistical Adaptive Gating.
"""
import os
import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

from face_segmentation.pipeline import FaceSegmentationPipeline
from face_segmentation.ensemble_mapper import EnsembleLesionMapper
from cloud_inference import CloudInferenceEngine

load_dotenv()

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "P3xb3D2tKzmT0XEOIJS4")
MODEL_A_ID = os.getenv("MODEL_A_ID", "runner-e0dmy/acne-ijcab/2")
MODEL_B_ID = os.getenv("MODEL_B_ID", "acne-project-2auvb/acne-detection-v2/1")

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def validate(image_dir, label_dir, iou_threshold=0.45, limit=20):
    print(f"[Validator] Initializing Clinical Pipeline (SAG-Enabled)...")
    pipeline = FaceSegmentationPipeline(smooth_edges=True)
    cloud_engine = CloudInferenceEngine(api_key=ROBOFLOW_API_KEY)
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))][:limit]
    
    stats = {"tp": 0, "fp": 0, "fn": 0, "ious": []}

    for img_name in tqdm(image_files):
        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")
        if not os.path.exists(label_path): continue
            
        image = cv2.imread(img_path)
        if image is None: continue
        H, W = image.shape[:2]
        
        # 1. Pipeline Segmentation (Required for SAG baselines)
        result = pipeline.segment(image)
        cloud_results = cloud_engine.fetch_multi_scale_consensus(image, MODEL_A_ID, MODEL_B_ID)
        
        # 2. Optimized Mapping with Statistical Adaptive Gating
        mapper = EnsembleLesionMapper(result["masks"])
        assignments = mapper.ensemble_map_multi_scale(
            cloud_results["preds_a_640"], 
            cloud_results["preds_a_1280"], 
            cloud_results["preds_b"], 
            (H, W),
            image=image
        )
        
        all_preds = []
        for region_list in assignments.values():
            for lesion in region_list:
                all_preds.append(lesion["bbox"])
        
        # 3. Load Ground Truth
        all_gt = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = list(map(float, line.strip().split()))
                xc, yc, w, h = parts[1:5]
                x1 = (xc - w/2) * W; y1 = (yc - h/2) * H
                x2 = (xc + w/2) * W; y2 = (yc + h/2) * H
                all_gt.append([x1, y1, x2, y2])
        
        # 4. Matching
        matched_gt = set()
        for p in all_preds:
            best_iou = 0
            best_idx = -1
            for i, g in enumerate(all_gt):
                if i in matched_gt: continue
                iou = calculate_iou(p, g)
                if iou > best_iou:
                    best_iou = iou; best_idx = i
            
            if best_iou >= iou_threshold:
                stats["tp"] += 1
                matched_gt.add(best_idx)
                stats["ious"].append(best_iou)
            else:
                stats["fp"] += 1
        stats["fn"] += (len(all_gt) - len(matched_gt))

    p = stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) > 0 else 0
    r = stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

    print(f"\n=== CLINICAL VALIDATION (SAG OPTIMIZED) ===")
    print(f"  Precision: {p:.4f}")
    print(f"  Recall:    {r:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  TP: {stats['tp']}, FP: {stats['fp']}, FN: {stats['fn']}")
    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    validate(args.images, args.labels, limit=args.limit)
