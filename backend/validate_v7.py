"""
V7 Clinical Validation Engine - True Accuracy Auditor (SAG Optimized)
Calculates Precision, Recall, and mAP@50 using Statistical Adaptive Gating.

Phase 5: Added timing instrumentation, per-image stats, JSON output.
Phase 7: Configurable SAG_Z_THRESHOLD, NMS_IOU_THRESHOLD, match IoU.
"""
import os
import sys
import cv2
import json
import time
import argparse
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
if not ROBOFLOW_API_KEY:
    print("[Fatal] ROBOFLOW_API_KEY environment variable is required. Set it in .env or export it.")
    sys.exit(1)
assert ROBOFLOW_API_KEY is not None
MODEL_A_ID = os.getenv("MODEL_A_ID", "runner-e0dmy/acne-ijcab/2")
MODEL_B_ID = os.getenv("MODEL_B_ID", "acne-project-2auvb/acne-detection-v2/1")

from utils import calculate_iou


def validate(image_dir, label_dir, iou_threshold=0.45, limit=20, output_json=None,
             sag_z=None, nms_iou=None):
    """Run validation benchmark on labeled images.

    Args:
        image_dir: Path to validation images.
        label_dir: Path to YOLO-format label files.
        iou_threshold: Minimum IoU for a true positive match.
        limit: Maximum images to process.
        output_json: Optional path to save detailed results as JSON.
        sag_z: Override SAG_Z_THRESHOLD for this run.
        nms_iou: Override NMS_IOU_THRESHOLD for this run.

    Returns:
        Dict with aggregate and per-image statistics.
    """
    # Apply threshold overrides via env vars before importing the mapper
    if sag_z is not None:
        os.environ['SAG_Z_THRESHOLD'] = str(sag_z)
    if nms_iou is not None:
        os.environ['NMS_IOU_THRESHOLD'] = str(nms_iou)

    # Import pipeline modules after setting env vars so they pick up overrides.
    # Use importlib to force re-read of env vars even if already imported.
    import importlib
    import face_segmentation.ensemble_mapper as em_mod
    importlib.reload(em_mod)
    from face_segmentation.pipeline import FaceSegmentationPipeline
    from face_segmentation.ensemble_mapper import EnsembleLesionMapper
    from cloud_inference import CloudInferenceEngine

    effective_sag_z = float(os.environ.get('SAG_Z_THRESHOLD', '0.5'))
    effective_nms_iou = float(os.environ.get('NMS_IOU_THRESHOLD', '0.30'))

    print(f"[Validator] Initializing Clinical Pipeline (SAG-Enabled)...")
    print(f"[Validator] SAG_Z_THRESHOLD={effective_sag_z}, NMS_IOU_THRESHOLD={effective_nms_iou}, match_iou={iou_threshold}")
    pipeline = FaceSegmentationPipeline(smooth_edges=True)
    cloud_engine = CloudInferenceEngine(api_key=ROBOFLOW_API_KEY)
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))][:limit]
    
    stats = {"tp": 0, "fp": 0, "fn": 0, "ious": []}
    per_image = []
    timing_samples = []

    for img_name in tqdm(image_files):
        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")
        if not os.path.exists(label_path):
            continue
            
        image = cv2.imread(img_path)
        if image is None:
            continue
        H, W = image.shape[:2]
        
        t0 = time.perf_counter()

        # 1. Pipeline Segmentation (Required for SAG baselines)
        result = pipeline.segment(image)

        t_seg = time.perf_counter()

        cloud_results = cloud_engine.fetch_multi_scale_consensus(image, MODEL_A_ID, MODEL_B_ID)
        
        t_cloud = time.perf_counter()

        # 2. Optimized Mapping with Statistical Adaptive Gating
        mapper = EnsembleLesionMapper(result["masks"])
        assignments = mapper.ensemble_map_multi_scale(
            cloud_results["preds_a_640"], 
            cloud_results["preds_a_1280"], 
            cloud_results["preds_b"], 
            (H, W),
            image=image
        )

        t_map = time.perf_counter()

        # Extract pipeline metrics if present
        pipeline_metrics = assignments.pop('_pipeline_metrics', None)
        
        all_preds = []
        for region, region_list in assignments.items():
            if region.startswith('_') or not isinstance(region_list, list):
                continue
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
        image_ious = []
        for p in all_preds:
            best_iou = 0
            best_idx = -1
            for i, g in enumerate(all_gt):
                if i in matched_gt:
                    continue
                iou = calculate_iou(p, g)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            
            if best_iou >= iou_threshold:
                stats["tp"] += 1
                matched_gt.add(best_idx)
                stats["ious"].append(best_iou)
                image_ious.append(best_iou)
            else:
                stats["fp"] += 1
        fn_count = len(all_gt) - len(matched_gt)
        stats["fn"] += fn_count

        t_end = time.perf_counter()
        img_timing = {
            'segmentation_ms': round((t_seg - t0) * 1000, 1),
            'cloud_ms': round((t_cloud - t_seg) * 1000, 1),
            'mapping_ms': round((t_map - t_cloud) * 1000, 1),
            'total_ms': round((t_end - t0) * 1000, 1),
        }
        timing_samples.append(img_timing)

        img_tp = len(matched_gt)
        img_fp = len(all_preds) - img_tp
        img_stats = {
            'image': img_name,
            'gt_count': len(all_gt),
            'pred_count': len(all_preds),
            'tp': img_tp,
            'fp': img_fp,
            'fn': fn_count,
            'mean_iou': round(float(np.mean(image_ious)), 4) if image_ious else 0.0,
            'timing': img_timing,
        }
        if pipeline_metrics:
            img_stats['pipeline_metrics'] = pipeline_metrics
        # Cloud timing from Phase 5 instrumentation
        cloud_timing = cloud_results.get('_timing')
        if cloud_timing:
            img_stats['cloud_timing'] = cloud_timing
        per_image.append(img_stats)

    p = stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) > 0 else 0
    r = stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    mean_iou = float(np.mean(stats["ious"])) if stats["ious"] else 0.0

    # Timing aggregation
    avg_timing = {}
    if timing_samples:
        for key in timing_samples[0]:
            vals = [s[key] for s in timing_samples if key in s]
            avg_timing[f'{key}_mean'] = round(sum(vals) / len(vals), 1) if vals else None

    print(f"\n=== CLINICAL VALIDATION (SAG OPTIMIZED) ===")
    print(f"  Images processed: {len(per_image)}")
    print(f"  Precision: {p:.4f}")
    print(f"  Recall:    {r:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Mean IoU:  {mean_iou:.4f}")
    print(f"  TP: {stats['tp']}, FP: {stats['fp']}, FN: {stats['fn']}")
    if avg_timing:
        print(f"  Avg Total Time: {avg_timing.get('total_ms_mean', 'N/A')}ms")

    full_results = {
        'config': {
            'image_dir': image_dir,
            'label_dir': label_dir,
            'iou_threshold': iou_threshold,
            'sag_z_threshold': effective_sag_z,
            'nms_iou_threshold': effective_nms_iou,
            'limit': limit,
            'images_processed': len(per_image),
            'model_a_id': MODEL_A_ID,
            'model_b_id': MODEL_B_ID,
        },
        'aggregate': {
            'precision': round(p, 4),
            'recall': round(r, 4),
            'f1_score': round(f1, 4),
            'mean_iou': round(mean_iou, 4),
            'tp': stats['tp'],
            'fp': stats['fp'],
            'fn': stats['fn'],
        },
        'timing': avg_timing,
        'per_image': per_image,
    }

    if output_json:
        with open(output_json, 'w') as f:
            json.dump(full_results, f, indent=2)
        print(f"  Results saved to: {output_json}")

    return full_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--match-iou", type=float, default=0.45,
                        help="IoU threshold for TP matching (default 0.45)")
    parser.add_argument("--sag-z", type=float, default=None,
                        help="Override SAG_Z_THRESHOLD (default from env or 0.5)")
    parser.add_argument("--nms-iou", type=float, default=None,
                        help="Override NMS_IOU_THRESHOLD (default from env or 0.30)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save JSON results file")
    args = parser.parse_args()
    validate(args.images, args.labels, iou_threshold=args.match_iou,
             limit=args.limit, output_json=args.output,
             sag_z=args.sag_z, nms_iou=args.nms_iou)
