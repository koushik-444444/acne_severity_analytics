"""
Face Region Segmentation - API-ONLY GOLD EDITION
STRICTLY NO MODEL DOWNLOADS. USES ROBOFLOW HOSTED API + LOCAL WBF FUSION.
"""
import argparse
import os
import sys
import json
import time

import cv2
import numpy as np
from roboflow import Roboflow
from dotenv import load_dotenv

# Load clinical environment
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from face_segmentation.pipeline import FaceSegmentationPipeline
from face_segmentation.ensemble_mapper import EnsembleLesionMapper
from cloud_inference import CloudInferenceEngine
from usage_tracker import log_api_call, get_usage_stats
from face_segmentation.utils.visualization import (
    draw_parsing_map,
    draw_region_masks,
    draw_landmarks,
    create_mask_grid,
    save_individual_masks,
    anonymize_image
)

# --- Configuration (Load from .env or fallback) ---
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "P3xb3D2tKzmT0XEOIJS4")
MODEL_A_ID = os.getenv("MODEL_A_ID", "runner-e0dmy/acne-ijcab/2")
MODEL_B_ID = os.getenv("MODEL_B_ID", "acne-project-2auvb/acne-detection-v2/1")
# Max resolution for API upload (10MB limit / Latency optimization)
MAX_API_DIM = int(os.getenv("MAX_API_DIM", 2048))

def parse_args():
    parser = argparse.ArgumentParser(description="Clinical API-Only Acne Diagnostic Engine")
    parser.add_argument("--image", type=str, required=True, help="Path to clinical photo")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--visualize", action="store_true", help="Save clinical overlays")
    parser.add_argument("--anonymize", action="store_true", help="Enable Privacy Mode")
    parser.add_argument("--smooth", action="store_true", help="Apply clinical soft-masking")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # 1. Initialize Local Infrastructure (Segmentation & Landmarks)
    pipeline = FaceSegmentationPipeline(
        bisenet_weights="weights/79999_iter.pth",
        smooth_edges=args.smooth
    )

    image = cv2.imread(args.image)
    if image is None: return 1
    H, W = image.shape[:2]

    # Run Local Segmentation
    t_start = time.perf_counter()
    print(f"[V7-API] Segmenting patient photo: {args.image}")
    result = pipeline.segment(image, return_intermediates=True)
    t_seg = time.perf_counter() - t_start
    print(f"  -> Segmentation Complete: {t_seg:.2f}s")

    # 2. Remote Clinical Inference (Parallel API-Consensus)
    t_api_start = time.perf_counter()
    print(f"[V7-Consensus] Starting Parallel Multi-Scale API Audit...")
    cloud_engine = CloudInferenceEngine(api_key=ROBOFLOW_API_KEY)
    cloud_results = cloud_engine.fetch_multi_scale_consensus(
        image, 
        MODEL_A_ID, 
        MODEL_B_ID
    )
    t_api = time.perf_counter() - t_api_start
    print(f"  -> Cloud API Consensus (3 Parallel Streams) Complete: {t_api:.2f}s")

    # 3. Local Fusion & Multi-Scale Consensus
    t_fusion_start = time.perf_counter()
    mapper = EnsembleLesionMapper(result["masks"])
    print("[V7-Consensus] Fusing 3 streams using Localization Averaging & Confidence Voting...")
    assignments = mapper.ensemble_map_multi_scale(
        cloud_results["preds_a_640"], 
        cloud_results["preds_a_1280"], 
        cloud_results["preds_b"], 
        (H, W)
    )
    clinical_report = mapper.get_clinical_report(assignments)
    t_fusion = time.perf_counter() - t_fusion_start
    print(f"  -> Local WBF Fusion Complete: {t_fusion:.2f}s")
    
    total_engine_time = t_seg + t_api + t_fusion
    print(f"\n[Performance Summary] Total Engine Time: {total_engine_time:.2f}s (Initialization Overlap: {time.perf_counter() - t_start:.2f}s)")

    # 4. Reporting
    result["metadata"]["clinical_analysis"] = clinical_report
    result["metadata"]["lesions"] = assignments

    print(f"\n=== CLINICAL REPORT (API-GOLD) ===")
    print(f"  Severity Grade: {clinical_report['clinical_severity']}")
    print(f"  GAGS Score:     {clinical_report['gags_total_score']}")
    print(f"  Lesion Imbalance: {clinical_report['symmetry_delta']}%")
    print(f"  API Quota Used:   {get_usage_stats()} calls")

    # Save Outputs
    base = os.path.splitext(os.path.basename(args.image))[0]
    with open(os.path.join(args.output, f"{base}_results.json"), "w") as f:
        json.dump(result["metadata"], f, indent=4)

    if args.visualize:
        overlay = draw_region_masks(image, result["masks"], lesions=assignments, clinical_report=clinical_report)
        cv2.imwrite(os.path.join(args.output, f"{base}_diagnostic.jpg"), overlay)
        if args.anonymize:
            anon = anonymize_image(image, result["masks"])
            cv2.imwrite(os.path.join(args.output, f"{base}_anonymized.jpg"), anon)
        print(f"[Saved] Clinical Workspace results to {args.output}")

    return 0

if __name__ == "__main__":
    main()
