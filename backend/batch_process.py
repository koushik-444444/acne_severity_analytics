"""
Batch processing script - API-ONLY GOLD EDITION
Iterates through a directory and organizes results into clinical folders.
"""
import os
import cv2
import argparse
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from roboflow import Roboflow
from dotenv import load_dotenv

# Load clinical environment
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from face_segmentation.pipeline import FaceSegmentationPipeline
from face_segmentation.ensemble_mapper import EnsembleLesionMapper
from cloud_inference import CloudInferenceEngine
from usage_tracker import log_api_call
from face_segmentation.utils.visualization import (
    draw_region_masks,
    save_individual_masks,
    anonymize_image
)
try:
    from extract_parts import extract_parts
except ImportError:
    extract_parts = None

# Clinical Configuration (Environment fallback)
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
if not ROBOFLOW_API_KEY:
    print("[Fatal] ROBOFLOW_API_KEY environment variable is required. Set it in .env or export it.")
    sys.exit(1)
assert ROBOFLOW_API_KEY is not None
MODEL_A_ID = os.getenv("MODEL_A_ID", "runner-e0dmy/acne-ijcab/2")
MODEL_B_ID = os.getenv("MODEL_B_ID", "acne-project-2auvb/acne-detection-v2/1")
MAX_API_DIM = int(os.getenv("MAX_API_DIM", 2048))

def process_batch(input_dir, output_root, smooth=True, anonymize=False):
    # Initialize pipeline
    pipeline = FaceSegmentationPipeline(
        bisenet_weights="weights/79999_iter.pth",
        smooth_edges=smooth
    )
    
    cloud_engine = CloudInferenceEngine(api_key=ROBOFLOW_API_KEY)
    input_path = Path(input_dir)
    image_extensions = [".jpg", ".jpeg", ".png"]
    
    image_files = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]
    if not image_files:
        for d in input_path.iterdir():
            if d.is_dir():
                image_files.extend([f for f in d.iterdir() if f.suffix.lower() in image_extensions])

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Starting API-GOLD Batch Audit on {len(image_files)} patients...")
    all_stats = []

    for img_file in image_files:
        img_name = img_file.stem
        img_output_dir = Path(output_root) / img_name
        img_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            image = cv2.imread(str(img_file))
            if image is None: continue
            H, W = image.shape[:2]
            
            # 1. Local Segmentation
            result = pipeline.segment(image, return_intermediates=True)
            
            # 2. Remote API Ensemble (Parallel)
            print(f"  [{img_name}] Calling APIs in parallel...")
            cloud_results = cloud_engine.fetch_multi_scale_consensus(
                image, 
                MODEL_A_ID, 
                MODEL_B_ID
            )

            # 3. Local Fusion
            mapper = EnsembleLesionMapper(result["masks"])
            assignments = mapper.ensemble_map_multi_scale(
                cloud_results["preds_a_640"], 
                cloud_results["preds_a_1280"], 
                cloud_results["preds_b"], 
                (H, W),
                image=image
            )
            clinical_analysis = mapper.get_clinical_report(assignments)

            # 4. Record Results
            row = {
                "image_name": img_name,
                "total_face_pixels": result["metadata"]["total_face_pixels"],
                "total_lesions": clinical_analysis.get("total_lesions", 0),
                "gags_score": clinical_analysis.get("gags_total_score", 0),
                "severity": clinical_analysis.get("clinical_severity", "None"),
                "symmetry_delta": clinical_analysis.get("symmetry_delta", 0)
            }
            for region, data in clinical_analysis.get("regions", {}).items():
                row[f"{region}_lpi"] = data["lpi"]
                row[f"{region}_count"] = data["count"]
            all_stats.append(row)

            # 5. Save Artifacts
            result["metadata"]["clinical_analysis"] = clinical_analysis
            result["metadata"]["lesions"] = assignments
            with open(img_output_dir / f"{img_name}_results.json", "w") as f:
                json.dump(result["metadata"], f, indent=4)

            # Save individual PNG masks to a subfolder for extract_parts to find
            mask_save_dir = img_output_dir / "masks"
            save_individual_masks(result["masks"], str(mask_save_dir), prefix=f"{img_name}_")
            
            overlay = draw_region_masks(image, result["masks"], lesions=assignments, clinical_report=clinical_analysis)
            cv2.imwrite(str(img_output_dir / f"{img_name}_diagnostic.jpg"), overlay)
            
            if anonymize:
                anon = anonymize_image(image, result["masks"])
                cv2.imwrite(str(img_output_dir / f"{img_name}_anonymized.jpg"), anon)

            # Pass the correct mask directory to extract_parts
            if extract_parts is not None:
                extract_parts(str(img_file), str(mask_save_dir), str(img_output_dir / "face_parts"))
            print(f"  [{img_name}] Completed - Grade: {row['severity']}")

            
        except Exception as e:
            print(f"  [Error] {img_name}: {e}")

    if all_stats:
        pd.DataFrame(all_stats).to_csv(Path(output_root) / "batch_clinical_report.csv", index=False)
        print(f"\n[Done] Final Audit Report saved to {output_root}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="test_images/test_images_face")
    parser.add_argument("--output", default="batch_api_audit")
    parser.add_argument("--anonymize", action="store_true")
    parser.add_argument("--no-smooth", action="store_true")
    args = parser.parse_args()
    process_batch(args.input, args.output, not args.no_smooth, args.anonymize)
