"""
Visualization utilities for face segmentation results.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# Distinct colors for each face region (BGR format)
REGION_COLORS = {
    "nose": (0, 165, 255),       # Orange
    "left_cheek": (0, 255, 0),   # Green
    "right_cheek": (255, 0, 0),  # Blue
    "forehead": (255, 0, 255),   # Magenta
    "chin": (0, 255, 255),       # Yellow
    "jawline_neck": (150, 150, 150), # Gray for neck
}

# Colors for BiSeNet 19-class parsing visualization
PARSING_COLORS = [
    [0, 0, 0],        # 0  background
    [204, 0, 0],      # 1  skin
    [76, 153, 0],     # 2  l_brow
    [204, 204, 0],    # 3  r_brow
    [51, 51, 255],    # 4  l_eye
    [204, 0, 204],    # 5  r_eye
    [0, 255, 255],    # 6  eye_g
    [255, 204, 204],  # 7  l_ear
    [102, 51, 0],     # 8  r_ear
    [255, 0, 0],      # 9  ear_r
    [102, 204, 0],    # 10 nose
    [255, 255, 0],    # 11 mouth
    [0, 0, 153],      # 12 u_lip
    [0, 0, 204],      # 13 l_lip
    [255, 51, 153],   # 14 neck
    [0, 204, 204],    # 15 necklace
    [0, 51, 0],       # 16 cloth
    [255, 153, 51],   # 17 hair
    [0, 204, 0],      # 18 hat
]


def draw_parsing_map(
    image: np.ndarray,
    parsing: np.ndarray,
    alpha: float = 0.6,
) -> np.ndarray:
    """
    Overlay the 19-class BiSeNet parsing map on the original image.

    Args:
        image: BGR image, shape (H, W, 3).
        parsing: Parsing map, shape (H, W), values 0-18.
        alpha: Blend weight for the overlay.

    Returns:
        Blended image, shape (H, W, 3).
    """
    vis = image.copy()
    color_map = np.zeros_like(image)

    for label_id in range(1, 19):
        mask = parsing == label_id
        color = PARSING_COLORS[label_id]
        color_map[mask] = color

    result = cv2.addWeighted(vis, 1 - alpha, color_map, alpha, 0)
    return result


def anonymize_image(
    image: np.ndarray,
    masks: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    De-identify the image by blacking out eyes, hair, and background.
    Only keeps the 6 clinical regions visible.
    """
    # Create an aggregate mask of all clinical regions
    combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for mask in masks.values():
        if mask.shape[:2] != combined_mask.shape:
            mask = cv2.resize(mask, (combined_mask.shape[1], combined_mask.shape[0]))
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # Apply to image
    anonymized = cv2.bitwise_and(image, image, mask=combined_mask)
    return anonymized


def calculate_erythema_index(
    image: np.ndarray,
    mask: np.ndarray,
) -> float:
    """
    Calculate a digital Erythema (redness) Index for a region.
    Proxy formula: (R - G) / (R + G)
    """
    if np.sum(mask) == 0:
        return 0.0
        
    # Extract pixels in mask
    pixels = image[mask > 0] # BGR
    mean_b = np.mean(pixels[:, 0])
    mean_g = np.mean(pixels[:, 1])
    mean_r = np.mean(pixels[:, 2])
    
    # Normalized Redness Index
    ei = (mean_r - mean_g) / (mean_r + mean_g + 1e-6)
    # Scale to 0-100 for readability
    return round(float(ei) * 100, 2)


def draw_region_masks(
    image: np.ndarray,
    masks: Dict[str, np.ndarray],
    alpha: float = 0.5,
    draw_labels: bool = True,
    lesions: Optional[Dict[str, List[Dict]]] = None,
    clinical_report: Optional[Dict] = None,
) -> np.ndarray:
    """
    Overlay the 5 region masks on the original image with distinct colors.
    Optionally draw region-specific color-coded lesions and a clinical summary.

    Args:
        image: BGR image, shape (H, W, 3).
        masks: Dict of region name -> binary mask (H, W).
        alpha: Blend weight for the overlay.
        draw_labels: Whether to draw region names on the image.
        lesions: Optional dict of assigned lesions from LesionMapper.
        clinical_report: Optional dict containing GAGS score and severity.

    Returns:
        Blended image with region overlays and diagnostic data.
    """
    overlay = image.copy()
    color_layer = np.zeros_like(image)
    for region_name, mask in masks.items():
        color = REGION_COLORS.get(region_name, (128, 128, 128))
        if mask.shape[:2] != color_layer.shape[:2]:
            mask = cv2.resize(mask, (color_layer.shape[1], color_layer.shape[0]))
        color_layer[mask > 0] = color

    overlay = cv2.addWeighted(overlay, 1 - alpha, color_layer, alpha, 0)

    if draw_labels:
        for region_name, mask in masks.items():
            ys, xs = np.where(mask > 0)
            if len(xs) == 0: continue
            cx, cy = int(np.mean(xs)), int(np.mean(ys))
            color = REGION_COLORS.get(region_name, (128, 128, 128))
            cv2.putText(overlay, region_name, (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(overlay, region_name, (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw color-coded lesions
    if lesions:
        for region_name, detections in lesions.items():
            color = REGION_COLORS.get(region_name, (128, 128, 128))
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                # Ensemble Confidence Visuals
                conf_level = det.get("confidence_level", "Unknown")
                thickness = 2
                
                if "High" in conf_level:
                    thickness = 3 # Bold for consensus
                elif "Review" in conf_level:
                    color = (255, 255, 255) # White for Model B unique
                
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
                cv2.rectangle(overlay, (x1-1, y1-1), (x2+1, y2+1), (255, 255, 255), 1)
                cx, cy = det["center"]
                cv2.circle(overlay, (cx, cy), 3, color, -1)
                
                # Small text for confidence if high-res
                if overlay.shape[0] > 1000:
                    cv2.putText(overlay, conf_level.split("(")[0], (x1, y1-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # Draw Clinical Summary Watermark
    if clinical_report:
        overlay = _draw_watermark(overlay, clinical_report)

    return overlay


def draw_lesion_boxes(
    image: np.ndarray,
    lesions: Optional[Dict[str, List[Dict]]] = None,
    clinical_report: Optional[Dict] = None,
) -> np.ndarray:
    """
    Draw only acne lesion bounding boxes on the original image.
    No facial region segmentation is shown to the user.
    """
    overlay = image.copy()

    if lesions:
        for region_name, detections in lesions.items():
            if region_name == 'unassigned':
                color = (255, 255, 255)
            else:
                color = REGION_COLORS.get(region_name, (0, 242, 255))

            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                conf_level = det.get('confidence_level', 'Unknown')
                thickness = 2

                if 'High' in conf_level:
                    thickness = 3
                elif 'Review' in conf_level:
                    color = (255, 255, 255)

                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
                cv2.rectangle(overlay, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 255, 255), 1)
                cx, cy = det['center']
                cv2.circle(overlay, (cx, cy), 3, color, -1)

                if overlay.shape[0] > 1000:
                    cv2.putText(
                        overlay,
                        conf_level.split('(')[0],
                        (x1, max(12, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (255, 255, 255),
                        1,
                    )

    if clinical_report:
        overlay = _draw_watermark(overlay, clinical_report)

    return overlay

def _draw_watermark(image: np.ndarray, report: Dict) -> np.ndarray:
    """Draw a professional clinical summary box in the top-left corner."""
    H, W = image.shape[:2]
    # Box dimensions
    bw, bh = 280, 140
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (10 + bw, 10 + bh), (0, 0, 0), -1)
    image = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

    # Text content
    lines = [
        f"CLINICAL ACNE REPORT",
        f"--------------------",
        f"Severity: {report['clinical_severity']}",
        f"GAGS Score: {report['gags_total_score']}",
        f"Total Lesions: {report['total_lesions']}",
        f"Symmetry Delta: {report['symmetry_delta']}%"
    ]
    
    # Highlight severity color
    sev_colors = {
        "None": (0, 255, 0),
        "Mild": (0, 255, 255),
        "Moderate": (0, 165, 255),
        "Severe": (0, 0, 255),
        "Very Severe / Cystic": (0, 0, 139)
    }
    sev_color = sev_colors.get(report['clinical_severity'], (255, 255, 255))

    y_off = 35
    for i, line in enumerate(lines):
        color = (255, 255, 255)
        if "Severity:" in line: color = sev_color
        cv2.putText(image, line, (20, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1 if i > 0 else 2)
        y_off += 20
        
    return image


def draw_landmarks(
    image: np.ndarray,
    landmarks: np.ndarray,
    draw_indices: bool = False,
    color: Tuple[int, int, int] = (0, 255, 0),
    radius: int = 2,
) -> np.ndarray:
    """
    Draw facial landmarks on an image.

    Args:
        image: BGR image.
        landmarks: (N, 2) array of (x, y) coordinates (468 for MediaPipe).
        draw_indices: Whether to draw landmark index numbers.
        color: BGR color for landmark dots.
        radius: Radius of landmark dots.

    Returns:
        Image with landmarks drawn.
    """
    vis = image.copy()
    for i, (x, y) in enumerate(landmarks):
        cv2.circle(vis, (int(x), int(y)), radius, color, -1)
        if draw_indices:
            cv2.putText(
                vis, str(i), (int(x) + 3, int(y) - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1,
            )
    return vis


def create_mask_grid(
    image: np.ndarray,
    masks: Dict[str, np.ndarray],
    cols: int = 3,
) -> np.ndarray:
    """
    Create a grid visualization showing original image + each region mask.

    Returns:
        Grid image with all regions shown individually.
    """
    H, W = image.shape[:2]
    # Resize for grid
    cell_w, cell_h = 320, 320

    cells = []
    # Original image
    resized_orig = cv2.resize(image, (cell_w, cell_h))
    cv2.putText(
        resized_orig, "Original", (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
    )
    cells.append(resized_orig)

    for name, mask in masks.items():
        # Create colored mask on black background
        cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
        mask_resized = cv2.resize(mask, (cell_w, cell_h))
        color = REGION_COLORS.get(name, (128, 128, 128))
        cell[mask_resized > 0] = color
        # Also show mask on original
        orig_small = cv2.resize(image, (cell_w, cell_h))
        cell = cv2.addWeighted(orig_small, 0.4, cell, 0.6, 0)
        cv2.putText(
            cell, name, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
        # Show pixel count
        pix_count = int(np.sum(mask > 0))
        cv2.putText(
            cell, f"{pix_count}px", (10, cell_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1,
        )
        cells.append(cell)

    # Pad to fill grid
    while len(cells) % cols != 0:
        cells.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))

    # Assemble grid
    rows = []
    for i in range(0, len(cells), cols):
        row = np.hstack(cells[i : i + cols])
        rows.append(row)
    grid = np.vstack(rows)
    return grid


def save_individual_masks(
    masks: Dict[str, np.ndarray],
    output_dir: str,
    prefix: str = "",
) -> List[str]:
    """
    Save each region mask as a separate PNG file.

    Returns:
        List of saved file paths.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    saved = []
    for name, mask in masks.items():
        fname = f"{prefix}{name}.png" if prefix else f"{name}.png"
        path = os.path.join(output_dir, fname)
        cv2.imwrite(path, mask)
        saved.append(path)
    return saved
