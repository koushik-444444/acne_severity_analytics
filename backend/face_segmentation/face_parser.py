"""
Stage 1: BiSeNet Face Parser

Loads the pretrained BiSeNet model (CelebAMask-HQ, 19 classes) and produces
per-pixel semantic segmentation maps. Extracts the 'nose' mask directly
and the 'skin' mask for downstream geometric partitioning.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from face_segmentation.models.bisenet import BiSeNet


class FaceParser:
    """
    BiSeNet-based face parser.

    Produces a 19-class parsing map from a face image using a pretrained
    BiSeNet model trained on CelebAMask-HQ.

    CelebAMask-HQ label mapping:
        0: background    1: skin        2: l_brow      3: r_brow
        4: l_eye         5: r_eye       6: eye_g       7: l_ear
        8: r_ear         9: ear_r      10: nose       11: mouth
       12: u_lip        13: l_lip      14: neck       15: necklace
       16: cloth        17: hair       18: hat
    """

    LABEL_NAMES = {
        0: "background", 1: "skin", 2: "l_brow", 3: "r_brow",
        4: "l_eye", 5: "r_eye", 6: "eye_g", 7: "l_ear",
        8: "r_ear", 9: "ear_r", 10: "nose", 11: "mouth",
        12: "u_lip", 13: "l_lip", 14: "neck", 15: "necklace",
        16: "cloth", 17: "hair", 18: "hat",
    }

    # Indices of face-interior components (NOT background, hair, neck, accessories)
    FACE_COMPONENT_IDS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}

    def __init__(
        self,
        weight_path: str,
        n_classes: int = 19,
        device: Optional[str] = None,
        input_size: Tuple[int, int] = (512, 512),
    ):
        """
        Args:
            weight_path: Path to pretrained BiSeNet weights (.pth file).
            n_classes: Number of output classes (19 for CelebAMask-HQ).
            device: 'cuda', 'cpu', or None for auto-detect.
            input_size: (H, W) to resize input images for the network.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.n_classes = n_classes
        self.input_size = input_size

        self.net = BiSeNet(n_classes=n_classes)
        self._load_weights(weight_path)
        self.net.to(self.device)
        self.net.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    def _load_weights(self, weight_path: str) -> None:
        """Load pretrained weights, handling both GPU and CPU saved models."""
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(
                f"BiSeNet weights not found at: {weight_path}\n"
                f"Download from: https://drive.google.com/open?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812\n"
                f"Or run: python download_weights.py"
            )
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
        self.net.load_state_dict(state_dict)

    @torch.no_grad()
    def parse(self, image: np.ndarray) -> np.ndarray:
        """
        Run face parsing on a BGR image (OpenCV format).

        Args:
            image: BGR image as numpy array, shape (H, W, 3).

        Returns:
            Parsing map of shape (H, W) with integer class labels 0-18.
        """
        orig_h, orig_w = image.shape[:2]

        # Convert BGR -> RGB -> PIL
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        resized = pil_img.resize(
            (self.input_size[1], self.input_size[0]), Image.Resampling.BILINEAR
        )

        # Transform and infer
        tensor = self.transform(resized).unsqueeze(0).to(self.device)
        out = self.net(tensor)[0]  # Main output head

        # Argmax over classes
        parsing = out.squeeze(0).cpu().numpy().argmax(0)  # (512, 512)

        # Resize back to original resolution
        if (orig_h, orig_w) != self.input_size:
            parsing = cv2.resize(
                parsing.astype(np.uint8),
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )

        return parsing.astype(np.uint8)

    def get_component_mask(
        self, parsing: np.ndarray, label_id: int
    ) -> np.ndarray:
        """
        Extract binary mask for a specific component from parsing map.

        Args:
            parsing: Parsing map from parse(), shape (H, W).
            label_id: Class ID (0-18).

        Returns:
            Binary mask, shape (H, W), dtype uint8 (0 or 255).
        """
        return ((parsing == label_id) * 255).astype(np.uint8)

    def get_nose_mask(self, parsing: np.ndarray) -> np.ndarray:
        """Extract nose mask (label 10)."""
        return self.get_component_mask(parsing, 10)

    def get_skin_mask(self, parsing: np.ndarray) -> np.ndarray:
        """Extract skin mask (label 1). Includes cheeks, forehead, chin."""
        return self.get_component_mask(parsing, 1)

    def get_face_mask(self, parsing: np.ndarray) -> np.ndarray:
        """
        Extract combined face region mask (all face components).
        Useful for defining the overall face boundary.
        """
        mask = np.zeros(parsing.shape[:2], dtype=np.uint8)
        for label_id in self.FACE_COMPONENT_IDS:
            mask[parsing == label_id] = 255
        return mask

    def get_all_masks(self, parsing: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract all 19 component masks from a parsing map.

        Returns:
            Dict mapping label names to binary masks.
        """
        masks = {}
        for label_id, label_name in self.LABEL_NAMES.items():
            if label_id == 0:
                continue
            masks[label_name] = self.get_component_mask(parsing, label_id)
        return masks
