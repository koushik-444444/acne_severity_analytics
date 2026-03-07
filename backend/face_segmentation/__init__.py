"""
Face Segmentation Package - Two-Stage Pipeline
================================================
Stage 1: BiSeNet face parsing (semantic segmentation of facial components)
Stage 2: Landmark-based geometric partitioning (cheek, forehead, chin extraction)

Combined output: pixel-accurate masks for nose, left_cheek, right_cheek, forehead, chin.
"""


def __getattr__(name):
    """Lazy imports to avoid requiring all dependencies at import time."""
    if name == "FaceSegmentationPipeline":
        from face_segmentation.pipeline import FaceSegmentationPipeline
        return FaceSegmentationPipeline
    if name == "FaceParser":
        from face_segmentation.face_parser import FaceParser
        return FaceParser
    if name == "LandmarkRegionExtractor":
        from face_segmentation.landmark_extractor import LandmarkRegionExtractor
        return LandmarkRegionExtractor
    if name == "RegionCombiner":
        from face_segmentation.region_combiner import RegionCombiner
        return RegionCombiner
    raise AttributeError(f"module 'face_segmentation' has no attribute {name}")


__all__ = [
    "FaceSegmentationPipeline",
    "FaceParser",
    "LandmarkRegionExtractor",
    "RegionCombiner",
]
