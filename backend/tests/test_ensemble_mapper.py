"""Unit tests for EnsembleLesionMapper class label preservation."""
import numpy as np
import pytest

from face_segmentation.ensemble_mapper import EnsembleLesionMapper, _is_typed_label


def _make_ensemble_mapper(regions=None):
    """Create an EnsembleLesionMapper with simple 200x200 region masks.

    Each region covers the full height and a horizontal stripe.
    """
    if regions is None:
        regions = ['forehead', 'nose', 'left_cheek', 'right_cheek']
    masks = {}
    h, w = 200, 200
    step = w // len(regions)
    for i, name in enumerate(regions):
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[:, i * step:(i + 1) * step] = 255
        masks[name] = mask
    return EnsembleLesionMapper(masks)


def _rf_pred(x, y, w, h, conf, cls='Acne'):
    """Build a single Roboflow-style prediction dict."""
    return {
        'x': x, 'y': y, 'width': w, 'height': h,
        'confidence': conf, 'class': cls,
    }


# --- _is_typed_label ---

class TestIsTypedLabel:
    def test_generic_labels(self):
        assert not _is_typed_label('acne')
        assert not _is_typed_label('Acne')
        assert not _is_typed_label('acne_detected')
        assert not _is_typed_label('lesion')
        assert not _is_typed_label('')

    def test_typed_labels(self):
        assert _is_typed_label('pustule')
        assert _is_typed_label('Papules')
        assert _is_typed_label('blackheads')
        assert _is_typed_label('nodule')
        assert _is_typed_label('cyst')
        assert _is_typed_label('dark spot')


# --- ensemble_map_multi_scale class label preservation ---

class TestEnsembleClassLabels:
    def test_model_b_typed_label_preserved(self):
        """Model B returns typed labels — they should survive the ensemble."""
        mapper = _make_ensemble_mapper()
        # Place a detection in the 'forehead' region (x=25, which is in [0, 50))
        preds_a_640 = []
        preds_a_1280 = []
        preds_b = [_rf_pred(25, 100, 20, 20, 0.7, 'pustule')]

        result = mapper.ensemble_map_multi_scale(
            preds_a_640, preds_a_1280, preds_b,
            (200, 200), image=None,
        )
        forehead_lesions = result['forehead']
        assert len(forehead_lesions) == 1
        assert forehead_lesions[0]['class_name'] == 'pustule'
        assert forehead_lesions[0]['severity_grade'] == 3

    def test_model_a_generic_label_default_grade(self):
        """Model A returns generic 'Acne' — grade defaults to 2."""
        mapper = _make_ensemble_mapper()
        preds_a_640 = [_rf_pred(25, 100, 20, 20, 0.8, 'Acne')]
        preds_a_1280 = []
        preds_b = []

        result = mapper.ensemble_map_multi_scale(
            preds_a_640, preds_a_1280, preds_b,
            (200, 200), image=None,
        )
        forehead_lesions = result['forehead']
        assert len(forehead_lesions) == 1
        assert forehead_lesions[0]['class_name'] == 'Acne'
        assert forehead_lesions[0]['severity_grade'] == 2

    def test_typed_label_promoted_during_nms(self):
        """When overlapping detections are merged, the typed label wins."""
        mapper = _make_ensemble_mapper()
        # Two nearly-overlapping detections in the same spot
        # Model A (higher conf but generic), Model B (lower conf but typed)
        preds_a_640 = [_rf_pred(25, 100, 20, 20, 0.9, 'Acne')]
        preds_a_1280 = []
        preds_b = [_rf_pred(26, 101, 20, 20, 0.6, 'nodule')]

        result = mapper.ensemble_map_multi_scale(
            preds_a_640, preds_a_1280, preds_b,
            (200, 200), image=None,
        )
        # Should keep only 1 lesion (NMS) but with 'nodule' promoted
        forehead_lesions = result['forehead']
        assert len(forehead_lesions) == 1
        assert forehead_lesions[0]['class_name'] == 'nodule'
        assert forehead_lesions[0]['severity_grade'] == 4

    def test_empty_predictions_returns_empty(self):
        mapper = _make_ensemble_mapper()
        result = mapper.ensemble_map_multi_scale(
            [], [], [], (200, 200), image=None,
        )
        total = sum(len(items) for items in result.values())
        assert total == 0

    def test_missing_class_key_defaults_to_acne(self):
        """Prediction without a 'class' key should default to 'acne'."""
        mapper = _make_ensemble_mapper()
        pred = {'x': 25, 'y': 100, 'width': 20, 'height': 20, 'confidence': 0.7}
        result = mapper.ensemble_map_multi_scale(
            [pred], [], [], (200, 200), image=None,
        )
        forehead_lesions = result['forehead']
        assert len(forehead_lesions) == 1
        assert forehead_lesions[0]['class_name'] == 'acne'
        assert forehead_lesions[0]['severity_grade'] == 2

    def test_multiple_typed_labels_across_regions(self):
        """Multiple detections in different regions preserve their own labels."""
        mapper = _make_ensemble_mapper()
        # forehead region [0, 50), nose region [50, 100)
        preds_b = [
            _rf_pred(25, 100, 20, 20, 0.8, 'blackheads'),
            _rf_pred(75, 100, 20, 20, 0.7, 'papules'),
        ]
        result = mapper.ensemble_map_multi_scale(
            [], [], preds_b, (200, 200), image=None,
        )
        forehead = result['forehead']
        nose = result['nose']
        assert len(forehead) == 1
        assert forehead[0]['class_name'] == 'blackheads'
        assert forehead[0]['severity_grade'] == 1
        assert len(nose) == 1
        assert nose[0]['class_name'] == 'papules'
        assert nose[0]['severity_grade'] == 2
