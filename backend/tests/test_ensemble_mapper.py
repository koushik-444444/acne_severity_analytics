"""Unit tests for EnsembleLesionMapper class label preservation and proximity propagation."""
import numpy as np
import pytest

from face_segmentation.ensemble_mapper import (
    EnsembleLesionMapper,
    PROXIMITY_THRESHOLD,
    _is_typed_label,
)


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
        assert forehead_lesions[0]['type_source'] == 'direct'

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
        assert forehead_lesions[0]['type_source'] == 'none'

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
        assert forehead_lesions[0]['type_source'] == 'direct'

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


# --- type_source field ---

class TestTypeSourceField:
    def test_type_source_direct_for_typed(self):
        """Directly typed detections get type_source='direct'."""
        mapper = _make_ensemble_mapper()
        preds_b = [_rf_pred(25, 100, 20, 20, 0.8, 'pustule')]
        result = mapper.ensemble_map_multi_scale(
            [], [], preds_b, (200, 200), image=None,
        )
        assert result['forehead'][0]['type_source'] == 'direct'

    def test_type_source_none_for_generic_no_neighbors(self):
        """Generic detections with no nearby typed source keep type_source='none'."""
        mapper = _make_ensemble_mapper()
        # Only Model A, no Model B typed sources
        preds_a = [_rf_pred(25, 100, 20, 20, 0.8, 'Acne')]
        result = mapper.ensemble_map_multi_scale(
            preds_a, [], [], (200, 200), image=None,
        )
        assert result['forehead'][0]['type_source'] == 'none'

    def test_type_source_present_on_all_detections(self):
        """Every detection dict includes a type_source key."""
        mapper = _make_ensemble_mapper()
        preds_a = [_rf_pred(25, 100, 20, 20, 0.8, 'Acne')]
        preds_b = [_rf_pred(75, 100, 20, 20, 0.7, 'pustule')]
        result = mapper.ensemble_map_multi_scale(
            preds_a, [], preds_b, (200, 200), image=None,
        )
        for region, dets in result.items():
            for det in dets:
                assert 'type_source' in det, f'Missing type_source in {region}'


# --- Proximity propagation ---

class TestProximityPropagation:
    def test_nearby_generic_gets_typed_label(self):
        """A generic 'Acne' detection near a Model B typed detection gets propagated."""
        mapper = _make_ensemble_mapper()
        # Model A detection at (25, 100) — in forehead [0, 50)
        preds_a = [_rf_pred(25, 100, 20, 20, 0.9, 'Acne')]
        # Model B typed detection VERY close: (27, 102) — within PROXIMITY_THRESHOLD
        # But non-overlapping with NMS (different enough bbox) — actually NMS
        # overlap at IoU > 0.35 will merge these.
        # Use a well-separated pair instead: model A at (25, 50), model B
        # typed at (27, 52) with boxes that overlap (NMS will merge, promote).
        # For proximity test, we need a detection that survives NMS as generic
        # AND a nearby model B typed raw detection.
        # Strategy: place model A detection in forehead at x=25, model B
        # typed detection in forehead at x=30 — close enough for NMS merge,
        # so the typed label will be promoted during NMS (not proximity).
        #
        # Better: place model A at (25, 50) and model B typed at (25, 60)
        # with SMALL boxes that do NOT overlap (IoU < 0.35).
        # Both survive NMS independently.  The model A one stays generic.
        # The model B one is directly typed.
        # Now the model A detection should be PROXIMITY propagated from the
        # model B raw source at (25, 60).
        preds_a_640 = [_rf_pred(25, 50, 10, 10, 0.9, 'Acne')]
        preds_b = [_rf_pred(25, 60, 10, 10, 0.5, 'pustule')]

        result = mapper.ensemble_map_multi_scale(
            preds_a_640, [], preds_b,
            (200, 200), image=None,
        )
        forehead = result['forehead']
        assert len(forehead) == 2
        # Find the one that was generic (higher confidence = model A)
        generic_det = [d for d in forehead if d['confidence'] == 0.9]
        assert len(generic_det) == 1
        det = generic_det[0]
        # With centres at (25, 50) and (25, 60) on a 200x200 image,
        # normalised distance = sqrt(0 + ((50-60)/200)^2) = 0.05 < 0.06
        assert det['class_name'] == 'pustule'
        assert det['type_source'] == 'proximity'
        assert det['severity_grade'] == 3

    def test_distant_generic_stays_generic(self):
        """A generic detection far from any typed source keeps its label."""
        mapper = _make_ensemble_mapper()
        # Model A in forehead (x=25), Model B typed in right_cheek (x=175)
        preds_a = [_rf_pred(25, 100, 20, 20, 0.9, 'Acne')]
        preds_b = [_rf_pred(175, 100, 20, 20, 0.6, 'pustule')]

        result = mapper.ensemble_map_multi_scale(
            preds_a, [], preds_b,
            (200, 200), image=None,
        )
        forehead = result['forehead']
        assert len(forehead) == 1
        assert forehead[0]['class_name'] == 'Acne'
        assert forehead[0]['type_source'] == 'none'

    def test_proximity_chooses_nearest_source(self):
        """When multiple typed sources exist, the nearest one wins."""
        mapper = _make_ensemble_mapper()
        # Model A generic detection at (25, 50)
        preds_a = [_rf_pred(25, 50, 10, 10, 0.9, 'Acne')]
        # Two model B typed detections: pustule at (25, 60) and nodule at (25, 58)
        # Both within proximity threshold of the model A detection.
        preds_b = [
            _rf_pred(25, 60, 10, 10, 0.5, 'pustule'),
            _rf_pred(25, 58, 10, 10, 0.4, 'nodule'),
        ]

        result = mapper.ensemble_map_multi_scale(
            preds_a, [], preds_b,
            (200, 200), image=None,
        )
        forehead = result['forehead']
        generic_det = [d for d in forehead if d['confidence'] == 0.9]
        assert len(generic_det) == 1
        det = generic_det[0]
        # Nodule at (25, 58) is closer to (25, 50) than pustule at (25, 60)
        # Distance to nodule: |58-50|/200 = 0.04
        # Distance to pustule: |60-50|/200 = 0.05
        assert det['class_name'] == 'nodule'
        assert det['type_source'] == 'proximity'

    def test_no_propagation_when_no_model_b_typed(self):
        """Without any Model B typed detections, nothing is propagated."""
        mapper = _make_ensemble_mapper()
        preds_a = [_rf_pred(25, 100, 20, 20, 0.9, 'Acne')]
        # Model B returns generic 'Acne' too
        preds_b = [_rf_pred(27, 102, 20, 20, 0.4, 'Acne')]

        result = mapper.ensemble_map_multi_scale(
            preds_a, [], preds_b,
            (200, 200), image=None,
        )
        forehead = result['forehead']
        # NMS may merge these — regardless, no typed propagation
        for det in forehead:
            assert det['class_name'] == 'Acne'
            assert det['type_source'] == 'none'

    def test_already_typed_not_overwritten_by_proximity(self):
        """A detection with a direct typed label is never overwritten."""
        mapper = _make_ensemble_mapper()
        # Model B typed detection at (25, 50) with 'pustule'
        # Another model B typed detection very close at (25, 55) with 'nodule'
        preds_b = [
            _rf_pred(25, 50, 10, 10, 0.8, 'pustule'),
            _rf_pred(25, 55, 10, 10, 0.5, 'nodule'),
        ]

        result = mapper.ensemble_map_multi_scale(
            [], [], preds_b,
            (200, 200), image=None,
        )
        forehead = result['forehead']
        # Both may survive NMS (small boxes).
        # The higher-conf pustule should keep its direct type.
        pustule_dets = [d for d in forehead if d['confidence'] == 0.8]
        if pustule_dets:
            assert pustule_dets[0]['class_name'] == 'pustule'
            assert pustule_dets[0]['type_source'] == 'direct'

    def test_proximity_threshold_boundary(self):
        """Detection just inside PROXIMITY_THRESHOLD distance gets propagated."""
        mapper = _make_ensemble_mapper()
        # Place generic model A at (75, 100) in 200x200 image
        # and typed model B at a distance just inside PROXIMITY_THRESHOLD
        # Normalised distance = sqrt((dx/W)^2 + (dy/H)^2)
        # Use dy only: dy/H = 11/200 = 0.055 < 0.06
        preds_a = [_rf_pred(75, 100, 10, 10, 0.9, 'Acne')]
        preds_b = [_rf_pred(75, 111, 10, 10, 0.5, 'papules')]

        result = mapper.ensemble_map_multi_scale(
            preds_a, [], preds_b,
            (200, 200), image=None,
        )
        nose = result['nose']
        generic_det = [d for d in nose if d['confidence'] == 0.9]
        assert len(generic_det) == 1
        det = generic_det[0]
        # Distance = 11/200 = 0.055 < PROXIMITY_THRESHOLD (should propagate)
        assert det['class_name'] == 'papules'
        assert det['type_source'] == 'proximity'
