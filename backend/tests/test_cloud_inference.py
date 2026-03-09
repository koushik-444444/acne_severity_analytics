"""Unit tests for CloudInferenceEngine configuration and task dispatch."""
import os
import cv2
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MODEL_A = 'runner-e0dmy/acne-ijcab/2'
MODEL_B = 'acne-project-2auvb/acne-detection-v2/1'


def _make_engine(**env_overrides):
    """Construct a CloudInferenceEngine with controlled env vars.

    ``env_overrides`` are temporarily injected into ``os.environ`` so
    the constructor picks them up via ``os.getenv``.
    The ``Roboflow`` client is mocked to avoid real API key validation.
    """
    env = {
        'MAX_API_DIM': '2048',
        'MAX_MODEL_B_DIM': '1280',
        'JPEG_QUALITY': '85',
        'ENABLE_DUAL_SCALE_A': 'false',
    }
    env.update(env_overrides)
    with patch.dict(os.environ, env, clear=False), \
         patch('cloud_inference.Roboflow') as MockRoboflow:
        from cloud_inference import CloudInferenceEngine
        engine = CloudInferenceEngine(api_key='test-key')
    return engine


# ---------------------------------------------------------------------------
# Constructor / env-var tests
# ---------------------------------------------------------------------------

class TestCloudInferenceInit:
    def test_defaults(self):
        engine = _make_engine()
        assert engine.max_api_dim == 2048
        assert engine.max_model_b_dim == 1280
        assert engine.enable_dual_scale_a is False

    def test_custom_model_b_dim(self):
        engine = _make_engine(MAX_MODEL_B_DIM='1024')
        assert engine.max_model_b_dim == 1024

    def test_dual_scale_enabled(self):
        for val in ('true', 'True', '1', 'yes'):
            engine = _make_engine(ENABLE_DUAL_SCALE_A=val)
            assert engine.enable_dual_scale_a is True, f'Failed for {val}'

    def test_dual_scale_disabled(self):
        for val in ('false', 'False', '0', 'no', ''):
            engine = _make_engine(ENABLE_DUAL_SCALE_A=val)
            assert engine.enable_dual_scale_a is False, f'Failed for {val}'


# ---------------------------------------------------------------------------
# JPEG quality constant
# ---------------------------------------------------------------------------

class TestJpegQuality:
    def test_default_jpeg_quality(self):
        with patch.dict(os.environ, {'JPEG_QUALITY': '85'}, clear=False):
            # Re-import to pick up the env var at module level
            import importlib
            import cloud_inference
            importlib.reload(cloud_inference)
            assert cloud_inference.JPEG_QUALITY == 85

    def test_custom_jpeg_quality(self):
        with patch.dict(os.environ, {'JPEG_QUALITY': '70'}, clear=False):
            import importlib
            import cloud_inference
            importlib.reload(cloud_inference)
            assert cloud_inference.JPEG_QUALITY == 70


# ---------------------------------------------------------------------------
# fetch_multi_scale_consensus — task dispatch
# ---------------------------------------------------------------------------

class TestFetchMultiScaleConsensus:
    """Verify the correct tasks are dispatched depending on config."""

    def _run_consensus(self, engine, model_a=MODEL_A, model_b=MODEL_B):
        """Call fetch_multi_scale_consensus with _fetch_single_scale mocked."""
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Mock _fetch_single_scale to return a recognisable list
        def side_effect(image, model_id, target_dim):
            return [{'model': model_id, 'dim': target_dim}]

        with patch.object(engine, '_fetch_single_scale', side_effect=side_effect) as mock_fetch:
            result = engine.fetch_multi_scale_consensus(dummy_image, model_a, model_b)
        return result, mock_fetch

    def test_dual_scale_disabled_sends_two_tasks(self):
        engine = _make_engine(ENABLE_DUAL_SCALE_A='false')
        result, mock_fetch = self._run_consensus(engine)

        # Should have called _fetch_single_scale exactly 2 times
        assert mock_fetch.call_count == 2
        # preds_a_640 should be empty
        assert result['preds_a_640'] == []
        # preds_a_1280 should have data
        assert len(result['preds_a_1280']) == 1
        assert result['preds_a_1280'][0]['dim'] == 1280
        # preds_b should have data
        assert len(result['preds_b']) == 1
        assert result['preds_b'][0]['dim'] == 1280  # max_model_b_dim

    def test_dual_scale_enabled_sends_three_tasks(self):
        engine = _make_engine(ENABLE_DUAL_SCALE_A='true')
        result, mock_fetch = self._run_consensus(engine)

        assert mock_fetch.call_count == 3
        # All three should have data
        assert len(result['preds_a_640']) == 1
        assert result['preds_a_640'][0]['dim'] == 640
        assert len(result['preds_a_1280']) == 1
        assert result['preds_a_1280'][0]['dim'] == 1280
        assert len(result['preds_b']) == 1

    def test_model_b_uses_max_model_b_dim(self):
        engine = _make_engine(MAX_MODEL_B_DIM='1024')
        result, mock_fetch = self._run_consensus(engine)

        assert len(result['preds_b']) == 1
        assert result['preds_b'][0]['dim'] == 1024

    def test_error_in_one_task_returns_empty_list(self):
        engine = _make_engine(ENABLE_DUAL_SCALE_A='true')
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)

        call_count = [0]

        def side_effect(image, model_id, target_dim):
            call_count[0] += 1
            if target_dim == 640:
                raise RuntimeError('Simulated 413 error')
            return [{'model': model_id, 'dim': target_dim}]

        with patch.object(engine, '_fetch_single_scale', side_effect=side_effect):
            result = engine.fetch_multi_scale_consensus(dummy_image, MODEL_A, MODEL_B)

        # Failed task should produce empty list
        assert result['preds_a_640'] == []
        # Other tasks should succeed
        assert len(result['preds_a_1280']) == 1
        assert len(result['preds_b']) == 1

    def test_all_keys_always_present(self):
        """Return dict always has preds_a_640, preds_a_1280, preds_b keys."""
        for dual in ('true', 'false'):
            engine = _make_engine(ENABLE_DUAL_SCALE_A=dual)
            dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)

            with patch.object(engine, '_fetch_single_scale', return_value=[]):
                result = engine.fetch_multi_scale_consensus(dummy_image, MODEL_A, MODEL_B)

            assert 'preds_a_640' in result
            assert 'preds_a_1280' in result
            assert 'preds_b' in result


# ---------------------------------------------------------------------------
# _fetch_single_scale — scaling logic
# ---------------------------------------------------------------------------

class TestFetchSingleScale:
    """Test the scaling and JPEG quality logic (Roboflow model mocked)."""

    def _mock_engine(self):
        """Create an engine with the Roboflow client fully mocked."""
        engine = _make_engine()
        mock_model = MagicMock()
        mock_model.predict.return_value.json.return_value = {
            'predictions': [
                {'x': 50, 'y': 50, 'width': 10, 'height': 10, 'confidence': 0.8, 'class': 'Acne'},
            ]
        }
        engine.rf = MagicMock()
        engine.rf.workspace.return_value.project.return_value.version.return_value.model = mock_model
        return engine, mock_model

    @patch('cloud_inference.log_api_call')
    @patch('cloud_inference.os.remove')
    def test_downscaling_applied_for_large_image(self, mock_remove, mock_log):
        engine, mock_model = self._mock_engine()
        # Image larger than target_dim (640)
        large_image = np.zeros((1000, 1000, 3), dtype=np.uint8)

        from cloud_inference import SCALED_CONFIDENCE
        result = engine._fetch_single_scale(large_image, MODEL_A, 640)

        # Should have predictions with re-scaled coords
        assert len(result) == 1
        # Coords should be scaled back: 50 / (640/1000) = 78.125
        scale = 640 / 1000
        assert abs(result[0]['x'] - 50 / scale) < 0.1
        # Model was called with scaled confidence
        mock_model.predict.assert_called_once()
        call_args = mock_model.predict.call_args
        assert call_args[1]['confidence'] == SCALED_CONFIDENCE

    @patch('cloud_inference.log_api_call')
    @patch('cloud_inference.os.remove')
    def test_no_downscaling_for_small_image(self, mock_remove, mock_log):
        engine, mock_model = self._mock_engine()
        # Image smaller than target_dim
        small_image = np.zeros((500, 500, 3), dtype=np.uint8)

        from cloud_inference import NATIVE_CONFIDENCE
        result = engine._fetch_single_scale(small_image, MODEL_A, 1280)

        assert len(result) == 1
        # Coords should NOT be re-scaled
        assert result[0]['x'] == 50
        # Model was called with native confidence
        call_args = mock_model.predict.call_args
        assert call_args[1]['confidence'] == NATIVE_CONFIDENCE

    @patch('cloud_inference.log_api_call')
    @patch('cloud_inference.os.remove')
    @patch('cloud_inference.cv2.imwrite')
    def test_jpeg_quality_passed_to_imwrite(self, mock_imwrite, mock_remove, mock_log):
        engine, mock_model = self._mock_engine()
        image = np.zeros((1000, 1000, 3), dtype=np.uint8)

        engine._fetch_single_scale(image, MODEL_A, 640)

        # cv2.imwrite should have been called with JPEG quality params
        assert mock_imwrite.called
        call_args = mock_imwrite.call_args
        from cloud_inference import JPEG_QUALITY
        assert call_args[0][2] == [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
