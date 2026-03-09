"""Integration tests for the full ML analysis pipeline."""
import io
import os
import uuid
import numpy as np
import cv2
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# Set env vars
os.environ['ROBOFLOW_API_KEY'] = 'test-key'

from api_bridge import app, BridgeStore


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Setup a fully mocked environment for pipeline tests."""
    test_db = tmp_path / 'test.db'
    test_uploads = tmp_path / 'uploads'
    test_outputs = tmp_path / 'outputs'
    test_reports = tmp_path / 'reports'
    test_uploads.mkdir()
    test_outputs.mkdir()
    test_reports.mkdir()

    monkeypatch.setattr('api_bridge.UPLOAD_DIR', test_uploads)
    monkeypatch.setattr('api_bridge.OUTPUT_DIR', test_outputs)
    monkeypatch.setattr('api_bridge.REPORT_DIR', test_reports)

    store = BridgeStore(test_db)
    
    # Mock resources
    mock_pipeline = MagicMock()
    mock_masks = {region: np.zeros((100, 100), dtype=np.uint8) for region in [
        'forehead', 'nose', 'left_cheek', 'right_cheek', 'chin'
    ]}
    mock_pipeline.segment.return_value = {
        'masks': mock_masks,
        'metadata': {'seg_mode': 'full'}
    }

    mock_cloud = MagicMock()
    mock_cloud.fetch_multi_scale_consensus.return_value = {
        'preds_a_640': [{'class': 'Acne', 'confidence': 0.8, 'bbox': [10, 10, 20, 20]}],
        'preds_a_1280': [],
        'preds_b': [{'class': 'pustule', 'confidence': 0.9, 'bbox': [10, 10, 20, 20]}],
        'timing_ms': {'total': 500},
        'file_sizes': {'original': 1000}
    }

    # Patch the functions that retrieve the resources
    monkeypatch.setattr('api_bridge.get_pipeline', lambda: mock_pipeline)
    monkeypatch.setattr('api_bridge.get_cloud_engine', lambda: mock_cloud)
    monkeypatch.setattr('api_bridge.get_store', lambda: store)
    
    # Initialize app.state.resources to satisfy the API endpoints
    app.state.resources = {
        'pipeline': mock_pipeline,
        'cloud_engine': mock_cloud,
        'store': store
    }
    
    # Mock visualizers and runtime imports to avoid heavy deps
    monkeypatch.setattr('api_bridge.ensure_runtime_imports', lambda: None)
    monkeypatch.setattr('api_bridge.draw_lesion_boxes', MagicMock(side_effect=lambda img, **kwargs: img))
    
    mock_mapper_inst = MagicMock()
    mock_mapper_inst.ensemble_map_multi_scale.return_value = {
        'forehead': [], 'nose': [], 'left_cheek': [], 'right_cheek': [], 'chin': [],
        '_pipeline_metrics': {'total_time': 100}
    }
    mock_mapper_inst.get_clinical_report.return_value = {
        'clinical_severity': 'Mild',
        'gags_total_score': 5,
        'total_lesions': 3,
        'symmetry_delta': 0.1,
        'regions': {}
    }
    monkeypatch.setattr('api_bridge.EnsembleLesionMapper', MagicMock(return_value=mock_mapper_inst))

    from api_bridge import limiter
    limiter.reset()

    with TestClient(app) as c:
        yield c
    
    store.close()


def test_analyze_full_pipeline_success(client):
    """Verify that /analyze coordinates all components and stores results."""
    # Create a dummy JPEG image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = img_encoded.tobytes()

    session_id = f"test-{uuid.uuid4().hex[:8]}"
    
    response = client.post(
        '/analyze',
        data={
            'session_id': session_id,
            'profile_id': 'test-profile',
            'privacy_mode': 'false',
            'retention_hours': '72'
        },
        files={
            'file': ('test.jpg', img_bytes, 'image/jpeg')
        }
    )

    if response.status_code != 200:
        print(f"Error detail: {response.text}")
    assert response.status_code == 200
    data = response.json()
    assert data['session_id'] == session_id
    assert 'results' in data
    assert data['lesion_count'] > 0
    
    # Check status was updated
    status_resp = client.get(f'/status/{session_id}')
    assert status_resp.status_code == 200
    assert status_resp.json()['stage'] == 'completed'

    # Check database persistence
    session_resp = client.get(f'/session/{session_id}')
    assert session_resp.status_code == 200
    assert session_resp.json()['results'] is not None
    assert session_resp.json()['profile_id'] == 'test-profile'


def test_analyze_parallel_execution_safety(client):
    """Ensure that calling /analyze doesn't crash even if state is shared."""
    # This test primarily ensures the background thread handling doesn't hang the client
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = img_encoded.tobytes()

    r1 = client.post('/analyze', data={'session_id': 's1'}, files={'file': ('t1.jpg', img_bytes, 'image/jpeg')})
    r2 = client.post('/analyze', data={'session_id': 's2'}, files={'file': ('t2.jpg', img_bytes, 'image/jpeg')})
    
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()['session_id'] == 's1'
    assert r2.json()['session_id'] == 's2'


def test_analyze_invalid_image_format_rejection(client):
    """Verify that invalid files are caught before the pipeline starts."""
    response = client.post(
        '/analyze',
        data={'session_id': 'fail-test'},
        files={'file': ('test.txt', b'not an image', 'text/plain')}
    )
    assert response.status_code == 415
    assert 'Only JPEG, PNG, and WEBP' in response.json()['detail']


def test_analyze_empty_file_rejection(client):
    """Verify that empty uploads are rejected."""
    response = client.post(
        '/analyze',
        data={'session_id': 'empty-test'},
        files={'file': ('empty.jpg', b'', 'image/jpeg')}
    )
    assert response.status_code == 400
    assert 'empty' in response.json()['detail'].lower()
