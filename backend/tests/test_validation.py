"""Unit tests for validation and helper functions in api_bridge.py."""
import pytest
from unittest.mock import MagicMock

from api_bridge import (
    validate_session_id,
    normalize_retention,
    validate_upload,
    consensus_summary,
    compare_payload,
)
from fastapi import HTTPException


# --- validate_session_id ---

def test_validate_session_id_valid_alphanumeric():
    assert validate_session_id('abc123') == 'abc123'


def test_validate_session_id_valid_with_dashes_underscores():
    assert validate_session_id('my-session_01') == 'my-session_01'


def test_validate_session_id_max_length():
    sid = 'a' * 128
    assert validate_session_id(sid) == sid


def test_validate_session_id_too_long():
    sid = 'a' * 129
    with pytest.raises(HTTPException) as exc:
        validate_session_id(sid)
    assert exc.value.status_code == 400


def test_validate_session_id_empty():
    with pytest.raises(HTTPException) as exc:
        validate_session_id('')
    assert exc.value.status_code == 400


def test_validate_session_id_special_chars():
    with pytest.raises(HTTPException) as exc:
        validate_session_id('session id!')
    assert exc.value.status_code == 400


def test_validate_session_id_path_traversal():
    with pytest.raises(HTTPException) as exc:
        validate_session_id('../etc/passwd')
    assert exc.value.status_code == 400


# --- normalize_retention ---

def test_normalize_retention_normal():
    assert normalize_retention(72) == 72


def test_normalize_retention_below_minimum():
    assert normalize_retention(0) == 1
    assert normalize_retention(-5) == 1


def test_normalize_retention_above_maximum():
    from api_bridge import MAX_RETENTION_HOURS
    assert normalize_retention(999999) == MAX_RETENTION_HOURS


# --- validate_upload ---

def test_validate_upload_wrong_content_type():
    upload = MagicMock()
    upload.content_type = 'image/gif'
    payload = b'\x47\x49\x46\x38'  # GIF magic bytes
    with pytest.raises(HTTPException) as exc:
        validate_upload(upload, payload)
    assert exc.value.status_code == 415


def test_validate_upload_magic_bytes_mismatch_still_valid():
    """Implementation only checks payload starts with *any* valid sig, not cross-check."""
    upload = MagicMock()
    upload.content_type = 'image/jpeg'
    payload = b'\x89PNG' + b'\x00' * 100  # PNG magic bytes claimed as JPEG
    # Should NOT raise — content_type vs magic-byte cross-check is not enforced
    validate_upload(upload, payload)


def test_validate_upload_invalid_magic_bytes():
    """Payload with no recognized image signature should be rejected."""
    upload = MagicMock()
    upload.content_type = 'image/jpeg'
    payload = b'\x00\x00\x00\x00' + b'\x00' * 100
    with pytest.raises(HTTPException) as exc:
        validate_upload(upload, payload)
    assert exc.value.status_code == 400


def test_validate_upload_empty_payload():
    upload = MagicMock()
    upload.content_type = 'image/jpeg'
    payload = b''
    with pytest.raises(HTTPException):
        validate_upload(upload, payload)


def test_validate_upload_valid_jpeg():
    upload = MagicMock()
    upload.content_type = 'image/jpeg'
    payload = b'\xff\xd8\xff' + b'\x00' * 100
    # Should not raise
    validate_upload(upload, payload)


def test_validate_upload_valid_png():
    upload = MagicMock()
    upload.content_type = 'image/png'
    payload = b'\x89PNG' + b'\x00' * 100
    validate_upload(upload, payload)


# --- consensus_summary ---

def test_consensus_summary_empty():
    result = consensus_summary({})
    assert result['verified_lesions'] == 0
    assert result['average_confidence'] == 0.0
    assert result['summary'] == 'No verified lesions detected'


def test_consensus_summary_with_lesions():
    assignments = {
        'nose': [
            {'confidence': 0.8, 'bbox': [0, 0, 10, 10]},
            {'confidence': 0.6, 'bbox': [5, 5, 15, 15]},
        ],
        'left_cheek': [
            {'confidence': 0.9, 'bbox': [20, 20, 30, 30]},
        ],
        'unassigned': [
            {'confidence': 0.3, 'bbox': [50, 50, 60, 60]},
        ],
    }
    result = consensus_summary(assignments)
    assert result['verified_lesions'] == 3
    assert result['unassigned_count'] == 1
    assert len(result['top_regions']) == 2
    assert result['top_regions'][0]['region'] == 'nose'
    assert result['top_regions'][0]['count'] == 2


def test_consensus_summary_excludes_unassigned_from_count():
    assignments = {
        'unassigned': [{'confidence': 0.5}] * 10,
    }
    result = consensus_summary(assignments)
    assert result['verified_lesions'] == 0


# --- compare_payload ---

def test_compare_payload_returns_none_when_no_previous():
    result = compare_payload(None, {'session_id': 's1', 'results': {}})
    assert result is None


def test_compare_payload_returns_none_when_no_results():
    prev = {'session_id': 's0', 'results': None}
    curr = {'session_id': 's1', 'results': {'clinical_analysis': {}}}
    result = compare_payload(prev, curr)
    assert result is None


def test_compare_payload_computes_deltas():
    prev = {
        'session_id': 's0',
        'timestamp': '2025-01-01T00:00:00+00:00',
        'results': {
            'clinical_analysis': {
                'total_lesions': 10,
                'gags_total_score': 20,
                'clinical_severity': 'Moderate',
                'symmetry_delta': 5.0,
                'regions': {
                    'nose': {'count': 3, 'lpi': 1.5},
                },
            },
        },
    }
    curr = {
        'session_id': 's1',
        'timestamp': '2025-01-15T00:00:00+00:00',
        'results': {
            'clinical_analysis': {
                'total_lesions': 7,
                'gags_total_score': 15,
                'clinical_severity': 'Mild',
                'symmetry_delta': 3.0,
                'regions': {
                    'nose': {'count': 2, 'lpi': 1.0},
                },
            },
        },
    }
    result = compare_payload(prev, curr)
    assert result is not None
    assert result['lesion_delta'] == -3
    assert result['gags_delta'] == -5
    assert result['regions']['nose']['count_delta'] == -1
