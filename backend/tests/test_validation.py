"""Unit tests for validation and helper functions in api_bridge.py."""
import pytest
from unittest.mock import MagicMock

from api_bridge import (
    validate_session_id,
    validate_profile_id,
    validate_cursor,
    normalize_retention,
    validate_upload,
    consensus_summary,
    compare_payload,
    summarize_stream_provenance,
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


# --- validate_profile_id ---

def test_validate_profile_id_valid_simple():
    assert validate_profile_id('user-1') == 'user-1'


def test_validate_profile_id_valid_with_dots():
    assert validate_profile_id('user.name_01') == 'user.name_01'


def test_validate_profile_id_max_length():
    pid = 'a' * 64
    assert validate_profile_id(pid) == pid


def test_validate_profile_id_too_long():
    pid = 'a' * 65
    with pytest.raises(HTTPException) as exc:
        validate_profile_id(pid)
    assert exc.value.status_code == 400


def test_validate_profile_id_empty():
    with pytest.raises(HTTPException) as exc:
        validate_profile_id('')
    assert exc.value.status_code == 400


def test_validate_profile_id_special_chars():
    with pytest.raises(HTTPException) as exc:
        validate_profile_id('profile id!')
    assert exc.value.status_code == 400


def test_validate_profile_id_path_traversal():
    with pytest.raises(HTTPException) as exc:
        validate_profile_id('../etc/passwd')
    assert exc.value.status_code == 400


def test_validate_profile_id_html_injection():
    with pytest.raises(HTTPException) as exc:
        validate_profile_id('<script>alert(1)</script>')
    assert exc.value.status_code == 400


# --- validate_cursor ---

def test_validate_cursor_valid_iso_timestamp():
    assert validate_cursor('2025-01-15T00:00:00+00:00') == '2025-01-15T00:00:00+00:00'


def test_validate_cursor_valid_utc_z():
    assert validate_cursor('2025-01-15T00:00:00Z') == '2025-01-15T00:00:00Z'


def test_validate_cursor_too_long():
    cursor = '2025-01-15T00:00:00+00:00' + '0' * 50
    with pytest.raises(HTTPException) as exc:
        validate_cursor(cursor)
    assert exc.value.status_code == 400


def test_validate_cursor_invalid_chars():
    with pytest.raises(HTTPException) as exc:
        validate_cursor('DROP TABLE sessions')
    assert exc.value.status_code == 400


def test_validate_cursor_sql_injection():
    with pytest.raises(HTTPException) as exc:
        validate_cursor("'; DROP TABLE sessions; --")
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


def test_validate_upload_cross_validation_rejects_mismatch():
    """Content-Type declared as JPEG but payload is PNG — now rejected."""
    upload = MagicMock()
    upload.content_type = 'image/jpeg'
    payload = b'\x89PNG' + b'\x00' * 100  # PNG magic bytes claimed as JPEG
    with pytest.raises(HTTPException) as exc:
        validate_upload(upload, payload)
    assert exc.value.status_code == 400
    assert 'Content-Type' in exc.value.detail


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
    with pytest.raises(HTTPException) as exc:
        validate_upload(upload, payload)
    assert exc.value.status_code == 400
    assert 'empty' in exc.value.detail


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


def test_validate_upload_valid_webp():
    """Valid WEBP: RIFF header + WEBP identifier at bytes 8-12."""
    upload = MagicMock()
    upload.content_type = 'image/webp'
    # RIFF<size>WEBP + padding
    payload = b'RIFF' + b'\x00\x00\x00\x00' + b'WEBP' + b'\x00' * 100
    validate_upload(upload, payload)


def test_validate_upload_riff_not_webp_rejected():
    """RIFF container that is NOT WEBP (e.g. WAV) should be rejected."""
    upload = MagicMock()
    upload.content_type = 'image/webp'
    # RIFF<size>WAVE — not a WEBP
    payload = b'RIFF' + b'\x00\x00\x00\x00' + b'WAVE' + b'\x00' * 100
    with pytest.raises(HTTPException) as exc:
        validate_upload(upload, payload)
    assert exc.value.status_code == 400


def test_validate_upload_riff_too_short_rejected():
    """RIFF payload too short to contain WEBP identifier."""
    upload = MagicMock()
    upload.content_type = 'image/webp'
    payload = b'RIFF' + b'\x00\x00'  # only 6 bytes, not enough for WEBP at 8-12
    with pytest.raises(HTTPException) as exc:
        validate_upload(upload, payload)
    assert exc.value.status_code == 400


# --- consensus_summary ---

def test_consensus_summary_empty():
    result = consensus_summary({})
    assert result['verified_lesions'] == 0
    assert result['average_confidence'] == 0.0
    assert result['summary'] == 'No verified lesions detected'
    assert result['lesions'] == []
    assert result['type_counts'] == {}


def test_consensus_summary_with_lesions():
    assignments = {
        'nose': [
            {'confidence': 0.8, 'bbox': [0, 0, 10, 10], 'class_name': 'pustule'},
            {'confidence': 0.6, 'bbox': [5, 5, 15, 15], 'class_name': 'acne'},
        ],
        'left_cheek': [
            {'confidence': 0.9, 'bbox': [20, 20, 30, 30], 'class_name': 'pustule'},
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
    # New: lesions flat list and type_counts
    assert len(result['lesions']) == 3
    assert all('region' in l for l in result['lesions'])
    assert result['type_counts'] == {'pustule': 2, 'acne': 1}


def test_consensus_summary_excludes_unassigned_from_count():
    assignments = {
        'unassigned': [{'confidence': 0.5}] * 10,
    }
    result = consensus_summary(assignments)
    assert result['verified_lesions'] == 0
    assert result['lesions'] == []


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


# --- summarize_stream_provenance ---

def test_stream_provenance_empty():
    result = summarize_stream_provenance({})
    assert result['stream_total'] == 0
    assert result['stream_classes'] == {}


def test_stream_provenance_with_typed_classes():
    cloud_results = {
        'preds_a_640': [
            {'class': 'Acne', 'confidence': 0.8},
            {'class': 'Acne', 'confidence': 0.7},
        ],
        'preds_b': [
            {'class': 'pustule', 'confidence': 0.9},
            {'class': 'nodule', 'confidence': 0.85},
            {'class': 'pustule', 'confidence': 0.6},
        ],
    }
    result = summarize_stream_provenance(cloud_results)
    assert result['streams']['model_a_640'] == 2
    assert result['streams']['model_b_native'] == 3
    assert result['stream_total'] == 5
    assert result['stream_classes']['model_a_640'] == {'Acne': 2}
    assert result['stream_classes']['model_b_native'] == {'pustule': 2, 'nodule': 1}


def test_stream_provenance_strongest_stream():
    cloud_results = {
        'preds_a_640': [{'class': 'Acne'}] * 10,
        'preds_a_1280': [{'class': 'Acne'}] * 5,
        'preds_b': [{'class': 'pustule'}] * 3,
    }
    result = summarize_stream_provenance(cloud_results)
    assert result['strongest_stream'] == 'model_a_640'
