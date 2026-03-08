"""Unit tests for backend/utils.py — calculate_iou."""
from utils import calculate_iou


def test_calculate_iou_perfect_overlap():
    box = [10, 10, 50, 50]
    assert calculate_iou(box, box) == 1.0


def test_calculate_iou_no_overlap():
    assert calculate_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_calculate_iou_partial_overlap():
    # 5x5 overlap area = 25
    # box1 area = 100, box2 area = 100, union = 175
    iou = calculate_iou([0, 0, 10, 10], [5, 5, 15, 15])
    assert abs(iou - 25 / 175) < 1e-6


def test_calculate_iou_zero_area():
    # Degenerate box (line) has zero area
    assert calculate_iou([5, 5, 5, 10], [5, 5, 5, 10]) == 0.0


def test_calculate_iou_contained():
    # Small box fully inside large box
    # Inner area = 4, outer area = 100, union = 100
    iou = calculate_iou([0, 0, 10, 10], [4, 4, 6, 6])
    assert abs(iou - 4 / 100) < 1e-6


def test_calculate_iou_touching_edges():
    # Boxes share an edge but no interior overlap
    assert calculate_iou([0, 0, 10, 10], [10, 0, 20, 10]) == 0.0
