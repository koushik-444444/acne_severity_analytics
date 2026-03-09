"""Unit tests for LesionMapper severity grading and GAGS calculation."""
import numpy as np
import pytest

from face_segmentation.mapping import LesionMapper


def _make_mapper(regions=None):
    """Create a LesionMapper with simple 100x100 region masks."""
    if regions is None:
        regions = ['forehead', 'nose', 'left_cheek', 'right_cheek']
    masks = {}
    h, w = 100, 100
    step = w // len(regions)
    for i, name in enumerate(regions):
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[:, i * step:(i + 1) * step] = 255
        masks[name] = mask
    return LesionMapper(masks)


# --- _get_severity_grade ---

class TestSeverityGradeMapping:
    def test_blackhead_grade_1(self):
        mapper = _make_mapper()
        assert mapper._get_severity_grade('blackhead') == 1
        assert mapper._get_severity_grade('Blackheads') == 1

    def test_whitehead_grade_1(self):
        mapper = _make_mapper()
        assert mapper._get_severity_grade('whitehead') == 1
        assert mapper._get_severity_grade('Whiteheads') == 1

    def test_comedone_grade_1(self):
        mapper = _make_mapper()
        assert mapper._get_severity_grade('comedone') == 1

    def test_papule_grade_2(self):
        mapper = _make_mapper()
        assert mapper._get_severity_grade('papule') == 2
        assert mapper._get_severity_grade('Papules') == 2

    def test_pustule_grade_3(self):
        mapper = _make_mapper()
        assert mapper._get_severity_grade('pustule') == 3
        assert mapper._get_severity_grade('Pustules') == 3

    def test_nodule_grade_4(self):
        mapper = _make_mapper()
        assert mapper._get_severity_grade('nodule') == 4
        assert mapper._get_severity_grade('Nodules') == 4

    def test_cyst_grade_4(self):
        mapper = _make_mapper()
        assert mapper._get_severity_grade('cyst') == 4
        assert mapper._get_severity_grade('cystic') == 4

    def test_unknown_defaults_to_2(self):
        mapper = _make_mapper()
        assert mapper._get_severity_grade('acne') == 2
        assert mapper._get_severity_grade('Acne') == 2
        assert mapper._get_severity_grade('random_class') == 2

    def test_case_insensitive(self):
        mapper = _make_mapper()
        assert mapper._get_severity_grade('PUSTULE') == 3
        assert mapper._get_severity_grade('Cyst') == 4


# --- get_clinical_report GAGS ---

class TestClinicalReportGAGS:
    def test_empty_assignments_zero_score(self):
        mapper = _make_mapper()
        assignments = {name: [] for name in mapper.region_names}
        assignments['unassigned'] = []
        report = mapper.get_clinical_report(assignments)
        assert report['gags_total_score'] == 0
        assert report['total_lesions'] == 0
        assert report['clinical_severity'] == 'None'

    def test_single_region_with_typed_lesions(self):
        mapper = _make_mapper()
        assignments = {name: [] for name in mapper.region_names}
        assignments['unassigned'] = []
        # Forehead (weight 2) with a pustule (grade 3) -> GAGS = 2 * 3 = 6
        assignments['forehead'] = [
            {'bbox': [0, 0, 5, 5], 'center': [2, 2], 'confidence': 0.8,
             'class_name': 'pustule', 'severity_grade': 3},
        ]
        report = mapper.get_clinical_report(assignments)
        assert report['gags_total_score'] == 6
        assert report['total_lesions'] == 1
        assert report['regions']['forehead']['gags_score'] == 6

    def test_mixed_grades_uses_max(self):
        mapper = _make_mapper()
        assignments = {name: [] for name in mapper.region_names}
        assignments['unassigned'] = []
        # Nose (weight 1) with blackhead (grade 1) + nodule (grade 4) -> max=4, GAGS = 1 * 4 = 4
        assignments['nose'] = [
            {'bbox': [30, 0, 35, 5], 'center': [32, 2], 'confidence': 0.7,
             'class_name': 'blackhead', 'severity_grade': 1},
            {'bbox': [35, 0, 40, 5], 'center': [37, 2], 'confidence': 0.9,
             'class_name': 'nodule', 'severity_grade': 4},
        ]
        report = mapper.get_clinical_report(assignments)
        assert report['regions']['nose']['gags_score'] == 4
        assert report['total_lesions'] == 2

    def test_severity_band_mild(self):
        mapper = _make_mapper()
        assignments = {name: [] for name in mapper.region_names}
        assignments['unassigned'] = []
        # Create a score of 6 (Mild: 1-18)
        assignments['forehead'] = [
            {'bbox': [0, 0, 5, 5], 'center': [2, 2], 'confidence': 0.8,
             'class_name': 'pustule', 'severity_grade': 3},
        ]
        report = mapper.get_clinical_report(assignments)
        assert report['clinical_severity'] == 'Mild'

    def test_severity_band_moderate(self):
        mapper = _make_mapper()
        regions = ['forehead', 'left_cheek', 'right_cheek', 'nose', 'chin']
        masks = {}
        for i, name in enumerate(regions):
            mask = np.zeros((100, 200), dtype=np.uint8)
            mask[:, i * 40:(i + 1) * 40] = 255
            masks[name] = mask
        mapper = LesionMapper(masks)
        assignments = {name: [] for name in mapper.region_names}
        assignments['unassigned'] = []
        # forehead(2)*4=8, left_cheek(2)*4=8, right_cheek(2)*4=8 = 24 -> Moderate (19-30)
        for region in ['forehead', 'left_cheek', 'right_cheek']:
            assignments[region] = [
                {'bbox': [0, 0, 5, 5], 'center': [2, 2], 'confidence': 0.8,
                 'class_name': 'cyst', 'severity_grade': 4},
            ]
        report = mapper.get_clinical_report(assignments)
        assert report['clinical_severity'] == 'Moderate'
