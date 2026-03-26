"""
Prediction Service Unit Tests.
Tests the feature engineering and risk classification logic directly.
"""

import pytest

from backend.schemas.student import StudentFeatures
from backend.services.prediction_service import PredictionService


@pytest.fixture(scope="module")
def service():
    """Load the prediction service once for all tests in this module."""
    svc = PredictionService()
    svc.load_model()
    return svc


@pytest.fixture
def sample_features() -> StudentFeatures:
    """Valid StudentFeatures instance."""
    return StudentFeatures(
        age=20,
        gpa=3.2,
        semester_gpa=3.0,
        cgpa=3.1,
        study_hours_per_day=4.5,
        attendance_rate=85.0,
        assignment_delay_days=2.0,
        family_income=50000.0,
        travel_time_minutes=30.0,
        stress_index=5.0,
        gender="Male",
        internet_access="Yes",
        part_time_job="No",
        scholarship="Yes",
        department="Engineering",
        semester="Year 2",
        parental_education="Bachelor",
    )


class TestFeatureEngineering:
    """Test that feature engineering produces correct output."""

    def test_feature_count(self, service, sample_features):
        """Engineered features should match expected column count."""
        df = service._engineer_features(sample_features)
        assert df.shape[0] == 1  # Single row
        assert df.shape[1] == len(service.expected_features)

    def test_feature_columns_match(self, service, sample_features):
        """Column names must exactly match expected features."""
        df = service._engineer_features(sample_features)
        assert list(df.columns) == service.expected_features

    def test_binary_encoding(self, service, sample_features):
        """Binary fields should be encoded to 0/1."""
        df = service._engineer_features(sample_features)
        assert df["Gender"].iloc[0] == 1  # Male = 1
        assert df["Internet_Access"].iloc[0] == 1  # Yes = 1
        assert df["Part_Time_Job"].iloc[0] == 0  # No = 0
        assert df["Scholarship"].iloc[0] == 1  # Yes = 1

    def test_derived_features(self, service, sample_features):
        """Derived features should be correctly computed."""
        df = service._engineer_features(sample_features)

        # Study_Attendance_Ratio = 4.5 / (85.0 + 1e-6)
        expected_ratio = 4.5 / (85.0 + 1e-6)
        assert abs(df["Study_Attendance_Ratio"].iloc[0] - expected_ratio) < 1e-4

        # Income_per_Travel = 50000 / (30 + 1e-6)
        expected_income = 50000.0 / (30.0 + 1e-6)
        assert abs(df["Income_per_Travel"].iloc[0] - expected_income) < 1e-1

        # Overloaded_Flag: stress=5 (<= 7) → 0
        assert df["Overloaded_Flag"].iloc[0] == 0

    def test_overloaded_flag_when_stressed(self, service):
        """Overloaded flag should be 1 when stress > 7 and study < 2."""
        features = StudentFeatures(
            age=20, gpa=2.0, semester_gpa=2.0, cgpa=2.0,
            study_hours_per_day=1.0,  # < 2
            attendance_rate=50.0,
            assignment_delay_days=5.0,
            family_income=30000.0,
            travel_time_minutes=60.0,
            stress_index=8.5,  # > 7
            gender="Female", internet_access="No",
            part_time_job="Yes", scholarship="No",
            department="Arts", semester="Year 1",
            parental_education="High School",
        )
        df = service._engineer_features(features)
        assert df["Overloaded_Flag"].iloc[0] == 1

    def test_one_hot_encoding(self, service, sample_features):
        """One-hot encoding should correctly set department columns."""
        df = service._engineer_features(sample_features)
        # Department = Engineering → Department_Engineering should be 1
        assert df["Department_Engineering"].iloc[0] == 1
        assert df["Department_Business"].iloc[0] == 0
        assert df["Department_CS"].iloc[0] == 0
        assert df["Department_Science"].iloc[0] == 0


class TestPrediction:
    """Test the full prediction pipeline."""

    def test_predict_returns_valid_score(self, service, sample_features):
        """Risk score should be between 0 and 1."""
        score, status, rec = service.predict(sample_features)
        assert 0.0 <= score <= 1.0

    def test_predict_returns_valid_status(self, service, sample_features):
        """Risk status should be one of the three levels."""
        _, status, _ = service.predict(sample_features)
        assert status in ["🟢 Low", "🟡 Medium", "🔴 High"]

    def test_predict_returns_recommendation(self, service, sample_features):
        """Recommendation should be a non-empty string."""
        _, _, rec = service.predict(sample_features)
        assert isinstance(rec, str)
        assert len(rec) > 0


class TestRiskClassification:
    """Test the risk classification logic."""

    def test_high_risk(self, service):
        assert service._classify_risk(0.75) == "🔴 High"
        assert service._classify_risk(0.60) == "🔴 High"

    def test_medium_risk(self, service):
        assert service._classify_risk(0.50) == "🟡 Medium"
        assert service._classify_risk(0.35) == "🟡 Medium"

    def test_low_risk(self, service):
        assert service._classify_risk(0.20) == "🟢 Low"
        assert service._classify_risk(0.0) == "🟢 Low"

    def test_boundary_high(self, service):
        assert service._classify_risk(0.60) == "🔴 High"
        assert service._classify_risk(0.59) == "🟡 Medium"

    def test_boundary_medium(self, service):
        assert service._classify_risk(0.35) == "🟡 Medium"
        assert service._classify_risk(0.34) == "🟢 Low"
