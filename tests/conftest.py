"""
Pytest Fixtures for the Student Dropout EWS API tests.
Provides:
  - test_db: In-memory SQLite database
  - client: FastAPI TestClient with test database
  - sample_student_data: Valid student input dictionary
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from fastapi.testclient import TestClient

from backend.database import Base, get_db
from backend.main import app


# ══════════════════════════════════════════════════════════════
# Test Database (in-memory SQLite)
# ══════════════════════════════════════════════════════════════
TEST_DATABASE_URL = "sqlite:///./test_ews.db"

test_engine = create_engine(
    TEST_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestSessionLocal = sessionmaker(
    autocommit=False, autoflush=False, bind=test_engine
)


def override_get_db():
    """Override the get_db dependency to use the test database."""
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


# ══════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════

@pytest.fixture(scope="function")
def test_db():
    """Create a fresh test database for each test function."""
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture(scope="function")
def client(test_db):
    """
    FastAPI TestClient with the test database injected.
    Forces model loading before tests run.
    """
    app.dependency_overrides[get_db] = override_get_db

    # Ensure the prediction service model is loaded
    from backend.services.prediction_service import prediction_service
    if prediction_service.model is None:
        prediction_service.load_model()

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


@pytest.fixture
def sample_student_data() -> dict:
    """
    Valid sample student data matching the API schema.
    This represents a typical student profile.
    """
    return {
        "age": 20,
        "gpa": 3.2,
        "semester_gpa": 3.0,
        "cgpa": 3.1,
        "study_hours_per_day": 4.5,
        "attendance_rate": 85.0,
        "assignment_delay_days": 2.0,
        "family_income": 50000.0,
        "travel_time_minutes": 30.0,
        "stress_index": 5.0,
        "gender": "Male",
        "internet_access": "Yes",
        "part_time_job": "No",
        "scholarship": "Yes",
        "department": "Engineering",
        "semester": "Year 2",
        "parental_education": "Bachelor",
    }


@pytest.fixture
def high_risk_student_data() -> dict:
    """
    Sample data for a student likely to be flagged as high risk.
    Low GPA, low attendance, high stress, no support.
    """
    return {
        "age": 22,
        "gpa": 1.2,
        "semester_gpa": 1.0,
        "cgpa": 1.5,
        "study_hours_per_day": 1.0,
        "attendance_rate": 35.0,
        "assignment_delay_days": 10.0,
        "family_income": 15000.0,
        "travel_time_minutes": 90.0,
        "stress_index": 9.0,
        "gender": "Female",
        "internet_access": "No",
        "part_time_job": "Yes",
        "scholarship": "No",
        "department": "Arts",
        "semester": "Year 3",
        "parental_education": "High School",
    }
