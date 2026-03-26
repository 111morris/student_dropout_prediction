"""
Student ORM Model.
SQLAlchemy table definition for the students table.
Stores all raw input features plus prediction results.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, Float, Integer, DateTime
from backend.database import Base


def generate_uuid() -> str:
    """Generate a new UUID4 string for student IDs."""
    return str(uuid.uuid4())


class Student(Base):
    """
    Students table — stores raw features and prediction results.

    Columns:
        id              : UUID primary key (auto-generated)
        -- Raw input features --
        age, gender, family_income, internet_access, study_hours_per_day,
        attendance_rate, assignment_delay_days, travel_time_minutes,
        part_time_job, scholarship, stress_index, gpa, semester_gpa,
        cgpa, semester, department, parental_education
        -- Prediction results --
        risk_score      : Model output probability (0.0 – 1.0)
        risk_status     : Categorical label (🟢 Low / 🟡 Medium / 🔴 High)
        recommendation  : Actionable text for staff
        -- Timestamps --
        created_at      : Row creation time
        updated_at      : Last update time
    """

    __tablename__ = "students"

    # Primary key
    id = Column(String, primary_key=True, default=generate_uuid)

    # ── Raw Numerical Features ────────────────────────────────
    age = Column(Integer, nullable=False)
    gpa = Column(Float, nullable=False)
    semester_gpa = Column(Float, nullable=False)
    cgpa = Column(Float, nullable=False)
    study_hours_per_day = Column(Float, nullable=False)
    attendance_rate = Column(Float, nullable=False)
    assignment_delay_days = Column(Float, nullable=False)
    family_income = Column(Float, nullable=False)
    travel_time_minutes = Column(Float, nullable=False)
    stress_index = Column(Float, nullable=False)

    # ── Raw Categorical Features ──────────────────────────────
    gender = Column(String, nullable=False)
    internet_access = Column(String, nullable=False)
    part_time_job = Column(String, nullable=False)
    scholarship = Column(String, nullable=False)
    department = Column(String, nullable=False)
    semester = Column(String, nullable=False)
    parental_education = Column(String, nullable=False)

    # ── Prediction Results ────────────────────────────────────
    risk_score = Column(Float, nullable=True)
    risk_status = Column(String, nullable=True)
    recommendation = Column(String, nullable=True)

    # ── Timestamps ────────────────────────────────────────────
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<Student(id={self.id!r}, gpa={self.gpa}, "
            f"risk_status={self.risk_status!r})>"
        )
