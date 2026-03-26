"""
Pydantic Schemas for the Student Dropout EWS API.
Defines request/response models with full validation.
Uses Enums for categorical fields to enforce valid values.
"""

from enum import Enum
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════
# Enums — enforce valid categorical values
# ══════════════════════════════════════════════════════════════

class Gender(str, Enum):
    """Valid gender values."""
    MALE = "Male"
    FEMALE = "Female"


class YesNo(str, Enum):
    """Binary yes/no fields."""
    YES = "Yes"
    NO = "No"


class Department(str, Enum):
    """
    Valid department values.
    These must match the values seen during model training.
    The model used pd.get_dummies with drop_first=True, so one
    category becomes the baseline (alphabetically first = Arts).
    """
    ARTS = "Arts"
    BUSINESS = "Business"
    CS = "CS"
    ENGINEERING = "Engineering"
    SCIENCE = "Science"


class Semester(str, Enum):
    """
    Valid semester values.
    Model was trained with: Year 1, Year 2, Year 3, Year 4
    (drop_first=True → Year 1 is baseline).
    """
    YEAR_1 = "Year 1"
    YEAR_2 = "Year 2"
    YEAR_3 = "Year 3"
    YEAR_4 = "Year 4"


class ParentalEducation(str, Enum):
    """
    Valid parental education levels.
    Model was trained with: Bachelor, High School, Master, PhD
    (drop_first=True → Bachelor is baseline).
    """
    BACHELOR = "Bachelor"
    HIGH_SCHOOL = "High School"
    MASTER = "Master"
    PHD = "PhD"


class RiskStatus(str, Enum):
    """Risk level categories."""
    LOW = "🟢 Low"
    MEDIUM = "🟡 Medium"
    HIGH = "🔴 High"


# ══════════════════════════════════════════════════════════════
# Request Schemas
# ══════════════════════════════════════════════════════════════

class StudentFeatures(BaseModel):
    """
    Input features for prediction.
    These are the RAW features before any preprocessing.
    The prediction service handles feature engineering internally.
    """

    # Numerical features with validation
    age: int = Field(
        ..., ge=10, le=60, description="Student age (10–60)", examples=[20]
    )
    gpa: float = Field(
        ..., ge=0.0, le=4.0, description="Current GPA (0.0–4.0)", examples=[3.2]
    )
    semester_gpa: float = Field(
        ..., ge=0.0, le=4.0, description="Semester GPA (0.0–4.0)", examples=[3.0]
    )
    cgpa: float = Field(
        ..., ge=0.0, le=4.0,
        description="Cumulative GPA (0.0–4.0)", examples=[3.1]
    )
    study_hours_per_day: float = Field(
        ..., ge=0.0, le=24.0,
        description="Study hours per day (0–24)", examples=[4.5]
    )
    attendance_rate: float = Field(
        ..., ge=0.0, le=100.0,
        description="Attendance percentage (0–100)", examples=[85.0]
    )
    assignment_delay_days: float = Field(
        ..., ge=0.0,
        description="Average assignment delay in days", examples=[2.0]
    )
    family_income: float = Field(
        ..., ge=0.0,
        description="Family income (annual, in local currency)", examples=[50000.0]
    )
    travel_time_minutes: float = Field(
        ..., ge=0.0,
        description="Daily travel time in minutes", examples=[30.0]
    )
    stress_index: float = Field(
        ..., ge=0.0, le=10.0,
        description="Stress level index (0–10)", examples=[5.0]
    )

    # Categorical features (using Enums)
    gender: Gender = Field(..., description="Student gender", examples=["Male"])
    internet_access: YesNo = Field(
        ..., description="Has internet access?", examples=["Yes"]
    )
    part_time_job: YesNo = Field(
        ..., description="Has a part-time job?", examples=["No"]
    )
    scholarship: YesNo = Field(
        ..., description="Receives a scholarship?", examples=["Yes"]
    )
    department: Department = Field(
        ..., description="Academic department", examples=["Engineering"]
    )
    semester: Semester = Field(
        ..., description="Current semester/year", examples=["Year 2"]
    )
    parental_education: ParentalEducation = Field(
        ..., description="Highest parental education level", examples=["Bachelor"]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
            ]
        }
    }


# ══════════════════════════════════════════════════════════════
# Response Schemas
# ══════════════════════════════════════════════════════════════

class PredictionResponse(BaseModel):
    """Response from the /predict endpoint."""
    risk_score: float = Field(
        ..., ge=0.0, le=1.0, description="Dropout risk probability (0.0–1.0)"
    )
    risk_status: str = Field(
        ..., description="Risk level: 🟢 Low, 🟡 Medium, or 🔴 High"
    )
    recommendation: str = Field(
        ..., description="Actionable recommendation for staff"
    )


class StudentResponse(BaseModel):
    """Full student record returned from the API."""
    id: str
    # Raw features
    age: int
    gpa: float
    semester_gpa: float
    cgpa: float
    study_hours_per_day: float
    attendance_rate: float
    assignment_delay_days: float
    family_income: float
    travel_time_minutes: float
    stress_index: float
    gender: str
    internet_access: str
    part_time_job: str
    scholarship: str
    department: str
    semester: str
    parental_education: str
    # Prediction results
    risk_score: Optional[float] = None
    risk_status: Optional[str] = None
    recommendation: Optional[str] = None
    # Timestamps
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class StudentListResponse(BaseModel):
    """Paginated list of students."""
    total: int
    students: list[StudentResponse]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    database_connected: bool
