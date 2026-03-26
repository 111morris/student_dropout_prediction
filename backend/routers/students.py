"""
Students Router.
CRUD endpoints for managing student records.
Each new student automatically gets a risk prediction.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.schemas.student import (
    StudentFeatures,
    StudentResponse,
    StudentListResponse,
)
from backend.services import crud_service

logger = logging.getLogger("ews.router.students")

router = APIRouter(prefix="/students", tags=["Students"])


@router.post(
    "",
    response_model=StudentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add a new student",
    description=(
        "Create a new student record and automatically run risk prediction. "
        "The student's features are saved to the database along with "
        "the computed risk score, risk status, and recommendation."
    ),
)
def create_student(
    features: StudentFeatures,
    db: Session = Depends(get_db),
) -> StudentResponse:
    """Create a new student with automatic risk prediction."""
    student = crud_service.create_student(db, features)
    logger.info(f"Created student {student.id}")
    return student


@router.get(
    "",
    response_model=StudentListResponse,
    summary="List all students",
    description=(
        "Retrieve all students with optional pagination and risk filtering."
    ),
)
def list_students(
    skip: int = Query(0, ge=0, description="Records to skip"),
    limit: int = Query(100, ge=1, le=500, description="Max records to return"),
    risk_filter: Optional[str] = Query(
        None,
        description="Filter by risk status (e.g., '🔴 High', '🟡 Medium', '🟢 Low')",
    ),
    db: Session = Depends(get_db),
) -> StudentListResponse:
    """Get paginated list of all students."""
    students, total = crud_service.get_all_students(
        db, skip=skip, limit=limit, risk_filter=risk_filter
    )
    return StudentListResponse(total=total, students=students)


@router.get(
    "/{student_id}",
    response_model=StudentResponse,
    summary="Get a student by ID",
    description="Retrieve a single student record including their risk profile.",
)
def get_student(
    student_id: str,
    db: Session = Depends(get_db),
) -> StudentResponse:
    """Get a single student by their UUID."""
    return crud_service.get_student(db, student_id)


@router.delete(
    "/{student_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a student",
    description="Remove a student record from the database.",
)
def delete_student(
    student_id: str,
    db: Session = Depends(get_db),
) -> None:
    """Delete a student by their UUID."""
    crud_service.delete_student(db, student_id)
    logger.info(f"Deleted student {student_id}")
