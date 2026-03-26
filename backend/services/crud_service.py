"""
CRUD Service for Student records.
Handles all database operations with proper error handling.
"""

import logging
from typing import Optional

from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from backend.models.student import Student
from backend.schemas.student import StudentFeatures
from backend.services.prediction_service import prediction_service

logger = logging.getLogger("ews.crud_service")


def create_student(db: Session, features: StudentFeatures) -> Student:
    """
    Create a new student record with automatic risk prediction.

    Steps:
      1. Run prediction on the input features.
      2. Create a Student ORM object with raw features + prediction results.
      3. Save to database.

    Args:
        db: SQLAlchemy session.
        features: Validated student input features.

    Returns:
        The created Student ORM object.
    """
    # Run prediction
    risk_score, risk_status, recommendation = prediction_service.predict(features)

    # Create ORM object
    student = Student(
        age=features.age,
        gpa=features.gpa,
        semester_gpa=features.semester_gpa,
        cgpa=features.cgpa,
        study_hours_per_day=features.study_hours_per_day,
        attendance_rate=features.attendance_rate,
        assignment_delay_days=features.assignment_delay_days,
        family_income=features.family_income,
        travel_time_minutes=features.travel_time_minutes,
        stress_index=features.stress_index,
        gender=features.gender.value,
        internet_access=features.internet_access.value,
        part_time_job=features.part_time_job.value,
        scholarship=features.scholarship.value,
        department=features.department.value,
        semester=features.semester.value,
        parental_education=features.parental_education.value,
        risk_score=risk_score,
        risk_status=risk_status,
        recommendation=recommendation,
    )

    db.add(student)
    db.commit()
    db.refresh(student)

    logger.info(
        f"Created student {student.id} — "
        f"risk_score={risk_score:.4f}, status={risk_status}"
    )
    return student


def get_student(db: Session, student_id: str) -> Student:
    """
    Retrieve a single student by ID.

    Raises:
        HTTPException 404 if not found.
    """
    student = db.query(Student).filter(Student.id == student_id).first()
    if student is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student with id '{student_id}' not found.",
        )
    return student


def get_all_students(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    risk_filter: Optional[str] = None,
) -> tuple[list[Student], int]:
    """
    Retrieve all students with optional pagination and filtering.

    Args:
        db: SQLAlchemy session.
        skip: Number of records to skip (offset).
        limit: Maximum records to return.
        risk_filter: Optional filter by risk status (e.g., "🔴 High").

    Returns:
        Tuple of (list of students, total count).
    """
    query = db.query(Student)

    if risk_filter:
        query = query.filter(Student.risk_status == risk_filter)

    total = query.count()
    students = query.order_by(Student.created_at.desc()).offset(skip).limit(limit).all()

    logger.info(f"Retrieved {len(students)} students (total: {total})")
    return students, total


def delete_student(db: Session, student_id: str) -> None:
    """
    Delete a student record by ID.

    Raises:
        HTTPException 404 if not found.
    """
    student = get_student(db, student_id)
    db.delete(student)
    db.commit()
    logger.info(f"Deleted student {student_id}")
