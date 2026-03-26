"""
Prediction Router.
Provides the POST /predict endpoint for instant risk assessment
without saving to the database.
"""

import logging

from fastapi import APIRouter, status

from backend.schemas.student import StudentFeatures, PredictionResponse
from backend.services.prediction_service import prediction_service

logger = logging.getLogger("ews.router.predict")

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post(
    "",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict dropout risk",
    description=(
        "Submit student features for an instant dropout risk prediction. "
        "This does NOT save the student to the database. "
        "Use POST /students to save and predict simultaneously."
    ),
)
def predict_risk(features: StudentFeatures) -> PredictionResponse:
    """
    Run dropout risk prediction on the provided student features.

    Returns:
        - risk_score: probability between 0.0 and 1.0
        - risk_status: 🟢 Low / 🟡 Medium / 🔴 High
        - recommendation: actionable advice for school staff
    """
    risk_score, risk_status, recommendation = prediction_service.predict(features)

    logger.info(
        f"Quick prediction — score={risk_score:.4f}, status={risk_status}"
    )

    return PredictionResponse(
        risk_score=round(risk_score, 4),
        risk_status=risk_status,
        recommendation=recommendation,
    )
