"""
Prediction Service.
Replicates the exact feature engineering pipeline from src/feature_engineer.py,
then runs inference using the trained best_model.joblib.

This is the critical bridge between raw user input and the ML model.
The feature engineering steps MUST match exactly what the model was trained on,
otherwise predictions will be incorrect.

Pipeline (matches training):
  1. Drop Student_ID, Dropout (not present in API input)
  2. Drop Semester_GPA, CGPA (removed during training due to multicollinearity)
  3. Binary-encode: Gender, Internet_Access, Part_Time_Job, Scholarship
  4. Create derived features: Study_Attendance_Ratio, Income_per_Travel, Overloaded_Flag
  5. One-hot encode: Department, Semester, Parental_Education (drop_first=True)
  6. Ensure exact column order matches model's training features
  7. Run model.predict_proba()
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.config import (
    MODEL_PATH,
    SCALER_PATH,
    RISK_THRESHOLD_HIGH,
    RISK_THRESHOLD_MEDIUM,
)
from backend.schemas.student import StudentFeatures

logger = logging.getLogger("ews.prediction_service")


class PredictionService:
    """
    Loads the trained model and provides prediction with full
    feature engineering replication.
    """

    def __init__(self) -> None:
        self.model = None
        self.scaler = None
        # The exact columns the model was trained on (from cleaned_dataset.csv header)
        self.expected_features: list[str] = [
            "Age", "Gender", "Family_Income", "Internet_Access",
            "Study_Hours_per_Day", "Attendance_Rate", "Assignment_Delay_Days",
            "Travel_Time_Minutes", "Part_Time_Job", "Scholarship",
            "Stress_Index", "GPA",
            # Derived features
            "Study_Attendance_Ratio", "Income_per_Travel", "Overloaded_Flag",
            # One-hot encoded (drop_first=True)
            "Department_Business", "Department_CS",
            "Department_Engineering", "Department_Science",
            "Semester_Year 2", "Semester_Year 3", "Semester_Year 4",
            "Parental_Education_High School", "Parental_Education_Master",
            "Parental_Education_PhD",
        ]

    def load_model(self) -> None:
        """Load the trained model (and scaler if available) from disk."""
        model_path = Path(MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}. "
                "Run the training pipeline (main.py) first."
            )

        self.model = joblib.load(MODEL_PATH)
        logger.info(f"✓ Model loaded from {MODEL_PATH}")

        scaler_path = Path(SCALER_PATH)
        if scaler_path.exists():
            self.scaler = joblib.load(SCALER_PATH)
            logger.info(f"✓ Scaler loaded from {SCALER_PATH}")
        else:
            logger.info("No scaler file found — model does not require scaling")

    def _engineer_features(self, features: StudentFeatures) -> pd.DataFrame:
        """
        Replicate the exact feature engineering from src/feature_engineer.py
        and src/data_cleaner.py for a single student's raw input.

        Steps match the training pipeline:
          1. Build raw DataFrame
          2. Drop columns removed during training (Semester_GPA, CGPA)
          3. Binary encode categoricals
          4. Create derived features
          5. One-hot encode multi-class categoricals
          6. Align columns to match model's expected feature order
        """
        # Step 1: Build raw DataFrame from input
        raw_data = {
            "Age": [features.age],
            "Gender": [features.gender.value],
            "Family_Income": [features.family_income],
            "Internet_Access": [features.internet_access.value],
            "Study_Hours_per_Day": [features.study_hours_per_day],
            "Attendance_Rate": [features.attendance_rate],
            "Assignment_Delay_Days": [features.assignment_delay_days],
            "Travel_Time_Minutes": [features.travel_time_minutes],
            "Part_Time_Job": [features.part_time_job.value],
            "Scholarship": [features.scholarship.value],
            "Stress_Index": [features.stress_index],
            "GPA": [features.gpa],
            "Semester_GPA": [features.semester_gpa],
            "CGPA": [features.cgpa],
            "Semester": [features.semester.value],
            "Department": [features.department.value],
            "Parental_Education": [features.parental_education.value],
        }
        df = pd.DataFrame(raw_data)

        # Step 2: Drop columns removed during training (multicollinearity)
        # The cleaning pipeline drops CGPA and Semester_GPA
        # GPA_CGPA_Diff is also not in the final feature set
        df = df.drop(columns=["CGPA", "Semester_GPA"], errors="ignore")

        # Step 3: Binary encode
        binary_maps = {
            "Gender": {"Male": 1, "Female": 0},
            "Internet_Access": {"Yes": 1, "No": 0},
            "Part_Time_Job": {"Yes": 1, "No": 0},
            "Scholarship": {"Yes": 1, "No": 0},
        }
        for col, mapping in binary_maps.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)

        # Step 4: Derived features (same as src/feature_engineer.py)
        df["Study_Attendance_Ratio"] = (
            df["Study_Hours_per_Day"] / (df["Attendance_Rate"] + 1e-6)
        )
        df["Income_per_Travel"] = (
            df["Family_Income"] / (df["Travel_Time_Minutes"] + 1e-6)
        )
        df["Overloaded_Flag"] = (
            (df["Stress_Index"] > 7) & (df["Study_Hours_per_Day"] < 2)
        ).astype(int)

        # Step 5: Manual one-hot encoding with known categories.
        # NOTE: pd.get_dummies(drop_first=True) does NOT work for single-row
        # DataFrames because it only sees one category and drops it.
        # Instead, we manually create the one-hot columns using the full
        # category lists from training (drop_first=True → first alpha is baseline).
        one_hot_specs = {
            # Column name → list of categories that get their own column
            #   (the first alphabetical category is the baseline / dropped)
            "Department": ["Business", "CS", "Engineering", "Science"],  # "Arts" = baseline
            "Semester": ["Year 2", "Year 3", "Year 4"],                  # "Year 1" = baseline
            "Parental_Education": ["High School", "Master", "PhD"],      # "Bachelor" = baseline
        }

        for col, categories in one_hot_specs.items():
            value = df[col].iloc[0]
            for cat in categories:
                col_name = f"{col}_{cat}"
                df[col_name] = int(value == cat)
            df = df.drop(columns=[col])

        # Step 6: Enforce exact column order to match model's training features
        df = df[self.expected_features]

        logger.debug(f"Engineered features: {list(df.columns)}")
        return df

    def predict(
        self, features: StudentFeatures
    ) -> tuple[float, str, str]:
        """
        Run the full prediction pipeline for a single student.

        Args:
            features: Validated student input features.

        Returns:
            Tuple of (risk_score, risk_status, recommendation).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Feature engineering
        X = self._engineer_features(features)

        # Predict probability of dropout (class 1)
        if hasattr(self.model, "predict_proba"):
            risk_score = float(self.model.predict_proba(X)[0, 1])
        else:
            risk_score = float(self.model.predict(X)[0])

        # Classify risk level
        risk_status = self._classify_risk(risk_score)

        # Generate actionable recommendation
        recommendation = self._generate_recommendation(features, risk_score)

        logger.info(
            f"Prediction: score={risk_score:.4f}, "
            f"status={risk_status}, dept={features.department.value}"
        )

        return risk_score, risk_status, recommendation

    @staticmethod
    def _classify_risk(risk_score: float) -> str:
        """Classify a risk score into a categorical level."""
        if risk_score >= RISK_THRESHOLD_HIGH:
            return "🔴 High"
        elif risk_score >= RISK_THRESHOLD_MEDIUM:
            return "🟡 Medium"
        else:
            return "🟢 Low"

    @staticmethod
    def _generate_recommendation(
        features: StudentFeatures, risk_score: float
    ) -> str:
        """
        Generate an actionable recommendation based on the student's
        specific features and risk score.
        """
        recommendations: list[str] = []

        # High-risk interventions
        if risk_score >= RISK_THRESHOLD_HIGH:
            recommendations.append(
                "⚠️ URGENT: This student is at high risk of dropping out. "
                "Immediate intervention is recommended."
            )

        # Feature-specific recommendations
        if features.attendance_rate < 60:
            recommendations.append(
                "📉 Attendance is critically low ({:.0f}%). "
                "Consider a meeting with the student to discuss barriers "
                "to attendance.".format(features.attendance_rate)
            )
        elif features.attendance_rate < 75:
            recommendations.append(
                "📉 Attendance is below average ({:.0f}%). "
                "Monitor and encourage regular class participation.".format(
                    features.attendance_rate
                )
            )

        if features.gpa < 2.0:
            recommendations.append(
                "📚 GPA is below 2.0 ({:.2f}). Recommend academic tutoring "
                "and study skills workshops.".format(features.gpa)
            )
        elif features.gpa < 2.5:
            recommendations.append(
                "📚 GPA is below average ({:.2f}). Consider assigning "
                "a peer tutor or academic mentor.".format(features.gpa)
            )

        if features.stress_index > 7:
            recommendations.append(
                "😰 High stress level ({:.1f}/10). Refer to counseling "
                "services and consider workload adjustment.".format(
                    features.stress_index
                )
            )

        if features.study_hours_per_day < 2:
            recommendations.append(
                "⏰ Very low study hours ({:.1f} hrs/day). "
                "Encourage structured study schedules and "
                "time management workshops.".format(features.study_hours_per_day)
            )

        if features.internet_access == "No":
            recommendations.append(
                "🌐 No internet access at home. Ensure the student has "
                "access to campus computer labs and library resources."
            )

        if features.scholarship == "No" and features.family_income < 30000:
            recommendations.append(
                "💰 Low family income without scholarship support. "
                "Explore financial aid options and emergency fund eligibility."
            )

        if features.assignment_delay_days > 5:
            recommendations.append(
                "📝 Frequently late on assignments ({:.0f} days avg delay). "
                "Work with the student to create a submission plan "
                "with intermediate deadlines.".format(
                    features.assignment_delay_days
                )
            )

        if features.part_time_job == "Yes" and features.study_hours_per_day < 3:
            recommendations.append(
                "💼 Part-time job with low study hours. Discuss whether "
                "work schedule adjustments are possible."
            )

        # Low-risk positive reinforcement
        if risk_score < RISK_THRESHOLD_MEDIUM:
            recommendations.append(
                "✅ This student is currently at low risk. Continue monitoring "
                "and provide positive reinforcement."
            )

        if not recommendations:
            recommendations.append(
                "📋 No specific concerns identified. Continue standard "
                "academic monitoring."
            )

        return " | ".join(recommendations)


# ──────────────────────────────────────────────────────────────
# Singleton instance (loaded once on app startup)
# ──────────────────────────────────────────────────────────────
prediction_service = PredictionService()
