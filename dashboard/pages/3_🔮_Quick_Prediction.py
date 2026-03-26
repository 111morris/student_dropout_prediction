"""
🔮 Quick Prediction Page
Instant dropout risk prediction without saving to the database.
"""

import streamlit as st
import requests
import os

st.set_page_config(page_title="Quick Prediction", page_icon="🔮", layout="wide")

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")

st.markdown("# 🔮 Quick Risk Prediction")
st.markdown(
    "Enter student details for an **instant** dropout risk assessment. "
    "This does **not** save the student to the database."
)
st.markdown("---")

# ══════════════════════════════════════════════════════════════
# Quick Prediction Form
# ══════════════════════════════════════════════════════════════
with st.form("quick_predict_form"):
    st.markdown("### 📋 Student Features")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        age = st.number_input("Age", min_value=10, max_value=60, value=20, step=1)
        gpa = st.number_input("GPA", min_value=0.0, max_value=4.0, value=3.0, step=0.1, format="%.2f")
        semester_gpa = st.number_input("Semester GPA", min_value=0.0, max_value=4.0, value=3.0, step=0.1, format="%.2f")
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=4.0, value=3.0, step=0.1, format="%.2f")
        gender = st.selectbox("Gender", options=["Male", "Female"])

    with col2:
        study_hours_per_day = st.number_input("Study Hrs/Day", min_value=0.0, max_value=24.0, value=4.0, step=0.5)
        attendance_rate = st.number_input("Attendance %", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
        assignment_delay_days = st.number_input("Assignment Delay (days)", min_value=0.0, max_value=60.0, value=2.0, step=1.0)
        stress_index = st.slider("Stress (0–10)", min_value=0.0, max_value=10.0, value=5.0, step=0.5)

    with col3:
        family_income = st.number_input("Family Income", min_value=0.0, value=50000.0, step=5000.0, format="%.0f")
        travel_time_minutes = st.number_input("Travel Time (min)", min_value=0.0, value=30.0, step=5.0)
        internet_access = st.selectbox("Internet Access", options=["Yes", "No"])
        part_time_job = st.selectbox("Part-Time Job", options=["No", "Yes"])

    with col4:
        department = st.selectbox("Department", options=["Arts", "Business", "CS", "Engineering", "Science"])
        semester = st.selectbox("Semester", options=["Year 1", "Year 2", "Year 3", "Year 4"])
        parental_education = st.selectbox("Parental Education", options=["Bachelor", "High School", "Master", "PhD"])
        scholarship = st.selectbox("Scholarship", options=["Yes", "No"])

    submitted = st.form_submit_button(
        "🔮 Predict Risk Now",
        use_container_width=True,
        type="primary",
    )

# ══════════════════════════════════════════════════════════════
# Display Results
# ══════════════════════════════════════════════════════════════
if submitted:
    payload = {
        "age": age, "gpa": gpa, "semester_gpa": semester_gpa, "cgpa": cgpa,
        "study_hours_per_day": study_hours_per_day,
        "attendance_rate": attendance_rate,
        "assignment_delay_days": assignment_delay_days,
        "family_income": family_income,
        "travel_time_minutes": travel_time_minutes,
        "stress_index": stress_index,
        "gender": gender, "internet_access": internet_access,
        "part_time_job": part_time_job, "scholarship": scholarship,
        "department": department, "semester": semester,
        "parental_education": parental_education,
    }

    try:
        with st.spinner("🧠 Running AI model..."):
            response = requests.post(
                f"{API_URL}/predict", json=payload, timeout=10
            )

        if response.status_code == 200:
            data = response.json()

            st.markdown("---")
            st.markdown("## 🎯 Prediction Result")

            # Big risk display
            col_r1, col_r2 = st.columns([1, 2])

            with col_r1:
                risk_score = data["risk_score"]
                risk_status = data["risk_status"]

                st.metric("Risk Score", f"{risk_score:.4f}")
                st.metric("Risk Level", risk_status)

                # Visual progress bar
                st.markdown("**Risk Meter:**")
                st.progress(min(risk_score, 1.0))

            with col_r2:
                st.markdown("### 💡 Recommendation")
                recs = data["recommendation"].split(" | ")
                for rec in recs:
                    if "URGENT" in rec or "⚠️" in rec:
                        st.error(rec)
                    elif "✅" in rec:
                        st.success(rec)
                    else:
                        st.info(rec)

            # Interpretation guide
            with st.expander("📖 How to interpret the results"):
                st.markdown("""
                | Risk Score | Level | Meaning |
                |-----------|-------|---------|
                | 0.00 – 0.34 | 🟢 Low | Student is likely to continue |
                | 0.35 – 0.59 | 🟡 Medium | Some risk factors present — monitor closely |
                | 0.60 – 1.00 | 🔴 High | High dropout risk — immediate action needed |

                **What to do:**
                - 🟢 **Low**: Continue regular monitoring
                - 🟡 **Medium**: Schedule a check-in meeting
                - 🔴 **High**: Contact student immediately, arrange support
                """)
        else:
            error = response.json().get("detail", "Unknown error")
            st.error(f"Prediction failed: {error}")

    except requests.ConnectionError:
        st.error("Cannot connect to backend API. Is it running?")
    except Exception as e:
        st.error(f"Error: {e}")
