"""
➕ Add New Student Page
Form to add a new student to the database with automatic risk prediction.
"""

import streamlit as st
import requests
import os

st.set_page_config(page_title="Add Student", page_icon="➕", layout="wide")

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")

st.markdown("# ➕ Add New Student")
st.markdown(
    "Fill in the student details below. The system will automatically "
    "predict their dropout risk and save to the database."
)
st.markdown("---")

# ══════════════════════════════════════════════════════════════
# Student Information Form
# ══════════════════════════════════════════════════════════════
with st.form("add_student_form", clear_on_submit=True):
    st.markdown("### 📋 Personal Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=10, max_value=60, value=20, step=1)
        gender = st.selectbox("Gender", options=["Male", "Female"])
    with col2:
        department = st.selectbox(
            "Department",
            options=["Arts", "Business", "CS", "Engineering", "Science"],
        )
        semester = st.selectbox(
            "Semester / Year",
            options=["Year 1", "Year 2", "Year 3", "Year 4"],
        )
    with col3:
        parental_education = st.selectbox(
            "Parental Education",
            options=["Bachelor", "High School", "Master", "PhD"],
        )
        scholarship = st.selectbox("Scholarship", options=["Yes", "No"])

    st.markdown("---")
    st.markdown("### 📊 Academic Performance")
    col4, col5, col6 = st.columns(3)

    with col4:
        gpa = st.number_input(
            "GPA (0.0 – 4.0)", min_value=0.0, max_value=4.0,
            value=3.0, step=0.1, format="%.2f"
        )
        semester_gpa = st.number_input(
            "Semester GPA (0.0 – 4.0)", min_value=0.0, max_value=4.0,
            value=3.0, step=0.1, format="%.2f"
        )
    with col5:
        cgpa = st.number_input(
            "CGPA (0.0 – 4.0)", min_value=0.0, max_value=4.0,
            value=3.0, step=0.1, format="%.2f"
        )
        attendance_rate = st.number_input(
            "Attendance Rate (%)", min_value=0.0, max_value=100.0,
            value=80.0, step=1.0, format="%.1f"
        )
    with col6:
        study_hours_per_day = st.number_input(
            "Study Hours per Day", min_value=0.0, max_value=24.0,
            value=4.0, step=0.5, format="%.1f"
        )
        assignment_delay_days = st.number_input(
            "Assignment Delay (days)", min_value=0.0, max_value=60.0,
            value=2.0, step=1.0, format="%.0f"
        )

    st.markdown("---")
    st.markdown("### 🏠 Personal & Wellbeing")
    col7, col8, col9 = st.columns(3)

    with col7:
        family_income = st.number_input(
            "Family Income (annual)", min_value=0.0,
            value=50000.0, step=5000.0, format="%.0f"
        )
        internet_access = st.selectbox("Internet Access", options=["Yes", "No"])
    with col8:
        travel_time_minutes = st.number_input(
            "Travel Time (minutes)", min_value=0.0,
            value=30.0, step=5.0, format="%.0f"
        )
        part_time_job = st.selectbox("Part-Time Job", options=["No", "Yes"])
    with col9:
        stress_index = st.slider(
            "Stress Index (0–10)", min_value=0.0, max_value=10.0,
            value=5.0, step=0.5
        )

    st.markdown("---")
    submitted = st.form_submit_button(
        "🎯 Add Student & Predict Risk",
        use_container_width=True,
        type="primary",
    )

# ══════════════════════════════════════════════════════════════
# Handle Form Submission
# ══════════════════════════════════════════════════════════════
if submitted:
    payload = {
        "age": age,
        "gpa": gpa,
        "semester_gpa": semester_gpa,
        "cgpa": cgpa,
        "study_hours_per_day": study_hours_per_day,
        "attendance_rate": attendance_rate,
        "assignment_delay_days": assignment_delay_days,
        "family_income": family_income,
        "travel_time_minutes": travel_time_minutes,
        "stress_index": stress_index,
        "gender": gender,
        "internet_access": internet_access,
        "part_time_job": part_time_job,
        "scholarship": scholarship,
        "department": department,
        "semester": semester,
        "parental_education": parental_education,
    }

    try:
        with st.spinner("Running AI prediction..."):
            response = requests.post(
                f"{API_URL}/students", json=payload, timeout=10
            )

        if response.status_code == 201:
            data = response.json()
            st.success("✅ Student added successfully!")

            st.markdown("---")
            st.markdown("### 🎯 Prediction Result")

            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                st.metric("Risk Score", f"{data['risk_score']:.3f}")
            with col_r2:
                st.metric("Risk Level", data["risk_status"])
            with col_r3:
                st.metric("Student ID", data["id"][:8] + "...")

            # Recommendation box
            st.markdown("---")
            st.markdown("### 💡 AI Recommendation")
            # Split recommendations by pipe separator
            recs = data["recommendation"].split(" | ")
            for rec in recs:
                if "URGENT" in rec or "⚠️" in rec:
                    st.error(rec)
                elif "✅" in rec:
                    st.success(rec)
                else:
                    st.warning(rec)
        else:
            error_detail = response.json().get("detail", "Unknown error")
            st.error(f"❌ Failed to add student: {error_detail}")

    except requests.ConnectionError:
        st.error(
            "Cannot connect to the backend API. "
            "Make sure it's running on port 8000."
        )
    except Exception as e:
        st.error(f"Error: {e}")
