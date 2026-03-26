"""
👤 Student Detail Page
View individual student details, risk factors, and AI recommendation.
"""

import streamlit as st
import requests
import os

st.set_page_config(page_title="Student Detail", page_icon="👤", layout="wide")

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")

st.markdown("# 👤 Student Detail View")
st.markdown("Enter a student ID to view their full profile and risk assessment.")
st.markdown("---")

# ══════════════════════════════════════════════════════════════
# Student ID Input
# ══════════════════════════════════════════════════════════════
student_id = st.text_input(
    "🔎 Student ID",
    placeholder="Paste the student UUID here...",
    help="You can find student IDs in the Student Dashboard table.",
)

if student_id:
    try:
        with st.spinner("Fetching student data..."):
            response = requests.get(
                f"{API_URL}/students/{student_id.strip()}", timeout=5
            )

        if response.status_code == 200:
            student = response.json()

            # ── Header with Risk Badge ───────────────────────
            col_h1, col_h2 = st.columns([3, 1])
            with col_h1:
                st.markdown(f"### 🎓 Student Profile")
                st.markdown(f"**ID:** `{student['id']}`")
            with col_h2:
                risk_status = student.get("risk_status", "Unknown")
                if "High" in risk_status:
                    st.error(f"**{risk_status}**")
                elif "Medium" in risk_status:
                    st.warning(f"**{risk_status}**")
                else:
                    st.success(f"**{risk_status}**")

            st.markdown("---")

            # ── Risk Assessment Summary ──────────────────────
            st.markdown("### 🎯 Risk Assessment")
            col_r1, col_r2, col_r3 = st.columns(3)

            with col_r1:
                score = student.get("risk_score", 0)
                st.metric("Risk Score", f"{score:.4f}" if score else "N/A")
                st.progress(min(score or 0, 1.0))

            with col_r2:
                st.metric("Risk Level", risk_status)

            with col_r3:
                created = student.get("created_at", "")[:10]
                st.metric("Date Added", created)

            st.markdown("---")

            # ── Student Details ──────────────────────────────
            st.markdown("### 📋 Student Information")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("#### 🧑 Personal")
                st.markdown(f"- **Age:** {student['age']}")
                st.markdown(f"- **Gender:** {student['gender']}")
                st.markdown(f"- **Department:** {student['department']}")
                st.markdown(f"- **Semester:** {student['semester']}")
                st.markdown(f"- **Parental Education:** {student['parental_education']}")

            with col2:
                st.markdown("#### 📊 Academic")
                st.markdown(f"- **GPA:** {student['gpa']:.2f}")
                st.markdown(f"- **Semester GPA:** {student['semester_gpa']:.2f}")
                st.markdown(f"- **CGPA:** {student['cgpa']:.2f}")
                st.markdown(f"- **Attendance Rate:** {student['attendance_rate']:.1f}%")
                st.markdown(f"- **Study Hours/Day:** {student['study_hours_per_day']:.1f}")
                st.markdown(f"- **Assignment Delay:** {student['assignment_delay_days']:.0f} days")

            with col3:
                st.markdown("#### 🏠 Personal & Wellbeing")
                st.markdown(f"- **Family Income:** {student['family_income']:,.0f}")
                st.markdown(f"- **Travel Time:** {student['travel_time_minutes']:.0f} min")
                st.markdown(f"- **Internet Access:** {student['internet_access']}")
                st.markdown(f"- **Part-Time Job:** {student['part_time_job']}")
                st.markdown(f"- **Scholarship:** {student['scholarship']}")
                st.markdown(f"- **Stress Index:** {student['stress_index']:.1f}/10")

            st.markdown("---")

            # ── AI Recommendation ────────────────────────────
            st.markdown("### 💡 AI Recommendation")
            recommendation = student.get("recommendation", "No recommendation available.")
            recs = recommendation.split(" | ")

            for rec in recs:
                if "URGENT" in rec or "⚠️" in rec:
                    st.error(rec)
                elif "✅" in rec:
                    st.success(rec)
                elif "📉" in rec or "📚" in rec or "😰" in rec:
                    st.warning(rec)
                else:
                    st.info(rec)

            # ── Risk Factor Analysis ─────────────────────────
            st.markdown("---")
            st.markdown("### 🔍 Key Risk Factors")

            risk_factors = []
            if student["attendance_rate"] < 60:
                risk_factors.append(("🔴", "Critical", f"Attendance very low: {student['attendance_rate']:.1f}%"))
            elif student["attendance_rate"] < 75:
                risk_factors.append(("🟡", "Warning", f"Attendance below average: {student['attendance_rate']:.1f}%"))
            else:
                risk_factors.append(("🟢", "Good", f"Attendance: {student['attendance_rate']:.1f}%"))

            if student["gpa"] < 2.0:
                risk_factors.append(("🔴", "Critical", f"GPA very low: {student['gpa']:.2f}"))
            elif student["gpa"] < 2.5:
                risk_factors.append(("🟡", "Warning", f"GPA below average: {student['gpa']:.2f}"))
            else:
                risk_factors.append(("🟢", "Good", f"GPA: {student['gpa']:.2f}"))

            if student["stress_index"] > 7:
                risk_factors.append(("🔴", "Critical", f"High stress: {student['stress_index']:.1f}/10"))
            elif student["stress_index"] > 5:
                risk_factors.append(("🟡", "Warning", f"Moderate stress: {student['stress_index']:.1f}/10"))
            else:
                risk_factors.append(("🟢", "Good", f"Stress: {student['stress_index']:.1f}/10"))

            if student["study_hours_per_day"] < 2:
                risk_factors.append(("🔴", "Critical", f"Very low study: {student['study_hours_per_day']:.1f} hrs/day"))
            elif student["study_hours_per_day"] < 3:
                risk_factors.append(("🟡", "Warning", f"Low study: {student['study_hours_per_day']:.1f} hrs/day"))
            else:
                risk_factors.append(("🟢", "Good", f"Study: {student['study_hours_per_day']:.1f} hrs/day"))

            if student["assignment_delay_days"] > 5:
                risk_factors.append(("🟡", "Warning", f"Late assignments: {student['assignment_delay_days']:.0f} days avg"))

            if student["internet_access"] == "No":
                risk_factors.append(("🟡", "Warning", "No internet access"))

            # Display risk factors as a table
            for icon, severity, detail in risk_factors:
                if severity == "Critical":
                    st.error(f"{icon} **{severity}**: {detail}")
                elif severity == "Warning":
                    st.warning(f"{icon} **{severity}**: {detail}")
                else:
                    st.success(f"{icon} **{severity}**: {detail}")

            # ── Delete Student ───────────────────────────────
            st.markdown("---")
            with st.expander("⚠️ Danger Zone"):
                st.markdown("**Delete this student record permanently.**")
                if st.button("🗑️ Delete Student", type="secondary"):
                    del_response = requests.delete(
                        f"{API_URL}/students/{student_id.strip()}", timeout=5
                    )
                    if del_response.status_code == 204:
                        st.success("Student deleted successfully.")
                        st.rerun()
                    else:
                        st.error("Failed to delete student.")

        elif response.status_code == 404:
            st.warning("🔍 Student not found. Please check the ID and try again.")
        else:
            st.error(f"API error: {response.status_code}")

    except requests.ConnectionError:
        st.error("Cannot connect to backend API. Is it running?")
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info(
        "💡 **Tip:** Go to the **📊 Student Dashboard** page, find a student "
        "in the table, copy their ID, and paste it above."
    )
