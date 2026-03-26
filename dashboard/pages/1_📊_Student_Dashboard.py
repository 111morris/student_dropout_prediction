"""
📊 Student Dashboard Page
Displays all students in a table with color-coded risk badges.
"""

import streamlit as st
import requests
import pandas as pd
import os

st.set_page_config(page_title="Student Dashboard", page_icon="📊", layout="wide")

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")


def get_risk_badge(status: str) -> str:
    """Return an HTML badge for the risk status."""
    if "High" in status:
        return '<span class="risk-high">🔴 High Risk</span>'
    elif "Medium" in status:
        return '<span class="risk-medium">🟡 Medium Risk</span>'
    else:
        return '<span class="risk-low">🟢 Low Risk</span>'


# Custom CSS
st.markdown("""
<style>
    .risk-high { background: #ff6b6b; color: white; padding: 4px 12px; border-radius: 12px; font-weight: 600; }
    .risk-medium { background: #ffc048; color: white; padding: 4px 12px; border-radius: 12px; font-weight: 600; }
    .risk-low { background: #2ed573; color: white; padding: 4px 12px; border-radius: 12px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


st.markdown("# 📊 Student Dashboard")
st.markdown("View and monitor all students in the system.")
st.markdown("---")

# ── Filters ──────────────────────────────────────────────────
col1, col2 = st.columns([1, 3])
with col1:
    risk_filter = st.selectbox(
        "Filter by Risk Level",
        options=["All", "🔴 High", "🟡 Medium", "🟢 Low"],
        index=0,
    )

# ── Fetch Students ───────────────────────────────────────────
try:
    params = {"limit": 500}
    if risk_filter != "All":
        params["risk_filter"] = risk_filter

    response = requests.get(f"{API_URL}/students", params=params, timeout=5)

    if response.status_code == 200:
        data = response.json()
        students = data["students"]

        if not students:
            st.info("👋 No students found. Add one from the **➕ Add Student** page!")
        else:
            st.markdown(f"**Showing {len(students)} of {data['total']} students**")

            # Convert to DataFrame for display
            df = pd.DataFrame(students)

            # Select and rename columns for display
            display_cols = {
                "id": "ID",
                "age": "Age",
                "gender": "Gender",
                "department": "Department",
                "semester": "Semester",
                "gpa": "GPA",
                "cgpa": "CGPA",
                "attendance_rate": "Attendance %",
                "stress_index": "Stress",
                "risk_score": "Risk Score",
                "risk_status": "Risk Level",
            }

            df_display = df[list(display_cols.keys())].rename(columns=display_cols)
            df_display["Risk Score"] = df_display["Risk Score"].apply(
                lambda x: f"{x:.3f}" if x is not None else "N/A"
            )

            # Color-code risk level
            def color_risk(val):
                if "High" in str(val):
                    return "background-color: #ffe0e0; color: #c0392b; font-weight: 600;"
                elif "Medium" in str(val):
                    return "background-color: #fff3cd; color: #856404; font-weight: 600;"
                elif "Low" in str(val):
                    return "background-color: #d4edda; color: #155724; font-weight: 600;"
                return ""

            styled = df_display.style.applymap(
                color_risk, subset=["Risk Level"]
            )

            st.dataframe(
                styled,
                use_container_width=True,
                height=min(400, len(students) * 40 + 50),
            )

            # ── Quick Summary ────────────────────────────────
            st.markdown("---")
            st.markdown("### 📈 Risk Distribution")
            col_a, col_b, col_c = st.columns(3)

            high = sum(1 for s in students if "High" in (s.get("risk_status") or ""))
            med = sum(1 for s in students if "Medium" in (s.get("risk_status") or ""))
            low = sum(1 for s in students if "Low" in (s.get("risk_status") or ""))

            with col_a:
                st.metric("🔴 High Risk", high)
            with col_b:
                st.metric("🟡 Medium Risk", med)
            with col_c:
                st.metric("🟢 Low Risk", low)

            # ── Clickable student list ───────────────────────
            st.markdown("---")
            st.markdown("### 🔍 View Student Details")
            st.markdown(
                "Copy a student ID from the table above and paste it in the "
                "**👤 Student Detail** page to see their full profile."
            )
    else:
        st.error(f"API error: {response.status_code}")

except requests.ConnectionError:
    st.warning(
        "⚠️ Cannot connect to the backend API. "
        "Make sure it's running on port 8000."
    )
except Exception as e:
    st.error(f"Error: {e}")
