"""
Student Dropout Early Warning System — Dashboard
Main Streamlit application entry point.

Run with:
    streamlit run dashboard/app.py

Requires the FastAPI backend to be running at http://localhost:8000.
"""

import streamlit as st
import requests

# ══════════════════════════════════════════════════════════════
# Page Configuration
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Student Dropout EWS",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# Backend API URL
# ══════════════════════════════════════════════════════════════
import os
API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")


# ══════════════════════════════════════════════════════════════
# Custom CSS for premium look
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* Main page styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    /* Risk badges */
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .risk-medium {
        background: linear-gradient(135deg, #ffc048, #ff9f43);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .risk-low {
        background: linear-gradient(135deg, #2ed573, #26de81);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2d3436;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #636e72;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: #e0e0e0;
    }
    /* Info boxes */
    .info-box {
        background: #f0f4ff;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎓 Student EWS")
    st.markdown("---")
    st.markdown(
        "**Early Warning System** for identifying students at risk of dropping out."
    )
    st.markdown("---")
    st.markdown("### Navigation")
    st.markdown(
        "Use the pages in the sidebar to:\n"
        "- 📊 View the Student Dashboard\n"
        "- ➕ Add a New Student\n"
        "- 🔮 Run a Quick Prediction\n"
        "- 👤 View Student Details"
    )
    st.markdown("---")
    st.markdown(
        "<small>Powered by Machine Learning<br>"
        "Built with FastAPI + Streamlit</small>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════
# Main Page — Overview Dashboard
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="main-header">🎓 Student Dropout Early Warning System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">AI-powered risk assessment to help schools support students before it\'s too late.</div>',
    unsafe_allow_html=True,
)

# Fetch summary stats from the API
try:
    response = requests.get(f"{API_URL}/students?limit=500", timeout=5)
    if response.status_code == 200:
        data = response.json()
        total_students = data["total"]
        students = data["students"]

        # Count risk levels
        high_risk = sum(1 for s in students if "High" in (s.get("risk_status") or ""))
        medium_risk = sum(1 for s in students if "Medium" in (s.get("risk_status") or ""))
        low_risk = sum(1 for s in students if "Low" in (s.get("risk_status") or ""))

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                label="📚 Total Students",
                value=total_students,
            )
        with col2:
            st.metric(
                label="🔴 High Risk",
                value=high_risk,
                delta=f"{high_risk/max(total_students,1)*100:.0f}%" if total_students > 0 else "0%",
                delta_color="inverse",
            )
        with col3:
            st.metric(
                label="🟡 Medium Risk",
                value=medium_risk,
                delta=f"{medium_risk/max(total_students,1)*100:.0f}%" if total_students > 0 else "0%",
                delta_color="off",
            )
        with col4:
            st.metric(
                label="🟢 Low Risk",
                value=low_risk,
                delta=f"{low_risk/max(total_students,1)*100:.0f}%" if total_students > 0 else "0%",
                delta_color="normal",
            )

        st.markdown("---")

        if total_students > 0:
            st.markdown("### ⚡ Quick Stats")
            col_a, col_b = st.columns(2)
            with col_a:
                avg_gpa = sum(s["gpa"] for s in students) / total_students
                avg_attendance = sum(s["attendance_rate"] for s in students) / total_students
                st.info(f"📊 **Average GPA:** {avg_gpa:.2f}")
                st.info(f"📊 **Average Attendance:** {avg_attendance:.1f}%")
            with col_b:
                avg_risk = sum(s["risk_score"] for s in students if s["risk_score"] is not None) / max(total_students, 1)
                st.info(f"📊 **Average Risk Score:** {avg_risk:.3f}")
                st.info(f"📊 **Students Monitored:** {total_students}")
        else:
            st.info(
                "👋 No students in the database yet. "
                "Use **➕ Add Student** to get started!"
            )
    else:
        st.error(f"Failed to fetch data from API (status: {response.status_code})")

except requests.ConnectionError:
    st.warning(
        "⚠️ **Cannot connect to the backend API.**\n\n"
        "Make sure the FastAPI server is running:\n"
        "```bash\n"
        "uvicorn backend.main:app --reload --port 8000\n"
        "```"
    )
except Exception as e:
    st.error(f"Unexpected error: {e}")


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #999;'>"
    "🎓 Student Dropout Early Warning System v1.0.0 | "
    "Built for school staff — teachers & administrators"
    "</div>",
    unsafe_allow_html=True,
)
