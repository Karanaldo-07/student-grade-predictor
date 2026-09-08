from pathlib import Path

import streamlit as st

from src.model import FEATURES, load_model, predict_grade

MODEL_PATH = Path("models/student_grade_model.joblib")

st.set_page_config(page_title="Student Grade Predictor", page_icon="🎓", layout="centered")

st.title("🎓 Student Grade Predictor")
st.caption("A lightweight machine-learning demo for educational use.")

if not MODEL_PATH.exists():
    st.error("The trained model is not available yet. Run `python train.py` first.")
    st.stop()

model = load_model(MODEL_PATH)

with st.form("prediction_form"):
    study_hours = st.slider("Study hours per day", 0.0, 12.0, 5.0, 0.5)
    attendance = st.slider("Attendance (%)", 0, 100, 85)
    previous_grade = st.slider("Previous grade (%)", 0, 100, 70)
    assignments_completed = st.slider("Assignments completed (%)", 0, 100, 80)
    sleep_hours = st.slider("Sleep hours per night", 0.0, 12.0, 7.0, 0.5)
    extracurricular_hours = st.slider("Extracurricular hours per week", 0.0, 20.0, 5.0, 0.5)
    submitted = st.form_submit_button("Predict grade")

if submitted:
    values = {
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_grade": previous_grade,
        "assignments_completed": assignments_completed,
        "sleep_hours": sleep_hours,
        "extracurricular_hours": extracurricular_hours,
    }
    prediction = predict_grade(model, values)
    st.metric("Predicted final grade", f"{prediction:.1f}%")
    if prediction >= 90:
        st.success("Estimated performance: Excellent")
    elif prediction >= 75:
        st.info("Estimated performance: Good")
    elif prediction >= 60:
        st.warning("Estimated performance: Needs improvement")
    else:
        st.error("Estimated performance: At risk")

st.divider()
st.caption("The prediction is an estimate from the demo dataset and is not a guarantee of actual academic performance.")
