import streamlit as st
import requests

st.title("DeepCare")
st.subheader("Reducing Drop-Off, Enhancing Care")

# Collect raw inputs
age = st.number_input("Age", min_value=0, max_value=120, value=40, step=1)
chronic_conditions = st.checkbox("Has Chronic Conditions", value=False)
avg_monthly_income = st.number_input("Average Monthly Income (₹)", min_value=0.0, value=30000.0, step=1000.0)
num_visits = st.number_input("Number of Visits", min_value=0, value=5, step=1)
total_spent = st.number_input("Total Spent (₹)", min_value=0.0, value=1000.0, step=100.0)
time_in_waiting = st.number_input("Time in Waiting (minutes)", min_value=0.0, value=15.0, step=1.0)
visit_duration = st.number_input("Visit Duration (minutes)", min_value=0.0, value=30.0, step=1.0)
satisfaction_score = st.slider("Satisfaction Score", min_value=0.0, max_value=10.0, value=5.0, step=1.0)
log_count = st.number_input("Log Count", min_value=0, value=2, step=1)
critical_logs = st.number_input("Critical Logs", min_value=0, value=0, step=1)
unique_events = st.number_input("Unique Events", min_value=0, value=1, step=1)
patient_segment = st.selectbox("Patient Segment", options=[0, 1, 2], index=0)

if st.button("Predict Drop-Off"):
    features = {
        "age": age,
        "chronic_conditions": 1 if chronic_conditions else 0,
        "avg_monthly_income": avg_monthly_income,
        "num_visits": num_visits,
        "total_spent": total_spent,
        "time_in_waiting": time_in_waiting,
        "visit_duration": visit_duration,
        "satisfaction_score": satisfaction_score,
        "log_count": log_count,
        "critical_logs": critical_logs,
        "unique_events": unique_events,
        "patient_segment": patient_segment
    }
    try:
        response = requests.post("http://localhost:8000/predict", json=features)
        if response.status_code == 200:
            result = response.json()
            probability = result["probability"]
            risk = result["drop_off_risk"]
            
            if risk == "high":
                st.error(f"⚠️ High Risk of Drop-Off (Probability: {probability:.2%})")
            else:
                st.success(f"✅ Low Risk of Drop-Off (Probability: {probability:.2%})")
        else:
            st.error(f"Error: {response.json().get('detail', response.text)}")
    except Exception as e:
        st.error(f"Request failed: {e}")