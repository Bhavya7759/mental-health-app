import streamlit as st
import pickle
import numpy as np

with open('mental_health_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('model_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

st.set_page_config(page_title="Student Depression Risk Predictor", page_icon="🧠", layout="centered")

st.title("🧠 Student Depression Risk Predictor")
st.markdown("Answer the questions below honestly. This tool predicts whether a student may be at risk of depression based on academic, lifestyle, and personal factors.")
st.divider()

st.subheader("Personal Information")
col1, col2 = st.columns(2)
with col1:
    age    = st.slider("Age", 16, 35, 20)
    gender = st.selectbox("Gender", ["Male", "Female"])
with col2:
    city       = st.selectbox("City", ["Delhi", "Mumbai", "Bangalore", "Chennai", "Hyderabad", "Kolkata", "Pune", "Other"])
    profession = st.selectbox("Profession", ["Student", "Working Professional", "Other"])

st.subheader("Academic & Work")
col3, col4 = st.columns(2)
with col3:
    academic_pressure  = st.selectbox("Academic Pressure (1=Low, 5=High)", [1, 2, 3, 4, 5])
    work_pressure      = st.selectbox("Work Pressure (1=Low, 5=High)", [1, 2, 3, 4, 5])
    cgpa               = st.slider("CGPA (out of 10)", 0.0, 10.0, 7.0, step=0.1)
with col4:
    study_satisfaction = st.selectbox("Study Satisfaction (1=Low, 5=High)", [1, 2, 3, 4, 5])
    job_satisfaction   = st.selectbox("Job Satisfaction (1=Low, 5=High)", [1, 2, 3, 4, 5])
    work_study_hours   = st.slider("Work/Study Hours per day", 0, 16, 6)
    degree             = st.selectbox("Degree", ["B.Tech", "MBA", "M.Tech", "BCA", "MCA", "B.Sc", "M.Sc", "Other"])

st.subheader("Lifestyle & Mental Health")
col5, col6 = st.columns(2)
with col5:
    sleep_duration    = st.selectbox("Sleep Duration", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"])
    dietary_habits    = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])
    financial_stress  = st.selectbox("Financial Stress (1=Low, 5=High)", [1, 2, 3, 4, 5])
with col6:
    suicidal_thoughts        = st.selectbox("Have you ever had suicidal thoughts?", ["No", "Yes"])
    family_history           = st.selectbox("Family History of Mental Illness?", ["No", "Yes"])

st.divider()

# Encoding maps
gender_map    = {"Male": 1, "Female": 0}
city_map      = {"Delhi": 0, "Mumbai": 1, "Bangalore": 2, "Chennai": 3,
                 "Hyderabad": 4, "Kolkata": 5, "Pune": 6, "Other": 7}
profession_map = {"Student": 2, "Working Professional": 3, "Other": 1}
sleep_map     = {"Less than 5 hours": 1, "5-6 hours": 2, "7-8 hours": 3, "More than 8 hours": 4}
diet_map      = {"Healthy": 1, "Moderate": 2, "Unhealthy": 3}
yes_no        = {"Yes": 1, "No": 0}
degree_map    = {"B.Tech": 0, "MBA": 1, "M.Tech": 2, "BCA": 3,
                 "MCA": 4, "B.Sc": 5, "M.Sc": 6, "Other": 7}

input_dict = {
    "gender":                           gender_map[gender],
    "age":                              age,
    "city":                             city_map[city],
    "profession":                       profession_map[profession],
    "academic_pressure":                academic_pressure,
    "work_pressure":                    work_pressure,
    "cgpa":                             cgpa,
    "study_satisfaction":               study_satisfaction,
    "job_satisfaction":                 job_satisfaction,
    "sleep_duration":                   sleep_map[sleep_duration],
    "dietary_habits":                   diet_map[dietary_habits],
    "degree":                           degree_map[degree],
    "have_you_ever_had_suicidal_thoughts": yes_no[suicidal_thoughts],
    "work_study_hours":                 work_study_hours,
    "financial_stress":                 financial_stress,
    "family_history_of_mental_illness": yes_no[family_history],
}

input_array = np.array([[input_dict[col] for col in model_columns]])

if st.button("Predict Depression Risk", use_container_width=True):
    probability = model.predict_proba(input_array)[0][1]
    risk        = round(probability * 100, 1)
    prediction  = 1 if probability >= 0.40 else 0

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ Higher Depression Risk Detected — {risk}% likelihood")
        st.markdown("Based on these responses, this student may be at risk of depression. Please consider speaking to a counselor or mental health professional. **This is not a clinical diagnosis.**")
    else:
        st.success(f"✅ Lower Depression Risk — {risk}% likelihood")
        st.markdown("Based on these responses, depression risk appears lower. Keep maintaining healthy study habits, sleep, and social connections.")

    st.caption("This tool is for educational purposes only and does not replace professional medical advice.")