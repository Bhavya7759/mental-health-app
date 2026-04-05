import streamlit as st
import pickle
import numpy as np

with open('mental_health_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('model_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

st.set_page_config(page_title="Mental Health Risk Predictor", page_icon="🧠", layout="centered")

st.title("Mental Health Risk Predictor")
st.markdown("Answer the questions below. This tool predicts whether someone may benefit from seeking mental health treatment.")
st.divider()

st.subheader("About you")
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 75, 30)
with col2:
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

st.subheader("Your workplace")
col3, col4 = st.columns(2)
with col3:
    self_employed    = st.selectbox("Are you self-employed?", ["No", "Yes"])
    no_employees     = st.selectbox("Company size", ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
    remote_work      = st.selectbox("Do you work remotely?", ["No", "Yes"])
    tech_company     = st.selectbox("Is it a tech company?", ["Yes", "No"])
with col4:
    benefits         = st.selectbox("Mental health benefits?", ["Yes", "No", "Don't know"])
    care_options     = st.selectbox("Care options available?", ["Yes", "No", "Not sure"])
    wellness_program = st.selectbox("Wellness program?", ["Yes", "No", "Don't know"])
    seek_help        = st.selectbox("Employer encourages seeking help?", ["Yes", "No", "Don't know"])

st.subheader("Mental health history and perceptions")
col5, col6 = st.columns(2)
with col5:
    family_history            = st.selectbox("Family history of mental illness?", ["No", "Yes"])
    work_interfere            = st.selectbox("Does mental health interfere with work?", ["Never", "Rarely", "Sometimes", "Often"])
    anonymity                 = st.selectbox("Is anonymity protected?", ["Yes", "No", "Don't know"])
    leave                     = st.selectbox("How easy to take mental health leave?", ["Very easy", "Somewhat easy", "Somewhat difficult", "Very difficult", "Don't know"])
with col6:
    mental_health_consequence = st.selectbox("Negative consequences for discussing mental health?", ["No", "Yes", "Maybe"])
    phys_health_consequence   = st.selectbox("Same for physical health?", ["No", "Yes", "Maybe"])
    coworkers                 = st.selectbox("Comfortable discussing with coworkers?", ["Yes", "No", "Some of them"])
    supervisor                = st.selectbox("Comfortable discussing with supervisor?", ["Yes", "No", "Some of them"])

col7, col8 = st.columns(2)
with col7:
    mental_health_interview = st.selectbox("Bring up mental health in interview?", ["Yes", "No", "Maybe"])
    phys_health_interview   = st.selectbox("Bring up physical health in interview?", ["Yes", "No", "Maybe"])
with col8:
    mental_vs_physical = st.selectbox("Employer takes mental health as seriously as physical?", ["Yes", "No", "Don't know"])
    obs_consequence    = st.selectbox("Seen negative consequences for others?", ["No", "Yes"])

st.divider()

gender_map      = {"Male": 1, "Female": 0, "Other": 2}
yes_no          = {"Yes": 1, "No": 0}
yes_no_dk       = {"Yes": 2, "No": 1, "Don't know": 0}
care_map        = {"Yes": 2, "No": 0, "Not sure": 1}
interfere_map   = {"Never": 1, "Rarely": 2, "Sometimes": 3, "Often": 0}
leave_map       = {"Very easy": 4, "Somewhat easy": 3, "Don't know": 0, "Somewhat difficult": 2, "Very difficult": 1}
consequence_map = {"No": 0, "Yes": 2, "Maybe": 1}
coworker_map    = {"Yes": 2, "No": 0, "Some of them": 1}
employee_map    = {"1-5": 0, "6-25": 1, "26-100": 2, "100-500": 3, "500-1000": 4, "More than 1000": 5}
mvp_map         = {"Yes": 2, "No": 0, "Don't know": 1}

input_dict = {
    "Age":                        age,
    "Gender":                     gender_map[gender],
    "self_employed":              yes_no[self_employed],
    "family_history":             yes_no[family_history],
    "work_interfere":             interfere_map[work_interfere],
    "no_employees":               employee_map[no_employees],
    "remote_work":                yes_no[remote_work],
    "tech_company":               yes_no[tech_company],
    "benefits":                   yes_no_dk[benefits],
    "care_options":               care_map[care_options],
    "wellness_program":           yes_no_dk[wellness_program],
    "seek_help":                  yes_no_dk[seek_help],
    "anonymity":                  yes_no_dk[anonymity],
    "leave":                      leave_map[leave],
    "mental_health_consequence":  consequence_map[mental_health_consequence],
    "phys_health_consequence":    consequence_map[phys_health_consequence],
    "coworkers":                  coworker_map[coworkers],
    "supervisor":                 coworker_map[supervisor],
    "mental_health_interview":    consequence_map[mental_health_interview],
    "phys_health_interview":      consequence_map[phys_health_interview],
    "mental_vs_physical":         mvp_map[mental_vs_physical],
    "obs_consequence":            yes_no[obs_consequence],
}

input_array = np.array([[input_dict[col] for col in model_columns]])

if st.button("Predict", use_container_width=True):
    probability = model.predict_proba(input_array)[0][1]
    risk        = round(probability * 100, 1)
    prediction  = 1 if probability >= 0.40 else 0

    st.divider()
    if prediction == 1:
        st.error(f"Higher Risk Detected — {risk}% likelihood of needing treatment")
        st.markdown("Based on these responses, this person may benefit from seeking mental health support. This is **not a diagnosis** — please consult a qualified professional.")
    else:
        st.success(f"Lower Risk — {risk}% likelihood of needing treatment")
        st.markdown("Based on these responses, mental health risk appears lower. Staying aware and maintaining healthy habits is always a good idea.")

    st.caption("This tool is for educational purposes only and does not replace professional medical advice.")
