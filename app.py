import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Diabetes Risk Detector", page_icon="🩺")

st.title("🩺 Diabetes Risk Predictor")
st.write("Enter patient medical measurements below to evaluate diabetes risk.")

# Load exported model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# Input layout
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=40.0, max_value=300.0, value=120.0)
    bp = st.number_input("Blood Pressure (mm Hg)", min_value=30.0, max_value=180.0, value=70.0)
    skin = st.number_input("Skin Thickness (mm)", min_value=5.0, max_value=100.0, value=20.0)

with col2:
    insulin = st.number_input("Insulin Level (mu U/ml)", min_value=10.0, max_value=900.0, value=80.0)
    bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0)
    pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.05, max_value=3.0, value=0.47)
    age = st.number_input("Age", min_value=1, max_value=120, value=33)

# Prediction
if st.button("Assess Diabetes Risk 🚀"):
    raw_input = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, pedigree, age]])
    scaled_input = scaler.transform(raw_input)
    
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1] * 100

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ **Result: High Risk of Diabetes** (Probability: {probability:.1f}%)")
    else:
        st.success(f"✅ **Result: Low Risk of Diabetes** (Probability: {probability:.1f}%)")
