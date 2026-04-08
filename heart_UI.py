import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('heart_stacked_lr_model.pkl')
scaler = joblib.load('heart_scale.pkl')

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("❤ Heart Disease Prediction")
st.markdown("Clinical-style input form")

# ================= UI =================
Age = st.number_input("Age", min_value=1, max_value=120, value=30)
Sex = st.selectbox("Sex", ["M", "F"])
RestingBP = st.number_input("Resting Blood Pressure", value=120)
Cholesterol = st.number_input("Cholesterol", value=200)
MaxHR = st.number_input("Max Heart Rate", value=150)
Oldpeak = st.number_input("Oldpeak", value=1.0)

ST_Slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
ExerciseAngina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
ChestPainType = st.selectbox("Chest Pain Type", ["ASY", "NAP", "ATA", "TA"])
FastingBS = st.selectbox("Fasting Blood Sugar", ["Yes", "No"])
RestingECG = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])

# ================= PREPROCESS =================

# Binary mappings
isMale = 1 if Sex == 'M' else 0
isExerciseAngina = 1 if ExerciseAngina == 'Yes' else 0
isFastingBS = 1 if FastingBS == 'Yes' else 0

# ST Slope encoding
ST_Slope_Up = 1 if ST_Slope == 'Up' else 0
ST_Slope_Flat = 1 if ST_Slope == 'Flat' else 0

# Chest Pain encoding
ChestPainType_ATA = 1 if ChestPainType == 'ATA' else 0
ChestPainType_NAP = 1 if ChestPainType == 'NAP' else 0

# Resting ECG encoding
RestingECG_ST = 1 if RestingECG == 'ST' else 0
RestingECG_Normal = 1 if RestingECG == 'Normal' else 0

# Log transform Oldpeak
Oldpeak_log = np.log1p(Oldpeak)

# Build full feature vector (14 features)
full_features = np.array([[ 
    Age, RestingBP, Cholesterol, MaxHR, Oldpeak_log,
    ST_Slope_Up, ST_Slope_Flat, isExerciseAngina,
    ChestPainType_ATA, isMale, isFastingBS,
    ChestPainType_NAP, RestingECG_ST, RestingECG_Normal
]])

# Scale ALL features (as per trained pipeline)
final_features = scaler.transform(full_features)

# ================= PREDICTION =================

if st.button("Predict"):
    prediction = model.predict(final_features)[0]
    probability = model.predict_proba(final_features)[0][1]

    if prediction == 1:
        st.error(f"High Risk of Heart Disease ⚠️ (Probability: {probability:.2f})")
    else:
        st.success(f"Low Risk of Heart Disease ✅ (Probability: {probability:.2f})")
