#ui file using streamlit
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib

model = joblib.load("/home/stranger/Desktop/Heart_Disease_Project/results/final_model.joblib")
st.title("❤️ Heart Disease Prediction App")

st.markdown("Enter your health details below to check the risk of heart disease.")

# --- User Inputs ---
age = st.number_input("Age", 18, 100, 30)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", [0, 1])
restecg = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 70, 250, 150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 10.0, 1.0)
slope = st.selectbox("Slope of Peak Exercise ST Segment (0–2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect)", [0, 1, 2])

# Collect input in correct format
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])

# --- Prediction ---
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ High chance of Heart Disease")
    else:
        st.success("✅ Low chance of Heart Disease")

# --- Data Visualization (Bonus) ---
st.subheader("📊 Heart Disease Trends in Dataset")
uploaded_file = st.file_uploader("Upload dataset (CSV)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    st.bar_chart(df["target"].value_counts())
    st.line_chart(df[["age", "chol"]])
