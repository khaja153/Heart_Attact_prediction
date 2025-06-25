import streamlit as st
import numpy as np
import pandas as pd
import joblib

# 🔹 Load model and scaler
model = joblib.load("xgb_heart_model.pkl")
scaler = joblib.load("scaler.pkl")

# 🔹 Set title
st.title("💓 Heart Disease Prediction App")
st.write("Enter patient info to predict the risk of heart disease.")

# 🔹 Define input fields (match your model's features)
age = st.slider("Age", 18, 100, 45)
gender = st.radio("Gender", ["Male", "Female"])
height = st.number_input("Height (cm)", 100, 250, 170)
weight = st.number_input("Weight (kg)", 30, 200, 70)
bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
daily_steps = st.number_input("Daily Steps", 0, 50000, 8000)
calories = st.number_input("Calories Intake", 500, 5000, 2000)
sleep = st.number_input("Hours of Sleep", 0.0, 24.0, 7.0)
heart_rate = st.number_input("Heart Rate", 30, 200, 80)
bp = st.text_input("Blood Pressure (e.g. 120/80)", "120/80")
exercise = st.number_input("Exercise Hours per Week", 0.0, 40.0, 3.0)
smoker = st.radio("Smoker", ["Yes", "No"])
alcohol = st.number_input("Alcohol Consumption per Week", 0.0, 50.0, 0.0)
diabetic = st.radio("Diabetic", ["Yes", "No"])

# 🔹 Split BP into systolic and diastolic
try:
    systolic, diastolic = map(float, bp.split("/"))
except:
    systolic, diastolic = 120.0, 80.0

# 🔹 Prepare input data (order matters)
input_data = pd.DataFrame([[
    age,
    1 if gender == "Male" else 0,
    height,
    weight,
    bmi,
    daily_steps,
    calories,
    sleep,
    heart_rate,
    systolic,
    diastolic,
    exercise,
    1 if smoker == "Yes" else 0,
    alcohol,
    1 if diabetic == "Yes" else 0
]], columns=[
    'Age', 'Gender', 'Height_cm', 'Weight_kg', 'BMI', 'Daily_Steps',
    'Calories_Intake', 'Hours_of_Sleep', 'Heart_Rate', 'Systolic_BP', 'Diastolic_BP',
    'Exercise_Hours_per_Week', 'Smoker', 'Alcohol_Consumption_per_Week', 'Diabetic'
])

# Debug: Show expected and actual columns
st.write("Scaler expects:", scaler.feature_names_in_)
st.write("Your input columns:", input_data.columns)

# Force correct order
input_data = input_data[scaler.feature_names_in_]

# Scale input
scaled_input = scaler.transform(input_data)

# 🔹 Predict
if st.button("Predict"):
    pred = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1]
    
    if pred == 1:
        st.error(f"⚠️ High Risk! Probability: {prob:.2f}")
    else:
        st.success(f"✅ Low Risk. Probability: {prob:.2f}")
