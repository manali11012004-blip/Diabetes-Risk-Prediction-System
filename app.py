import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# =============================
# LOAD MODEL
# =============================
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# =============================
# RISK FUNCTION
# =============================
def risk_level(prob):
    if prob < 0.3:
        return "Low Risk"
    elif prob < 0.7:
        return "Medium Risk"
    else:
        return "High Risk"

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺")

st.title("🩺 Diabetes Risk Prediction System")
st.write("Enter patient details below:")

# =============================
# SIDEBAR GUIDE
# =============================
st.sidebar.title("ℹ️ How to Fill the Form")

st.sidebar.write("""
### 🧾 Enter your details:

**Glucose (Blood Sugar)**
- Use a glucometer (finger test)
- Normal: 70 – 140

**Blood Pressure**
- Use BP machine
- Enter upper value (e.g. 120 from 120/80)

**BMI**
- Use calculator below (don’t guess)

**Family Diabetes Risk**
- 0.1 → No history  
- 0.5 → Some relatives  
- 1.0 → Strong history  

**Age**
- Your current age
""")

# =============================
# BMI CALCULATOR
# =============================
st.sidebar.subheader("🧮 BMI Calculator")

weight = st.sidebar.number_input("Weight (kg)", 30, 200, 70)
height = st.sidebar.number_input("Height (cm)", 100, 220, 170)

bmi_calc = weight / ((height/100) ** 2)
st.sidebar.write(f"Your BMI: {round(bmi_calc,2)}")

# =============================
# INPUT FORM
# =============================
col1, col2 = st.columns(2)

with col1:
    preg = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose", 0, 300, 120)
    bp = st.number_input("Blood Pressure", 0, 200, 70)

with col2:
    bmi = st.number_input("BMI", 0.0, 60.0, float(round(bmi_calc,2)))
    dpf = st.slider("Family Diabetes Risk", 0.1, 1.0, 0.5)
    age = st.number_input("Age", 1, 100, 30)

# =============================
# PREDICT BUTTON
# =============================
if st.button("Predict"):

    # Create dataframe
    df = pd.DataFrame([[preg, glucose, bp, bmi, dpf, age]],
        columns=[
            "Pregnancies","Glucose","BloodPressure",
            "BMI","DiabetesPedigreeFunction","Age"
        ]
    )

    # Feature engineering
    df["AgeGroup"] = pd.cut(df["Age"], bins=[20,30,40,50,60,100], labels=False)
    df["BMICategory"] = pd.cut(df["BMI"], bins=[0,18.5,25,30,100], labels=False)
    df["GlucoseLevel"] = pd.cut(df["Glucose"], bins=[0,100,140,200], labels=False)

    # Scale
    scaled = scaler.transform(df)

    # Predict
    prob = model.predict_proba(scaled)[0][1]
    threshold = 0.4
    prediction = "Diabetic" if prob > threshold else "Not Diabetic"

    # =============================
    # RESULT
    # =============================
    st.subheader("Result")

    if prediction == "Diabetic":
        st.error(f"⚠️ {prediction}")
    else:
        st.success(f"✅ {prediction}")

    st.write(f"Probability: {round(prob, 2)}")
    st.write(f"Risk Level: {risk_level(prob)}")

    # =============================
    # XAI (Feature Importance)
    # =============================
    st.subheader("🔍 Why this prediction?")

    try:
        xgb_model = model.estimators_[1]
        importances = xgb_model.feature_importances_

        features = [
            "Pregnancies","Glucose","BloodPressure",
            "BMI","DPF","Age",
            "AgeGroup","BMICategory","GlucoseLevel"
        ]

        fig, ax = plt.subplots()
        ax.barh(features, importances)
        ax.set_title("Feature Importance")

        st.pyplot(fig)

    except:
        st.write("Explanation not available")