
# 🩺 Diabetes Risk Prediction System

A Machine Learning-based web application that predicts the risk of diabetes using patient health data.  
The system uses an ensemble model and provides probability-based predictions along with risk levels and explainability.

---

## 🚀 Features

- Predicts diabetes (Diabetic / Not Diabetic)
- Displays:
  - Probability score
  - Risk level (Low / Medium / High)
- User-friendly Streamlit interface
- Built-in BMI Calculator
- Explainable AI (feature importance)
- Threshold tuning for better real-world performance

📸 This repository includes application screenshots (input form, prediction results, and explainable AI output) for quick visualization.

---

## 🧠 Project Overview

This project implements a complete Machine Learning pipeline:

1. Data preprocessing and feature engineering (in Jupyter Notebook)
2. Model training using ensemble methods
3. Model saving using `.pkl` files
4. Deployment using Streamlit web application
5. Explainable AI for transparency

---

## 📊 Dataset

- Dataset: PIMA Indians Diabetes Dataset
- Contains medical attributes such as:
  - Glucose level
  - BMI
  - Age
  - Blood pressure
  - Family history

---

## ⚙️ Model Details

- Models Used:
  - Random Forest
  - XGBoost
- Final Model:
  - Voting Classifier (Ensemble)

### Techniques Used
- Data cleaning (handling missing values)
- Feature engineering:
  - AgeGroup
  - BMICategory
  - GlucoseLevel
- Feature scaling using StandardScaler
- Threshold optimization (0.4 instead of default 0.5)

---

## 📈 Performance

- Accuracy: ~75–80%
- Focus: **Higher recall to reduce missed diabetic cases**
- Model uses probability-based predictions instead of fixed classification

---

## 📊 Input Features

| Feature | Description |
|--------|------------|
| Pregnancies | Number of pregnancies |
| Glucose | Blood sugar level |
| Blood Pressure | Systolic BP |
| BMI | Body Mass Index |
| Family Risk | Diabetes family history |
| Age | Age |

---

## 🧪 Example Inputs & Expected Outputs

### 🟢 Low Risk Example
      Pregnancies: 1
Glucose: 85
Blood Pressure: 66
BMI: 26.6
Family Risk: 0.2
Age: 30

👉 Output:
- Prediction: Not Diabetic  
- Probability: ~0.03  
- Risk Level: Low  

---

### 🟡 Medium Risk Example

Pregnancies: 2
Glucose: 120
Blood Pressure: 70
BMI: 30.5
Family Risk: 0.45
Age: 40

👉 Output:- Prediction: Borderline  
- Probability: ~0.35–0.45  
- Risk Level: Medium  

---

### 🔴 High Risk Example

Pregnancies: 6
Glucose: 148
Blood Pressure: 72
BMI: 33.6
Family Risk: 0.627
Age: 50

👉 Output:
- Prediction: Diabetic  
- Probability: ~0.8+  
- Risk Level: High  
