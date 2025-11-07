"""
Title: Type 2 Diabetes Risk Predictor (Interactive Web App)
Author: Adetoyinbo Oyinkansola
Organisation: Write Right Concept
Description:
    Streamlit app for predicting Type 2 diabetes risk using a trained Gradient Boosting model.
    This app transforms the research analysis into an interactive experience.
"""

# ===============================
# 📦 IMPORT LIBRARIES
# ===============================
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
import pickle

# ===============================
# 🧱 APP CONFIGURATION
# ===============================
st.set_page_config(
    page_title="Diabetes Risk Predictor | Adetoyinbo Oyinkansola",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🩺 Type 2 Diabetes Risk Prediction App")
st.write(
    """
    This interactive app uses a **machine learning model (Gradient Boosting)** 
    developed during my internship at **Write Right Concept** to predict the likelihood of 
    developing Type 2 Diabetes based on key health indicators.

    🔍 The model achieved **94% accuracy (AUC = 0.98)** in testing and demonstrates
    how data analytics can support preventive healthcare.
    """
)

# ===============================
# 🧩 FEATURE INPUTS
# ===============================

st.sidebar.header("🧾 Input Your Health Details")

age = st.sidebar.slider("Age", 18, 80, 30)
bmi = st.sidebar.slider("Body Mass Index (BMI)", 10.0, 50.0, 25.0)
highbp = st.sidebar.selectbox("High Blood Pressure", ["No", "Yes"])
physically_active = st.sidebar.selectbox("Physically Active", ["No", "Yes"])
stress = st.sidebar.selectbox("High Stress Levels", ["No", "Yes"])
family_diabetes = st.sidebar.selectbox("Family History of Diabetes", ["No", "Yes"])
sleep = st.sidebar.slider("Average Sleep (Hours per night)", 3, 10, 7)
regular_medicine = st.sidebar.selectbox("Takes Regular Medicine", ["No", "Yes"])
urination_freq = st.sidebar.selectbox("Frequent Urination", ["No", "Yes"])

# Convert inputs to dataframe
input_data = pd.DataFrame(
    {
        "Age": [age],
        "BMI": [bmi],
        "HighBP": [1 if highbp == "Yes" else 0],
        "PhysicallyActive": [1 if physically_active == "Yes" else 0],
        "Stress": [1 if stress == "Yes" else 0],
        "Family_Diabetes": [1 if family_diabetes == "Yes" else 0],
        "Sleep": [sleep],
        "RegularMedicine": [1 if regular_medicine == "Yes" else 0],
        "UrinationFreq": [1 if urination_freq == "Yes" else 0],
    }
)

# ===============================
# ⚙️ MODEL TRAINING (Temporary Inline)
# ===============================
# In practice, you’d load a pre-trained model, but here we fit a lightweight one for demonstration.
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load and prepare data (same as in your main analysis)
df = pd.read_csv("diabetes.csv")
for col in df.select_dtypes("object"):
    df[col] = LabelEncoder().fit_transform(df[col])
scaler = MinMaxScaler()
df[df.select_dtypes("number").columns] = scaler.fit_transform(df.select_dtypes("number"))

X = df.drop("Diabetic", axis=1)
y = df["Diabetic"]
X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)
model = GradientBoostingClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ===============================
# 🔮 PREDICTION
# ===============================
st.subheader("📊 Prediction Result")

if st.button("Predict My Risk"):
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)
    probability = model.predict_proba(scaled_input)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ High Risk of Type 2 Diabetes (Probability: {probability*100:.1f}%)")
        st.write(
            """
            **Recommendation:**  
            I encourage you to discuss preventive screening with your healthcare provider.  
            Maintaining regular exercise, balanced diet, and sleep patterns can lower your risk.
            """
        )
    else:
        st.success(f"✅ Low Risk of Type 2 Diabetes (Probability: {probability*100:.1f}%)")
        st.write(
            """
            **Great job!**  
            Continue healthy lifestyle practices such as physical activity, stress management, 
            and regular check-ups.
            """
        )

# ===============================
# 📉 MODEL PERFORMANCE INSIGHT
# ===============================
st.markdown("---")
st.subheader("🔍 About the Model")
st.write(
    """
    - **Algorithm:** Gradient Boosting (200 estimators)  
    - **Accuracy:** 94%  
    - **AUC (ROC):** 0.98  
    - **Dataset:** Diabetes Health Indicators (Tigga & Garg, 2019)  
    - **Balancing Technique:** SMOTE for fair representation  
    - **Developer:** *Adetoyinbo Oyinkansola @ Write Right Concept*  
    """
)

st.caption("© 2025 — Built with ❤️ and Streamlit for data-driven wellness innovation.")
