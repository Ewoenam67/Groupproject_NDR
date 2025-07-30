import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_option_menu import option_menu

# ------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------
st.set_page_config(page_title="Loan Default Classifier - Group 3", layout="wide")

# ------------------------------------------------
# LOAD TRAINED ARTIFACTS
# ------------------------------------------------
model = joblib.load("gradient_boost_model.pkl")
scaler = joblib.load("scaler.pkl")
X_columns = joblib.load("X_columns.pkl")

# ------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "Predictor", "About"],
        icons=["house", "bar-chart", "info-circle"],
        default_index=1,
        styles={
            "container": {"padding": "0!important", "background-color": "#ecfdf5"},
            "icon": {"color": "#047857", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "--hover-color": "#a7f3d0"},
            "nav-link-selected": {"background-color": "#6ee7b7", "color": "#000"},
        }
    )

# ------------------------------------------------
# HOME TAB
# ------------------------------------------------
if selected == "Home":
    st.title("🏠 Welcome to the Loan Default Classifier - Group 3")
    st.write("Use this app to predict whether a customer is likely to default on a loan based on their input features.")

# ------------------------------------------------
# PREDICTOR TAB
# ------------------------------------------------
elif selected == "Predictor":
    st.title("🤖 Loan Default Prediction")
    st.write("Enter the required customer details to predict loan default status.")

    input_data = {}
    for col in X_columns:
        if col.lower() in ['loan_amount', 'interest_rate', 'credit_score', 'loan_term']:
            input_data[col] = st.number_input(col.replace("_", " ").title(), min_value=0.0, value=50.0)
        elif col.lower() in ['employment_status']:
            input_data[col] = st.selectbox("Employment Status", ['Employed', 'Self-employed', 'Unemployed'])
        elif col.lower() in ['gender']:
            input_data[col] = st.selectbox("Gender", ['Male', 'Female'])
        else:
            input_data[col] = st.text_input(f"{col.replace('_', ' ').title()}")

    # Manual encoding
    encoders = {
        'Gender': {'Male': 1, 'Female': 0},
        'Employment_Status': {'Employed': 0, 'Self-employed': 1, 'Unemployed': 2}
    }

    for col, mapping in encoders.items():
        if col in input_data:
            input_data[col] = mapping[input_data[col]]

    input_df = pd.DataFrame([input_data])
  # Ensure all expected columns exist and are in correct order
for col in X_columns:
    if col not in input_df.columns:
        input_df[col] = 0  # or another neutral value
input_df = input_df[X_columns]
input_scaled = scaler.transform(input_df)

    if st.button("Predict Loan Default"):
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][int(pred)]
        outcome = "Default" if pred == 1 else "No Default"
        st.success(f"📌 Prediction: {outcome}")
        st.write(f"Confidence Score: {prob * 100:.2f}%")

# ------------------------------------------------
# ABOUT TAB
# ------------------------------------------------
elif selected == "About":
    st.title("ℹ About This App")
    st.markdown("""
        This app was developed by *Group 3* to predict whether a customer will default on a loan.

        *Model Used:* Gradient Boosting Classifier  
        *Tools:* Python, Scikit-learn, Streamlit  
        *Dataset:* Xente Loan Default Dataset  
        *Goal:* Improve decision-making in customer loan approvals
    """)
