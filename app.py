import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ======================================
# LOAD MODEL FILES
# ======================================

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("features.pkl")

# ======================================
# PAGE CONFIG
# ======================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# ======================================
# TITLE
# ======================================

st.title("📊 Customer Churn Prediction App")
st.write("Fill customer details below.")

# ======================================
# INPUT SECTION
# ======================================

input_data = {}

for feature in feature_names:

    # ======================================
    # GENDER
    # ======================================

    if feature == "gender":

        value = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        input_data[feature] = 1 if value == "Male" else 0

    # ======================================
    # SENIOR CITIZEN
    # ======================================

    elif feature == "SeniorCitizen":

        value = st.selectbox(
            "Senior Citizen",
            ["No", "Yes"]
        )

        input_data[feature] = 1 if value == "Yes" else 0

    # ======================================
    # YES / NO FEATURES
    # ======================================

    elif feature in [
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]:

        value = st.selectbox(
            feature,
            ["No", "Yes"]
        )

        input_data[feature] = 1 if value == "Yes" else 0

    # ======================================
    # CONTRACT
    # ======================================

    elif feature == "Contract":

        value = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"]
        )

        contract_map = {
            "Month-to-month": 0,
            "One year": 1,
            "Two year": 2
        }

        input_data[feature] = contract_map[value]

    # ======================================
    # INTERNET SERVICE
    # ======================================

    elif feature == "InternetService":

        value = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        internet_map = {
            "DSL": 0,
            "Fiber optic": 1,
            "No": 2
        }

        input_data[feature] = internet_map[value]

    # ======================================
    # PAYMENT METHOD
    # ======================================

    elif feature == "PaymentMethod":

        value = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer",
                "Credit card"
            ]
        )

        payment_map = {
            "Electronic check": 0,
            "Mailed check": 1,
            "Bank transfer": 2,
            "Credit card": 3
        }

        input_data[feature] = payment_map[value]

    # ======================================
    # NUMERIC FEATURES
    # ======================================

    else:

        input_data[feature] = st.number_input(
            feature,
            min_value=0.0,
            value=0.0
        )

# ======================================
# CREATE DATAFRAME
# ======================================

input_df = pd.DataFrame([input_data])

# Ensure exact same column order
input_df = input_df[feature_names]

# ======================================
# SCALE INPUT
# ======================================

scaled_data = scaler.transform(input_df)

# ======================================
# PREDICTION
# ======================================

if st.button("Predict Churn"):

    prediction = model.predict(scaled_data)[0]

    probability = model.predict_proba(scaled_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error("⚠️ Customer is likely to Churn")

    else:

        st.success("✅ Customer is likely to Stay")

    st.write(f"### Churn Probability: {probability:.2f}")

    st.progress(float(probability))