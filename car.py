import streamlit as st
import pandas as pd
import joblib

# ------------------- Page Configuration -------------------
st.set_page_config(
    page_title="Vehicle Maintenance Cost Prediction",
    page_icon="🚗",
    layout="wide"
)

# ------------------- Load Dataset -------------------
df = pd.read_csv("vehicle_maintenance.csv")

# ------------------- Load Model -------------------
reg_model = joblib.load("vehicle_model.pkl")

# ------------------- Load Encoders -------------------
brand_encoder = joblib.load("brand_encoder.pkl")
car_encoder = joblib.load("car_encoder.pkl")
issue_encoder = joblib.load("issue_encoder.pkl")

# ------------------- Title -------------------
st.title("🚗 Vehicle Maintenance Cost Prediction System")
st.write("Estimate the maintenance cost based on your vehicle details.")

st.divider()

# ------------------- Input Fields -------------------
col1, col2, col3, col4, col5 = st.columns(5)

# Brand Selection
with col1:
    brand = st.selectbox(
        "Vehicle Brand",
        sorted(df["Brand"].unique())
    )

# Filter Cars Based on Selected Brand
available_cars = sorted(
    df[df["Brand"] == brand]["Car_Name"].unique()
)

# Car Selection
with col2:
    car_name = st.selectbox(
        "Car Name",
        available_cars
    )

# Issue Selection
with col3:
    issue = st.selectbox(
        "Vehicle Issue",
        sorted(df["Issue"].unique())
    )

# Model Year
with col4:
    model_year = st.number_input(
        "Model Year",
        min_value=2000,
        max_value=2026,
        value=2020
    )

# KMs Driven
with col5:
    kms_driven = st.number_input(
        "KMs Driven",
        min_value=0,
        max_value=500000,
        value=30000,
        step=1000
    )

st.write("")

# ------------------- Prediction -------------------
if st.button("🔧 Predict Maintenance Cost", use_container_width=True):

    # Encode Inputs
    brand_encoded = brand_encoder.transform([brand])[0]
    car_encoded = car_encoder.transform([car_name])[0]
    issue_encoded = issue_encoder.transform([issue])[0]

    # Prepare Input
    input_data = [[
        brand_encoded,
        car_encoded,
        issue_encoded,
        model_year,
        kms_driven
    ]]

    # Predict
    predicted_cost = reg_model.predict(input_data)[0]

    st.success(f"## 💰 Estimated Maintenance Cost: ₹ {predicted_cost:,.2f}")

st.divider()

st.caption("Built with ❤️ using Streamlit & Scikit-Learn")
