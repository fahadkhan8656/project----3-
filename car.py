import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ------------------- Page Configuration -------------------
st.set_page_config(
    page_title="Car Maintenance Cost Prediction Using Regression",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------- Custom CSS (Dark/Light Mode Compatible) -------------------
st.markdown("""
    <style>
    .main { padding-top: 1rem; }
    
    /* Transparent semi-dark background compatible with Streamlit Dark & Light theme */
    [data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .cost-card {
        background-color: rgba(46, 125, 50, 0.15);
        border-left: 5px solid #2e7d32;
        padding: 20px;
        border-radius: 8px;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)


# ------------------- Data & Model Loaders (Cached) -------------------
@st.cache_data
def load_data():
    return pd.read_csv("vehicle_maintenance.csv")

@st.cache_resource
def load_models():
    reg_model = joblib.load("vehicle_model.pkl")
    brand_encoder = joblib.load("brand_encoder.pkl")
    car_encoder = joblib.load("car_encoder.pkl")
    issue_encoder = joblib.load("issue_encoder.pkl")
    return reg_model, brand_encoder, car_encoder, issue_encoder

df = load_data()
reg_model, brand_encoder, car_encoder, issue_encoder = load_models()


# ------------------- Header Section -------------------
col_head, col_badge = st.columns([3, 1])
with col_head:
    st.title("🚗 Car Maintenance Cost Prediction Using Regression")
    st.write("Get accurate repair cost estimates powered by Machine Learning based on real historical data.")

with col_badge:
    st.metric(label="Total Records Analyzed", value=f"{len(df):,}")

st.divider()


# ------------------- Sidebar Controls -------------------
st.sidebar.header("🛠️ Vehicle Configurations")

# Brand Selection
selected_brand = st.sidebar.selectbox(
    "1. Select Brand",
    options=sorted(df["Brand"].unique())
)

# Filter Cars
available_cars = sorted(df[df["Brand"] == selected_brand]["Car_Name"].unique())
selected_car = st.sidebar.selectbox(
    "2. Select Model",
    options=available_cars
)

# Issue Selection
selected_issue = st.sidebar.selectbox(
    "3. Vehicle Issue / Maintenance Type",
    options=sorted(df["Issue"].unique())
)

st.sidebar.subheader("📋 Usage Specifications")

# Model Year
model_year = st.sidebar.number_input(
    "Model Year",
    min_value=2000,
    max_value=2026,
    value=2020,
    step=1
)

# KMs Driven
kms_driven = st.sidebar.number_input(
    "Total Kilometers Driven",
    min_value=0,
    max_value=500000,
    value=45000,
    step=2500
)

predict_btn = st.sidebar.button("🔧 Calculate Maintenance Cost", type="primary", use_container_width=True)


# ------------------- Main Display & Results -------------------
tab1, tab2 = st.tabs(["📊 Estimation & Analysis", "📈 Market Insights"])

with tab1:
    col_summary, col_prediction = st.columns([1, 1], gap="medium")

    with col_summary:
        st.subheader("📋 Selected Summary")
        
        # Vehicle Specs Display
        st.markdown(f"""
        * **Brand & Model:** {selected_brand} - {selected_car}
        * **Reported Issue:** `{selected_issue}`
        * **Manufacturing Year:** {model_year} (Age: {2026 - model_year} years)
        * **Odometer Reading:** {kms_driven:,} km
        """)

        # Contextual Statistics from Dataset
        matched_df = df[(df["Brand"] == selected_brand) & (df["Issue"] == selected_issue)]
        if not matched_df.empty and "Cost" in matched_df.columns:
            avg_hist_cost = matched_df["Cost"].mean()
            st.info(f"💡 **Historical Average** for `{selected_issue}` in {selected_brand} vehicles: **₹ {avg_hist_cost:,.2f}**")

    with col_prediction:
        st.subheader("💰 Cost Prediction")
        
        if predict_btn:
            try:
                # Encode Inputs
                brand_encoded = brand_encoder.transform([selected_brand])[0]
                car_encoded = car_encoder.transform([selected_car])[0]
                issue_encoded = issue_encoder.transform([selected_issue])[0]

                # Prepare Input Vector
                input_data = [[
                    brand_encoded,
                    car_encoded,
                    issue_encoded,
                    model_year,
                    kms_driven
                ]]

                # Predict
                predicted_cost = reg_model.predict(input_data)[0]

                # Output Card
                st.markdown(f"""
                <div class="cost-card">
                    <h4 style="margin:0; color:#2e7d32;">Estimated Total Repair Cost</h4>
                    <h1 style="margin:5px 0; color:#4caf50;">₹ {predicted_cost:,.2f}</h1>
                    <small>Estimate includes estimated labor and component replacement costs.</small>
                </div>
                """, unsafe_allow_html=True)

                st.caption("⚠️ *Estimates are generated via predictive modeling and may vary based on exact workshop rates.*")

            except Exception as e:
                st.error(f"Error executing prediction: {e}")
        else:
            st.warning("👈 Adjust your vehicle details in the sidebar and click **Calculate Maintenance Cost**.")

with tab2:
    st.subheader("📊 Historical Cost Distribution")
    
    if "Cost" in df.columns:
        fig = px.box(
            df, 
            x="Brand", 
            y="Cost", 
            color="Brand", 
            title="Maintenance Cost Range by Brand",
            labels={"Cost": "Cost (₹)"}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload/Include a 'Cost' column in your dataset to display historical market insights.")

# ------------------- Footer -------------------
st.divider()
st.caption("🚗 Vehicle Cost Analytics Dashboard | Powered by Scikit-Learn & Streamlit")
