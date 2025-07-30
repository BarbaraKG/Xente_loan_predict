import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_option_menu import option_menu

# ------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------
st.set_page_config(page_title="Loan Default Predictor - Xente", layout="wide")

# ------------------------------------------------
# CUSTOM STYLING (New color scheme)
# ------------------------------------------------
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #DFF0D8 !important;
        }
        [data-testid="stSidebar"] > div:first-child {
            border-right: none;
        }
        html, body, [data-testid="stAppViewContainer"] > .main {
            background-color: #F9F9F9 !important;
            color: #000000 !important;
        }
        .stSelectbox > div,
        .stSelectbox div[data-baseweb="select"] > div {
            background-color: #DFF0D8 !important;
            border-radius: 8px;
        }
        input[type="number"] {
            background-color: #DFF0D8 !important;
            border-radius: 8px;
            padding: 0.4rem;
        }
        div[data-baseweb="slider"] > div > div > div:nth-child(2) {
            background: #3C763D !important;
        }
        div[data-baseweb="slider"] > div > div > div:nth-child(3) {
            background: #d0d0d0 !important;
        }
        div[data-baseweb="slider"] [role="slider"] {
            background-color: #3C763D !important;
        }
        div.stButton > button {
            background-color: #3C763D !important;
            color: white !important;
            border-radius: 8px !important;
            height: 3em;
            padding: 0.6rem 1.5rem;
            border: none;
        }
        div.stButton > button:hover {
            background-color: #2E5E2A !important;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# LOAD ARTIFACTS
# ------------------------------------------------
model = joblib.load("loan_default_prediction.pkl")
scaler = joblib.load("scaler.pkl")
X_columns = joblib.load("X_columns.pkl")

# ------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------
with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Home", "Predictor", "About"],
        icons=["house", "bar-chart", "info-circle"],
        default_index=1,
        orientation="vertical",
        styles={
            "container": {
                "padding": "0!important",
                "background-color": "#DFF0D8"
            },
            "icon": {"color": "#3C763D", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#C8E5BC",
                "color": "#333333"
            },
            "nav-link-selected": {
                "background-color": "#B2D8A3",
                "color": "#000000"
            },
        },
    )

# ------------------------------------------------
# HOME TAB
# ------------------------------------------------
if selected == "Home":
    st.title("🏠 Welcome to the Xente Loan Default Predictor")
    st.write("This app uses machine learning to predict the likelihood of a customer defaulting on a loan.")

# ------------------------------------------------
# PREDICT TAB
# ------------------------------------------------
elif selected == "Predictor":
    st.title("📊 Predict Loan Default")
    st.write("Enter customer loan details below:")

    # Input fields
    product_category = st.selectbox("Product Category", ['Airtime', 'Data Bundles', 'Retail', 'Utility Bills', 'TV', 'Financial Services', 'Movies'])  # LabelEncoded
    amount_loan = st.number_input("Amount of Loan", min_value=50.0, max_value=100000.0, value=5000.0)
    investor_id = st.selectbox("Investor ID", [1, 2])
    total_amount = st.number_input("Total Amount", min_value=50.0, max_value=100000.0, value=5000.0)

    if st.button("Predict Default Probability"):
        new_data = pd.DataFrame({
            'ProductCategory': [product_category],
            'AmountLoan': [amount_loan],
            'InvestorId': [investor_id],
            'TotalAmount': [total_amount]
        })

        new_data_scaled = scaler.transform(new_data)
        for col in X_columns:
            if col not in new_data.columns:
                new_data[col] = 0

        new_data = pd.DataFrame(new_data_scaled, columns=X_columns)
        prob = model.predict_proba(new_data)[0][1] * 100
        st.success(f"Predicted Default Probability: {prob:.2f}%")

# ------------------------------------------------
# ABOUT TAB
# ------------------------------------------------
elif selected == "About":
    st.title("ℹ️ About This App")
    st.markdown("""
        This Streamlit app predicts the probability that a customer may default on a loan based on features from Xente's financial dataset.

        Built by: Group 7  
        Model Used: Random Forest Classifier (with oversampling)  
        Libraries: scikit-learn, pandas, numpy, streamlit
    """)
