from pathlib import Path

import pandas as pd
import streamlit as st

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Bank Marketing Analytics",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bank_marketing_clean.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "reports"
    / "findings"
    / "model_performance.csv"
)

AGE_SEGMENT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "findings"
    / "age_segment_conversion.csv"
)

BALANCE_SEGMENT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "findings"
    / "balance_segment_conversion.csv"
)

CAMPAIGN_PATH = (
    PROJECT_ROOT
    / "reports"
    / "findings"
    / "campaign_intensity_conversion.csv"
)

PREVIOUS_CONTACT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "findings"
    / "previous_contact_conversion.csv"
)

# =====================================================
# DATA LOADING
# =====================================================

@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_PATH)

df = load_dataset()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select page",
    [
        "Executive Summary",
        "Customer Segmentation",
        "Campaign Performance",
        "Predictive Modeling"
    ]
)

# =====================================================
# EXECUTIVE SUMMARY
# =====================================================

if page == "Executive Summary":

    st.title(
        "📊 Bank Marketing Campaign Analysis"
    )

    total_customers = len(df)

    conversion_rate = round(
        df["subscribed_target"].mean() * 100,
        2
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Customers",
            f"{total_customers:,}"
        )

    with col2:
        st.metric(
            "Conversion Rate",
            f"{conversion_rate}%"
        )

    st.markdown("---")

    st.subheader(
        "Executive Insights"
    )

    st.success(
        """
        • Customers aged 60+ are the highest-converting segment.

        • Previously contacted customers convert more frequently.

        • Higher account balances improve subscription probability.

        • Campaign fatigue appears after repeated contacts.

        • Predictive models can accurately identify likely subscribers.
        """
    )

# =====================================================
# CUSTOMER SEGMENTATION
# =====================================================

elif page == "Customer Segmentation":

    st.title(
        "👥 Customer Segmentation"
    )

    st.subheader(
        "Age Segment Conversion"
    )

    age_df = pd.read_csv(
        AGE_SEGMENT_PATH
    )

    st.dataframe(
        age_df,
        use_container_width=True
    )

    st.subheader(
        "Balance Segment Conversion"
    )

    balance_df = pd.read_csv(
        BALANCE_SEGMENT_PATH
    )

    st.dataframe(
        balance_df,
        use_container_width=True
    )

# =====================================================
# CAMPAIGN PERFORMANCE
# =====================================================

elif page == "Campaign Performance":

    st.title(
        "📞 Campaign Performance"
    )

    st.subheader(
        "Campaign Intensity Analysis"
    )

    campaign_df = pd.read_csv(
        CAMPAIGN_PATH
    )

    st.dataframe(
        campaign_df,
        use_container_width=True
    )

    st.subheader(
        "Previous Contact Analysis"
    )

    previous_df = pd.read_csv(
        PREVIOUS_CONTACT_PATH
    )

    st.dataframe(
        previous_df,
        use_container_width=True
    )

# =====================================================
# PREDICTIVE MODELING
# =====================================================

elif page == "Predictive Modeling":

    st.title(
        "🤖 Predictive Modeling"
    )

    models = pd.read_csv(
        MODEL_PATH
    )

    st.dataframe(
        models,
        use_container_width=True
    )

    best_model = (
        models
        .sort_values(
            "roc_auc",
            ascending=False
        )
        .iloc[0]
    )

    st.success(
        f"""
Best Model: {best_model['model_name']}

ROC-AUC: {best_model['roc_auc']}
"""
    )
