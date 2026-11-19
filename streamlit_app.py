import sqlite3
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
# DATABASE CONFIGURATION
# =====================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
)

DATABASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "marketing.db"
)

# =====================================================
# DATABASE CONNECTION
# =====================================================

@st.cache_resource
def get_connection():

    return sqlite3.connect(
        str(DATABASE_PATH),
        check_same_thread=False
    )


conn = get_connection()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title(
    "Navigation"
)

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

    df = pd.read_sql(
        """
        SELECT *
        FROM customers
        """,
        conn
    )

    total_customers = len(df)

    conversion_rate = round(
        df["subscribed_target"].mean() * 100,
        2
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Customers",
        f"{total_customers:,}"
    )

    col2.metric(
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

    age_df = pd.read_sql(
        """
        SELECT *
        FROM age_analysis
        """,
        conn
    )

    st.subheader(
        "Age Group Conversion"
    )

    st.dataframe(
        age_df,
        use_container_width=True
    )

    balance_df = pd.read_sql(
        """
        SELECT *
        FROM balance_analysis
        """,
        conn
    )

    st.subheader(
        "Balance Segment Conversion"
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

    campaign_df = pd.read_sql(
        """
        SELECT *
        FROM campaign_performance
        """,
        conn
    )

    st.subheader(
        "Campaign Intensity Analysis"
    )

    st.dataframe(
        campaign_df,
        use_container_width=True
    )

    engagement_df = pd.read_sql(
        """
        SELECT *
        FROM engagement_analysis
        """,
        conn
    )

    st.subheader(
        "Previous Contact Analysis"
    )

    st.dataframe(
        engagement_df,
        use_container_width=True
    )

# =====================================================
# PREDICTIVE MODELING
# =====================================================

elif page == "Predictive Modeling":

    st.title(
        "🤖 Predictive Modeling"
    )

    models = pd.read_sql(
        """
        SELECT *
        FROM model_metrics
        """,
        conn
    )

    st.subheader(
        "Model Performance"
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
