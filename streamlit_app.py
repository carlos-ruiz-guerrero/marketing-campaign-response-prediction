import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Bank Marketing Analytics",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# DATABASE CONNECTION
# =====================================================

from pathlib import Path

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

DATABASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "marketing.db"
)

@st.cache_resource
def get_connection():
    return sqlite3.connect(
        str(DATABASE_PATH),
        check_same_thread=False
    )

conn = get_connection()

st.write(
    f"Database: {DATABASE_PATH}"
)

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

        • Previously contacted customers convert significantly more frequently.

        • Higher account balances improve subscription probability.

        • Campaign fatigue appears after repeated contacts.

        • Predictive models can accurately identify likely subscribers.
        """
    )

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

    fig = px.bar(
        age_df,
        x="age_group",
        y="conversion_rate",
        title="Age Group Conversion Rate"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    balance_df = pd.read_sql(
        """
        SELECT *
        FROM balance_analysis
        """,
        conn
    )

    fig2 = px.bar(
        balance_df,
        x="balance_segment",
        y="conversion_rate",
        title="Balance Segment Conversion Rate"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

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

    fig = px.bar(
        campaign_df,
        x="campaign_group",
        y="conversion_rate",
        title="Campaign Intensity Impact"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    engagement_df = pd.read_sql(
        """
        SELECT *
        FROM engagement_analysis
        """,
        conn
    )

    fig2 = px.bar(
        engagement_df,
        x="contacted_before",
        y="conversion_rate",
        title="Previous Contact Impact"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

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

    st.dataframe(
        models,
        use_container_width=True
    )

    fig = px.bar(
        models,
        x="model_name",
        y="roc_auc",
        title="Model ROC-AUC Comparison"
    )

    st.plotly_chart(
        fig,
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
            Best Model:
            {best_model['model_name']}

            ROC-AUC:
            {best_model['roc_auc']}
            """
    )

