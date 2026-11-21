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
# PATHS (se localizan solos, sin depender de la estructura)
# =====================================================

APP_DIR = Path(__file__).resolve().parent
SEARCH_ROOTS = [APP_DIR, APP_DIR.parent]


def find_file(name: str) -> Path:
    for root in SEARCH_ROOTS:
        for match in root.rglob(name):
            return match
    st.error(f"No se encuentra el archivo `{name}` en el repositorio.")
    st.stop()


DATASET_PATH = find_file("bank_marketing_clean.csv")
FINDINGS_DIR = find_file("model_performance.csv").parent

# =====================================================
# DATA LOADING
# =====================================================

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_findings(folder: str) -> dict:
    """Lee todos los CSV de la carpeta findings. Clave = nombre sin extensión."""
    return {p.stem: pd.read_csv(p) for p in sorted(Path(folder).glob("*.csv"))}


df = load_csv(str(DATASET_PATH))
FINDINGS = load_findings(str(FINDINGS_DIR))


def show_finding(key: str, title: str):
    st.subheader(title)
    if key in FINDINGS:
        st.dataframe(FINDINGS[key], width="stretch")
    else:
        st.warning(f"No se encontró `{key}.csv` en la carpeta findings.")


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
        "Predictive Modeling",
        "Statistical Analysis",
        "All Findings",
    ]
)

# =====================================================
# EXECUTIVE SUMMARY
# =====================================================

if page == "Executive Summary":

    st.title("📊 Bank Marketing Campaign Analysis")

    total_customers = len(df)
    conversion_rate = round(df["subscribed_target"].mean() * 100, 2)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Customers", f"{total_customers:,}")
    with col2:
        st.metric("Conversion Rate", f"{conversion_rate}%")

    st.markdown("---")
    st.subheader("Executive Insights")

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

    st.title("👥 Customer Segmentation")

    show_finding("age_segment_conversion", "Age Segment Conversion")
    show_finding("balance_segment_conversion", "Balance Segment Conversion")
    show_finding("job_conversion", "Job Conversion")
    show_finding("education_conversion", "Education Conversion")
    show_finding("marital_conversion", "Marital Status Conversion")
    show_finding("top_customer_profiles", "Top Customer Profiles")

# =====================================================
# CAMPAIGN PERFORMANCE
# =====================================================

elif page == "Campaign Performance":

    st.title("📞 Campaign Performance")

    show_finding("campaign_intensity_conversion", "Campaign Intensity Analysis")
    show_finding("previous_contact_conversion", "Previous Contact Analysis")
    show_finding("contact_conversion", "Contact Channel Conversion")
    show_finding("month_conversion", "Monthly Conversion")
    show_finding("poutcome_conversion", "Previous Outcome Conversion")

# =====================================================
# PREDICTIVE MODELING
# =====================================================

elif page == "Predictive Modeling":

    st.title("🤖 Predictive Modeling")

    models = FINDINGS.get("model_performance")

    if models is None:
        st.warning("No se encontró `model_performance.csv`.")
    else:
        st.dataframe(models, width="stretch")

        if {"model_name", "roc_auc"}.issubset(models.columns):
            best_model = models.sort_values("roc_auc", ascending=False).iloc[0]
            st.success(
                f"Best Model: {best_model['model_name']}\n\n"
                f"ROC-AUC: {best_model['roc_auc']}"
            )
        else:
            st.info(f"Columnas disponibles: {list(models.columns)}")

    show_finding("feature_importance", "Feature Importance")

# =====================================================
# STATISTICAL ANALYSIS
# =====================================================

elif page == "Statistical Analysis":

    st.title("📐 Statistical Analysis")

    show_finding("chi_square_results", "Chi-Square Results")

# =====================================================
# ALL FINDINGS (explorador genérico)
# =====================================================

elif page == "All Findings":

    st.title("🗂️ All Findings")

    if not FINDINGS:
        st.warning("La carpeta findings no contiene archivos CSV.")
    else:
        selected = st.selectbox("Selecciona un archivo", list(FINDINGS.keys()))
        selected_df = FINDINGS[selected]

        st.caption(f"{len(selected_df)} filas · {len(selected_df.columns)} columnas")
        st.dataframe(selected_df, width="stretch")

        st.download_button(
            "Descargar CSV",
            selected_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected}.csv",
            mime="text/csv",
        )
