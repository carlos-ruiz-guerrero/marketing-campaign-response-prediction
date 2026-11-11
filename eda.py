import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from config import (
    RAW_DATASET,
    FIGURES_DIR,
    FINDINGS_DIR
)

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(RAW_DATASET)

# =====================================================
# DATASET OVERVIEW
# =====================================================

print("=" * 60)
print("BANK MARKETING CAMPAIGN ANALYSIS")
print("=" * 60)

print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")

print("\nSubscription Distribution (%)")

print(
    round(
        df["subscribed"]
        .value_counts(normalize=True) * 100,
        2
    )
)

print("\nDuplicate Records:")
print(df.duplicated().sum())

print("\nMissing Values:")
print(df.isnull().sum().sum())

# =====================================================
# COLUMN CLASSIFICATION
# =====================================================

numerical_cols = df.select_dtypes(
    include="number"
).columns.tolist()

categorical_cols = [
    col for col in df.columns
    if col not in numerical_cols
]

print("\nNumerical Columns:")
print(numerical_cols)

print("\nCategorical Columns:")
print(categorical_cols)

# =====================================================
# TARGET DISTRIBUTION
# =====================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="subscribed"
)

plt.title("Term Deposit Subscription Distribution")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "subscription_distribution.png"
)

plt.show()

# =====================================================
# CONVERSION DRIVER ANALYSIS
# =====================================================

print("\n" + "=" * 60)
print("CONVERSION DRIVER ANALYSIS")
print("=" * 60)

# =====================================================
# JOB CONVERSION RATE
# =====================================================

job_conversion = pd.crosstab(
    df["job"],
    df["subscribed"],
    normalize="index"
) * 100

job_conversion = job_conversion.sort_values(
    by="yes",
    ascending=False
)

print("\nJOB CONVERSION RATE")
print(job_conversion)

plt.figure(figsize=(12, 6))

sns.barplot(
    x=job_conversion.index,
    y=job_conversion["yes"]
)

plt.xticks(rotation=45)

plt.title("Subscription Rate by Job")
plt.ylabel("Conversion Rate (%)")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "job_conversion_rate.png"
)

plt.show()

# =====================================================
# EDUCATION CONVERSION RATE
# =====================================================

education_conversion = pd.crosstab(
    df["education"],
    df["subscribed"],
    normalize="index"
) * 100

education_conversion = education_conversion.sort_values(
    by="yes",
    ascending=False
)

print("\nEDUCATION CONVERSION RATE")
print(education_conversion)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=education_conversion.index,
    y=education_conversion["yes"]
)

plt.xticks(rotation=30)

plt.title("Subscription Rate by Education")
plt.ylabel("Conversion Rate (%)")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "education_conversion_rate.png"
)

plt.show()

# =====================================================
# MARITAL STATUS CONVERSION RATE
# =====================================================

marital_conversion = pd.crosstab(
    df["marital"],
    df["subscribed"],
    normalize="index"
) * 100

print("\nMARITAL STATUS CONVERSION RATE")
print(marital_conversion)

plt.figure(figsize=(8, 5))

marital_conversion["yes"].plot(
    kind="bar",
    color="orange"
)

plt.title("Subscription Rate by Marital Status")
plt.ylabel("Conversion Rate (%)")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "marital_conversion_rate.png"
)

plt.show()

# =====================================================
# CONTACT METHOD PERFORMANCE
# =====================================================

contact_conversion = pd.crosstab(
    df["contact"],
    df["subscribed"],
    normalize="index"
) * 100

print("\nCONTACT METHOD PERFORMANCE")
print(contact_conversion)

plt.figure(figsize=(8, 5))

contact_conversion["yes"].plot(
    kind="bar",
    color="purple"
)

plt.title("Subscription Rate by Contact Method")
plt.ylabel("Conversion Rate (%)")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "contact_conversion_rate.png"
)

plt.show()

# =====================================================
# MONTH PERFORMANCE
# =====================================================

month_conversion = pd.crosstab(
    df["month"],
    df["subscribed"],
    normalize="index"
) * 100

month_conversion = month_conversion.sort_values(
    by="yes",
    ascending=False
)

print("\nMONTH PERFORMANCE")
print(month_conversion)

plt.figure(figsize=(12, 6))

sns.barplot(
    x=month_conversion.index,
    y=month_conversion["yes"]
)

plt.title("Subscription Rate by Month")
plt.ylabel("Conversion Rate (%)")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "month_conversion_rate.png"
)

plt.show()

# =====================================================
# PREVIOUS CAMPAIGN OUTCOME
# =====================================================

poutcome_conversion = pd.crosstab(
    df["poutcome"],
    df["subscribed"],
    normalize="index"
) * 100

poutcome_conversion = poutcome_conversion.sort_values(
    by="yes",
    ascending=False
)

print("\nPREVIOUS CAMPAIGN OUTCOME")
print(poutcome_conversion)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=poutcome_conversion.index,
    y=poutcome_conversion["yes"]
)

plt.title(
    "Subscription Rate by Previous Campaign Outcome"
)

plt.ylabel("Conversion Rate (%)")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "poutcome_conversion_rate.png"
)

plt.show()

# =====================================================
# EXPORT FINDINGS
# =====================================================

job_conversion.to_csv(
    FINDINGS_DIR / "job_conversion.csv"
)

education_conversion.to_csv(
    FINDINGS_DIR / "education_conversion.csv"
)

marital_conversion.to_csv(
    FINDINGS_DIR / "marital_conversion.csv"
)

contact_conversion.to_csv(
    FINDINGS_DIR / "contact_conversion.csv"
)

month_conversion.to_csv(
    FINDINGS_DIR / "month_conversion.csv"
)

poutcome_conversion.to_csv(
    FINDINGS_DIR / "poutcome_conversion.csv"
)

# =====================================================
# EXECUTIVE SUMMARY
# =====================================================

overall_rate = (
    (df["subscribed"] == "yes").mean()
) * 100

print("\n" + "=" * 60)
print("EXECUTIVE SUMMARY")
print("=" * 60)

print(
    f"Overall subscription rate: "
    f"{overall_rate:.2f}%"
)

print(
    "\nFocus future campaigns on "
    "high-converting customer segments."
)