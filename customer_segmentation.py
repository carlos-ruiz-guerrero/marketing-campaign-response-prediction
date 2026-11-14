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

print("=" * 60)
print("CUSTOMER SEGMENTATION ANALYSIS")
print("=" * 60)

# =====================================================
# AGE GROUP SEGMENTATION
# =====================================================

df["age_group"] = pd.cut(
    df["age"],
    bins=[18, 30, 40, 50, 60, 100],
    labels=[
        "18-30",
        "31-40",
        "41-50",
        "51-60",
        "60+"
    ]
)

age_segment = pd.crosstab(
    df["age_group"],
    df["subscribed"],
    normalize="index"
) * 100

age_segment = age_segment.sort_values(
    by="yes",
    ascending=False
)

print("\nAGE SEGMENT CONVERSION")
print(age_segment)

plt.figure(figsize=(8, 5))

sns.barplot(
    x=age_segment.index,
    y=age_segment["yes"]
)

plt.title("Subscription Rate by Age Group")
plt.ylabel("Conversion Rate (%)")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "age_group_conversion.png"
)

plt.show()

# =====================================================
# BALANCE SEGMENTATION
# =====================================================

df["balance_segment"] = pd.qcut(
    df["balance"],
    q=4,
    labels=[
        "Low",
        "Medium",
        "High",
        "Very High"
    ]
)

balance_segment = pd.crosstab(
    df["balance_segment"],
    df["subscribed"],
    normalize="index"
) * 100

balance_segment = balance_segment.sort_values(
    by="yes",
    ascending=False
)

print("\nBALANCE SEGMENT CONVERSION")
print(balance_segment)

plt.figure(figsize=(8, 5))

sns.barplot(
    x=balance_segment.index,
    y=balance_segment["yes"]
)

plt.title("Subscription Rate by Balance Segment")
plt.ylabel("Conversion Rate (%)")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "balance_segment_conversion.png"
)

plt.show()

# =====================================================
# CAMPAIGN INTENSITY
# =====================================================

df["campaign_group"] = pd.cut(
    df["campaign"],
    bins=[0, 1, 3, 5, 100],
    labels=[
        "1 Contact",
        "2-3 Contacts",
        "4-5 Contacts",
        "6+ Contacts"
    ]
)

campaign_segment = pd.crosstab(
    df["campaign_group"],
    df["subscribed"],
    normalize="index"
) * 100

campaign_segment = campaign_segment.sort_values(
    by="yes",
    ascending=False
)

print("\nCAMPAIGN INTENSITY")
print(campaign_segment)

plt.figure(figsize=(8, 5))

sns.barplot(
    x=campaign_segment.index,
    y=campaign_segment["yes"]
)

plt.title("Subscription Rate by Campaign Intensity")
plt.ylabel("Conversion Rate (%)")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "campaign_intensity_conversion.png"
)

plt.show()

# =====================================================
# PREVIOUS CONTACT HISTORY
# =====================================================

df["contacted_before"] = (
    df["previous"] > 0
).astype(int)

contact_history = pd.crosstab(
    df["contacted_before"],
    df["subscribed"],
    normalize="index"
) * 100

print("\nPREVIOUS CONTACT HISTORY")
print(contact_history)

plt.figure(figsize=(6, 5))

sns.barplot(
    x=["No Previous Contact", "Previous Contact"],
    y=contact_history["yes"]
)

plt.title("Previous Contact vs Subscription")
plt.ylabel("Conversion Rate (%)")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "previous_contact_conversion.png"
)

plt.show()

# =====================================================
# TOP CUSTOMER PROFILES
# =====================================================

profile = (
    df.groupby(
        [
            "age_group",
            "job"
        ],
        observed=True
    )["subscribed"]
    .apply(
        lambda x:
        (x == "yes").mean() * 100
    )
    .reset_index()
)

profile.columns = [
    "Age_Group",
    "Job",
    "Conversion_Rate"
]

profile = profile.sort_values(
    "Conversion_Rate",
    ascending=False
)

print("\nTOP CUSTOMER PROFILES")
print(profile.head(20))

# =====================================================
# SAVE FINDINGS
# =====================================================

age_segment.to_csv(
    FINDINGS_DIR / "age_segment_conversion.csv"
)

balance_segment.to_csv(
    FINDINGS_DIR / "balance_segment_conversion.csv"
)

campaign_segment.to_csv(
    FINDINGS_DIR / "campaign_intensity_conversion.csv"
)

contact_history.to_csv(
    FINDINGS_DIR / "previous_contact_conversion.csv"
)

profile.to_csv(
    FINDINGS_DIR / "top_customer_profiles.csv",
    index=False
)

# =====================================================
# EXECUTIVE SUMMARY
# =====================================================

print("\n" + "=" * 60)
print("CUSTOMER SEGMENTATION SUMMARY")
print("=" * 60)

best_age = age_segment["yes"].idxmax()

print(
    f"\nHighest converting age group: {best_age}"
)

best_balance = balance_segment["yes"].idxmax()

print(
    f"Highest converting balance segment: {best_balance}"
)

print(
    "\nUse these findings to prioritize "
    "future marketing campaigns."
)