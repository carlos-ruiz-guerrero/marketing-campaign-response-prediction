import pandas as pd


from config import (
    RAW_DATASET,
    PROCESSED_DATA_DIR
)

###from config import (
    ### RAW_DATASET,
   ### PROCESSED_DATA_DIR
### )

# =====================================================
# LOAD DATA
# =====================================================

print("RAW_DATASET =", RAW_DATASET)
print("Exists =", RAW_DATASET.exists())

df = pd.read_csv(RAW_DATASET)

print("=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# =====================================================
# AGE GROUPS
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

# =====================================================
# BALANCE SEGMENTS
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

# =====================================================
# PREVIOUS CONTACT FLAG
# =====================================================

df["contacted_before"] = (
    df["previous"] > 0
).astype(int)

# =====================================================
# CUSTOMER TENURE STYLE FEATURE
# =====================================================

df["recently_contacted"] = (
    (df["pdays"] >= 0)
    &
    (df["pdays"] <= 30)
).astype(int)

# =====================================================
# HIGH BALANCE FLAG
# =====================================================

balance_threshold = (
    df["balance"]
    .quantile(0.75)
)

df["high_balance_customer"] = (
    df["balance"] >= balance_threshold
).astype(int)

# =====================================================
# CAMPAIGN FATIGUE FLAG
# =====================================================

df["campaign_fatigue"] = (
    df["campaign"] > 5
).astype(int)

# =====================================================
# TARGET ENCODING
# =====================================================

df["subscribed_target"] = (
    df["subscribed"]
    .map({
        "no": 0,
        "yes": 1
    })
)

# =====================================================
# PREVIEW
# =====================================================

print("\nNew Features Created:\n")

new_features = [
    "age_group",
    "balance_segment",
    "campaign_group",
    "contacted_before",
    "recently_contacted",
    "high_balance_customer",
    "campaign_fatigue",
    "subscribed_target"
]

print(new_features)

# =====================================================
# SAVE CLEAN DATASET
# =====================================================

output_file = (
    PROCESSED_DATA_DIR
    / "bank_marketing_clean.csv"
)

df.to_csv(
    output_file,
    index=False
)

print(
    f"\nDataset saved:\n{output_file}"
)

print(
    f"\nRows: {df.shape[0]:,}"
)

## print(df.columns.tolist())

print(
    f"Columns: {df.shape[1]}"
)

print("\nEngineered Features Preview")

print(
    df[
        [
            "age_group",
            "balance_segment",
            "campaign_group",
            "contacted_before",
            "recently_contacted",
            "high_balance_customer",
            "campaign_fatigue",
            "subscribed_target"
        ]
    ].head()
)

