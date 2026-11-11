# src/data_preparation.py

from ucimlrepo import fetch_ucirepo
import pandas as pd
import os

# =====================================================
# Create project folders
# =====================================================

folders = [
    "data/raw",
    "data/processed",
    "reports/figures",
    "reports/findings",
    "src",
    "notebooks"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

# =====================================================
# Download dataset
# =====================================================

bank_marketing = fetch_ucirepo(id=222)

X = bank_marketing.data.features
y = bank_marketing.data.targets

# =====================================================
# Build master dataframe
# =====================================================

df = X.copy()

# Sometimes y comes as a DataFrame
df["subscribed"] = y.iloc[:, 0]

# =====================================================
# Save raw dataset
# =====================================================

df.to_csv(
    "data/raw/bank_marketing_raw.csv",
    index=False
)

# =====================================================
# Basic validation
# =====================================================

print("Dataset saved successfully.")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nColumns:")
print(df.columns.tolist())
