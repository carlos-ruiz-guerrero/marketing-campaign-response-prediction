from ucimlrepo import fetch_ucirepo
import pandas as pd

from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    FIGURES_DIR,
    FINDINGS_DIR,
    RAW_DATASET
)

# =====================================================
# CREATE PROJECT FOLDERS
# =====================================================

RAW_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PROCESSED_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FINDINGS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =====================================================
# DOWNLOAD DATASET
# =====================================================

bank_marketing = fetch_ucirepo(id=222)

X = bank_marketing.data.features
y = bank_marketing.data.targets

# =====================================================
# BUILD DATAFRAME
# =====================================================

df = X.copy()

df["subscribed"] = y.iloc[:, 0]

# =====================================================
# SAVE RAW DATASET
# =====================================================

df.to_csv(
    RAW_DATASET,
    index=False
)

# =====================================================
# VALIDATION
# =====================================================

print("Dataset saved successfully.")

print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")

print("\nColumns:")
print(df.columns.tolist())
