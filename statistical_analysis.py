import pandas as pd
import numpy as np

from scipy.stats import chi2_contingency

from config import (
    RAW_DATASET,
    FINDINGS_DIR
)

# =====================================================
# CRAMER'S V
# =====================================================

def cramers_v(confusion_matrix):

    chi2 = chi2_contingency(
        confusion_matrix
    )[0]

    n = confusion_matrix.sum().sum()

    phi2 = chi2 / n

    r, k = confusion_matrix.shape

    return np.sqrt(
        phi2 / min(k - 1, r - 1)
    )

# =====================================================
# EFFECT SIZE INTERPRETATION
# =====================================================

def interpret_effect_size(v):

    if v < 0.10:
        return "Negligible"

    elif v < 0.20:
        return "Weak"

    elif v < 0.40:
        return "Moderate"

    elif v < 0.60:
        return "Strong"

    else:
        return "Very Strong"

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(RAW_DATASET)

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
# VARIABLES TO TEST
# =====================================================

variables = [
    "age_group",
    "job",
    "education",
    "marital",
    "contact",
    "poutcome"
]

results = []

print("=" * 60)
print("CHI-SQUARE AND CRAMER'S V ANALYSIS")
print("=" * 60)

# =====================================================
# CHI-SQUARE TESTS
# =====================================================

for variable in variables:

    contingency_table = pd.crosstab(
        df[variable],
        df["subscribed"]
    )

    chi2, p_value, dof, expected = (
        chi2_contingency(contingency_table)
    )

    cramers_v_value = cramers_v(
        contingency_table
    )

    significant = p_value < 0.05

    results.append({
        "Variable": variable,
        "Chi2": round(chi2, 2),
        "P_Value": p_value,
        "Degrees_of_Freedom": dof,
        "Cramers_V": round(
            cramers_v_value,
            3
        ),
        "Effect_Size": interpret_effect_size(
            cramers_v_value
        ),
        "Significant": significant
    })

    print(f"\nVariable: {variable}")

    print(
        f"Chi-Square Statistic: "
        f"{chi2:.2f}"
    )

    print(
        f"P-Value: "
        f"{p_value:.10f}"
    )

    print(
        f"Cramer's V: "
        f"{cramers_v_value:.3f}"
    )

    print(
        f"Effect Size: "
        f"{interpret_effect_size(cramers_v_value)}"
    )

    if significant:
        print("Result: SIGNIFICANT")
    else:
        print("Result: NOT SIGNIFICANT")

# =====================================================
# RESULTS TABLE
# =====================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Cramers_V",
    ascending=False
)

print("\n" + "=" * 60)
print("SUMMARY RESULTS")
print("=" * 60)

print(results_df)

# =====================================================
# EXPORT RESULTS
# =====================================================

results_df.to_csv(
    FINDINGS_DIR / "chi_square_results.csv",
    index=False
)

print(
    "\nResults saved to "
    "chi_square_results.csv"
)

# =====================================================
# TOP PREDICTORS
# =====================================================

print("\n" + "=" * 60)
print("TOP PREDICTORS")
print("=" * 60)

for _, row in results_df.iterrows():

    print(
        f"{row['Variable']} | "
        f"Cramer's V = {row['Cramers_V']} | "
        f"{row['Effect_Size']}"
    )