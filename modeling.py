## Modeling workflow

# 1. Load bank_marketing_clean.csv
# 2. Select target
# 3. One-hot encoder - categoricals
# 4. Train/Test Split
# 5. Logistic Regression
# 6. Random Forest
# 7. Evaluation Metrics
# 8. Feature Importance

## Models

# LogisticRegression
# RandomForestClassifier

## Metrics

# accuracy
# precision
# recall
# f1-score
# roc_auc

## Output

# Confusion Matrix
# Feature Importance
# ROC Curve

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from imblearn.over_sampling import SMOTE

from config import (
    PROCESSED_DATA_DIR,
    FIGURES_DIR,
    FINDINGS_DIR
)

# =====================================================
# LOAD DATA
# =====================================================

DATASET = (
    PROCESSED_DATA_DIR /
    "bank_marketing_clean.csv"
)

df = pd.read_csv(DATASET)

print("=" * 60)
print("PREDICTIVE MODELING")
print("=" * 60)

# =====================================================
# FEATURES / TARGET
# =====================================================

X = df.drop(
    columns=[
        "subscribed",
        "subscribed_target"
    ]
)

y = df["subscribed_target"]

# =====================================================
# ONE-HOT ENCODING
# =====================================================

X = pd.get_dummies(
    X,
    drop_first=True
)

# =====================================================
# TRAIN / TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)

print(
    f"Training rows: {len(X_train):,}"
)

print(
    f"Testing rows: {len(X_test):,}"
)

# =====================================================
# EVALUATION FUNCTION
# =====================================================

def evaluate_model(
    model_name,
    y_true,
    predictions,
    probabilities
):

    return {
        "Model": model_name,
        "Accuracy": round(
            accuracy_score(
                y_true,
                predictions
            ),
            4
        ),
        "Precision": round(
            precision_score(
                y_true,
                predictions
            ),
            4
        ),
        "Recall": round(
            recall_score(
                y_true,
                predictions
            ),
            4
        ),
        "F1_Score": round(
            f1_score(
                y_true,
                predictions
            ),
            4
        ),
        "ROC_AUC": round(
            roc_auc_score(
                y_true,
                probabilities
            ),
            4
        )
    }

results = []

# =====================================================
# LOGISTIC REGRESSION + SCALING
# =====================================================

log_model = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),
        (
            "logistic",
            LogisticRegression(
                max_iter=5000,
                random_state=42
            )
        )
    ]
)

log_model.fit(
    X_train,
    y_train
)

log_pred = log_model.predict(
    X_test
)

log_prob = log_model.predict_proba(
    X_test
)[:, 1]

results.append(
    evaluate_model(
        "Logistic Regression",
        y_test,
        log_pred,
        log_prob
    )
)

# =====================================================
# RANDOM FOREST BASELINE
# =====================================================

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

rf_pred = rf_model.predict(
    X_test
)

rf_prob = rf_model.predict_proba(
    X_test
)[:, 1]

results.append(
    evaluate_model(
        "Random Forest",
        y_test,
        rf_pred,
        rf_prob
    )
)

# =====================================================
# RANDOM FOREST WITHOUT DURATION
# =====================================================

X_no_duration = X.drop(
    columns=["duration"]
)

X_train_nd, X_test_nd, y_train_nd, y_test_nd = (
    train_test_split(
        X_no_duration,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)

rf_nd = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_nd.fit(
    X_train_nd,
    y_train_nd
)

rf_pred_nd = rf_nd.predict(
    X_test_nd
)

rf_prob_nd = rf_nd.predict_proba(
    X_test_nd
)[:, 1]

results.append(
    evaluate_model(
        "Random Forest (No Duration)",
        y_test_nd,
        rf_pred_nd,
        rf_prob_nd
    )
)

# =====================================================
# SMOTE
# =====================================================

smote = SMOTE(
    random_state=42
)

X_train_smote, y_train_smote = (
    smote.fit_resample(
        X_train,
        y_train
    )
)

print("\nClass Distribution After SMOTE")
print(
    pd.Series(
        y_train_smote
    ).value_counts()
)

# =====================================================
# RANDOM FOREST + TUNING
# =====================================================

rf_base = RandomForestClassifier(
    random_state=42
)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    cv=3,
    scoring="roc_auc",
    n_jobs=-1
)

grid_search.fit(
    X_train_smote,
    y_train_smote
)

best_rf = (
    grid_search.best_estimator_
)

print("\nBest Parameters")
print(
    grid_search.best_params_
)

best_pred = best_rf.predict(
    X_test
)

best_prob = best_rf.predict_proba(
    X_test
)[:, 1]

results.append(
    evaluate_model(
        "RF + SMOTE + Tuning",
        y_test,
        best_pred,
        best_prob
    )
)

# =====================================================
# MODEL COMPARISON
# =====================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="ROC_AUC",
    ascending=False
)

print("\nMODEL PERFORMANCE")
print(results_df)

results_df.to_csv(
    FINDINGS_DIR /
    "model_performance.csv",
    index=False
)

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

importance = pd.DataFrame(
    {
        "Feature": X.columns,
        "Importance":
        best_rf.feature_importances_
    }
)

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

importance.to_csv(
    FINDINGS_DIR /
    "feature_importance.csv",
    index=False
)

print("\nTOP FEATURES")
print(
    importance.head(15)
)

# =====================================================
# FEATURE IMPORTANCE CHART
# =====================================================

importance.head(15).plot(
    kind="barh",
    x="Feature",
    y="Importance",
    figsize=(10, 6)
)

plt.title(
    "Top 15 Feature Importances"
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR /
    "feature_importance.png"
)

plt.show()

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    y_test,
    best_pred
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title(
    "Best Model Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR /
    "confusion_matrix.png"
)

plt.show()

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

print(
    "\nBEST MODEL CLASSIFICATION REPORT\n"
)

print(
    classification_report(
        y_test,
        best_pred
    )
)