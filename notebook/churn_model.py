import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# =========================
# Load Dataset
# =========================

df = pd.read_csv("data/customer_churn.csv")

print("Dataset Shape:", df.shape)

# =========================
# Data Cleaning
# =========================

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Fill missing values
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# Drop Customer ID
if "customerID" in df.columns:
    df = df.drop("customerID", axis=1)

# Convert target variable
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# =========================
# Features & Target
# =========================

X = df.drop("Churn", axis=1)
y = df["Churn"]

# =========================
# Identify Columns
# =========================

categorical_cols = X.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_cols = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("\nCategorical Columns:")
print(categorical_cols)

print("\nNumeric Columns:")
print(numeric_cols)

# =========================
# Preprocessing
# =========================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_cols
        ),
        (
            "num",
            "passthrough",
            numeric_cols
        )
    ]
)

# =========================
# Model
# =========================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# =========================
# Train Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =========================
# Train Model
# =========================

pipeline.fit(
    X_train,
    y_train
)

# =========================
# Prediction
# =========================

y_pred = pipeline.predict(X_test)

# =========================
# Evaluation
# =========================

print("\n===== MODEL RESULTS =====")

print(
    "\nAccuracy:",
    accuracy_score(y_test, y_pred)
)

print(
    "\nConfusion Matrix:"
)

print(
    confusion_matrix(y_test, y_pred)
)

print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        y_pred
    )
)

# =========================
# Save Model
# =========================

joblib.dump(
    pipeline,
    "customer_churn_pipeline.pkl"
)

print(
    "\nModel saved successfully as customer_churn_pipeline.pkl"
)