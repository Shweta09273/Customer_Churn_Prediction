import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("data/customer_churn.csv")

print("Dataset Shape:", df.shape)

# ==========================
# Data Cleaning
# ==========================

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

# Convert target column
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# Encode categorical columns
for col in df.select_dtypes(include="object").columns:
    if col != "Churn":
        df[col] = df[col].astype("category").cat.codes

print("\nData Types:")
print(df.dtypes)

# ==========================
# Features and Target
# ==========================
X = df.drop("Churn", axis=1)
y = df["Churn"]

# ==========================
# Train Test Split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================
# Logistic Regression
# ==========================
lr = LogisticRegression(max_iter=2000)

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

print("\n===== Logistic Regression =====")

print("Accuracy:",
      accuracy_score(y_test, lr_pred))

print("Recall:",
      recall_score(y_test, lr_pred))

print("ROC AUC:",
      roc_auc_score(y_test, lr_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, lr_pred))

# ==========================
# Random Forest
# ==========================
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)
joblib.dump(rf, "customer_churn_model.pkl")

rf_pred = rf.predict(X_test)

print("\n===== Random Forest =====")

print("Accuracy:",
      accuracy_score(y_test, rf_pred))

print("Recall:",
      recall_score(y_test, rf_pred))

print("ROC AUC:",
      roc_auc_score(y_test, rf_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(6,4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("images/confusion_matrix.png")
plt.close()

print("Confusion Matrix image saved!")
# ==========================
# Feature Importance
# ==========================

importances = rf.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(10,6))

sns.barplot(
    data=feature_importance.head(10),
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Important Features")
plt.tight_layout()

plt.savefig("images/feature_importance.png")
plt.close()

print("Feature Importance image saved!")