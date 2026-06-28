import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("dataset/german_credit_data.csv")

print("========== First 5 Rows ==========")
print(df.head())

print("\n========== Dataset Information ==========")
print(df.info())

print("\n========== Missing Values ==========")
print(df.isnull().sum())

print("\n========== Dataset Shape ==========")
print(df.shape)

# ==========================================
# Features and Target
# ==========================================

X = df.drop("kredit", axis=1)
y = df["kredit"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# ==========================================
# Split Dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================
# Build Random Forest Model
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train Model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# ==========================================
# Accuracy
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("Model Accuracy")
print("==============================")
print(f"{accuracy * 100:.2f}%")

# ==========================================
# Classification Report
# ==========================================

print("\n==============================")
print("Classification Report")
print("==============================")
print(classification_report(y_test, y_pred))

# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\n==============================")
print("Confusion Matrix")
print("==============================")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# ==========================================
# Feature Importance
# ==========================================

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

importance = importance.sort_values(ascending=False)

plt.figure(figsize=(10,6))
importance.head(10).plot(kind="bar", color="green")

plt.title("Top 10 Important Features")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ==========================================
# ROC-AUC Score
# ==========================================

y_prob = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_prob)

print("\n==============================")
print("ROC-AUC Score")
print("==============================")
print(f"{roc_auc:.4f}")

# ==========================================
# ROC Curve
# ==========================================

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(8,6))

plt.plot(
    fpr,
    tpr,
    label=f"Random Forest (AUC = {roc_auc:.4f})",
    linewidth=2
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="red",
    label="Random Guess"
)

plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)

plt.show()