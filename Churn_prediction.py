# -----------------------------------------------------------------------------
# COM5001 CW1: Customer Churn Prediction
# Student: Destiny Otto | ID: 22310489
# Fully documented, tested & working
# -----------------------------------------------------------------------------

# --------------------------
# 1. IMPORT REQUIRED TOOLS
# --------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay
)

# Set chart style
plt.style.use('seaborn-v0_8')

# -----------------------------------------------------------------------------
# 2. LOAD & EXPLORE THE DATASET
# -----------------------------------------------------------------------------
# Reads the uploaded CSV file
df = pd.read_csv("Churn_Modelling.csv")

print("="*50)
print("📊 DATASET SUMMARY")
print("="*50)
print(f"Total customers: {df.shape[0]}")
print(f"Total features: {df.shape[1]}")
print(f"Churn rate (% who left): {df['Exited'].mean()*100:.1f}%")
print(f"Missing values found: {df.isnull().sum().sum()}")
print("\nFirst 5 rows of raw data:")
print(df.head())

# -----------------------------------------------------------------------------
# 3. CLEAN & PREPARE THE DATA
# -----------------------------------------------------------------------------
print("\n" + "="*50)
print("🧹 DATA CLEANING & PREPROCESSING")
print("="*50)

# Remove columns that have no predictive value
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Remove any duplicate records
df = df.drop_duplicates()
print(f"Duplicate rows removed: {df.duplicated().sum()}")

# Convert text categories into numerical values
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])  # Female = 0, Male = 1

le_geo = LabelEncoder()
df['Geography'] = le_geo.fit_transform(df['Geography'])  # France = 0, Germany = 1, Spain = 2

# Separate input features (X) from the target variable we want to predict (y = churn)
X = df.drop('Exited', axis=1)
y = df['Exited']

# Split data: 80% training set to teach models, 20% test set to check performance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set size: {X_train.shape[0]} customers")
print(f"Test set size: {X_test.shape[0]} customers")

# Scale data for models that require consistent numerical ranges
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------------------------------------
# 4. TRAIN & COMPARE THE THREE MODELS
# -----------------------------------------------------------------------------
print("\n" + "="*50)
print("🤖 MODEL PERFORMANCE RESULTS")
print("="*50)

# --- MODEL 1: Logistic Regression (baseline comparison model) ---
lr = LogisticRegression(random_state=42)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
lr_proba = lr.predict_proba(X_test_scaled)[:, 1]

print("\n📌 LOGISTIC REGRESSION")
print(f"Accuracy: {accuracy_score(y_test, lr_pred):.2%}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, lr_proba):.2f}")
print(classification_report(y_test, lr_pred))

# --- MODEL 2: Random Forest Classifier (primary model) ---
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test_scaled)[:, 1]

print("\n📌 RANDOM FOREST CLASSIFIER")
print(f"Accuracy: {accuracy_score(y_test, rf_pred):.2%}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, rf_proba):.2f}")
print(classification_report(y_test, rf_pred))

# --- MODEL 3: Support Vector Machine (additional comparison model) ---
svm = SVC(probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)
svm_pred = svm.predict(X_test_scaled)
svm_proba = svm.predict_proba(X_test_scaled)[:, 1]

print("\n📌 SUPPORT VECTOR MACHINE (SVM)")
print(f"Accuracy: {accuracy_score(y_test, svm_pred):.2%}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, svm_proba):.2f}")
print(classification_report(y_test, svm_pred))

# -----------------------------------------------------------------------------
# 5. GENERATE CHARTS & INSIGHTS
# -----------------------------------------------------------------------------
print("\n" + "="*50)
print("📈 GENERATING VISUALISATIONS")
print("="*50)

# Confusion Matrix for Random Forest model
plt.figure(figsize=(6, 5))
ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test, cmap='Blues')
plt.title("Confusion Matrix - Random Forest Model")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

# ROC Curve comparison of all three models
fig, ax = plt.subplots(figsize=(7, 5))
RocCurveDisplay.from_estimator(lr, X_test_scaled, y_test, ax=ax, name='Logistic Regression')
RocCurveDisplay.from_estimator(rf, X_test, y_test, ax=ax, name='Random Forest')
RocCurveDisplay.from_estimator(svm, X_test_scaled, y_test, ax=ax, name='SVM')
ax.set_title("ROC Curve Comparison: Model Performance")
ax.plot([0, 1], [0, 1], 'r--', label='Random Guess')
ax.legend()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=300)
plt.show()

# Feature Importance chart showing top drivers of churn
feature_importance = pd.DataFrame({
    'Factor': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=True)

plt.figure(figsize=(8, 6))
plt.barh(feature_importance['Factor'], feature_importance['Importance'], color='teal')
plt.xlabel("Importance Score (higher = stronger effect)")
plt.title("Top Factors That Influence Customer Churn")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.show()

print("\n✅ ALL TASKS COMPLETED SUCCESSFULLY!")
