# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_curve, auc)

# ==========================================
# 2. DATA LOADING
# ==========================================
# We load the data directly from the UCI Machine Learning Repository
url = "https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/04_Apply/Students_Alcohol_Consumption/student-mat.csv"
# The dataset uses semicolons ';' as separators, not commas
df = pd.read_csv(url)

print("Data Loaded Successfully!")
print(f"Total Student Records: {df.shape[0]}")
print("-" * 30)

# ==========================================
# 3. PREPROCESSING
# ==========================================

# Select the specific features mentioned in the Research Methodology
# studytime: 1-4 scale
# failures: number of past class failures
# absences: number of school absences
# internet: Internet access at home (yes/no)
# Dalc: Workday alcohol consumption (1-5 scale)
feature_cols = ['studytime', 'failures', 'absences', 'internet', 'Dalc']
target_col = 'G3' # Final Grade

# Create a clean dataframe with only these columns
data = df[feature_cols + [target_col]].copy()

# BINARY CLASSIFICATION TARGET:
# If Grade >= 10, Result is Pass (1). Else Fail (0).
data['result'] = data['G3'].apply(lambda x: 1 if x >= 10 else 0)

# ENCODING:
# 'internet' column is 'yes'/'no'. We convert this to 1/0.
le = LabelEncoder()
data['internet'] = le.fit_transform(data['internet'])

# Define X (Inputs) and y (Output)
X = data[feature_cols]
y = data['result']

# SPLITTING:
# 80% for Training the models, 20% for Testing performance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SCALING:
# SVM strictly requires scaled data (mean=0, variance=1) to work correctly.
# Random Forest doesn't strictly need it, but it doesn't hurt.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data Preprocessing Complete.")
print("-" * 30)

# ==========================================
# 4. MODEL CONSTRUCTION & TRAINING
# ==========================================

# --- MODEL A: RANDOM FOREST CLASSIFIER ---
# n_estimators=100: Creates 100 decision trees
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Generate Predictions
rf_pred = rf_model.predict(X_test)
rf_probs = rf_model.predict_proba(X_test)[:, 1] # Probability for ROC Curve


# --- MODEL B: SUPPORT VECTOR MACHINE (SVM) ---
# kernel='rbf': Radial Basis Function to handle non-linear data
# probability=True: Needed to calculate ROC-AUC later
svm_model = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Generate Predictions
svm_pred = svm_model.predict(X_test_scaled)
svm_probs = svm_model.predict_proba(X_test_scaled)[:, 1] # Probability for ROC Curve

print("Models Trained Successfully.")
print("-" * 30)

# ==========================================
# 5. EVALUATION & RESULTS GENERATION
# ==========================================

# Function to calculate metrics nicely
def calculate_metrics(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    return [model_name, round(acc*100, 2), round(prec*100, 2), round(rec*100, 2), round(f1, 3)]

# Get metrics for both
rf_metrics = calculate_metrics(y_test, rf_pred, "Random Forest")
svm_metrics = calculate_metrics(y_test, svm_pred, "SVM")

# Create a DataFrame for the Results Table (Table 4.1 in paper)
results_table = pd.DataFrame([rf_metrics, svm_metrics], 
                             columns=["Model", "Accuracy %", "Precision %", "Recall %", "F1-Score"])

print("\n=== FINAL RESULTS (Table 4.1) ===")
print(results_table)

# ==========================================
# 6. VISUALIZATION (Confusion Matrix & ROC)
# ==========================================

# Plot Confusion Matrices
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Random Forest Heatmap
sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title('Random Forest Confusion Matrix')
ax[0].set_xlabel('Predicted Label')
ax[0].set_ylabel('True Label')

# SVM Heatmap
sns.heatmap(confusion_matrix(y_test, svm_pred), annot=True, fmt='d', cmap='Reds', ax=ax[1])
ax[1].set_title('SVM Confusion Matrix')
ax[1].set_xlabel('Predicted Label')
ax[1].set_ylabel('True Label')

plt.tight_layout()
plt.show()

# Plot ROC-AUC Curve (Figure 4.1 in paper)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
roc_auc_rf = auc(fpr_rf, tpr_rf)

fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_probs)
roc_auc_svm = auc(fpr_svm, tpr_svm)

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, color='blue', lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
plt.plot(fpr_svm, tpr_svm, color='red', lw=2, label=f'SVM (AUC = {roc_auc_svm:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--') # Random guess line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC-AUC Curve Comparison')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()