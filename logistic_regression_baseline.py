import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, matthews_corrcoef, roc_auc_score, classification_report, confusion_matrix, roc_curve,average_precision_score, precision_score, recall_score, f1_score)



df = pd.read_csv('Cardiovascular_Dataset.csv')

X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"Baseline Logistic Regression (AUC = {roc_auc_score(y_test, y_prob):.3f})")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Baseline Logistic Regression")
plt.legend()
plt.show()


print("--- FINAL RESULTS!! ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC-ROC:  {roc_auc_score(y_test, y_prob):.4f}")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

print("\nFull Classification Report:")
print(classification_report(y_test, y_pred))

roc_area = roc_auc_score(y_test, y_prob)
prc_area = average_precision_score(y_test, y_prob)
mcc = matthews_corrcoef(y_test, y_pred)
recall = recall_score(y_test, y_pred)        # True Positive Rate
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nAdditional Metrics:")
print(f"ROC Area (AUC):                {roc_area:.3f}")
print(f"Precision–Recall Curve Area:   {prc_area:.3f}")
print(f"Matthews Correlation Coefficient (MCC): {mcc:.3f}")
print(f"Recall / True Positive Rate:   {recall:.3f}")
print(f"Precision:                     {precision:.3f}")
print(f"F1 Score:                       {f1:.3f}")

print("\nFeature Importance (Coefficients):")
coeffs = pd.DataFrame({'Feature': X.columns, 'Weight': model.coef_[0]})
print(coeffs.sort_values(by='Weight', ascending=False).to_string(index=False))

