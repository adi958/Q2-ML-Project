import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, matthews_corrcoef, roc_auc_score, classification_report, confusion_matrix, roc_curve,average_precision_score, precision_score, recall_score, f1_score)

def load_data(file_path):
    df = pd.read_csv(file_path)
    
    objective_cols = ['age', 'gender', 'restingBP', 'serumcholestrol', 
                      'fastingbloodsugar', 'maxheartrate', 'noofmajorvessels']
    subjective_cols = ['chestpain', 'restingrelectro', 'exerciseangia', 'oldpeak', 'slope']
    target_col = 'target'
    
    X_obj = df[objective_cols].values
    X_subj = df[subjective_cols].values
    y = df[target_col].values
    
    scaler_obj = StandardScaler()
    X_obj = scaler_obj.fit_transform(X_obj)
    
    scaler_subj = StandardScaler()
    X_subj = scaler_subj.fit_transform(X_subj)
    
    return X_obj, X_subj, y, objective_cols, subjective_cols

class RSWLogisticRegression(nn.Module):
    def __init__(self, n_obj, n_subj):
        super(RSWLogisticRegression, self).__init__()
        self.linear_obj = nn.Linear(n_obj, 1, bias=False)
        self.linear_subj = nn.Linear(n_subj, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x_obj, x_subj):
        out = self.linear_obj(x_obj) + self.linear_subj(x_subj) + self.bias
        return torch.sigmoid(out)

def rsw_loss(outputs, targets, model, alpha_penalty=0.05):
    bce_loss = nn.BCELoss()(outputs, targets)
    subj_penalty = torch.norm(model.linear_subj.weight, p=2) 
    return bce_loss + (alpha_penalty * subj_penalty)

file_path = 'Cardiovascular_Dataset.csv'
X_obj, X_subj, y, obj_names, subj_names = load_data(file_path)

X_obj_tr, X_obj_te, X_subj_tr, X_subj_te, y_tr, y_te = train_test_split(X_obj, X_subj, y, test_size=0.2, random_state=42)

X_obj_tr = torch.FloatTensor(X_obj_tr)
X_subj_tr = torch.FloatTensor(X_subj_tr)
y_tr = torch.FloatTensor(y_tr).view(-1, 1)

X_obj_te = torch.FloatTensor(X_obj_te)
X_subj_te = torch.FloatTensor(X_subj_te)

model = RSWLogisticRegression(n_obj=len(obj_names), n_subj=len(subj_names))
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("--- Training RSW-LR Model ---")
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_obj_tr, X_subj_tr)
    loss = rsw_loss(outputs, y_tr, model)
    loss.backward()
    
    # PRIORITY: Force objective features to have 1.5x influence on learning
    model.linear_obj.weight.grad *= 1.5 
    
    optimizer.step()
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/100 | Loss: {loss.item():.4f}")

# 5. EVALUATE
model.eval()
with torch.no_grad():
    probs = model(X_obj_te, X_subj_te).numpy()
    preds = (probs > 0.5).astype(int)


print("\n--- FINAL RESULTS!! ---")
print(f"Accuracy: {accuracy_score(y_te, preds):.4f}")
print(f"AUC-ROC:  {roc_auc_score(y_te, probs):.4f}")

cm = confusion_matrix(y_te, preds)
print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_te, preds))

roc_area = roc_auc_score(y_te, probs)
prc_area = average_precision_score(y_te, probs)   # PRC Area
mcc = matthews_corrcoef(y_te, preds)
recall = recall_score(y_te, preds)                 # TPR
precision = precision_score(y_te, preds)
f1 = f1_score(y_te, preds)

# PRINT RESULTS
print("\n--- Evaluation Metrics ---")
print(f"ROC Area (AUC):                {roc_area:.3f}")
print(f"Precision–Recall Curve Area:   {prc_area:.3f}")
print(f"Matthews Correlation Coefficient (MCC): {mcc:.3f}")
print(f"Recall / True Positive Rate:   {recall:.3f}")
print(f"Precision:                    {precision:.3f}")
print(f"F1 Score:                     {f1:.3f}")

fpr, tpr, thresholds = roc_curve(y_te, probs)

plt.figure()
plt.plot(
    fpr,
    tpr,
    label=f"RSW Logistic Regression (AUC = {roc_auc_score(y_te, probs):.3f})"
)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – RSW Subtype-Weighted Logistic Regression")
plt.legend()
plt.show()

print("\nObjective Feature Importance (Higher = More Influence):")
for name, weight in zip(obj_names, model.linear_obj.weight.data[0].numpy()):
    print(f" - {name}: {weight:.4f}")

    
