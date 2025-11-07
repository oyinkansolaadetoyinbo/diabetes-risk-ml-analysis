"""
File: diabetes_prediction.py
Author: Adetoyinbo Oyinkansola
Organisation: Write Right Concept
Description: End-to-end ML pipeline for predicting Type 2 Diabetes risk.
"""

# ====== 1. IMPORTS ======
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
import eli5
from eli5.sklearn import PermutationImportance

# ====== 2. LOAD DATA ======
df = pd.read_csv("diabetes.csv")
print("Initial shape:", df.shape)

# ====== 3. PREPROCESS ======
for col in df.select_dtypes(include="object"):
  df[col] = LabelEncoder().fit_transform(df[col])
scaler = MinMaxScaler()
df[df.select_dtypes("number").columns] = scaler.fit_transform(df.select_dtypes("number"))
X, y = df.drop("Diabetic", axis=1), df["Diabetic"]
X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
print("After SMOTE:", y_res.value_counts().to_dict())

# ====== 4. MODEL TRAINING ======
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)
models = {
  "Logistic Regression": LogisticRegression(max_iter=300),
  "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
  "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
  "SVM": SVC(probability=True, random_state=42),
  "KNN": KNeighborsClassifier(n_neighbors=5)
}
results = []
for name, model in models.items():
  model.fit(X_train, y_train)
  preds = model.predict(X_test)
  auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
  report = classification_report(y_test, preds, output_dict=True)
  results.append({
    "Model": name,
    "Accuracy": report["accuracy"],
    "Precision": report["1"]["precision"],
    "Recall": report["1"]["recall"],
    "F1": report["1"]["f1-score"],
    "AUC": auc
  })
results_df = pd.DataFrame(results)
print("\nPerformance Summary:\n", results_df)

# ====== 5. VISUALS ======
plt.figure(figsize=(7, 6))
for name, model in models.items():
  RocCurveDisplay.from_estimator(model, X_test, y_test, name=name)
plt.title("ROC Curves – Diabetes Risk Models")
plt.tight_layout()
plt.savefig("assets/diabetes_roc.png", dpi=300)
plt.show()

best_model = GradientBoostingClassifier(n_estimators=200, random_state=42)
best_model.fit(X_train, y_train)
preds = best_model.predict(X_test)
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix – Gradient Boosting Model")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("assets/diabetes_confusion.png", dpi=300)
plt.show()

perm = PermutationImportance(best_model, random_state=42).fit(X_test, y_test)
weights = eli5.explain_weights_df(perm, feature_names=list(X.columns))
plt.figure(figsize=(8, 6))
sns.barplot(data=weights.head(10), x="weight", y="feature", palette="viridis")
plt.title("Top 10 Important Features – Gradient Boosting")
plt.tight_layout()
plt.savefig("assets/diabetes_features.png", dpi=300)
plt.show()

# ====== 6. EXPORT ======
results_df.to_csv("model_performance_summary.csv", index=False)
print("✅ Analysis completed – outputs saved in /assets/")
