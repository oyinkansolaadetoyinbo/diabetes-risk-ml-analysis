# 🩺 Predicting Type 2 Diabetes Risk Using Machine Learning

### 💡 Conducted and Reported by **Adetoyinbo Oyinkansola** at *Write Right Concept*

---

## 📖 Project Overview

During my internship at **Write Right Concept**, I led a full-cycle data-analytics project to develop and compare machine-learning models for **predicting Type 2 Diabetes risk**.

The objective was to demonstrate how **data science can inform early health interventions**, guiding patients, clinicians, and policy stakeholders toward smarter, evidence-based prevention strategies.

This project combined **statistical reasoning**, **model interpretability**, and **visual storytelling**, showing how analytical work can create tangible public-health impact.

---

## 🎯 Objectives

* Develop and test multiple machine-learning algorithms for diabetes-risk prediction.
* Identify the most reliable and interpretable algorithm for healthcare use.
* Translate analytical findings into actionable recommendations for individuals, healthcare providers, and policymakers.

---

## 🧩 Dataset Overview

**Source:** [Kaggle – Diabetes Health Indicators Dataset (Tigga & Garg, 2019)](https://www.kaggle.com/datasets/tigganeha4/diabetes-dataset-2019)
**Records:** 952  **Features:** 17  **Target:** `Diabetic (Yes/No)`

Key variables: `Age`, `BMI`, `HighBP`, `Family_Diabetes`, `PhysicallyActive`, `Stress`, `RegularMedicine`, and `UrinationFreq`.

---

## ⚙️ Data Pre-processing and Setup

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE

df = pd.read_csv("diabetes.csv")
print("Initial shape:", df.shape)

# Encode and scale
for col in df.select_dtypes('object'):
    df[col] = LabelEncoder().fit_transform(df[col])
scaler = MinMaxScaler()
df[df.select_dtypes('number').columns] = scaler.fit_transform(df.select_dtypes('number'))

# Handle imbalance
X, y = df.drop('Diabetic', axis=1), df['Diabetic']
X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
print("After SMOTE:", y_res.value_counts().to_dict())
```

---

## 🧠 Model Building and Evaluation

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=300),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    print(f"{name}: AUC = {auc:.2f}")
    print(classification_report(y_test, preds))
```

---

## 📊 Model Performance Summary

| Model                       | Accuracy | Precision |  Recall  | F1-Score |    AUC   |
| :-------------------------- | :------: | :-------: | :------: | :------: | :------: |
| Logistic Regression         |   0.86   |    0.80   |   0.71   |   0.75   |   0.87   |
| KNN (k = 2)                 |   0.95   |    0.97   |   0.85   |   0.91   |   0.95   |
| SVM                         |   0.75   |    1.00   |   0.15   |   0.26   |   0.77   |
| Random Forest               |   0.93   |    0.89   |   0.89   |   0.89   |   0.94   |
| **Gradient Boosting (GBM)** | **0.94** |  **0.91** | **0.90** | **0.91** | **0.98** |

✅ **Gradient Boosting** achieved the highest AUC (0.98) and the best trade-off between sensitivity and specificity.

---

## 📈 Visual Insights

### ROC Curves – Model Comparison

![ROC Curves](assets/diabetes_roc.png)

### Feature Importance – Gradient Boosting Model

![Feature Importance](assets/diabetes_features.png)

---

## 💬 Practical Implications — My Recommendations

### 👩‍⚕️ For Individuals

I encourage individuals—especially those with family history, high BMI, or irregular medication—to **monitor lifestyle patterns** such as activity, diet, and sleep. Predictive tools like this can be embedded in **mobile health apps** to promote timely screening.

### 🏥 For Healthcare Professionals

I recommend embedding ML-driven alerts into **EHR systems** to flag high-risk patients early. This supports personalised counselling and better preventive outcomes.

### 🏛️ For Public Health Stakeholders

I advocate leveraging **predictive analytics** for population-risk mapping, enabling targeted education campaigns and efficient funding allocation.

---

## 🧠 Key Learnings — My Reflection

* **Ensemble algorithms** (GBM & Random Forest) captured complex non-linear interactions better than linear models.
* **SMOTE balancing** improved fairness by enhancing minority-class recall.
* I learned that **explainability** is as vital as accuracy — using SHAP and permutation importance to build stakeholder trust.

This reinforced my belief that effective analytics merge **technical precision with human empathy**.

---

## 🚀 Future Directions — My Recommendations

* I plan to **deploy this model on Streamlit** for real-time prediction.
* I aim to integrate **wearable and IoT health data** for continuous monitoring.
* I encourage collaboration between **health-tech startups and analysts** to scale predictive wellness systems globally.

---

## 🛠️ Tech Stack

| Tool                 | Function                       |
| :------------------- | :----------------------------- |
| Python 3.11          | Core environment               |
| Pandas / NumPy       | Data cleaning & transformation |
| Scikit-learn         | Model building & evaluation    |
| Imbalanced-learn     | SMOTE balancing                |
| Matplotlib / Seaborn | Visualisation                  |
| eli5 / SHAP          | Model interpretability         |

---

## 💼 Professional Reflection

Conducting this analysis at **Write Right Concept** strengthened my ability to design and communicate data-driven solutions that connect technical depth with social purpose.

> *“In data analytics, the real value is not just predicting the future — it’s helping people prepare for it.”*

---

## 🔗 Author & Contact

**👩‍💻 Adetoyinbo Oyinkansola**
Data & Business Intelligence Analyst | Machine Learning & Health Analytics Enthusiast
📍 United Kingdom | 
📧 [adetoyinbo.oyinkansola@gmail.com](mailto:adetoyinbo.oyinkansola@gmail.com)
🌐 [LinkedIn – Adetoyinbo Oyinkansola](https://www.linkedin.com/in/adetoyinbo-oyinkansola)
💼 GitHub Portfolio: [github.com/oyinkansolaadetoyinbo](https://github.com/oyinkansolaadetoyinbo)

---

## 📚 References

Tigga, N. P. and Garg, S. (2019) ‘Prediction of Type 2 Diabetes using Machine Learning Classification Methods’, *Procedia Computer Science*, 167, pp. 706–716.

Kaggle (2019) *Diabetes Health Indicators Dataset*. Available at: [https://www.kaggle.com/datasets/tigganeha4/diabetes-dataset-2019](https://www.kaggle.com/datasets/tigganeha4/diabetes-dataset-2019).

Chawla, N. V. et al. (2002) ‘SMOTE: Synthetic Minority Over-sampling Technique’, *Journal of Artificial Intelligence Research*, 16, pp. 321–357.

Pedregosa, F. et al. (2011) ‘Scikit-learn: Machine Learning in Python’, *Journal of Machine Learning Research*, 12, pp. 2825–2830.

Lundberg, S. M. and Lee, S. I. (2017) ‘A Unified Approach to Interpreting Model Predictions’, *Advances in Neural Information Processing Systems 30 (NIPS 2017)*.



---

