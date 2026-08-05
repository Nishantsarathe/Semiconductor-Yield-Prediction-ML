# 🔬 Semiconductor Manufacturing Yield Prediction

A machine learning project that builds a classifier to predict the **Pass/Fail yield** of semiconductor manufacturing processes using high-dimensional sensor data.

---

## 📌 Problem Statement

Modern semiconductor fabrication relies on hundreds of sensors that continuously monitor production. Not all sensor signals are equally informative — many carry noise or redundant information. This project identifies the most relevant signals and builds a robust classifier to predict whether a production unit will **pass or fail** in-house line testing, enabling:

- Faster defect detection
- Reduced per-unit production costs
- Decreased time to learning in the process line

---

## 📂 Dataset

| Property | Details |
|---|---|
| File | `signal-data.csv` |
| Shape | 1,567 rows × 592 columns |
| Features | 591 sensor signal measurements + 1 timestamp |
| Target | `Pass (-1)` / `Fail (1)` |

---

## 🗂️ Project Structure

```
├── NishantSarathe_DataScienceMajorProject.ipynb   # Main notebook
├── signal-data.csv                                 # Dataset
├── best_svm_model.pkl                              # Saved best model
└── README.md
```

---

## 🔄 Pipeline Overview

### 1. Data Import & Exploration
- Loaded dataset, inspected shape, dtypes, and basic statistics
- Identified 591 sensor features with mixed relevance

### 2. Data Cleansing
- Visualised missing values across all 591 features (Top 30 plotted)
- Dropped columns with **> 70% missing values**
- Imputed remaining missing values using **column-wise median** (robust to outliers)
- Checked and handled duplicate records

### 3. Exploratory Data Analysis (EDA)
- Univariate, bivariate, and multivariate analysis
- Target class distribution analysis
- Correlation heatmaps and distribution plots

### 4. Data Pre-processing
- Removed **low-variance features** (near-constant signals with no predictive power)
- Separated features (`X`) from target (`y`)
- Applied **SMOTE** to handle class imbalance in the training set
- Performed **Train / Test split**
- Applied **StandardScaler** for feature normalisation
- Verified statistical similarity between train, test, and original distributions

### 5. Model Training & Evaluation

Six classifiers were trained and evaluated:

| Model | Notes |
|---|---|
| Logistic Regression | Baseline linear model |
| Decision Tree | Interpretable tree-based model |
| Random Forest | Ensemble; used for feature importance |
| **Support Vector Machine (SVM)** | **Best model** |
| K-Nearest Neighbours (KNN) | Distance-based model |
| Naive Bayes | Probabilistic baseline |

Each model was evaluated using:
- Classification report (Precision, Recall, F1-Score)
- Cross-validation accuracy
- Train vs. Test accuracy comparison

### 6. Hyperparameter Tuning
GridSearchCV with 5-fold cross-validation was applied to the SVM model:

```python
param_grid = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"]
}
```

### 7. Feature Importance
Top 20 most influential sensor features identified via Random Forest feature importance scores.

### 8. Model Persistence
Best model saved and reloaded using `joblib`:

```python
import joblib
joblib.dump(best_svm, "best_svm_model.pkl")
loaded_model = joblib.load("best_svm_model.pkl")
```

---

## 🏆 Results

| Rank | Model | Test Accuracy |
|---|---|---|
| 🥇 1 | **Support Vector Machine** | **93.63%** |
| 🥈 2 | Random Forest | 93.31% |
| 🥉 3 | Logistic Regression | 83.76% |
| 4 | Decision Tree | 79.94% |
| 5 | K-Nearest Neighbors | 34.71% |
| 6 | Naive Bayes | 28.98% |

**Best SVM Configuration:**
- Kernel: `RBF`
- C: `10`
- Gamma: `scale`
- Cross-validation Accuracy: **~99.83%**
- ROC-AUC: Excellent (≥ 0.90)

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Visualisation |
| `scikit-learn` | ML models, preprocessing, evaluation |
| `imbalanced-learn` | SMOTE for class balancing |
| `joblib` | Model serialisation |

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/your-username/semiconductor-yield-prediction.git
cd semiconductor-yield-prediction

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib

# Launch notebook
jupyter notebook NishantSarathe_DataScienceMajorProject.ipynb
```

---

## 📋 Conclusion

This project demonstrates that a machine learning pipeline — combining missing value treatment, low-variance feature removal, SMOTE-based class balancing, and hyperparameter-tuned SVM — can achieve **93.63% test accuracy** on high-dimensional semiconductor sensor data. The SVM with an RBF kernel proved to be the strongest classifier, supported by excellent cross-validation performance and ROC-AUC scores, making it well-suited for deployment in a real-time manufacturing quality control system.

---

## 👤 Author

**Nishant Sarathe**  
Data Science Major Project — Capstone 2
