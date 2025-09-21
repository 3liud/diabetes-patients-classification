# Diabetes Patients Classification (Refactored)

This project applies supervised machine learning classification to the Pima Indians Diabetes dataset.
It has been **updated** from the original notebook to include modern, production-style practices.

## Dataset

* **File**: `diabetes.csv` (768 rows, 9 columns: 8 features + Outcome label).
* **Note**: Some physiological variables contain zeros that actually mean *missing*.
  These are properly imputed in the new pipeline.

## Models

### Original

* Logistic Regression
* KNN
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)
* Naive Bayes

### Updated

* Logistic Regression (regularized, balanced)
* Random Forest (tuned with halving search)
* RBF SVM (scaled, tuned with halving search)

## Key Improvements

* Zero-as-missing fix for `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`.
* Pipeline + ColumnTransformer with imputation and scaling (prevents leakage).
* Robust cross-validation (`RepeatedStratifiedKFold`).
* Hyperparameter search with **HalvingRandomSearchCV** optimized for PR-AUC.
* Probability calibration (sigmoid).
* Threshold tuning: maximize F1 subject to minimum recall (default 0.70).
* Metrics reported: PR-AUC, ROC-AUC, F1\@0.5, F1\@t\*, Brier score, confusion matrix.

## Usage

### Baseline comparison

```bash
python3 diabetes_refactor_baseline.py
```

### Constrained PR-AUC pipeline

```bash
python3 diabetes_pr_auc_constrained.py --csv diabetes.csv --min-recall 0.70
```

### Options

* `--min-recall` : recall constraint for threshold selection (default 0.70).
* `--test-size`  : test set fraction (default 0.25).
* `--use-smote`  : enable SMOTE oversampling (requires imbalanced-learn).
* `--out-dir`    : directory to save reports (default: `./reports`).

## Requirements

* Python 3.9+
* scikit-learn >= 1.1
* numpy, pandas, matplotlib, seaborn
* imbalanced-learn (optional, for SMOTE)

## Outputs

* Console: CV summary, holdout metrics, confusion matrix, classification report.
* CSV: `reports/diabetes_pr_auc_constrained_report.csv` with summary metrics.

---

This refactor modernizes the original exploratory notebook with better preprocessing, robust evaluation, probability calibration, and decision-threshold optimization under clinical-style constraints.
