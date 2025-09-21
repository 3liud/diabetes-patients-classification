#!/usr/bin/env python3
"""
Diabetes classification baseline (refactor).
- Fixes zero-as-missing issues
- Proper preprocessing via Pipeline/ColumnTransformer
- Compares three models with stratified CV
- Reports ROC AUC, PR AUC, F1
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def load_data(path="diabetes.csv", target="Outcome"):
    df = pd.read_csv(path)
    X = df.drop(columns=[target]).copy()
    y = df[target].astype(int)
    bad_zero = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
    X[bad_zero] = X[bad_zero].replace(0, np.nan)
    return X, y

def build_preprocessor(feature_names):
    num_pre = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    return ColumnTransformer([("num", num_pre, feature_names)], remainder="drop")

def main(csv_path="diabetes.csv"):
    X, y = load_data(csv_path)
    num_features = X.columns.tolist()
    pre = build_preprocessor(num_features)

    models = {
        "LogReg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "SVC-RBF": SVC(kernel="rbf", probability=True, class_weight="balanced"),
        "RF": RandomForestClassifier(n_estimators=400, random_state=42)
    }
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

    rows = []
    for name, clf in models.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        scores = cross_validate(
            pipe, X, y,
            scoring={"roc_auc":"roc_auc", "pr_auc":"average_precision", "f1":"f1"},
            cv=rkf, n_jobs=None
        )
        rows.append({
            "Model": name,
            "ROC_AUC_cv": scores["test_roc_auc"].mean(),
            "PR_AUC_cv": scores["test_pr_auc"].mean(),
            "F1_cv": scores["test_f1"].mean()
        })
    report = pd.DataFrame(rows).sort_values("ROC_AUC_cv", ascending=False)
    print("\nCross-validated comparison:\n", report.round(4).to_string(index=False))

    # Fit the best model and produce a simple holdout readout
    best = report.iloc[0]["Model"]
    pipe_best = Pipeline([("pre", pre), ("clf", models[best])])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    pipe_best.fit(X_tr, y_tr)
    proba = pipe_best.predict_proba(X_te)[:,1]

    thr_grid = np.linspace(0.2,0.8,61)
    f1s = [f1_score(y_te, (proba>=t).astype(int)) for t in thr_grid]
    t_best = float(thr_grid[int(np.argmax(f1s))])

    metrics = {
        "ROC_AUC": roc_auc_score(y_te, proba),
        "PR_AUC": average_precision_score(y_te, proba),
        "F1@0.5": f1_score(y_te, (proba>=0.5).astype(int)),
        "F1@t*": f1_score(y_te, (proba>=t_best).astype(int)),
        "Brier": brier_score_loss(y_te, proba),
        "t*": t_best
    }
    print("\nBest model:", best)
    for k,v in metrics.items():
        print(f"{k}: {v:.4f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
