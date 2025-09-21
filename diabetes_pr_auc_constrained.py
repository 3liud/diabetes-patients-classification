#!/usr/bin/env python3
"""
Diabetes classification: PR-AUC–optimized, recall-constrained, calibrated.
- Zero-as-missing fix for key physiological vars
- Pipeline with imputation + scaling (no leakage)
- Model selection via HalvingRandomSearchCV on PR-AUC
- Probability calibration (sigmoid)
- Threshold chosen to maximize F1 subject to recall >= MIN_RECALL
- Clean CLI: set --min-recall, --test-size, etc.

Requires: scikit-learn >=1.1
Optional: imbalanced-learn (if installed, toggles SMOTE via --use-smote)
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    brier_score_loss,
)
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

# Optional SMOTE (graceful fallback if imblearn is missing)
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE

    HAS_IMB = True
except Exception:
    HAS_IMB = False


BAD_ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_data(csv_path: Path, target="Outcome"):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[target]).copy()
    y = df[target].astype(int).values
    # Replace implausible zeros with NaN for imputation
    for c in BAD_ZERO_COLS:
        if c in X.columns:
            X.loc[X[c] == 0, c] = np.nan
    return X, y


def build_preprocessor(feature_names):
    num = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    return ColumnTransformer([("num", num, feature_names)], remainder="drop")


def model_space():
    # Three contenders with broad but sane ranges
    estimators = {
        "LogReg": (
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            {
                "clf__C": np.logspace(-3, 2, 20),
                "clf__penalty": ["l2"],
                "clf__solver": ["lbfgs", "liblinear"],
            },
        ),
        "SVC-RBF": (
            SVC(kernel="rbf", probability=True, class_weight="balanced"),
            {
                "clf__C": np.logspace(-2, 2, 20),
                "clf__gamma": np.logspace(-4, 1, 20),
            },
        ),
        "RF": (
            RandomForestClassifier(
                n_estimators=600, random_state=42, n_jobs=-1, class_weight=None
            ),
            {
                "clf__max_depth": [None, 4, 6, 8, 10, 14],
                "clf__min_samples_leaf": [1, 2, 4, 8],
                "clf__max_features": ["sqrt", "log2", 0.6, 0.8, 1.0],
            },
        ),
    }
    return estimators


def pr_auc_constrained_threshold(y_true, proba, min_recall=0.70):
    # Find threshold maximizing F1 with recall >= min_recall
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
    # precision_recall_curve returns thresholds of length n-1
    # Align arrays: skip first point where threshold is undefined
    best_t, best_f1 = 0.5, -1.0
    for p, r, t in zip(precisions[1:], recalls[1:], thresholds):
        if r >= min_recall:
            if (p + r) > 0:
                f1 = 2 * p * r / (p + r)
                if f1 > best_f1:
                    best_f1, best_t = f1, float(t)
    # Fallback: if nothing meets the recall constraint, use 0.5
    return best_t, best_f1


def build_pipeline(pre, clf, use_smote=False, random_state=42):
    if use_smote and HAS_IMB:
        return ImbPipeline(
            [
                ("pre", pre),
                ("smote", SMOTE(random_state=random_state)),
                ("clf", clf),
            ]
        )
    else:
        return Pipeline(
            [
                ("pre", pre),
                ("clf", clf),
            ]
        )


def main(args):
    X, y = load_data(Path(args.csv))
    feature_names = X.columns.tolist()
    pre = build_preprocessor(feature_names)

    # Train/test split (holdout for final eval)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=args.seed)

    # Search over models; pick best by PR-AUC
    best_model_name, best_search, best_score = None, None, -np.inf

    for name, (estimator, grid) in model_space().items():
        pipe = build_pipeline(pre, estimator, use_smote=args.use_smote)
        search = HalvingRandomSearchCV(
            estimator=pipe,
            param_distributions=grid,
            scoring="average_precision",
            n_candidates="exhaust",
            factor=3,
            cv=cv,
            random_state=args.seed,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_tr, y_tr)
        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_model_name, best_search = name, search

    print(f"\nBest by PR-AUC (CV): {best_model_name}  |  PR-AUC={best_score:.4f}")
    print("Best params:", best_search.best_params_)

    # Calibrate best with sigmoid on train via CV
    # Fit uncalibrated best on full train:
    best_uncal = best_search.best_estimator_
    # Wrap with calibration using the same CV scheme
    calibrated = CalibratedClassifierCV(best_uncal, method="sigmoid", cv=cv)
    calibrated.fit(X_tr, y_tr)

    # Evaluate on holdout
    proba = calibrated.predict_proba(X_te)[:, 1]
    pr_auc = average_precision_score(y_te, proba)
    roc_auc = roc_auc_score(y_te, proba)
    brier = brier_score_loss(y_te, proba)

    # Threshold selection under recall constraint
    t_star, f1_star = pr_auc_constrained_threshold(
        y_te, proba, min_recall=args.min_recall
    )
    y_hat_05 = (proba >= 0.5).astype(int)
    y_hat_t = (proba >= t_star).astype(int)

    # Metrics
    f1_05 = f1_score(y_te, y_hat_05)
    f1_t = f1_score(y_te, y_hat_t)

    # Reports
    print("\n=== Holdout (probability metrics) ===")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Brier:   {brier:.4f}")

    print("\n=== Thresholded metrics ===")
    print(f"F1 @ 0.5: {f1_05:.4f}")
    print(f"F1 @ t*:  {f1_t:.4f} (t*={t_star:.3f}, recall≥{args.min_recall:.2f})")

    cm = confusion_matrix(y_te, y_hat_t)
    print("\nConfusion matrix @ t*:\n", cm)
    print(
        "\nClassification report @ t*:\n",
        classification_report(y_te, y_hat_t, digits=4),
    )

    # Save a small CSV with summary
    out = pd.DataFrame(
        [
            {
                "best_model": best_model_name,
                "cv_pr_auc": best_score,
                "holdout_pr_auc": pr_auc,
                "holdout_roc_auc": roc_auc,
                "brier": brier,
                "f1_at_0.5": f1_05,
                "f1_at_t*": f1_t,
                "t*": t_star,
                "min_recall": args.min_recall,
                "use_smote": bool(args.use_smote and HAS_IMB),
            }
        ]
    )
    out_path = Path(args.out_dir) / "diabetes_pr_auc_constrained_report.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nSaved summary → {out_path.resolve()}")

    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="diabetes.csv")
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-recall", type=float, default=0.70)
    p.add_argument(
        "--use-smote",
        action="store_true",
        help="Use SMOTE in the pipeline (requires imbalanced-learn).",
    )
    p.add_argument("--out-dir", type=str, default="reports")
    args = p.parse_args()
    raise SystemExit(main(args))
