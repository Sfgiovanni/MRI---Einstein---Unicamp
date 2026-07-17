"""
Shared subject-level 5-fold CV training/evaluation harness, reused by every
feature source (BrainIAC embeddings, radiomics, volumetry) so all methods are
compared under an identical protocol.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42

CLASSIFIER_GRIDS = {
    "logreg": (
        LogisticRegression(max_iter=5000, random_state=SEED),
        {"clf__C": [0.01, 0.1, 1.0, 10.0]},
    ),
    "svm_linear": (
        SVC(kernel="linear", probability=True, random_state=SEED),
        {"clf__C": [0.01, 0.1, 1.0, 10.0]},
    ),
    "svm_rbf": (
        SVC(kernel="rbf", probability=True, random_state=SEED),
        {"clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale", "auto"]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=SEED, n_jobs=-1),
        {"clf__n_estimators": [200, 500], "clf__max_depth": [None, 5, 10]},
    ),
}

try:
    from xgboost import XGBClassifier

    CLASSIFIER_GRIDS["xgboost"] = (
        XGBClassifier(
            random_state=SEED, eval_metric="logloss", n_jobs=-1, verbosity=0
        ),
        {"clf__n_estimators": [100, 300], "clf__max_depth": [3, 5], "clf__learning_rate": [0.01, 0.1]},
    )
except ImportError:
    pass

try:
    from lightgbm import LGBMClassifier

    CLASSIFIER_GRIDS["lightgbm"] = (
        LGBMClassifier(random_state=SEED, n_jobs=-1, verbosity=-1),
        {"clf__n_estimators": [100, 300], "clf__max_depth": [3, 5, -1], "clf__learning_rate": [0.01, 0.1]},
    )
except ImportError:
    pass


def load_folds(folds_csv):
    return pd.read_csv(folds_csv)


def make_stratified_subject_folds(labels_df, n_folds=5, seed=SEED):
    """
    Shared fold-construction logic reused by every dataset's label-prep script, so the
    CV protocol (stratified, subject-level, same seed) is identical across datasets.
    labels_df must have subject_id, label columns. Returns labels_df with a 'fold' column.
    """
    from sklearn.model_selection import StratifiedKFold

    df = labels_df.reset_index(drop=True).copy()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    df["fold"] = -1
    for fold_idx, (_, test_idx) in enumerate(skf.split(df["subject_id"], df["label"])):
        df.loc[test_idx, "fold"] = fold_idx
    assert (df["fold"] >= 0).all()
    return df


def run_cv(features_df, folds_df, feature_cols, method_name, output_dir, classifiers=None):
    """
    features_df: subject_id, label, <feature_cols...>
    folds_df: subject_id, label, fold  (fixed subject-level 5-fold split, shared across methods)
    """
    os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)

    df = features_df.merge(folds_df[["subject_id", "fold"]], on="subject_id", how="inner")
    assert len(df) == len(folds_df), (
        f"{len(folds_df) - len(df)} subjects in folds_df missing from features_df for method={method_name}"
    )

    n_folds = df["fold"].nunique()
    classifiers = classifiers or list(CLASSIFIER_GRIDS.keys())

    all_metrics = []
    for clf_name in classifiers:
        base_clf, grid = CLASSIFIER_GRIDS[clf_name]
        fold_preds = []
        for fold in range(n_folds):
            train_df = df[df["fold"] != fold]
            test_df = df[df["fold"] == fold]

            X_train = train_df[feature_cols].values
            y_train = train_df["label"].values
            X_test = test_df[feature_cols].values
            y_test = test_df["label"].values

            pipe = Pipeline([("scaler", StandardScaler()), ("clf", base_clf)])
            inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
            search = GridSearchCV(pipe, grid, cv=inner_cv, scoring="roc_auc", n_jobs=-1)
            search.fit(X_train, y_train)
            best = search.best_estimator_

            y_prob = best.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            for sid, yt, yp, ypred in zip(test_df["subject_id"], y_test, y_prob, y_pred):
                fold_preds.append({
                    "subject_id": sid, "fold": fold, "y_true": yt,
                    "y_prob": yp, "y_pred": ypred,
                })

            joblib.dump(best, os.path.join(output_dir, "models", f"{method_name}_{clf_name}_fold{fold}.joblib"))
            with open(os.path.join(output_dir, "models", f"{method_name}_{clf_name}_fold{fold}_bestparams.json"), "w") as f:
                json.dump(search.best_params_, f, indent=2)

        preds_df = pd.DataFrame(fold_preds)
        preds_df.to_csv(os.path.join(output_dir, "predictions", f"{method_name}_{clf_name}.csv"), index=False)

        for fold in range(n_folds):
            fold_p = preds_df[preds_df["fold"] == fold]
            auc = roc_auc_score(fold_p["y_true"], fold_p["y_prob"])
            bal_acc = balanced_accuracy_score(fold_p["y_true"], fold_p["y_pred"])
            f1 = f1_score(fold_p["y_true"], fold_p["y_pred"])
            sens = recall_score(fold_p["y_true"], fold_p["y_pred"], pos_label=1)
            spec = recall_score(fold_p["y_true"], fold_p["y_pred"], pos_label=0)
            all_metrics.append({
                "method": method_name, "classifier": clf_name, "fold": fold,
                "auc": auc, "balanced_accuracy": bal_acc, "f1": f1,
                "sensitivity": sens, "specificity": spec,
            })
        fold_aucs = [m["auc"] for m in all_metrics if m["method"] == method_name and m["classifier"] == clf_name]
        print(f"[{method_name}/{clf_name}] AUC per fold: {[f'{a:.3f}' for a in fold_aucs]} "
              f"mean={np.mean(fold_aucs):.3f} std={np.std(fold_aucs):.3f}")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(output_dir, f"metrics_{method_name}.csv"), index=False)
    return metrics_df


def run_cross_dataset(train_df, test_df, feature_cols, method_name, direction_name, output_dir, classifiers=None):
    """
    Train on ALL subjects of one dataset, test on ALL subjects of another (no overlap
    by construction - distinct cohorts). Scaler and hyperparameters are fit only on the
    training-side data; the test side is never used for fitting anything.
    train_df/test_df: subject_id, label, <feature_cols...> (no 'fold' column needed here).
    """
    os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)

    classifiers = classifiers or list(CLASSIFIER_GRIDS.keys())
    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values

    all_metrics = []
    for clf_name in classifiers:
        base_clf, grid = CLASSIFIER_GRIDS[clf_name]
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", base_clf)])
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        search = GridSearchCV(pipe, grid, cv=inner_cv, scoring="roc_auc", n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_

        y_prob = best.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        preds_df = pd.DataFrame({
            "subject_id": test_df["subject_id"].values, "y_true": y_test,
            "y_prob": y_prob, "y_pred": y_pred,
        })
        preds_df.to_csv(os.path.join(output_dir, "predictions", f"{method_name}_{clf_name}_{direction_name}.csv"), index=False)

        joblib.dump(best, os.path.join(output_dir, "models", f"{method_name}_{clf_name}_{direction_name}.joblib"))
        with open(os.path.join(output_dir, "models", f"{method_name}_{clf_name}_{direction_name}_bestparams.json"), "w") as f:
            json.dump(search.best_params_, f, indent=2)

        auc = roc_auc_score(y_test, y_prob)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        sens = recall_score(y_test, y_pred, pos_label=1)
        spec = recall_score(y_test, y_pred, pos_label=0)
        all_metrics.append({
            "method": method_name, "classifier": clf_name, "direction": direction_name,
            "n_train": len(train_df), "n_test": len(test_df),
            "auc": auc, "balanced_accuracy": bal_acc, "f1": f1,
            "sensitivity": sens, "specificity": spec,
        })
        print(f"[{method_name}/{clf_name}] {direction_name}: AUC={auc:.3f} bal_acc={bal_acc:.3f}")

    return pd.DataFrame(all_metrics)
