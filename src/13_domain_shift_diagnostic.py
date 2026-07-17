"""
Domain-shift diagnostic: can a classifier tell which dataset (OASIS-1 vs OASIS-2) a
subject's features came from, ignoring the AD/CN label entirely? A high dataset-
membership AUC means the feature space is dominated by a batch effect, which puts an
upper bound on how much of the cross-dataset AUC (script 11) can be attributed to
transferred disease signal vs. cohort/acquisition artifact.

Run for both the OASIS-1 variants used in the cross-dataset experiment:
  - oasis1 (SUBJ_111, OASIS's own motion-corrected average) vs oasis2 (mpr-1, raw)
  - oasis1_mpr1 (raw mpr-1, matching oasis2's input level) vs oasis2 (mpr-1, raw)
so the effect of the input-processing-level asymmetry on separability can be isolated.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EMB_COLS = lambda df: [c for c in df.columns if c.startswith("emb_")]
RAD_COLS = lambda df: [c for c in df.columns if c not in ("subject_id", "label")]
VOL_COLS = lambda df: ["eTIV", "nWBV", "ASF"]


def check_domain_separability(feat1_path, feat2_path, feature_cols_fn, is_csv=False):
    df1 = pd.read_csv(feat1_path) if is_csv else pd.read_parquet(feat1_path)
    df2 = pd.read_csv(feat2_path) if is_csv else pd.read_parquet(feat2_path)
    cols = feature_cols_fn(df1)
    X = np.vstack([df1[cols].values, df2[cols].values])
    y = np.array([0] * len(df1) + [1] * len(df2))
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=5000, random_state=42))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    return scores.mean(), scores.std()


def main():
    rows = []

    mean, std = check_domain_separability(
        "features/oasis1_brainiac_features.parquet", "features/oasis2_brainiac_features.parquet", EMB_COLS)
    rows.append({"comparison": "oasis1(SUBJ_111) vs oasis2(mpr-1)", "method": "brainiac", "membership_auc": mean, "std": std})

    mean, std = check_domain_separability(
        "features/oasis1_radiomics_features.parquet", "features/oasis2_radiomics_features.parquet", RAD_COLS)
    rows.append({"comparison": "oasis1(SUBJ_111) vs oasis2(mpr-1)", "method": "radiomics", "membership_auc": mean, "std": std})

    mean, std = check_domain_separability(
        "data/oasis1_labels.csv", "data/oasis2_labels.csv", VOL_COLS, is_csv=True)
    rows.append({"comparison": "oasis1(SUBJ_111) vs oasis2(mpr-1)", "method": "volumetry", "membership_auc": mean, "std": std})

    mean, std = check_domain_separability(
        "features/oasis1_mpr1_brainiac_features.parquet", "features/oasis2_brainiac_features.parquet", EMB_COLS)
    rows.append({"comparison": "oasis1_mpr1(mpr-1) vs oasis2(mpr-1) [input-level matched]", "method": "brainiac", "membership_auc": mean, "std": std})

    mean, std = check_domain_separability(
        "features/oasis1_mpr1_radiomics_features.parquet", "features/oasis2_radiomics_features.parquet", RAD_COLS)
    rows.append({"comparison": "oasis1_mpr1(mpr-1) vs oasis2(mpr-1) [input-level matched]", "method": "radiomics", "membership_auc": mean, "std": std})

    df = pd.DataFrame(rows)
    df.to_csv("results_cross/domain_shift_diagnostic.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
