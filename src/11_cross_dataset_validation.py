"""
Etapa 6 (experimento principal) - Cross-dataset validation: train on ALL subjects of
one dataset, test on ALL subjects of the other, in both directions, for every feature
method. No subject overlap is possible by construction (OASIS-1 and OASIS-2 are
distinct, non-overlapping cohorts with disjoint ID namespaces - verified in
PROGRESS_dataset2.md). Reuses the exact same classifier grids/seeds as the intra-domain
CV (common_cv.CLASSIFIER_GRIDS) so hyperparameter search spaces are identical.

For each direction and classifier, also reports the "generalization gap" = intra-domain
CV AUC (mean across folds, from the source dataset's own results_*/metrics_<method>.csv)
minus the cross-dataset AUC. A foundation model's core claim is that this gap should be
smaller than for baselines trained from scratch on dataset-specific features.
"""
import argparse
import os

import pandas as pd

from common_cv import CLASSIFIER_GRIDS, run_cross_dataset

DATASET_RESULTS_DIR = {"oasis1": "results", "oasis2": "results_oasis2"}
DATASET_FEATURES = {
    "brainiac": "features/{ds}_brainiac_features.parquet",
    "radiomics": "features/{ds}_radiomics_features.parquet",
}

# For brainiac/radiomics ONLY, cross-dataset training/testing uses the input-level-matched
# oasis1_mpr1 variant (raw single repetition, same as oasis2) instead of oasis1's own
# SUBJ_111 (OASIS's motion-corrected average, already N4-corrected/atlas-resampled before
# BrainIAC preprocessing). A domain-shift diagnostic (dataset-membership classifier AUC)
# found AUC=1.0 separability between oasis1(SUBJ_111) and oasis2(mpr-1) for both brainiac
# embeddings and radiomics - i.e. those cross-dataset AUCs were dominated by an input-level
# batch effect, not disease signal. Volumetry (eTIV/nWBV/ASF) is unaffected: it comes from
# OASIS's own independently-computed demographics stats, not from our preprocessing, and its
# dataset-membership AUC was only 0.58. Intra-domain numbers (get_intra_domain_auc) still use
# the original oasis1 (SUBJ_111) results, unaffected - see PROGRESS_dataset2.md.
CROSS_FEATURE_DATASET = {"oasis1": "oasis1_mpr1", "oasis2": "oasis2"}


def load_features(method, dataset):
    if method == "volumetry":
        df = pd.read_csv(f"data/{dataset}_labels.csv")
        feature_cols = ["eTIV", "nWBV", "ASF"]
        return df[["subject_id", "label"] + feature_cols].dropna(), feature_cols
    ds_for_features = CROSS_FEATURE_DATASET[dataset]
    path = DATASET_FEATURES[method].format(ds=ds_for_features)
    df = pd.read_parquet(path)
    feature_cols = [c for c in df.columns if c not in ("subject_id", "label")]
    return df, feature_cols


# For brainiac/radiomics, the generalization gap must compare intra-domain AUC against
# cross-dataset AUC WITHIN THE SAME INPUT PIPELINE (mpr1), not the canonical OASIS-1
# SUBJ_111 intra-domain number - otherwise the gap conflates "cohort changed" with
# "pipeline changed" (mpr1 is single-repetition, noisier than SUBJ_111's motion-corrected
# average - same subjects/folds, intra-domain AUC drops 0.750->0.707 for brainiac,
# 0.748->0.734 for radiomics on mpr1 alone, before any cross-dataset transfer). This
# gap-only intra-domain number is computed once via
# `05_train_classifiers_brainiac.py --dataset oasis1_mpr1 --folds_csv data/oasis1_folds.csv
# --output_dir results_oasis1_mpr1` (and the radiomics equivalent with 07). Volumetry is
# unaffected (no oasis1_mpr1 variant exists or is needed - see CROSS_FEATURE_DATASET).
GAP_INTRA_DOMAIN_OVERRIDE = {
    ("brainiac", "oasis1"): "results_oasis1_mpr1",
    ("radiomics", "oasis1"): "results_oasis1_mpr1",
}


def get_intra_domain_auc(method, dataset, for_gap=False):
    results_dir = DATASET_RESULTS_DIR[dataset]
    if for_gap:
        results_dir = GAP_INTRA_DOMAIN_OVERRIDE.get((method, dataset), results_dir)
    metrics_path = os.path.join(results_dir, f"metrics_{method}.csv")
    if not os.path.exists(metrics_path):
        return None, None
    df = pd.read_csv(metrics_path)
    agg = df.groupby("classifier")["auc"].mean()
    best_clf = agg.idxmax()
    return agg[best_clf], best_clf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=["brainiac", "radiomics", "volumetry"])
    parser.add_argument("--output_dir", default="results_cross")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_metrics = []
    gap_rows = []

    for method in args.methods:
        df1, cols1 = load_features(method, "oasis1")
        df2, cols2 = load_features(method, "oasis2")
        assert cols1 == cols2, f"{method}: feature columns differ between datasets ({len(cols1)} vs {len(cols2)}) - not directly comparable"
        feature_cols = cols1
        print(f"\n=== {method}: {len(df1)} OASIS-1 subjects, {len(df2)} OASIS-2 subjects, {len(feature_cols)} features ===")

        for direction, (train_df, test_df) in [
            ("oasis1_to_oasis2", (df1, df2)),
            ("oasis2_to_oasis1", (df2, df1)),
        ]:
            m = run_cross_dataset(train_df, test_df, feature_cols, method, direction, args.output_dir)
            all_metrics.append(m)

            train_ds = direction.split("_to_")[0]
            intra_auc, intra_clf = get_intra_domain_auc(method, train_ds, for_gap=True)
            for _, row in m.iterrows():
                gap_rows.append({
                    "method": method, "classifier": row["classifier"], "direction": direction,
                    "intra_domain_auc": intra_auc if row["classifier"] == intra_clf else None,
                    "intra_domain_best_classifier": intra_clf,
                    "cross_dataset_auc": row["auc"],
                    "generalization_gap": (intra_auc - row["auc"]) if (intra_auc is not None and row["classifier"] == intra_clf) else None,
                })

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    metrics_df.to_csv(os.path.join(args.output_dir, "metrics_cross_dataset.csv"), index=False)

    gap_df = pd.DataFrame(gap_rows)
    gap_df.to_csv(os.path.join(args.output_dir, "generalization_gap.csv"), index=False)

    print("\n=== Generalization gap (intra-domain best-classifier AUC minus cross-dataset AUC) ===")
    print(gap_df[gap_df["generalization_gap"].notna()].to_string(index=False))


if __name__ == "__main__":
    main()
