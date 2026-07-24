"""
Shared helpers for the extra-tests suite (fusion/SHAP/hippocampus): picking the best
classifier per method from CV metrics, comparing a new method's pooled OOF predictions
against a dataset's fixed baseline via paired DeLong, and Holm-Bonferroni correction
across the family of primary comparisons run within a dataset.
"""
import glob
import os

import pandas as pd
from statsmodels.stats.multitest import multipletests

from delong import delong_roc_test

# The 3 methods produced by the base pipeline (src/05, 07, 08) - whichever of these has
# the highest mean CV AUC for a given dataset is that dataset's "best base method", auto-
# detected below instead of hardcoded, so a new dataset works with zero code changes here.
BASELINE_METHOD_CANDIDATES = ["brainiac", "radiomics", "volumetry"]


def resolve_results_dir(dataset):
    """
    Naming convention used throughout this repo: OASIS-1 (the original/default dataset)
    keeps the legacy unsuffixed `results/` directory; every other dataset - including any
    new one a collaborator adds - uses `results_{dataset}/`. Same convention src/05, 07,
    08, 10 already use for --output_dir defaults.
    """
    return "results" if dataset == "oasis1" else f"results_{dataset}"


def load_all_metrics(results_dir):
    frames = []
    for f in glob.glob(os.path.join(results_dir, "metrics_*.csv")):
        frames.append(pd.read_csv(f))
    if not frames:
        raise RuntimeError(f"No metrics_*.csv found in {results_dir}")
    return pd.concat(frames, ignore_index=True)


def best_classifier_per_method(metrics_df):
    agg = metrics_df.groupby(["method", "classifier"])["auc"].mean().reset_index()
    best = agg.loc[agg.groupby("method")["auc"].idxmax()]
    return dict(zip(best["method"], best["classifier"]))


def load_baseline_predictions(dataset, results_dir=None):
    """
    Auto-detects the dataset's best base method: whichever of {brainiac, radiomics,
    volumetry} has the highest mean CV AUC in that dataset's base-pipeline results dir
    (run src/01-09 first). No per-dataset hardcoding, so this works for any dataset name.
    """
    results_dir = results_dir or resolve_results_dir(dataset)
    metrics_df = load_all_metrics(results_dir)
    candidates = metrics_df[metrics_df["method"].isin(BASELINE_METHOD_CANDIDATES)]
    if candidates.empty:
        raise RuntimeError(
            f"No baseline metrics ({BASELINE_METHOD_CANDIDATES}) found in {results_dir} "
            f"for dataset={dataset}. Run src/01-09 (base pipeline) for this dataset first."
        )
    best_clf_per_method = best_classifier_per_method(candidates)
    best_method = candidates.groupby("method")["auc"].mean().idxmax()
    best_clf = best_clf_per_method[best_method]

    pred_path = os.path.join(results_dir, "predictions", f"{best_method}_{best_clf}.csv")
    df = pd.read_csv(pred_path).sort_values("subject_id").reset_index(drop=True)
    return best_method, best_clf, df


def delong_vs_baseline(dataset, new_pred_csv, new_label):
    """
    Paired DeLong test between a new method's pooled OOF predictions and the dataset's
    fixed baseline predictions, joined on subject_id (both are pooled 5-fold CV OOF
    predictions over the same data{dataset}_folds.csv split, so this join is exact and
    the pairing is by-subject valid).
    """
    base_method, base_clf, base_df = load_baseline_predictions(dataset)
    new_df = pd.read_csv(new_pred_csv).sort_values("subject_id").reset_index(drop=True)

    merged = new_df.merge(base_df, on="subject_id", suffixes=("_new", "_base"))
    assert len(merged) == len(new_df) == len(base_df), (
        f"subject set mismatch comparing {new_label} (n={len(new_df)}) vs baseline "
        f"{base_method}/{base_clf} (n={len(base_df)}) on {dataset} - not the same subjects/folds"
    )
    mismatched = (merged["y_true_new"].values != merged["y_true_base"].values).sum()
    assert mismatched == 0, f"{mismatched} label mismatches between {new_label} and baseline - fold/pairing bug"

    auc_new, auc_base, z, p, ci = delong_roc_test(
        merged["y_true_new"].values, merged["y_prob_new"].values, merged["y_prob_base"].values
    )
    return {
        "dataset": dataset,
        "comparison": f"{new_label} vs {base_method} ({base_clf}) [baseline]",
        "auc_new": auc_new, "auc_baseline": auc_base, "auc_diff": auc_new - auc_base,
        "z": z, "p_value": p, "ci95_diff_low": ci[0], "ci95_diff_high": ci[1],
        "n_subjects": len(merged),
    }


def holm_correct(comparisons_df, pcol="p_value", primary_mask=None):
    """
    Holm-Bonferroni correction applied to the family of PRIMARY comparisons within a
    dataset (each new method's best classifier vs that dataset's best base method).
    Exploratory comparisons (e.g. individual k sweeps in SHAP, the 3B OOD hippocampus
    embedding) are excluded from the correction family and just carry raw p-values,
    labeled exploratory in the output.
    """
    df = comparisons_df.copy()
    df["p_holm"] = float("nan")
    df["family"] = "exploratory"
    mask = primary_mask if primary_mask is not None else pd.Series(True, index=df.index)
    if mask.sum() > 0:
        _, p_corrected, _, _ = multipletests(df.loc[mask, pcol].values, method="holm")
        df.loc[mask, "p_holm"] = p_corrected
        df.loc[mask, "family"] = "primary"
    return df
