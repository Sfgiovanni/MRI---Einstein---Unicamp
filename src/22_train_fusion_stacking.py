"""
Teste 1B - Late fusion / stacking: combina as predicoes (probabilidades) OOF do melhor
classificador BrainIAC-only e do melhor classificador radiomics-only (do pipeline base
ja existente, results{,_oasis2}/predictions/*.csv), sem re-treinar nada sobre as
features originais.

- "mean": media simples das duas probabilidades (sem treino nenhum).
- "meta_logreg": meta-classificador (logreg) treinado sobre as probabilidades OOF dos
  dois modelos base, usando os MESMOS 5 folds. Isso e stacking sem vazamento: a prob
  OOF de cada sujeito em cada fold ja vem de um modelo base treinado sem aquele fold;
  o meta-modelo, por sua vez, so ve as prob-OOF dos outros folds para treinar e prediz
  a prob-OOF do fold de teste - nenhum modelo (base ou meta) ve dados do seu proprio
  fold de teste em nenhuma etapa.
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score

from common_cv import CLASSIFIER_GRIDS, load_folds, run_cv
from stats_utils import best_classifier_per_method, load_all_metrics

BASE_RESULTS_DIR = {"oasis1": "results", "oasis2": "results_oasis2"}


def build_meta_features(dataset, folds_df):
    base_dir = BASE_RESULTS_DIR[dataset]
    metrics_df = load_all_metrics(base_dir)
    best = best_classifier_per_method(metrics_df)
    assert "brainiac" in best and "radiomics" in best, f"missing base methods in {base_dir}: {best}"

    b_pred = pd.read_csv(os.path.join(base_dir, "predictions", f"brainiac_{best['brainiac']}.csv"))
    r_pred = pd.read_csv(os.path.join(base_dir, "predictions", f"radiomics_{best['radiomics']}.csv"))

    merged = b_pred.merge(r_pred, on=["subject_id", "fold", "y_true"], suffixes=("_brainiac", "_radiomics"))
    assert len(merged) == len(b_pred) == len(r_pred) == len(folds_df), (
        f"subject/fold mismatch building meta-features for {dataset}: "
        f"brainiac={len(b_pred)} radiomics={len(r_pred)} folds={len(folds_df)} merged={len(merged)}"
    )
    merged = merged.rename(columns={"y_prob_brainiac": "brainiac_prob", "y_prob_radiomics": "radiomics_prob"})
    merged["label"] = merged["y_true"]
    print(f"[{dataset}] stacking base models: brainiac={best['brainiac']}, radiomics={best['radiomics']}")
    return merged[["subject_id", "label", "fold", "brainiac_prob", "radiomics_prob"]], best


def run_mean_baseline(meta_df, output_dir):
    os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)
    meta_df = meta_df.copy()
    meta_df["y_prob"] = (meta_df["brainiac_prob"] + meta_df["radiomics_prob"]) / 2.0
    meta_df["y_pred"] = (meta_df["y_prob"] >= 0.5).astype(int)
    preds = meta_df.rename(columns={"label": "y_true"})[["subject_id", "fold", "y_true", "y_prob", "y_pred"]]
    preds.to_csv(os.path.join(output_dir, "predictions", "fusion_late_mean.csv"), index=False)

    rows = []
    for fold in sorted(preds["fold"].unique()):
        fp = preds[preds["fold"] == fold]
        rows.append({
            "method": "fusion_late", "classifier": "mean", "fold": fold,
            "auc": roc_auc_score(fp["y_true"], fp["y_prob"]),
            "balanced_accuracy": balanced_accuracy_score(fp["y_true"], fp["y_pred"]),
            "f1": f1_score(fp["y_true"], fp["y_pred"]),
            "sensitivity": recall_score(fp["y_true"], fp["y_pred"], pos_label=1),
            "specificity": recall_score(fp["y_true"], fp["y_pred"], pos_label=0),
        })
    metrics_df = pd.DataFrame(rows)
    aucs = metrics_df["auc"].values
    print(f"[fusion_late/mean] AUC per fold: {[f'{a:.3f}' for a in aucs]} mean={aucs.mean():.3f} std={aucs.std():.3f}")
    return metrics_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="oasis1")
    parser.add_argument("--folds_csv", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    folds_csv = args.folds_csv or f"data/{args.dataset}_folds.csv"
    output_dir = args.output_dir or f"results_fusion_{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)

    folds_df = load_folds(folds_csv)
    meta_df, base_choice = build_meta_features(args.dataset, folds_df)

    mean_metrics = run_mean_baseline(meta_df, output_dir)

    meta_metrics = run_cv(
        meta_df.drop(columns=["fold"]), folds_df, ["brainiac_prob", "radiomics_prob"], "fusion_late", output_dir,
        classifiers=["meta_logreg"],
        classifier_grids={"meta_logreg": CLASSIFIER_GRIDS["logreg"]},
    )

    # run_cv already wrote metrics_fusion_late.csv for meta_logreg only; append the
    # mean-baseline rows (computed separately above, no training) into the same file
    # so downstream consolidation sees both "classifiers" for method=fusion_late.
    combined = pd.concat([mean_metrics, meta_metrics], ignore_index=True)
    combined.to_csv(os.path.join(output_dir, "metrics_fusion_late.csv"), index=False)
    print(f"[{args.dataset}] stacking done. base classifiers used: {base_choice}")


if __name__ == "__main__":
    main()
