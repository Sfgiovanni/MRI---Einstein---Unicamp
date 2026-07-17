"""
Classical volumetry/morphometry baseline using each dataset's own official whole-brain
volumetric measures (eTIV, nWBV, ASF) - both OASIS-1 and OASIS-2 publish these with the
identical definition/pipeline (OASIS team's own SPM-based processing), so they are
directly comparable across the two datasets with no extra harmonization step needed.
Used instead of running FreeSurfer recon-all (too slow on this hardware) or downloading
104GB of FreeSurfer-derivative discs - see PROGRESS.md for the cost/benefit rationale.
"""
import argparse

import pandas as pd

from common_cv import load_folds, run_cv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="oasis1")
    parser.add_argument("--labels_csv", default=None)
    parser.add_argument("--folds_csv", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    labels_csv = args.labels_csv or f"data/{args.dataset}_labels.csv"
    folds_csv = args.folds_csv or f"data/{args.dataset}_folds.csv"
    output_dir = args.output_dir or ("results" if args.dataset == "oasis1" else f"results_{args.dataset}")

    df = pd.read_csv(labels_csv)
    feature_cols = ["eTIV", "nWBV", "ASF"]
    features_df = df[["subject_id", "label"] + feature_cols].dropna()

    folds_df = load_folds(folds_csv)
    run_cv(features_df, folds_df, feature_cols, "volumetry", output_dir)
