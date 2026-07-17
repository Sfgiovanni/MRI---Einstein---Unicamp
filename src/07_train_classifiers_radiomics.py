import argparse

import pandas as pd

from common_cv import load_folds, run_cv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="oasis1")
    parser.add_argument("--features", default=None)
    parser.add_argument("--folds_csv", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    features_path = args.features or f"features/{args.dataset}_radiomics_features.parquet"
    folds_csv = args.folds_csv or f"data/{args.dataset}_folds.csv"
    output_dir = args.output_dir or ("results" if args.dataset == "oasis1" else f"results_{args.dataset}")

    features_df = pd.read_parquet(features_path)
    folds_df = load_folds(folds_csv)
    feature_cols = [c for c in features_df.columns if c not in ("subject_id", "label")]

    run_cv(features_df, folds_df, feature_cols, "radiomics", output_dir)
