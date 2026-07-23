"""
Teste 3B (exploratorio) - Classificadores sobre embeddings BrainIAC extraidos do recorte
do hipocampo (src/27). Mesmo protocolo de CV 5-fold por sujeito. Resultado marcado como
exploratorio no consolidado (risco de OOD documentado em src/27).
"""
import argparse
import os

import pandas as pd

from common_cv import load_folds, run_cv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="oasis1")
    parser.add_argument("--features_parquet", default=None)
    parser.add_argument("--folds_csv", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    features_parquet = args.features_parquet or f"features/{args.dataset}_brainiac_hippo_features.parquet"
    folds_csv = args.folds_csv or f"data/{args.dataset}_folds.csv"
    output_dir = args.output_dir or f"results_hippocampus_{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)

    features_df = pd.read_parquet(features_parquet)
    feature_cols = [c for c in features_df.columns if c not in ("subject_id", "label")]
    folds_df = load_folds(folds_csv)

    print(f"[{args.dataset}] brainiac_hippo (exploratory): {len(features_df)} subjects x {len(feature_cols)} dims")
    run_cv(features_df, folds_df, feature_cols, "brainiac_hippo", output_dir)


if __name__ == "__main__":
    main()
