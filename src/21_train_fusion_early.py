"""
Teste 1A - Early fusion (concatenacao) BrainIAC+radiomics vs single-modality.
768+107=875 dims vs N moderado (~150-235) infla overfitting facilmente, entao usamos
SOMENTE classificadores com controle de dimensionalidade (FUSION_CLASSIFIER_GRIDS em
common_cv.py): logreg L1, logreg L2, SVM linear, PCA->logreg (n_components escolhido
dentro do treino de cada fold via GridSearchCV). Padronizacao/PCA/regularizacao ajustadas
apenas no treino de cada fold - common_cv.run_cv ja garante isso (Pipeline por fold).
"""
import argparse
import os

import pandas as pd

from common_cv import FUSION_CLASSIFIER_GRIDS, load_folds, run_cv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="oasis1")
    parser.add_argument("--features_parquet", default=None)
    parser.add_argument("--folds_csv", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    features_parquet = args.features_parquet or f"features/{args.dataset}_fusion_features.parquet"
    folds_csv = args.folds_csv or f"data/{args.dataset}_folds.csv"
    output_dir = args.output_dir or f"results_fusion_{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)

    features_df = pd.read_parquet(features_parquet)
    feature_cols = [c for c in features_df.columns if c not in ("subject_id", "label")]
    folds_df = load_folds(folds_csv)

    print(f"[{args.dataset}] early fusion: {len(features_df)} subjects x {len(feature_cols)} dims")
    run_cv(
        features_df, folds_df, feature_cols, "fusion_early", output_dir,
        classifiers=list(FUSION_CLASSIFIER_GRIDS.keys()),
        classifier_grids=FUSION_CLASSIFIER_GRIDS,
    )


if __name__ == "__main__":
    main()
