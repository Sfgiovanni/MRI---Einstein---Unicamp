"""
Teste 2 - Selecao de features via SHAP dentro da CV (sem vazamento: SHAP roda no treino
de cada fold, k escolhido por CV interna via GridSearchCV, ver src/shap_selector.py).
Roda sobre --feature_set in {radiomics, fusion, brainiac} (fusion = saida do Teste 1,
src/20_prepare_fusion_features.py; precisa rodar antes). Nos embeddings BrainIAC (768-d,
distribuidos, nao esparsos) a selecao SHAP e esperada para render menos - documentado
como tal no summary, nao escondido.
Alem da CV/DeLong, extrai estabilidade da selecao (quais features aparecem no top-k em
mais folds) - relevante para radiomics (nomes de features interpretaveis); para
fusion/brainiac tambem calculado mas menos interpretavel (colunas emb_i).
"""
import argparse
import os

import joblib
import pandas as pd

from common_cv import load_folds, run_cv
from shap_selector import build_shap_classifier_grids

FEATURE_PARQUET = {
    "radiomics": "features/{dataset}_radiomics_features.parquet",
    "fusion": "features/{dataset}_fusion_features.parquet",
    "brainiac": "features/{dataset}_brainiac_features.parquet",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="oasis1")
    parser.add_argument("--feature_set", required=True, choices=list(FEATURE_PARQUET.keys()))
    parser.add_argument("--folds_csv", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    features_parquet = FEATURE_PARQUET[args.feature_set].format(dataset=args.dataset)
    folds_csv = args.folds_csv or f"data/{args.dataset}_folds.csv"
    output_dir = args.output_dir or f"results_shap_{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)

    features_df = pd.read_parquet(features_parquet)
    feature_cols = [c for c in features_df.columns if c not in ("subject_id", "label")]
    folds_df = load_folds(folds_csv)
    method_name = f"shap_{args.feature_set}"

    print(f"[{args.dataset}/{args.feature_set}] {len(features_df)} subjects x {len(feature_cols)} dims, "
          f"SHAP-in-fold selection sweep k in {{10,20,50,100}}")
    run_cv(
        features_df, folds_df, feature_cols, method_name, output_dir,
        classifiers=["shap_logreg"],
        classifier_grids=build_shap_classifier_grids(),
    )

    # ---- selection stability across the 5 outer folds (from the saved fold models) ----
    rows = []
    k_chosen = []
    for fold in range(folds_df["fold"].nunique()):
        model_path = os.path.join(output_dir, "models", f"{method_name}_shap_logreg_fold{fold}.joblib")
        pipe = joblib.load(model_path)
        selector = pipe.named_steps["selector"]
        k_chosen.append(selector.k)
        for idx in selector.selected_idx_:
            rows.append({"fold": fold, "feature": feature_cols[idx], "k_chosen": selector.k})

    stability_df = pd.DataFrame(rows)
    stability_df.to_csv(os.path.join(output_dir, f"{args.feature_set}_selection_per_fold.csv"), index=False)

    freq = (
        stability_df.groupby("feature")["fold"].nunique().reset_index(name="n_folds_selected")
        .sort_values("n_folds_selected", ascending=False)
    )
    freq.to_csv(os.path.join(output_dir, f"{args.feature_set}_selection_stability.csv"), index=False)

    print(f"[{args.dataset}/{args.feature_set}] k chosen per fold (inner CV): {k_chosen}")
    print(f"Top-10 most stable features:\n{freq.head(10).to_string(index=False)}")


if __name__ == "__main__":
    main()
