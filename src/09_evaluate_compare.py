"""
Etapa 6 - Aggregate metrics across methods, run DeLong tests comparing BrainIAC+ML
against each baseline (using pooled cross-validated out-of-fold predictions on the
identical subject-level 5-fold split), and generate comparison figures + summary
tables.
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve

from delong import delong_roc_test

PRIMARY_METHOD = "brainiac"


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--figures_dir", default="figures")
    args = parser.parse_args()
    os.makedirs(args.figures_dir, exist_ok=True)

    metrics_df = load_all_metrics(args.results_dir)
    methods_present = sorted(metrics_df["method"].unique())
    print(f"Methods found: {methods_present}")

    best_clf = best_classifier_per_method(metrics_df)
    print("Best classifier per method (by mean CV AUC):", best_clf)

    pooled = {}
    for method, clf in best_clf.items():
        pred_path = os.path.join(args.results_dir, "predictions", f"{method}_{clf}.csv")
        pooled[method] = pd.read_csv(pred_path).sort_values("subject_id").reset_index(drop=True)

    # sanity check: identical subjects/labels pooled across methods (same fixed CV split)
    ref = pooled[PRIMARY_METHOD] if PRIMARY_METHOD in pooled else pooled[methods_present[0]]
    for method, df in pooled.items():
        assert list(df["subject_id"]) == list(ref["subject_id"]), f"subject order/set mismatch for {method}"
        assert (df["y_true"].values == ref["y_true"].values).all(), f"label mismatch for {method} - possible leakage/fold inconsistency"

    # ---- aggregated summary table (mean +/- std across folds), all classifiers ----
    summary_rows = []
    for (method, clf), g in metrics_df.groupby(["method", "classifier"]):
        row = {"method": method, "classifier": clf, "is_best_for_method": best_clf.get(method) == clf}
        for metric in ["auc", "balanced_accuracy", "f1", "sensitivity", "specificity"]:
            row[f"{metric}_mean"] = g[metric].mean()
            row[f"{metric}_std"] = g[metric].std()
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows).sort_values(["method", "auc_mean"], ascending=[True, False])
    summary_df.to_csv(os.path.join(args.results_dir, "summary.csv"), index=False)

    # ---- DeLong tests: primary method vs each baseline (best classifier each) ----
    delong_rows = []
    if PRIMARY_METHOD in pooled:
        for method in methods_present:
            if method == PRIMARY_METHOD:
                continue
            a = pooled[PRIMARY_METHOD]
            b = pooled[method]
            auc_a, auc_b, z, p, ci = delong_roc_test(a["y_true"].values, a["y_prob"].values, b["y_prob"].values)
            delong_rows.append({
                "comparison": f"{PRIMARY_METHOD} ({best_clf[PRIMARY_METHOD]}) vs {method} ({best_clf[method]})",
                "auc_a": auc_a, "auc_b": auc_b, "auc_diff": auc_a - auc_b,
                "z": z, "p_value": p, "ci95_diff_low": ci[0], "ci95_diff_high": ci[1],
            })
    delong_df = pd.DataFrame(delong_rows)
    delong_df.to_csv(os.path.join(args.results_dir, "delong_tests.csv"), index=False)

    # ---- Figure: overlaid ROC curves (pooled out-of-fold predictions) ----
    plt.figure(figsize=(6, 6))
    for method, df in pooled.items():
        fpr, tpr, _ = roc_curve(df["y_true"], df["y_prob"])
        auc = summary_df[(summary_df["method"] == method) & (summary_df["is_best_for_method"])]["auc_mean"].values[0]
        plt.plot(fpr, tpr, label=f"{method} ({best_clf[method]}) AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("1 - Specificity (FPR)")
    plt.ylabel("Sensitivity (TPR)")
    plt.title("AD vs CN classification - ROC (pooled 5-fold CV predictions)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.figures_dir, "roc_comparison.png"), dpi=150)
    plt.close()

    # ---- Figure: boxplot of per-fold AUC per method (using each method's best classifier) ----
    box_rows = []
    for method, clf in best_clf.items():
        sub = metrics_df[(metrics_df["method"] == method) & (metrics_df["classifier"] == clf)]
        for _, r in sub.iterrows():
            box_rows.append({"method": method, "fold": r["fold"], "auc": r["auc"]})
    box_df = pd.DataFrame(box_rows)
    plt.figure(figsize=(6, 5))
    sns.boxplot(data=box_df, x="method", y="auc")
    sns.stripplot(data=box_df, x="method", y="auc", color="black", size=4, jitter=True)
    plt.ylabel("AUC-ROC (per fold)")
    plt.title("Per-fold AUC by method (best classifier each)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.figures_dir, "auc_boxplot.png"), dpi=150)
    plt.close()

    # ---- results/summary.md ----
    lines = ["# Resultado comparativo: BrainIAC+ML vs baselines\n"]
    lines.append("## Melhor classificador por metodo (por AUC medio de CV)\n")
    lines.append("| Metodo | Classificador | AUC (media +/- dp) | Bal. Acc. | F1 | Sensibilidade | Especificidade |")
    lines.append("|---|---|---|---|---|---|---|")
    for method, clf in best_clf.items():
        r = summary_df[(summary_df["method"] == method) & (summary_df["classifier"] == clf)].iloc[0]
        lines.append(
            f"| {method} | {clf} | {r['auc_mean']:.3f} +/- {r['auc_std']:.3f} | "
            f"{r['balanced_accuracy_mean']:.3f} +/- {r['balanced_accuracy_std']:.3f} | "
            f"{r['f1_mean']:.3f} +/- {r['f1_std']:.3f} | "
            f"{r['sensitivity_mean']:.3f} +/- {r['sensitivity_std']:.3f} | "
            f"{r['specificity_mean']:.3f} +/- {r['specificity_std']:.3f} |"
        )
    lines.append(
        "\n**Nota:** os AUCs A/B na tabela de DeLong abaixo sao calculados sobre as predicoes "
        "*pooled* (todas as 5 fold-predictions concatenadas), por isso diferem ligeiramente do "
        "'AUC (media +/- dp)' da tabela acima, que e a media das 5 AUCs calculadas fold-a-fold. "
        "Sao duas quantidades validas mas distintas do mesmo conjunto de predicoes - nao e um erro.\n"
    )
    lines.append("## Comparacao estatistica (teste de DeLong, predicoes pooled de CV, pareadas por sujeito)\n")
    lines.append("| Comparacao | AUC A | AUC B | Diferenca | z | p | IC95% diferenca |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in delong_df.iterrows():
        lines.append(
            f"| {r['comparison']} | {r['auc_a']:.3f} | {r['auc_b']:.3f} | {r['auc_diff']:.3f} | "
            f"{r['z']:.3f} | {r['p_value']:.4f} | [{r['ci95_diff_low']:.3f}, {r['ci95_diff_high']:.3f}] |"
        )
    lines.append("\n## Tabela completa (todos os classificadores)\n")
    lines.append(summary_df.to_markdown(index=False))
    lines.append(
        "\n## Limitacoes\n"
        "- Dataset: OASIS-1 (nao OpenNeuro - ver PROGRESS.md para justificativa da mudanca).\n"
        "- Rotulo binario definido por CDR (Clinical Dementia Rating), nao por biomarcador molecular "
        "(ex. PET amiloide/tau) - reflete diagnostico clinico de demencia, nao confirmacao patologica de AD.\n"
        "- Confundimento de idade entre grupos (CN media ~69 anos vs AD media ~76.8 anos) nao corrigido "
        "estatisticamente neste pipeline.\n"
        "- Teste de DeLong aplicado a predicoes pooled de validacao cruzada (nao a um unico modelo fixo) - "
        "aproximacao pragmatica amplamente usada, mas tecnicamente as predicoes nao sao i.i.d. de um unico classificador.\n"
        "- N moderado (235 sujeitos, 135 CN / 100 AD) - intervalos de confianca amplos esperados.\n"
        "- O 'melhor classificador' de cada metodo foi selecionado pela mesma AUC de CV que e depois "
        "reportada e comparada via DeLong (selection-on-the-test-metric). Isso se aplica igualmente aos "
        "3 metodos; como a conclusao principal e 'nenhuma diferenca significativa', esse viés de selecao "
        "tende a favorecer a deteccao de diferencas, nao a escondê-las - ou seja, joga contra, nao a favor, "
        "da conclusao de nao-superioridade do BrainIAC.\n"
        "- QC de pre-processamento (ver `results/qc_preprocessing.csv` e `src/10_qc_preprocessing.py`): "
        "fracao de voxels nao-zero (mascara cerebral pos skull-strip) tem distribuicao estreita e consistente "
        "entre os 235 sujeitos (media 0.230, dp 0.021, min 0.185, max 0.296, mesmo shape em todos) - sem "
        "evidencia de skull-strips falhos que pudessem explicar artificialmente o desempenho do BrainIAC/radiomics "
        "por inputs corrompidos (a volumetria usa um pipeline de processamento totalmente independente do OASIS, "
        "entao essa checagem descarta uma assimetria de qualidade de dados entre os metodos).\n"
    )
    with open(os.path.join(args.results_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines))

    print("\n=== SUMMARY ===")
    print(summary_df.to_string(index=False))
    print("\n=== DELONG ===")
    print(delong_df.to_string(index=False))


if __name__ == "__main__":
    main()
