"""
Consolidado cross-dataset: results_extra/summary_overall.md com tabela lado a lado
OASIS-1 x OASIS-2 por metodo, destacando achados que se replicam nos dois datasets vs
achados que so aparecem em um. Figuras (ROC comparativas + boxplot de AUC por metodo,
por dataset) em figures/.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve

from stats_utils import BASELINE_PREDICTIONS

OUTPUT_DIR = "results_extra"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("figures", exist_ok=True)


def load_dataset_comparisons(ds):
    path = f"results_extra_{ds}/comparisons_primary_exploratory.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def main():
    dfs = {ds: load_dataset_comparisons(ds) for ds in ["oasis1", "oasis2"]}
    if any(v is None for v in dfs.values()):
        missing = [ds for ds, v in dfs.items() if v is None]
        raise RuntimeError(f"Missing per-dataset consolidation for: {missing}. "
                            f"Run src/30_consolidate_dataset.py for each dataset first.")

    merged = dfs["oasis1"][["label", "method_name", "auc_new", "auc_diff", "p_value", "p_holm", "family"]].merge(
        dfs["oasis2"][["label", "method_name", "auc_new", "auc_diff", "p_value", "p_holm", "family"]],
        on=["label", "method_name"], suffixes=("_oasis1", "_oasis2"),
    )

    def replicates(row):
        sig1 = row["family_oasis1"] == "primary" and pd.notna(row["p_holm_oasis1"]) and row["p_holm_oasis1"] < 0.05
        sig2 = row["family_oasis2"] == "primary" and pd.notna(row["p_holm_oasis2"]) and row["p_holm_oasis2"] < 0.05
        same_direction = (row["auc_diff_oasis1"] > 0) == (row["auc_diff_oasis2"] > 0)
        if sig1 and sig2 and same_direction:
            return "REPLICA (significativo nos 2, mesma direcao)"
        elif same_direction and row["auc_diff_oasis1"] > 0 and row["auc_diff_oasis2"] > 0:
            return "direcao consistente (ganho nos 2), sem significancia"
        elif same_direction:
            return "direcao consistente (perda nos 2)"
        else:
            return "inconsistente entre datasets"

    merged["replication"] = merged.apply(replicates, axis=1)

    lines = ["# Consolidado cross-dataset — OASIS-1 x OASIS-2\n"]
    lines.append(f"Baseline OASIS-1: {BASELINE_PREDICTIONS['oasis1']['method']} "
                 f"({BASELINE_PREDICTIONS['oasis1']['classifier']}). "
                 f"Baseline OASIS-2: {BASELINE_PREDICTIONS['oasis2']['method']} "
                 f"({BASELINE_PREDICTIONS['oasis2']['classifier']}).\n")
    lines.append("Cada dataset e um estudo separado (sem pool entre coortes). Comparacao "
                 "primaria = novo metodo vs melhor metodo base DAQUELE dataset; p-Holm so "
                 "existe para a familia primaria (6 comparacoes por dataset).\n")

    lines.append("## Tabela lado a lado\n")
    lines.append("| Metodo | AUC diff OASIS-1 | p-Holm OASIS-1 | AUC diff OASIS-2 | p-Holm OASIS-2 | Replicacao |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in merged.sort_values("label").iterrows():
        ph1 = f"{r['p_holm_oasis1']:.4f}" if pd.notna(r["p_holm_oasis1"]) else "-"
        ph2 = f"{r['p_holm_oasis2']:.4f}" if pd.notna(r["p_holm_oasis2"]) else "-"
        lines.append(f"| {r['label']} | {r['auc_diff_oasis1']:+.3f} | {ph1} | "
                     f"{r['auc_diff_oasis2']:+.3f} | {ph2} | {r['replication']} |")

    n_replicated = (merged["replication"] == "REPLICA (significativo nos 2, mesma direcao)").sum()
    lines.append(f"\n**Achados que replicam (significativos nos dois datasets, mesma direcao): {n_replicated}/{len(merged)}**\n")

    lines.append(
        "\n## Ressalvas obrigatorias\n"
        "- Confundimento de idade nao corrigido em nenhum dos dois datasets (OASIS-1: AD mais velho, "
        "delta~+7.7 anos; OASIS-2: quase pareado, delta~-0.87 anos - direcoes opostas, o que ja e "
        "por si um resultado relevante para interpretar diferencas entre os dois baselines).\n"
        "- N moderado nos dois datasets (OASIS-1 n=235, OASIS-2 n=150) - intervalos de confianca amplos.\n"
        "- Teste multiplo: familia primaria (6 comparacoes) corrigida por Holm-Bonferroni EM CADA dataset "
        "separadamente (nao ha correcao conjunta cross-dataset - os dois sao estudos paralelos).\n"
        "- Teste 3B (BrainIAC no ROI do hipocampo) e exploratorio/OOD nos dois datasets - nao entra na "
        "familia primaria nem na conclusao de replicacao.\n"
        "- Regra do grupo 'Converted' no OASIS-2: classificado pela CDR na baseline (nao pelo status "
        "futuro) - 13 sujeitos 'Converted' com CDR=0 na baseline contam como CN.\n"
        "- OASIS-1 raw (PROCESSED/MPRAGE/SUBJ_111, ANALYZE 7.5) tinha um defeito de orientacao "
        "(qform/sform invalidos, eixos do array rotulados errado pelo affine fallback do nibabel) que "
        "quebrava a segmentacao hipocampal (FastSurfer E SynthSeg falhavam identicamente) - corrigido "
        "via reorientacao determinada empiricamente antes da segmentacao (ver PROGRESS_extra.md); "
        "OASIS-2 nao precisou de correcao (qform ja valido).\n"
    )

    with open(os.path.join(OUTPUT_DIR, "summary_overall.md"), "w") as f:
        f.write("\n".join(lines))
    merged.to_csv(os.path.join(OUTPUT_DIR, "overall_comparison_table.csv"), index=False)
    print(f"Written {OUTPUT_DIR}/summary_overall.md")
    print(merged[["label", "auc_diff_oasis1", "p_holm_oasis1", "auc_diff_oasis2", "p_holm_oasis2", "replication"]].to_string(index=False))

    make_figures()


RESULTS_DIRS = {
    "oasis1": {
        "baseline": "results", "fusion": "results_fusion_oasis1",
        "shap": "results_shap_oasis1", "hippocampus": "results_hippocampus_oasis1",
    },
    "oasis2": {
        "baseline": "results_oasis2", "fusion": "results_fusion_oasis2",
        "shap": "results_shap_oasis2", "hippocampus": "results_hippocampus_oasis2",
    },
}


def make_figures():
    for ds, dirs in RESULTS_DIRS.items():
        comp = load_dataset_comparisons(ds)
        if comp is None:
            continue

        # baseline pooled predictions
        base_method = BASELINE_PREDICTIONS[ds]["method"]
        base_clf = BASELINE_PREDICTIONS[ds]["classifier"]
        base_pred = pd.read_csv(os.path.join(dirs["baseline"], "predictions", f"{base_method}_{base_clf}.csv"))

        # gather predictions for each new method's best classifier
        pred_lookup = {"baseline": ("baseline", base_pred)}
        for _, r in comp.iterrows():
            results_dir = {
                "fusion_early": dirs["fusion"], "fusion_late": dirs["fusion"],
                "hippocampus": dirs["hippocampus"], "brainiac_hippo": dirs["hippocampus"],
                "shap_radiomics": dirs["shap"], "shap_fusion": dirs["shap"], "shap_brainiac": dirs["shap"],
            }.get(r["method_name"])
            if results_dir is None:
                continue
            from stats_utils import load_all_metrics, best_classifier_per_method
            try:
                metrics_df = load_all_metrics(results_dir)
                metrics_df = metrics_df[metrics_df["method"] == r["method_name"]]
                best = best_classifier_per_method(metrics_df)[r["method_name"]]
                pred_path = os.path.join(results_dir, "predictions", f"{r['method_name']}_{best}.csv")
                pred_lookup[r["label"]] = (r["label"], pd.read_csv(pred_path))
            except Exception as e:
                print(f"skip figure series {r['label']} ({ds}): {e}")

        plt.figure(figsize=(7, 7))
        for name, (label, df) in pred_lookup.items():
            fpr, tpr, _ = roc_curve(df["y_true"], df["y_prob"])
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(df["y_true"], df["y_prob"])
            plt.plot(fpr, tpr, label=f"{label} AUC={auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("1 - Specificity (FPR)")
        plt.ylabel("Sensitivity (TPR)")
        plt.title(f"ROC comparativa — {ds} (baseline + testes extra)")
        plt.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        plt.savefig(f"figures/extra_roc_comparison_{ds}.png", dpi=150)
        plt.close()

        box_rows = []
        for name, (label, df) in pred_lookup.items():
            if "fold" not in df.columns:
                continue
            for fold in sorted(df["fold"].unique()):
                fp = df[df["fold"] == fold]
                from sklearn.metrics import roc_auc_score
                box_rows.append({"method": label, "fold": fold, "auc": roc_auc_score(fp["y_true"], fp["y_prob"])})
        if box_rows:
            import seaborn as sns
            box_df = pd.DataFrame(box_rows)
            plt.figure(figsize=(9, 5))
            sns.boxplot(data=box_df, x="method", y="auc")
            sns.stripplot(data=box_df, x="method", y="auc", color="black", size=4, jitter=True)
            plt.xticks(rotation=30, ha="right")
            plt.ylabel("AUC-ROC (per fold)")
            plt.title(f"AUC por metodo (por fold) — {ds}")
            plt.tight_layout()
            plt.savefig(f"figures/extra_auc_boxplot_{ds}.png", dpi=150)
            plt.close()
        print(f"[{ds}] figures saved: figures/extra_roc_comparison_{ds}.png, figures/extra_auc_boxplot_{ds}.png")


if __name__ == "__main__":
    main()
