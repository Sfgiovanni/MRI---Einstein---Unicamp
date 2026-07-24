"""
Consolidado cross-dataset: results_extra/summary_overall.md com tabela lado a lado de
todos os datasets passados em --datasets (2 ou mais, default oasis1 oasis2) por metodo,
destacando achados que se replicam em todos os datasets vs achados que so aparecem em
alguns. Figuras (ROC comparativas + boxplot de AUC por metodo, por dataset) em figures/.

Cada dataset em --datasets precisa ja ter passado por src/30_consolidate_dataset.py.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from stats_utils import best_classifier_per_method, load_all_metrics, load_baseline_predictions, resolve_results_dir

OUTPUT_DIR = "results_extra"


def load_dataset_comparisons(ds):
    path = f"results_extra_{ds}/comparisons_primary_exploratory.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def classify_replication(group):
    sig = (group["family"] == "primary") & group["p_holm"].notna() & (group["p_holm"] < 0.05)
    same_direction = (group["auc_diff"] > 0).nunique() == 1
    n = len(group)
    if same_direction and sig.all():
        return f"REPLICA (significativo nos {n}, mesma direcao)"
    elif same_direction and (group["auc_diff"] > 0).all():
        return f"direcao consistente (ganho nos {n}), sem significancia"
    elif same_direction:
        return f"direcao consistente (perda nos {n})"
    else:
        return "inconsistente entre datasets"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["oasis1", "oasis2"],
                         help="Nomes dos datasets a consolidar lado a lado (2+; cada um ja "
                              "processado por src/30_consolidate_dataset.py).")
    args = parser.parse_args()
    datasets = args.datasets
    if len(datasets) < 2:
        raise SystemExit("--datasets precisa de pelo menos 2 datasets para uma comparacao lado a lado.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    dfs = {ds: load_dataset_comparisons(ds) for ds in datasets}
    missing = [ds for ds, v in dfs.items() if v is None]
    if missing:
        raise RuntimeError(f"Faltando consolidacao por dataset para: {missing}. "
                            f"Rode src/30_consolidate_dataset.py --dataset {{ds}} para cada um primeiro.")

    long_rows = []
    for ds, df in dfs.items():
        d = df[["label", "method_name", "auc_new", "auc_diff", "p_value", "p_holm", "family"]].copy()
        d["dataset"] = ds
        long_rows.append(d)
    long_df = pd.concat(long_rows, ignore_index=True)

    replication = (
        long_df.groupby(["label", "method_name"])
        .apply(classify_replication, include_groups=False)
        .rename("replication").reset_index()
    )

    baseline_info = {ds: load_baseline_predictions(ds, results_dir=resolve_results_dir(ds))[:2] for ds in datasets}

    lines = [f"# Consolidado cross-dataset — {' x '.join(datasets)}\n"]
    for ds in datasets:
        method, clf = baseline_info[ds]
        lines.append(f"Baseline {ds}: {method} ({clf}).")
    lines.append("")
    lines.append("Cada dataset e um estudo separado (sem pool entre coortes). Comparacao "
                 "primaria = novo metodo vs melhor metodo base DAQUELE dataset; p-Holm so "
                 f"existe para a familia primaria (comparacoes primarias por dataset, ver "
                 f"`results_extra_{{dataset}}/summary.md`).\n")

    lines.append("## Tabela lado a lado\n")
    header = "| Metodo | " + " | ".join(f"AUC diff {ds} | p-Holm {ds}" for ds in datasets) + " | Replicacao |"
    lines.append(header)
    lines.append("|" + "---|" * (2 * len(datasets) + 2))

    labels = long_df[["label", "method_name"]].drop_duplicates().sort_values("label")
    for _, lm in labels.iterrows():
        row_cells = [lm["label"]]
        for ds in datasets:
            sub = long_df[(long_df["label"] == lm["label"]) & (long_df["dataset"] == ds)]
            if sub.empty:
                row_cells.append("-"); row_cells.append("-")
                continue
            r = sub.iloc[0]
            ph = f"{r['p_holm']:.4f}" if pd.notna(r["p_holm"]) else "-"
            row_cells.append(f"{r['auc_diff']:+.3f}")
            row_cells.append(ph)
        rep = replication[(replication["label"] == lm["label"]) & (replication["method_name"] == lm["method_name"])]
        row_cells.append(rep["replication"].iloc[0] if len(rep) else "-")
        lines.append("| " + " | ".join(row_cells) + " |")

    n_replicated = replication["replication"].str.startswith("REPLICA").sum()
    lines.append(f"\n**Achados que replicam (significativos em todos os {len(datasets)} datasets, "
                 f"mesma direcao): {n_replicated}/{len(replication)}**\n")

    lines.append(
        "\n## Ressalvas obrigatorias\n"
        "- Cada dataset e tratado como estudo independente - sem pool entre coortes.\n"
        "- N moderado - intervalos de confianca amplos esperados em todos os datasets.\n"
        "- Teste multiplo: familia primaria corrigida por Holm-Bonferroni EM CADA dataset "
        "separadamente (nao ha correcao conjunta cross-dataset).\n"
        "- Testes marcados como exploratorios/OOD (ex.: BrainIAC no ROI do hipocampo) nao entram "
        "na familia primaria nem na conclusao de replicacao.\n"
        "- Ver `results_extra_{dataset}/summary.md` de cada dataset para ressalvas especificas "
        "(confundimento de idade, regras de rotulagem, QC de segmentacao, etc.).\n"
    )

    with open(os.path.join(OUTPUT_DIR, "summary_overall.md"), "w") as f:
        f.write("\n".join(lines))

    wide = long_df.pivot(index=["label", "method_name"], columns="dataset",
                          values=["auc_diff", "p_holm"]).reset_index()
    wide.columns = ["_".join(c).strip("_") for c in wide.columns.to_flat_index()]
    wide = wide.merge(replication, on=["label", "method_name"])
    wide.to_csv(os.path.join(OUTPUT_DIR, "overall_comparison_table.csv"), index=False)

    print(f"Written {OUTPUT_DIR}/summary_overall.md")
    print(wide.to_string(index=False))

    make_figures(datasets)


def make_figures(datasets):
    for ds in datasets:
        comp = load_dataset_comparisons(ds)
        if comp is None:
            continue
        base_dir = resolve_results_dir(ds)
        base_method, base_clf, base_pred = load_baseline_predictions(ds, results_dir=base_dir)

        results_dir_lookup = {
            "fusion_early": f"results_fusion_{ds}", "fusion_late": f"results_fusion_{ds}",
            "hippocampus": f"results_hippocampus_{ds}", "brainiac_hippo": f"results_hippocampus_{ds}",
            "shap_radiomics": f"results_shap_{ds}", "shap_fusion": f"results_shap_{ds}", "shap_brainiac": f"results_shap_{ds}",
        }

        pred_lookup = {"baseline": ("baseline", base_pred)}
        for _, r in comp.iterrows():
            results_dir = results_dir_lookup.get(r["method_name"])
            if results_dir is None:
                continue
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
