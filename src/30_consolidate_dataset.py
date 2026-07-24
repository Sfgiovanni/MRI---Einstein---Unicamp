"""
Consolidacao por dataset: junta baseline + hipocampo (3A/3B) + fusao (1A/1B) + SHAP (2),
marcando primario vs exploratorio, com DeLong (p bruto e p-Holm) vs o melhor metodo base
daquele dataset. results_extra_{dataset}/summary.md.

Familia primaria (correcao de Holm-Bonferroni aplicada so a esta familia, por dataset):
cada um dos 6 "novos metodos" (fusion_early, fusion_late, hippocampus[3A], shap_radiomics,
shap_fusion, shap_brainiac) vs o melhor metodo base do dataset. hippocampus_brainiac[3B]
e as comparacoes "selecionado vs sem selecao" do SHAP sao exploratorias (nao entram na
correcao de Holm).
"""
import argparse
import os

import pandas as pd

from stats_utils import best_classifier_per_method, delong_vs_baseline, holm_correct, load_all_metrics, \
    load_baseline_predictions, resolve_results_dir

# (results_dir, method_name_in_metrics_csv, display_label, is_primary)
NEW_METHODS = [
    ("results_fusion_{ds}", "fusion_early", "Fusao early (BrainIAC+radiomics concat)", True),
    ("results_fusion_{ds}", "fusion_late", "Fusao late (stacking BrainIAC+radiomics)", True),
    ("results_hippocampus_{ds}", "hippocampus", "Volume hipocampal (SynthSeg, 3A)", True),
    ("results_hippocampus_{ds}", "brainiac_hippo", "BrainIAC no ROI do hipocampo (3B, exploratorio/OOD)", False),
    ("results_shap_{ds}", "shap_radiomics", "Radiomics + selecao SHAP", True),
    ("results_shap_{ds}", "shap_fusion", "Fusao + selecao SHAP", True),
    ("results_shap_{ds}", "shap_brainiac", "BrainIAC 768d + selecao SHAP", True),
]

# exploratory: shap-selected vs the SAME feature set without selection (not vs baseline)
SHAP_VS_NOSELECTION = [
    ("shap_radiomics", "results_shap_{ds}", "radiomics", "{base}", "Radiomics: SHAP-selecionado vs sem selecao"),
    ("shap_fusion", "results_shap_{ds}", "fusion_early", "results_fusion_{ds}", "Fusao: SHAP-selecionado vs sem selecao"),
    ("shap_brainiac", "results_shap_{ds}", "brainiac", "{base}", "BrainIAC 768d: SHAP-selecionado vs sem selecao"),
]


def get_best_pred(results_dir, method_name):
    metrics_df = load_all_metrics(results_dir)
    metrics_df = metrics_df[metrics_df["method"] == method_name]
    best = best_classifier_per_method(metrics_df)[method_name]
    pred_path = os.path.join(results_dir, "predictions", f"{method_name}_{best}.csv")
    return best, pred_path


def delong_between(pred_csv_a, label_a, pred_csv_b, label_b):
    from delong import delong_roc_test
    a = pd.read_csv(pred_csv_a).sort_values("subject_id").reset_index(drop=True)
    b = pd.read_csv(pred_csv_b).sort_values("subject_id").reset_index(drop=True)
    merged = a.merge(b, on="subject_id", suffixes=("_a", "_b"))
    assert len(merged) == len(a) == len(b), f"subject mismatch comparing {label_a} vs {label_b}"
    assert (merged["y_true_a"].values == merged["y_true_b"].values).all()
    auc_a, auc_b, z, p, ci = delong_roc_test(merged["y_true_a"].values, merged["y_prob_a"].values, merged["y_prob_b"].values)
    return {"comparison": f"{label_a} vs {label_b}", "auc_a": auc_a, "auc_b": auc_b,
            "auc_diff": auc_a - auc_b, "z": z, "p_value": p, "ci95_diff_low": ci[0], "ci95_diff_high": ci[1]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    ds = args.dataset
    base_dir = resolve_results_dir(ds)
    output_dir = f"results_extra_{ds}"
    os.makedirs(output_dir, exist_ok=True)

    baseline_method, baseline_clf, _ = load_baseline_predictions(ds, results_dir=base_dir)

    rows = []
    primary_mask = []
    best_clf_used = {}
    for results_dir_tmpl, method_name, label, is_primary in NEW_METHODS:
        results_dir = results_dir_tmpl.format(ds=ds)
        try:
            best_clf, pred_path = get_best_pred(results_dir, method_name)
        except Exception as e:
            print(f"SKIP {method_name} ({results_dir}): {e}")
            continue
        best_clf_used[method_name] = (results_dir, best_clf)
        cmp = delong_vs_baseline(ds, pred_path, f"{label} ({best_clf})")
        cmp["method_name"] = method_name
        cmp["label"] = label
        rows.append(cmp)
        primary_mask.append(is_primary)

    comparisons_df = pd.DataFrame(rows)
    primary_mask_s = pd.Series(primary_mask, index=comparisons_df.index)
    comparisons_df = holm_correct(comparisons_df, primary_mask=primary_mask_s)

    # ---- exploratory: shap-selected vs no-selection (same feature set) ----
    explore_rows = []
    for shap_method, shap_dir_tmpl, base_method_name, base_dir_tmpl, label in SHAP_VS_NOSELECTION:
        shap_dir = shap_dir_tmpl.format(ds=ds)
        base_dir_resolved = base_dir_tmpl.format(ds=ds, base=base_dir)
        try:
            shap_clf, shap_pred = get_best_pred(shap_dir, shap_method)
            noselect_clf, noselect_pred = get_best_pred(base_dir_resolved, base_method_name)
            cmp = delong_between(shap_pred, f"{label.split(':')[0]} SHAP ({shap_clf})",
                                  noselect_pred, f"sem selecao ({noselect_clf})")
            cmp["label"] = label
            explore_rows.append(cmp)
        except Exception as e:
            print(f"SKIP exploratory {label}: {e}")
    explore_df = pd.DataFrame(explore_rows)

    # ---- write summary.md ----
    lines = [f"# Resultados extra ({ds}) — fusao, SHAP, hipocampo\n"]
    lines.append(f"Baseline de referencia (melhor metodo base deste dataset): "
                 f"**{baseline_method} ({baseline_clf})** — ver `{base_dir}/summary.md`.\n")

    lines.append("## Comparacoes primarias (familia com correcao de Holm-Bonferroni)\n")
    lines.append("| Metodo novo | Classificador | AUC novo | AUC baseline | Diferenca | z | p bruto | p-Holm | IC95% diff |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in comparisons_df[comparisons_df["family"] == "primary"].iterrows():
        p_holm = f"{r['p_holm']:.4f}" if pd.notna(r["p_holm"]) else "-"
        clf = best_clf_used.get(r["method_name"], (None, "?"))[1]
        lines.append(f"| {r['label']} | {clf} | {r['auc_new']:.3f} | {r['auc_baseline']:.3f} | "
                     f"{r['auc_diff']:.3f} | {r['z']:.3f} | {r['p_value']:.4f} | {p_holm} | "
                     f"[{r['ci95_diff_low']:.3f}, {r['ci95_diff_high']:.3f}] |")

    lines.append("\n## Comparacoes exploratorias (sem correcao de Holm, p bruto apenas)\n")
    lines.append("### vs melhor metodo base do dataset\n")
    lines.append("| Metodo novo | Classificador | AUC novo | AUC baseline | Diferenca | z | p bruto | IC95% diff |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in comparisons_df[comparisons_df["family"] == "exploratory"].iterrows():
        clf = best_clf_used.get(r["method_name"], (None, "?"))[1]
        lines.append(f"| {r['label']} | {clf} | {r['auc_new']:.3f} | {r['auc_baseline']:.3f} | "
                     f"{r['auc_diff']:.3f} | {r['z']:.3f} | {r['p_value']:.4f} | "
                     f"[{r['ci95_diff_low']:.3f}, {r['ci95_diff_high']:.3f}] |")

    if len(explore_df):
        lines.append("\n### SHAP-selecionado vs mesma feature set sem selecao\n")
        lines.append("| Comparacao | AUC A | AUC B | Diferenca | z | p bruto | IC95% diff |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in explore_df.iterrows():
            lines.append(f"| {r['comparison']} | {r['auc_a']:.3f} | {r['auc_b']:.3f} | {r['auc_diff']:.3f} | "
                         f"{r['z']:.3f} | {r['p_value']:.4f} | [{r['ci95_diff_low']:.3f}, {r['ci95_diff_high']:.3f}] |")

    lines.append(
        "\n## Limitacoes (herdadas do baseline + especificas destes testes)\n"
        "- Confundimento de idade nao corrigido estatisticamente (ver baseline deste dataset).\n"
        "- N moderado - intervalos de confianca amplos esperados.\n"
        "- Teste multiplo: familia primaria (6 comparacoes/dataset) corrigida por Holm-Bonferroni; "
        "demais comparacoes sao exploratorias e rotuladas como tal, sem correcao.\n"
        "- Teste 3B (BrainIAC no ROI do hipocampo) e out-of-distribution para o encoder "
        "(campo de visao muito menor que o cerebro inteiro, e em espaco nativo nao registrado) - "
        "tratado como exploratorio, nao como comparacao primaria.\n"
    )
    if ds == "oasis2":
        lines.append(
            "- Grupo 'Converted' no OASIS-2 classificado pela regra de CDR na baseline (nao pelo "
            "status futuro) - ver src/01_prepare_dataset.py (funcao prepare_oasis2) para a regra completa.\n"
        )

    with open(os.path.join(output_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines))

    comparisons_df.to_csv(os.path.join(output_dir, "comparisons_primary_exploratory.csv"), index=False)
    if len(explore_df):
        explore_df.to_csv(os.path.join(output_dir, "comparisons_shap_vs_noselection.csv"), index=False)

    print(f"[{ds}] consolidation written to {output_dir}/summary.md")
    print(comparisons_df[["label", "auc_new", "auc_baseline", "auc_diff", "p_value", "p_holm", "family"]].to_string(index=False))


if __name__ == "__main__":
    main()
