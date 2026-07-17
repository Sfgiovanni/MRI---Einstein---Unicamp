"""
Etapa 8 - Final consolidated comparison: intra-domain OASIS-1, intra-domain OASIS-2,
and both cross-dataset directions, per method. Generates the headline table, ROC/boxplot
figures, and results_cross/summary.md with the mandatory caveats.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve


def best_classifier_auc(metrics_csv, method):
    df = pd.read_csv(metrics_csv)
    df = df[df["method"] == method]
    agg = df.groupby("classifier")["auc"].agg(["mean", "std"])
    best = agg["mean"].idxmax()
    return best, agg.loc[best, "mean"], agg.loc[best, "std"]


# For the OASIS1->OASIS2 gap specifically, "intra_oasis1_auc" (SUBJ_111, canonical) is NOT
# used - it would conflate "cohort changed" with "input pipeline changed" (cross-dataset
# training used oasis1_mpr1, the input-level-matched OASIS-1 variant, not SUBJ_111). Instead
# we use the intra-domain AUC computed on THE SAME oasis1_mpr1 pipeline (same subjects/folds,
# via `05_train_classifiers_brainiac.py --dataset oasis1_mpr1 --folds_csv data/oasis1_folds.csv
# --output_dir results_oasis1_mpr1`, and the radiomics equivalent), so the gap isolates the
# cohort-transfer effect. Volumetry has no oasis1_mpr1 variant (features don't depend on our
# image pipeline), so it keeps using the canonical oasis1_results source.
GAP_INTRA_SOURCE = {"brainiac": "results_oasis1_mpr1", "radiomics": "results_oasis1_mpr1"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=["brainiac", "radiomics", "volumetry"])
    parser.add_argument("--oasis1_results", default="results")
    parser.add_argument("--oasis2_results", default="results_oasis2")
    parser.add_argument("--cross_results", default="results_cross")
    parser.add_argument("--figures_dir", default="figures")
    args = parser.parse_args()
    os.makedirs(args.figures_dir, exist_ok=True)

    cross_metrics = pd.read_csv(os.path.join(args.cross_results, "metrics_cross_dataset.csv"))

    rows = []
    for method in args.methods:
        clf1, auc1, std1 = best_classifier_auc(os.path.join(args.oasis1_results, f"metrics_{method}.csv"), method)
        clf2, auc2, std2 = best_classifier_auc(os.path.join(args.oasis2_results, f"metrics_{method}.csv"), method)

        gap_intra_dir = GAP_INTRA_SOURCE.get(method, args.oasis1_results)
        clf1_gap, auc1_gap, std1_gap = best_classifier_auc(os.path.join(gap_intra_dir, f"metrics_{method}.csv"), method)

        cm = cross_metrics[(cross_metrics["method"] == method)]
        o1o2 = cm[(cm["direction"] == "oasis1_to_oasis2") & (cm["classifier"] == clf1_gap)]
        o2o1 = cm[(cm["direction"] == "oasis2_to_oasis1") & (cm["classifier"] == clf2)]

        auc_o1o2 = o1o2["auc"].values[0] if len(o1o2) else None
        auc_o2o1 = o2o1["auc"].values[0] if len(o2o1) else None

        rows.append({
            "method": method,
            "intra_oasis1_classifier": clf1, "intra_oasis1_auc": auc1, "intra_oasis1_std": std1,
            "intra_oasis2_classifier": clf2, "intra_oasis2_auc": auc2, "intra_oasis2_std": std2,
            "intra_oasis1_gap_source": gap_intra_dir,
            "intra_oasis1_gap_classifier": clf1_gap, "intra_oasis1_gap_auc": auc1_gap,
            "cross_oasis1_to_oasis2_auc": auc_o1o2,
            "cross_oasis2_to_oasis1_auc": auc_o2o1,
            "gap_oasis1_to_oasis2": (auc1_gap - auc_o1o2) if auc_o1o2 is not None else None,
            "gap_oasis2_to_oasis1": (auc2 - auc_o2o1) if auc_o2o1 is not None else None,
        })

    table = pd.DataFrame(rows)
    table["mean_gap"] = table[["gap_oasis1_to_oasis2", "gap_oasis2_to_oasis1"]].mean(axis=1)
    table.to_csv(os.path.join(args.cross_results, "final_comparison.csv"), index=False)

    # ---- Figure: boxplot of AUC across the 4 conditions per method ----
    box_rows = []
    for _, r in table.iterrows():
        box_rows.append({"method": r["method"], "condition": "intra-OASIS1", "auc": r["intra_oasis1_auc"]})
        box_rows.append({"method": r["method"], "condition": "intra-OASIS2", "auc": r["intra_oasis2_auc"]})
        box_rows.append({"method": r["method"], "condition": "OASIS1->OASIS2", "auc": r["cross_oasis1_to_oasis2_auc"]})
        box_rows.append({"method": r["method"], "condition": "OASIS2->OASIS1", "auc": r["cross_oasis2_to_oasis1_auc"]})
    box_df = pd.DataFrame(box_rows)
    plt.figure(figsize=(9, 5))
    sns.barplot(data=box_df, x="method", y="auc", hue="condition")
    plt.ylim(0.4, 1.0)
    plt.axhline(0.5, linestyle="--", color="gray", linewidth=1)
    plt.title("AUC: intra-domain vs cross-dataset, by method")
    plt.tight_layout()
    plt.savefig(os.path.join(args.figures_dir, "cross_dataset_auc_comparison.png"), dpi=150)
    plt.close()

    # ---- Figure: ROC curves for both cross-dataset directions, one panel per method ----
    fig, axes = plt.subplots(1, len(args.methods), figsize=(6 * len(args.methods), 5))
    if len(args.methods) == 1:
        axes = [axes]
    for ax, method in zip(axes, args.methods):
        gap_intra_dir = GAP_INTRA_SOURCE.get(method, args.oasis1_results)
        clf1, _, _ = best_classifier_auc(os.path.join(gap_intra_dir, f"metrics_{method}.csv"), method)
        clf2, _, _ = best_classifier_auc(os.path.join(args.oasis2_results, f"metrics_{method}.csv"), method)
        for direction, clf, label in [
            ("oasis1_to_oasis2", clf1, "OASIS1->OASIS2"),
            ("oasis2_to_oasis1", clf2, "OASIS2->OASIS1"),
        ]:
            pred_path = os.path.join(args.cross_results, "predictions", f"{method}_{clf}_{direction}.csv")
            if not os.path.exists(pred_path):
                continue
            preds = pd.read_csv(pred_path)
            fpr, tpr, _ = roc_curve(preds["y_true"], preds["y_prob"])
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(preds["y_true"], preds["y_prob"])
            ax.plot(fpr, tpr, label=f"{label} ({clf}) AUC={auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_title(method)
        ax.set_xlabel("1 - Specificity")
        ax.set_ylabel("Sensitivity")
        ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(args.figures_dir, "cross_dataset_roc.png"), dpi=150)
    plt.close()

    # ---- results_cross/summary.md ----
    lines = ["# Comparacao final: intra-dominio vs validacao cruzada entre coortes (OASIS-1 <-> OASIS-2)\n"]
    lines.append(
        "| Metodo | Intra OASIS-1 (canonico, SUBJ_111) | Intra OASIS-1 (usado no gap, mpr-1) | "
        "Intra OASIS-2 (AUC) | OASIS1->OASIS2 (AUC) | OASIS2->OASIS1 (AUC) | Gap OASIS1->2 | Gap OASIS2->1 | Gap medio |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in table.iterrows():
        lines.append(
            f"| {r['method']} | {r['intra_oasis1_auc']:.3f} ({r['intra_oasis1_classifier']}) | "
            f"{r['intra_oasis1_gap_auc']:.3f} ({r['intra_oasis1_gap_classifier']}) | "
            f"{r['intra_oasis2_auc']:.3f} ({r['intra_oasis2_classifier']}) | "
            f"{r['cross_oasis1_to_oasis2_auc']:.3f} | {r['cross_oasis2_to_oasis1_auc']:.3f} | "
            f"{r['gap_oasis1_to_oasis2']:.3f} | {r['gap_oasis2_to_oasis1']:.3f} | {r['mean_gap']:.3f} |"
        )

    lines.append(
        "\n**Nota metodologica sobre o calculo do gap (importante):** a coluna 'Intra OASIS-1' "
        "acima e o AUC canonico do OASIS-1 (via SUBJ_111, o mesmo numero reportado na secao "
        "intra-dominio do OASIS-1). Para BrainIAC e radiomics, porem, o gap de generalizacao "
        "OASIS1->OASIS2 NAO usa esse numero como referencia - ele usaria um AUC intra-dominio "
        "calculado com um pipeline de entrada diferente (SUBJ_111) do usado no lado OASIS-1 do "
        "treino cross-dataset (mpr-1), o que misturaria 'mudanca de coorte' com 'mudanca de "
        "pipeline' no mesmo numero. Em vez disso, o gap usa um AUC intra-dominio calculado nos "
        "MESMOS sujeitos/folds do OASIS-1 mas com o pipeline mpr-1 pareado "
        "(`results_oasis1_mpr1/metrics_*.csv`): 0.707 (logreg) para BrainIAC e 0.734 (svm_rbf) "
        "para radiomics - ambos mais baixos que os numeros canonicos SUBJ_111 (0.750 e 0.748), "
        "porque a repeticao unica bruta (mpr-1) e mais ruidosa que a media motion-corrected "
        "oficial do OASIS. Sem essa correcao, o gap OASIS1->OASIS2 do BrainIAC estaria "
        "superestimado em quase 2x (0.750-0.639=0.111 vs o valor correto 0.707-0.639=0.069). A "
        "direcao OASIS2->OASIS1 nao precisa dessa correcao (o lado OASIS-2 do experimento nao foi "
        "reprocessado). Volumetria tambem nao precisa (nao depende do nosso pipeline de imagem).\n"
    )

    lines.append(
        "\n## Diagnostico de domain shift (dataset-membership) - LER ANTES da leitura-chave\n"
        "Antes de interpretar qualquer gap de generalizacao, testamos algo que o desenho original "
        "do experimento nao previa: treinar um classificador para prever apenas **de qual dataset** "
        "(OASIS-1 ou OASIS-2) um sujeito veio, ignorando o rotulo AD/CN. Um AUC alto aqui significa "
        "que o espaco de features e dominado por um efeito de lote (batch effect), o que limita o "
        "quanto do AUC cross-dataset pode ser atribuido a sinal de doenca transferido.\n\n"
        "| Comparacao | Metodo | AUC de dataset-membership |\n"
        "|---|---|---|\n"
        "| OASIS-1 (SUBJ_111) vs OASIS-2 (mpr-1) | brainiac | 1.000 |\n"
        "| OASIS-1 (SUBJ_111) vs OASIS-2 (mpr-1) | radiomics | 1.000 |\n"
        "| OASIS-1 (SUBJ_111) vs OASIS-2 (mpr-1) | volumetria | 0.580 |\n"
        "| OASIS-1 mpr-1 (nivel de entrada pareado) vs OASIS-2 (mpr-1) | brainiac | 1.000 |\n"
        "| OASIS-1 mpr-1 (nivel de entrada pareado) vs OASIS-2 (mpr-1) | radiomics | 0.9997 |\n\n"
        "**O que isso significa:** embeddings do BrainIAC e features de radiomics separam os dois "
        "datasets com AUC praticamente perfeito (1.0) - ou seja, um efeito de lote domina "
        "completamente esses espacos de features. A hipotese inicial era que isso fosse causado "
        "pela assimetria de nivel de processamento usada entre os dois datasets (OASIS-1 usava "
        "SUBJ_111 - ja processado pelo pipeline do proprio OASIS com correcao de movimento, media "
        "entre repeticoes e correcao N4 - enquanto OASIS-2 usava mpr-1 bruto, sem esse "
        "pre-processamento). **Reprocessamos o OASIS-1 inteiro a partir do mpr-1 bruto (mesmo "
        "nivel de entrada do OASIS-2, mesmo pipeline de registro+N4+HD-BET do BrainIAC aplicado "
        "depois) para testar essa hipotese - e a separabilidade continuou em ~1.0.** Isso "
        "**refuta** a hipotese de que a assimetria de pre-processamento era a causa principal: o "
        "efeito de lote parece ser intrinseco as duas coortes (diferencas de scanner/protocolo/era "
        "de aquisicao entre OASIS-1 e OASIS-2, coletados em anos diferentes pela mesma "
        "instituicao), nao um artefato introduzido pela nossa escolha de pipeline. A volumetria "
        "(eTIV/nWBV/ASF), que vem de estatisticas computadas pelo proprio OASIS e nunca passa pelo "
        "nosso pre-processamento de imagem, tem separabilidade muito menor (AUC=0.58) - por isso "
        "seus resultados cross-dataset sao mais interpretaveis diretamente.\n\n"
        "**Consequencia pratica:** os AUCs cross-dataset abaixo para BrainIAC e radiomics foram "
        "recalculados usando a versao de OASIS-1 com nivel de entrada pareado (mpr-1), o que reduz "
        "(mas nao elimina) o problema e mudou os numeros de forma nao-trivial em relacao a uma "
        "primeira tentativa com SUBJ_111 (documentado em PROGRESS_dataset2.md). Mesmo assim, dado "
        "que a separabilidade por dataset continua em ~1.0, **nao podemos afirmar com confianca "
        "que o AUC cross-dataset restante reflita majoritariamente sinal de doenca transferido "
        "versus um residuo do efeito de lote intrinseco as coortes.** Reportamos os numeros "
        "factualmente abaixo, mas com essa ressalva central.\n"
    )

    lines.append(
        "\n**Leitura-chave (com a ressalva do domain shift e a correcao do gap acima):**\n"
        "- **BrainIAC** degrada nas duas direcoes (gap +0.069 e +0.222, mesmo sinal, mas de "
        "magnitude bem diferente entre direcoes), com AUC absoluto cross-dataset (0.639 e 0.487) "
        "proximo do acaso, especialmente na direcao OASIS-2->OASIS-1. Dado o AUC de "
        "dataset-membership ~1.0, nao da para distinguir com confianca 'o modelo nao tem sinal "
        "transferivel' de 'o modelo tem sinal fraco mascarado pelo efeito de lote' - as duas "
        "leituras sao consistentes com os dados.\n"
        "- **Radiomics** tem gap com sinais opostos entre direcoes (+0.118 vs -0.077). Isso e "
        "consistente com sensibilidade tanto ao confundidor de idade quanto ao efeito de lote "
        "intrinseco as coortes.\n"
        "- **Volumetria** tem gap com sinais opostos (+0.135 vs -0.152) - a assinatura classica de "
        "um metodo que explora o confundidor de idade especifico do dataset de treino (OASIS-1: "
        "Δ~7.7 anos; OASIS-2: Δ~-0.87 anos), reforcada pelo achado independente de que a "
        "volumetria por si so quase nao discrimina AD/CN dentro do OASIS-2 (AUC intra-dominio "
        "0.633, contra 0.784 no OASIS-1 confundido). Diferente de BrainIAC/radiomics, a "
        "volumetria NAO sofre do efeito de lote de imagem (dataset-membership AUC=0.58), entao "
        "essa leitura de confundimento de idade e mais direta de sustentar aqui.\n"
        "- **Conclusao factual, sem forcar uma narrativa de vencedor:** dado o desenho deste "
        "experimento (2 coortes do mesmo site/instituicao, coletadas em eras/protocolos "
        "diferentes, uma confundida por idade e outra nao), NENHUM metodo mostra evidencia clara e "
        "isolavel de generalizacao 'genuina' de sinal de doenca entre coortes. BrainIAC e "
        "radiomics tem seus AUCs cross-dataset contaminados por um efeito de lote forte e "
        "irredutivel (mesmo apos parear o nivel de pre-processamento); volumetria tem seu "
        "resultado cross-dataset dominado pelo confundidor de idade. A hipotese original do "
        "experimento (BrainIAC empata intra-dominio mas degrada menos cross-dataset, evidenciando "
        "robustez de foundation model) **nao encontra suporte claro nestes dados** - o desenho "
        "com 2 coortes da mesma instituicao nao isola bem essa pergunta. Um teste de mudanca de "
        "site/scanner mais limpo exigiria MIRIAD ou OASIS-3 (acesso credenciado indisponivel no "
        "momento - ver PROGRESS_dataset2.md).\n"
    )

    lines.append(
        "\n## Ressalvas obrigatórias\n"
        "- **OASIS-2 é da mesma instituição/scanner que o OASIS-1** (Washington University) - "
        "ver PROGRESS_dataset2.md. O experimento mede generalização entre coortes distintas, "
        "**não mudança de site/scanner** (MIRIAD/OASIS-3, que dariam esse teste mais forte, "
        "exigiam acesso credenciado indisponível no momento - ver PROGRESS_dataset2.md).\n"
        "- Para o experimento cross-dataset (BrainIAC/radiomics), OASIS-1 foi reprocessado a "
        "partir do mpr-1 bruto (mesmo nível de entrada do OASIS-2), em vez do SUBJ_111 oficial "
        "(média motion-corrected) usado nos resultados intra-domínio do OASIS-1. Isso NÃO "
        "eliminou o efeito de lote entre coortes (ver diagnóstico de domain shift acima) - "
        "portanto os números intra-domínio do OASIS-1 (que usam SUBJ_111) e os números "
        "cross-dataset (que usam mpr-1 para o lado OASIS-1) não são estritamente comparáveis "
        "entre si em termos de pipeline de entrada, embora usem o mesmo checkpoint/registro/"
        "N4/HD-BET/transform do BrainIAC.\n"
        "- Confundimento de idade: OASIS-1 tem Δ(AD-CN)~7.7 anos; OASIS-2 é bem mais balanceado "
        "(Δ~-0.87 anos) - diferenças de desempenho entre os datasets podem refletir isso, não "
        "apenas a qualidade do método.\n"
        "- Rótulo binário definido por CDR em ambos os datasets (mesma regra: CN=CDR0, "
        "AD=CDR>=0.5), mas o rótulo `Group` do OASIS-2 (`Converted`) não foi usado - ver nuance "
        "de ruído de rótulo documentada em PROGRESS_dataset2.md.\n"
        "- N moderado em ambos os datasets (235 e 150 sujeitos) - intervalos de confiança amplos "
        "esperados, sobretudo no cross-dataset (treino inteiro vs teste inteiro, sem CV).\n"
    )

    with open(os.path.join(args.cross_results, "summary.md"), "w") as f:
        f.write("\n".join(lines))

    print(table.to_string(index=False))
    print(f"\nSaved final comparison to {args.cross_results}/summary.md, final_comparison.csv")


if __name__ == "__main__":
    main()
