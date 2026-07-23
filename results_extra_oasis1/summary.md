# Resultados extra (oasis1) — fusao, SHAP, hipocampo

Baseline de referencia (melhor metodo base deste dataset): **volumetry (logreg)** — ver `results/summary.md`.

## Comparacoes primarias (familia com correcao de Holm-Bonferroni)

| Metodo novo | Classificador | AUC novo | AUC baseline | Diferenca | z | p bruto | p-Holm | IC95% diff |
|---|---|---|---|---|---|---|---|---|
| Fusao early (BrainIAC+radiomics concat) | pca_logreg | 0.752 | 0.776 | -0.024 | -1.042 | 0.2973 | 1.0000 | [-0.068, 0.021] |
| Fusao late (stacking BrainIAC+radiomics) | mean | 0.763 | 0.776 | -0.012 | -0.536 | 0.5920 | 1.0000 | [-0.057, 0.032] |
| Volume hipocampal (SynthSeg, 3A) | random_forest | 0.807 | 0.776 | 0.031 | 1.024 | 0.3058 | 1.0000 | [-0.028, 0.091] |
| Radiomics + selecao SHAP | shap_logreg | 0.720 | 0.776 | -0.055 | -2.388 | 0.0169 | 0.1016 | [-0.100, -0.010] |
| Fusao + selecao SHAP | shap_logreg | 0.769 | 0.776 | -0.006 | -0.259 | 0.7960 | 1.0000 | [-0.053, 0.041] |
| BrainIAC 768d + selecao SHAP | shap_logreg | 0.754 | 0.776 | -0.021 | -0.744 | 0.4567 | 1.0000 | [-0.078, 0.035] |

## Comparacoes exploratorias (sem correcao de Holm, p bruto apenas)

### vs melhor metodo base do dataset

| Metodo novo | Classificador | AUC novo | AUC baseline | Diferenca | z | p bruto | IC95% diff |
|---|---|---|---|---|---|---|---|
| BrainIAC no ROI do hipocampo (3B, exploratorio/OOD) | logreg | 0.634 | 0.776 | -0.142 | -3.864 | 0.0001 | [-0.213, -0.070] |

### SHAP-selecionado vs mesma feature set sem selecao

| Comparacao | AUC A | AUC B | Diferenca | z | p bruto | IC95% diff |
|---|---|---|---|---|---|---|
| Radiomics SHAP (shap_logreg) vs sem selecao (svm_rbf) | 0.720 | 0.731 | -0.011 | -0.476 | 0.6344 | [-0.056, 0.034] |
| Fusao SHAP (shap_logreg) vs sem selecao (pca_logreg) | 0.769 | 0.752 | 0.018 | 1.109 | 0.2672 | [-0.013, 0.049] |
| BrainIAC 768d SHAP (shap_logreg) vs sem selecao (svm_linear) | 0.754 | 0.751 | 0.003 | 0.169 | 0.8657 | [-0.031, 0.037] |

## Limitacoes (herdadas do baseline + especificas destes testes)
- Confundimento de idade nao corrigido estatisticamente (ver baseline deste dataset).
- N moderado - intervalos de confianca amplos esperados.
- Teste multiplo: familia primaria (6 comparacoes/dataset) corrigida por Holm-Bonferroni; demais comparacoes sao exploratorias e rotuladas como tal, sem correcao.
- Teste 3B (BrainIAC no ROI do hipocampo) e out-of-distribution para o encoder (campo de visao muito menor que o cerebro inteiro, e em espaco nativo nao registrado) - tratado como exploratorio, nao como comparacao primaria.
