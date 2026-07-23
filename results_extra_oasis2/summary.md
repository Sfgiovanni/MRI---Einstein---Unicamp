# Resultados extra (oasis2) — fusao, SHAP, hipocampo

Baseline de referencia (melhor metodo base deste dataset): **brainiac (logreg)** — ver `results_oasis2/summary.md`.

## Comparacoes primarias (familia com correcao de Holm-Bonferroni)

| Metodo novo | Classificador | AUC novo | AUC baseline | Diferenca | z | p bruto | p-Holm | IC95% diff |
|---|---|---|---|---|---|---|---|---|
| Fusao early (BrainIAC+radiomics concat) | svm_linear | 0.691 | 0.689 | 0.002 | 0.067 | 0.9469 | 1.0000 | [-0.051, 0.055] |
| Fusao late (stacking BrainIAC+radiomics) | mean | 0.705 | 0.689 | 0.016 | 1.109 | 0.2676 | 0.8029 | [-0.012, 0.044] |
| Volume hipocampal (SynthSeg, 3A) | logreg | 0.723 | 0.689 | 0.034 | 0.653 | 0.5139 | 1.0000 | [-0.067, 0.135] |
| Radiomics + selecao SHAP | shap_logreg | 0.602 | 0.689 | -0.087 | -1.525 | 0.1273 | 0.6367 | [-0.199, 0.025] |
| Fusao + selecao SHAP | shap_logreg | 0.626 | 0.689 | -0.063 | -1.660 | 0.0968 | 0.5811 | [-0.137, 0.011] |
| BrainIAC 768d + selecao SHAP | shap_logreg | 0.642 | 0.689 | -0.047 | -1.508 | 0.1316 | 0.6367 | [-0.107, 0.014] |

## Comparacoes exploratorias (sem correcao de Holm, p bruto apenas)

### vs melhor metodo base do dataset

| Metodo novo | Classificador | AUC novo | AUC baseline | Diferenca | z | p bruto | IC95% diff |
|---|---|---|---|---|---|---|---|
| BrainIAC no ROI do hipocampo (3B, exploratorio/OOD) | logreg | 0.567 | 0.689 | -0.121 | -1.890 | 0.0588 | [-0.247, 0.004] |

### SHAP-selecionado vs mesma feature set sem selecao

| Comparacao | AUC A | AUC B | Diferenca | z | p bruto | IC95% diff |
|---|---|---|---|---|---|---|
| Radiomics SHAP (shap_logreg) vs sem selecao (svm_rbf) | 0.602 | 0.637 | -0.035 | -0.797 | 0.4255 | [-0.123, 0.052] |
| Fusao SHAP (shap_logreg) vs sem selecao (svm_linear) | 0.626 | 0.691 | -0.065 | -1.738 | 0.0822 | [-0.137, 0.008] |
| BrainIAC 768d SHAP (shap_logreg) vs sem selecao (logreg) | 0.642 | 0.689 | -0.047 | -1.508 | 0.1316 | [-0.107, 0.014] |

## Limitacoes (herdadas do baseline + especificas destes testes)
- Confundimento de idade nao corrigido estatisticamente (ver baseline deste dataset).
- N moderado - intervalos de confianca amplos esperados.
- Teste multiplo: familia primaria (6 comparacoes/dataset) corrigida por Holm-Bonferroni; demais comparacoes sao exploratorias e rotuladas como tal, sem correcao.
- Teste 3B (BrainIAC no ROI do hipocampo) e out-of-distribution para o encoder (campo de visao muito menor que o cerebro inteiro, e em espaco nativo nao registrado) - tratado como exploratorio, nao como comparacao primaria.

- Grupo 'Converted' no OASIS-2 classificado pela regra de CDR na baseline (nao pelo status futuro) - ver PROGRESS_extra.md / PROGRESS_dataset2.md para a regra completa.
