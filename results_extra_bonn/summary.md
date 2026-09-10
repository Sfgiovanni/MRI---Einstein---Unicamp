# Resultados extra (bonn) — fusao, SHAP, hipocampo

Baseline de referencia (melhor metodo base deste dataset): **radiomics (logreg)** — ver `results_bonn/summary.md`.

## Comparacoes primarias (familia com correcao de Holm-Bonferroni)

| Metodo novo | Classificador | AUC novo | AUC baseline | Diferenca | z | p bruto | p-Holm | IC95% diff |
|---|---|---|---|---|---|---|---|---|
| Fusao early (BrainIAC+radiomics concat) | svm_linear | 0.873 | 0.791 | 0.082 | 2.614 | 0.0090 | 0.0448 | [0.020, 0.143] |
| Fusao late (stacking BrainIAC+radiomics) | meta_logreg | 0.817 | 0.791 | 0.026 | 0.868 | 0.3856 | 1.0000 | [-0.033, 0.086] |
| Volume hipocampal (SynthSeg, 3A) | random_forest | 0.558 | 0.791 | -0.233 | -4.777 | 0.0000 | 0.0000 | [-0.328, -0.137] |
| Radiomics + selecao SHAP | shap_logreg | 0.797 | 0.791 | 0.006 | 0.567 | 0.5705 | 1.0000 | [-0.015, 0.028] |
| Fusao + selecao SHAP | shap_logreg | 0.837 | 0.791 | 0.047 | 1.541 | 0.1234 | 0.4937 | [-0.013, 0.106] |
| BrainIAC 768d + selecao SHAP | shap_logreg | 0.777 | 0.791 | -0.014 | -0.321 | 0.7481 | 1.0000 | [-0.099, 0.071] |

## Comparacoes exploratorias (sem correcao de Holm, p bruto apenas)

### vs melhor metodo base do dataset

| Metodo novo | Classificador | AUC novo | AUC baseline | Diferenca | z | p bruto | IC95% diff |
|---|---|---|---|---|---|---|---|
| BrainIAC no ROI do hipocampo (3B, exploratorio/OOD) | xgboost | 0.740 | 0.791 | -0.050 | -1.231 | 0.2182 | [-0.130, 0.030] |

### SHAP-selecionado vs mesma feature set sem selecao

| Comparacao | AUC A | AUC B | Diferenca | z | p bruto | IC95% diff |
|---|---|---|---|---|---|---|
| Radiomics SHAP (shap_logreg) vs sem selecao (logreg) | 0.797 | 0.791 | 0.006 | 0.567 | 0.5705 | [-0.015, 0.028] |
| Fusao SHAP (shap_logreg) vs sem selecao (svm_linear) | 0.837 | 0.873 | -0.035 | -1.550 | 0.1212 | [-0.080, 0.009] |
| BrainIAC 768d SHAP (shap_logreg) vs sem selecao (svm_linear) | 0.777 | 0.801 | -0.024 | -1.258 | 0.2085 | [-0.061, 0.013] |

## Limitacoes (herdadas do baseline + especificas destes testes)
- Confundimento de idade nao corrigido estatisticamente (ver baseline deste dataset).
- N moderado - intervalos de confianca amplos esperados.
- Teste multiplo: familia primaria (6 comparacoes/dataset) corrigida por Holm-Bonferroni; demais comparacoes sao exploratorias e rotuladas como tal, sem correcao.
- Teste 3B (BrainIAC no ROI do hipocampo) e out-of-distribution para o encoder (campo de visao muito menor que o cerebro inteiro, e em espaco nativo nao registrado) - tratado como exploratorio, nao como comparacao primaria.
