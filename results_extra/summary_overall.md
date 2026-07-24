# Consolidado cross-dataset — oasis1 x oasis2

Baseline oasis1: volumetry (logreg).
Baseline oasis2: brainiac (logreg).

Cada dataset e um estudo separado (sem pool entre coortes). Comparacao primaria = novo metodo vs melhor metodo base DAQUELE dataset; p-Holm so existe para a familia primaria (comparacoes primarias por dataset, ver `results_extra_{dataset}/summary.md`).

## Tabela lado a lado

| Metodo | AUC diff oasis1 | p-Holm oasis1 | AUC diff oasis2 | p-Holm oasis2 | Replicacao |
|---|---|---|---|---|---|
| BrainIAC 768d + selecao SHAP | -0.021 | 1.0000 | -0.047 | 0.6367 | direcao consistente (perda nos 2) |
| BrainIAC no ROI do hipocampo (3B, exploratorio/OOD) | -0.142 | - | -0.121 | - | direcao consistente (perda nos 2) |
| Fusao + selecao SHAP | -0.006 | 1.0000 | -0.063 | 0.5811 | direcao consistente (perda nos 2) |
| Fusao early (BrainIAC+radiomics concat) | -0.024 | 1.0000 | +0.002 | 1.0000 | inconsistente entre datasets |
| Fusao late (stacking BrainIAC+radiomics) | -0.012 | 1.0000 | +0.016 | 0.8029 | inconsistente entre datasets |
| Radiomics + selecao SHAP | -0.055 | 0.1016 | -0.087 | 0.6367 | direcao consistente (perda nos 2) |
| Volume hipocampal (SynthSeg, 3A) | +0.031 | 1.0000 | +0.034 | 1.0000 | direcao consistente (ganho nos 2), sem significancia |

**Achados que replicam (significativos em todos os 2 datasets, mesma direcao): 0/7**


## Ressalvas obrigatorias
- Cada dataset e tratado como estudo independente - sem pool entre coortes.
- N moderado - intervalos de confianca amplos esperados em todos os datasets.
- Teste multiplo: familia primaria corrigida por Holm-Bonferroni EM CADA dataset separadamente (nao ha correcao conjunta cross-dataset).
- Testes marcados como exploratorios/OOD (ex.: BrainIAC no ROI do hipocampo) nao entram na familia primaria nem na conclusao de replicacao.
- Ver `results_extra_{dataset}/summary.md` de cada dataset para ressalvas especificas (confundimento de idade, regras de rotulagem, QC de segmentacao, etc.).
