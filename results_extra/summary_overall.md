# Consolidado cross-dataset — OASIS-1 x OASIS-2

Baseline OASIS-1: volumetry (logreg). Baseline OASIS-2: brainiac (logreg).

Cada dataset e um estudo separado (sem pool entre coortes). Comparacao primaria = novo metodo vs melhor metodo base DAQUELE dataset; p-Holm so existe para a familia primaria (6 comparacoes por dataset).

## Tabela lado a lado

| Metodo | AUC diff OASIS-1 | p-Holm OASIS-1 | AUC diff OASIS-2 | p-Holm OASIS-2 | Replicacao |
|---|---|---|---|---|---|
| BrainIAC 768d + selecao SHAP | -0.021 | 1.0000 | -0.047 | 0.6367 | direcao consistente (perda nos 2) |
| BrainIAC no ROI do hipocampo (3B, exploratorio/OOD) | -0.142 | - | -0.121 | - | direcao consistente (perda nos 2) |
| Fusao + selecao SHAP | -0.006 | 1.0000 | -0.063 | 0.5811 | direcao consistente (perda nos 2) |
| Fusao early (BrainIAC+radiomics concat) | -0.024 | 1.0000 | +0.002 | 1.0000 | inconsistente entre datasets |
| Fusao late (stacking BrainIAC+radiomics) | -0.012 | 1.0000 | +0.016 | 0.8029 | inconsistente entre datasets |
| Radiomics + selecao SHAP | -0.055 | 0.1016 | -0.087 | 0.6367 | direcao consistente (perda nos 2) |
| Volume hipocampal (SynthSeg, 3A) | +0.031 | 1.0000 | +0.034 | 1.0000 | direcao consistente (ganho nos 2), sem significancia |

**Achados que replicam (significativos nos dois datasets, mesma direcao): 0/7**


## Ressalvas obrigatorias
- Confundimento de idade nao corrigido em nenhum dos dois datasets (OASIS-1: AD mais velho, delta~+7.7 anos; OASIS-2: quase pareado, delta~-0.87 anos - direcoes opostas, o que ja e por si um resultado relevante para interpretar diferencas entre os dois baselines).
- N moderado nos dois datasets (OASIS-1 n=235, OASIS-2 n=150) - intervalos de confianca amplos.
- Teste multiplo: familia primaria (6 comparacoes) corrigida por Holm-Bonferroni EM CADA dataset separadamente (nao ha correcao conjunta cross-dataset - os dois sao estudos paralelos).
- Teste 3B (BrainIAC no ROI do hipocampo) e exploratorio/OOD nos dois datasets - nao entra na familia primaria nem na conclusao de replicacao.
- Regra do grupo 'Converted' no OASIS-2: classificado pela CDR na baseline (nao pelo status futuro) - 13 sujeitos 'Converted' com CDR=0 na baseline contam como CN.
- OASIS-1 raw (PROCESSED/MPRAGE/SUBJ_111, ANALYZE 7.5) tinha um defeito de orientacao (qform/sform invalidos, eixos do array rotulados errado pelo affine fallback do nibabel) que quebrava a segmentacao hipocampal (FastSurfer E SynthSeg falhavam identicamente) - corrigido via reorientacao determinada empiricamente antes da segmentacao (ver PROGRESS_extra.md); OASIS-2 nao precisou de correcao (qform ja valido).
