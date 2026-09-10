# Resultado comparativo: BrainIAC+ML vs baselines

## Melhor classificador por metodo (por AUC medio de CV)

| Metodo | Classificador | AUC (media +/- dp) | Bal. Acc. | F1 | Sensibilidade | Especificidade |
|---|---|---|---|---|---|---|
| brainiac | svm_linear | 0.804 +/- 0.076 | 0.718 +/- 0.085 | 0.712 +/- 0.085 | 0.706 +/- 0.138 | 0.729 +/- 0.189 |
| radiomics | logreg | 0.791 +/- 0.038 | 0.694 +/- 0.049 | 0.691 +/- 0.046 | 0.682 +/- 0.053 | 0.706 +/- 0.083 |

**Nota:** os AUCs A/B na tabela de DeLong abaixo sao calculados sobre as predicoes *pooled* (todas as 5 fold-predictions concatenadas), por isso diferem ligeiramente do 'AUC (media +/- dp)' da tabela acima, que e a media das 5 AUCs calculadas fold-a-fold. Sao duas quantidades validas mas distintas do mesmo conjunto de predicoes - nao e um erro.

## Comparacao estatistica (teste de DeLong, predicoes pooled de CV, pareadas por sujeito)

| Comparacao | AUC A | AUC B | Diferenca | z | p | IC95% diferenca |
|---|---|---|---|---|---|---|
| brainiac (svm_linear) vs radiomics (logreg) | 0.801 | 0.791 | 0.010 | 0.236 | 0.8136 | [-0.073, 0.093] |

## Tabela completa (todos os classificadores)

| method    | classifier    | is_best_for_method   |   auc_mean |   auc_std |   balanced_accuracy_mean |   balanced_accuracy_std |   f1_mean |    f1_std |   sensitivity_mean |   sensitivity_std |   specificity_mean |   specificity_std |
|:----------|:--------------|:---------------------|-----------:|----------:|-------------------------:|------------------------:|----------:|----------:|-------------------:|------------------:|-------------------:|------------------:|
| brainiac  | svm_linear    | True                 |   0.804152 | 0.0763836 |                 0.717647 |               0.0847345 |  0.711815 | 0.0849375 |           0.705882 |         0.137953  |           0.729412 |         0.188786  |
| brainiac  | logreg        | False                |   0.799308 | 0.0707441 |                 0.723529 |               0.0896938 |  0.702416 | 0.100241  |           0.658824 |         0.120552  |           0.788235 |         0.135422  |
| brainiac  | xgboost       | False                |   0.756401 | 0.060756  |                 0.664706 |               0.0446051 |  0.633121 | 0.0693769 |           0.588235 |         0.110049  |           0.741176 |         0.0892103 |
| brainiac  | svm_rbf       | False                |   0.740484 | 0.0533815 |                 0.670588 |               0.0383482 |  0.654055 | 0.0392168 |           0.623529 |         0.0526134 |           0.717647 |         0.0644379 |
| brainiac  | random_forest | False                |   0.732526 | 0.0422656 |                 0.682353 |               0.0436247 |  0.624736 | 0.0846991 |           0.541176 |         0.113149  |           0.823529 |         0.0720438 |
| brainiac  | lightgbm      | False                |   0.728028 | 0.0986796 |                 0.670588 |               0.0383482 |  0.634987 | 0.0719825 |           0.588235 |         0.131533  |           0.752941 |         0.105227  |
| radiomics | logreg        | True                 |   0.791003 | 0.0384223 |                 0.694118 |               0.0492153 |  0.690768 | 0.0456329 |           0.682353 |         0.0526134 |           0.705882 |         0.083189  |
| radiomics | svm_linear    | False                |   0.775087 | 0.0211894 |                 0.688235 |               0.0335346 |  0.666912 | 0.0143021 |           0.623529 |         0.0526134 |           0.752941 |         0.113149  |
| radiomics | lightgbm      | False                |   0.767474 | 0.0290946 |                 0.670588 |               0.0436247 |  0.655026 | 0.024363  |           0.623529 |         0.0526134 |           0.717647 |         0.127526  |
| radiomics | xgboost       | False                |   0.755017 | 0.0128074 |                 0.705882 |               0.0360219 |  0.695145 | 0.0266311 |           0.670588 |         0.0670691 |           0.741176 |         0.114668  |
| radiomics | random_forest | False                |   0.747751 | 0.0398375 |                 0.694118 |               0.0335346 |  0.658269 | 0.0605876 |           0.6      |         0.113149  |           0.788235 |         0.0892103 |
| radiomics | svm_rbf       | False                |   0.738062 | 0.0407512 |                 0.682353 |               0.0246076 |  0.655226 | 0.0388486 |           0.611765 |         0.106858  |           0.752941 |         0.134138  |

## Limitacoes
- Dataset: OASIS-1 (nao OpenNeuro - ver PROGRESS.md para justificativa da mudanca).
- Rotulo binario definido por CDR (Clinical Dementia Rating), nao por biomarcador molecular (ex. PET amiloide/tau) - reflete diagnostico clinico de demencia, nao confirmacao patologica de AD.
- Confundimento de idade entre grupos (CN media ~69 anos vs AD media ~76.8 anos) nao corrigido estatisticamente neste pipeline.
- Teste de DeLong aplicado a predicoes pooled de validacao cruzada (nao a um unico modelo fixo) - aproximacao pragmatica amplamente usada, mas tecnicamente as predicoes nao sao i.i.d. de um unico classificador.
- N moderado (235 sujeitos, 135 CN / 100 AD) - intervalos de confianca amplos esperados.
- O 'melhor classificador' de cada metodo foi selecionado pela mesma AUC de CV que e depois reportada e comparada via DeLong (selection-on-the-test-metric). Isso se aplica igualmente aos 3 metodos; como a conclusao principal e 'nenhuma diferenca significativa', esse viés de selecao tende a favorecer a deteccao de diferencas, nao a escondê-las - ou seja, joga contra, nao a favor, da conclusao de nao-superioridade do BrainIAC.
- QC de pre-processamento (ver `results/qc_preprocessing.csv` e `src/10_qc_preprocessing.py`): fracao de voxels nao-zero (mascara cerebral pos skull-strip) tem distribuicao estreita e consistente entre os 235 sujeitos (media 0.230, dp 0.021, min 0.185, max 0.296, mesmo shape em todos) - sem evidencia de skull-strips falhos que pudessem explicar artificialmente o desempenho do BrainIAC/radiomics por inputs corrompidos (a volumetria usa um pipeline de processamento totalmente independente do OASIS, entao essa checagem descarta uma assimetria de qualidade de dados entre os metodos).
