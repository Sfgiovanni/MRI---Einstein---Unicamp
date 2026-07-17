# Resultado comparativo: BrainIAC+ML vs baselines

## Melhor classificador por metodo (por AUC medio de CV)

| Metodo | Classificador | AUC (media +/- dp) | Bal. Acc. | F1 | Sensibilidade | Especificidade |
|---|---|---|---|---|---|---|
| brainiac | svm_linear | 0.750 +/- 0.046 | 0.690 +/- 0.030 | 0.634 +/- 0.038 | 0.610 +/- 0.089 | 0.770 +/- 0.096 |
| radiomics | svm_rbf | 0.748 +/- 0.012 | 0.630 +/- 0.047 | 0.515 +/- 0.141 | 0.490 +/- 0.263 | 0.770 +/- 0.182 |
| volumetry | logreg | 0.784 +/- 0.039 | 0.689 +/- 0.048 | 0.625 +/- 0.073 | 0.600 +/- 0.162 | 0.778 +/- 0.141 |

**Nota:** os AUCs A/B na tabela de DeLong abaixo sao calculados sobre as predicoes *pooled* (todas as 5 fold-predictions concatenadas), por isso diferem ligeiramente do 'AUC (media +/- dp)' da tabela acima, que e a media das 5 AUCs calculadas fold-a-fold. Sao duas quantidades validas mas distintas do mesmo conjunto de predicoes - nao e um erro.

## Comparacao estatistica (teste de DeLong, predicoes pooled de CV, pareadas por sujeito)

| Comparacao | AUC A | AUC B | Diferenca | z | p | IC95% diferenca |
|---|---|---|---|---|---|---|
| brainiac (svm_linear) vs radiomics (svm_rbf) | 0.751 | 0.731 | 0.020 | 0.576 | 0.5648 | [-0.048, 0.087] |
| brainiac (svm_linear) vs volumetry (logreg) | 0.751 | 0.776 | -0.024 | -0.823 | 0.4105 | [-0.083, 0.034] |

## Tabela completa (todos os classificadores)

| method    | classifier    | is_best_for_method   |   auc_mean |   auc_std |   balanced_accuracy_mean |   balanced_accuracy_std |   f1_mean |    f1_std |   sensitivity_mean |   sensitivity_std |   specificity_mean |   specificity_std |
|:----------|:--------------|:---------------------|-----------:|----------:|-------------------------:|------------------------:|----------:|----------:|-------------------:|------------------:|-------------------:|------------------:|
| brainiac  | svm_linear    | True                 |   0.74963  | 0.0459506 |                 0.690185 |               0.0295253 |  0.633923 | 0.0380986 |               0.61 |         0.0894427 |           0.77037  |         0.095868  |
| brainiac  | logreg        | False                |   0.749259 | 0.044706  |                 0.686667 |               0.0504336 |  0.637571 | 0.060206  |               0.64 |         0.114018  |           0.733333 |         0.106058  |
| brainiac  | random_forest | False                |   0.72     | 0.0765629 |                 0.658333 |               0.0727662 |  0.619233 | 0.0733117 |               0.65 |         0.1       |           0.666667 |         0.114156  |
| brainiac  | lightgbm      | False                |   0.71037  | 0.0497869 |                 0.635741 |               0.0393143 |  0.584833 | 0.0280294 |               0.59 |         0.0547723 |           0.681481 |         0.10987   |
| brainiac  | svm_rbf       | False                |   0.706111 | 0.0622839 |                 0.65537  |               0.0589597 |  0.593154 | 0.0626601 |               0.57 |         0.103682  |           0.740741 |         0.159302  |
| brainiac  | xgboost       | False                |   0.698889 | 0.0567405 |                 0.624444 |               0.0437523 |  0.565913 | 0.0468996 |               0.56 |         0.0547723 |           0.688889 |         0.0721985 |
| radiomics | svm_rbf       | True                 |   0.747593 | 0.0117778 |                 0.630185 |               0.0467505 |  0.515198 | 0.141017  |               0.49 |         0.263154  |           0.77037  |         0.182198  |
| radiomics | svm_linear    | False                |   0.742593 | 0.0174703 |                 0.665185 |               0.0716639 |  0.57768  | 0.128821  |               0.56 |         0.248495  |           0.77037  |         0.137087  |
| radiomics | lightgbm      | False                |   0.737222 | 0.0523963 |                 0.64463  |               0.0764279 |  0.595846 | 0.100969  |               0.63 |         0.178885  |           0.659259 |         0.129365  |
| radiomics | logreg        | False                |   0.735926 | 0.0240598 |                 0.677778 |               0.050677  |  0.60609  | 0.101843  |               0.6  |         0.220794  |           0.755556 |         0.137586  |
| radiomics | random_forest | False                |   0.731667 | 0.0185    |                 0.662037 |               0.0399706 |  0.615577 | 0.0615861 |               0.65 |         0.169558  |           0.674074 |         0.121151  |
| radiomics | xgboost       | False                |   0.730741 | 0.0460289 |                 0.641852 |               0.0527916 |  0.580292 | 0.0789126 |               0.58 |         0.144049  |           0.703704 |         0.0944263 |
| volumetry | logreg        | True                 |   0.784444 | 0.0389902 |                 0.688889 |               0.0477054 |  0.624831 | 0.0732896 |               0.6  |         0.162019  |           0.777778 |         0.141033  |
| volumetry | svm_linear    | False                |   0.78     | 0.0433784 |                 0.688889 |               0.0477054 |  0.624831 | 0.0732896 |               0.6  |         0.162019  |           0.777778 |         0.141033  |
| volumetry | svm_rbf       | False                |   0.748519 | 0.0472748 |                 0.671667 |               0.0451174 |  0.613236 | 0.0730559 |               0.61 |         0.163554  |           0.733333 |         0.160161  |
| volumetry | lightgbm      | False                |   0.722963 | 0.105755  |                 0.636852 |               0.0811636 |  0.581573 | 0.0746787 |               0.57 |         0.115109  |           0.703704 |         0.211144  |
| volumetry | random_forest | False                |   0.706111 | 0.0998168 |                 0.635741 |               0.0857334 |  0.584865 | 0.0905286 |               0.59 |         0.119373  |           0.681481 |         0.189939  |
| volumetry | xgboost       | False                |   0.700185 | 0.107356  |                 0.645741 |               0.0810452 |  0.595034 | 0.098134  |               0.61 |         0.171026  |           0.681481 |         0.238019  |

## Limitacoes
- Dataset: OASIS-1 (nao OpenNeuro - ver PROGRESS.md para justificativa da mudanca).
- Rotulo binario definido por CDR (Clinical Dementia Rating), nao por biomarcador molecular (ex. PET amiloide/tau) - reflete diagnostico clinico de demencia, nao confirmacao patologica de AD.
- Confundimento de idade entre grupos (CN media ~69 anos vs AD media ~76.8 anos) nao corrigido estatisticamente neste pipeline.
- Teste de DeLong aplicado a predicoes pooled de validacao cruzada (nao a um unico modelo fixo) - aproximacao pragmatica amplamente usada, mas tecnicamente as predicoes nao sao i.i.d. de um unico classificador.
- N moderado (235 sujeitos, 135 CN / 100 AD) - intervalos de confianca amplos esperados.
- O 'melhor classificador' de cada metodo foi selecionado pela mesma AUC de CV que e depois reportada e comparada via DeLong (selection-on-the-test-metric). Isso se aplica igualmente aos 3 metodos; como a conclusao principal e 'nenhuma diferenca significativa', esse viés de selecao tende a favorecer a deteccao de diferencas, nao a escondê-las - ou seja, joga contra, nao a favor, da conclusao de nao-superioridade do BrainIAC.
- QC de pre-processamento (ver `results/qc_preprocessing.csv` e `src/10_qc_preprocessing.py`): fracao de voxels nao-zero (mascara cerebral pos skull-strip) tem distribuicao estreita e consistente entre os 235 sujeitos (media 0.230, dp 0.021, min 0.185, max 0.296, mesmo shape em todos) - sem evidencia de skull-strips falhos que pudessem explicar artificialmente o desempenho do BrainIAC/radiomics por inputs corrompidos (a volumetria usa um pipeline de processamento totalmente independente do OASIS, entao essa checagem descarta uma assimetria de qualidade de dados entre os metodos).
