# Resultado comparativo: BrainIAC+ML vs baselines

## Melhor classificador por metodo (por AUC medio de CV)

| Metodo | Classificador | AUC (media +/- dp) | Bal. Acc. | F1 | Sensibilidade | Especificidade |
|---|---|---|---|---|---|---|
| brainiac | logreg | 0.710 +/- 0.115 | 0.688 +/- 0.087 | 0.650 +/- 0.084 | 0.646 +/- 0.069 | 0.729 +/- 0.159 |
| radiomics | svm_rbf | 0.640 +/- 0.112 | 0.566 +/- 0.046 | 0.381 +/- 0.118 | 0.308 +/- 0.154 | 0.824 +/- 0.186 |
| volumetry | logreg | 0.633 +/- 0.086 | 0.575 +/- 0.120 | 0.442 +/- 0.182 | 0.385 +/- 0.196 | 0.765 +/- 0.072 |

**Nota:** os AUCs A/B na tabela de DeLong abaixo sao calculados sobre as predicoes *pooled* (todas as 5 fold-predictions concatenadas), por isso diferem ligeiramente do 'AUC (media +/- dp)' da tabela acima, que e a media das 5 AUCs calculadas fold-a-fold. Sao duas quantidades validas mas distintas do mesmo conjunto de predicoes - nao e um erro.

## Comparacao estatistica (teste de DeLong, predicoes pooled de CV, pareadas por sujeito)

| Comparacao | AUC A | AUC B | Diferenca | z | p | IC95% diferenca |
|---|---|---|---|---|---|---|
| brainiac (logreg) vs radiomics (svm_rbf) | 0.689 | 0.637 | 0.052 | 0.850 | 0.3951 | [-0.067, 0.170] |
| brainiac (logreg) vs volumetry (logreg) | 0.689 | 0.630 | 0.059 | 0.993 | 0.3209 | [-0.058, 0.177] |

## Tabela completa (todos os classificadores)

| method    | classifier    | is_best_for_method   |   auc_mean |   auc_std |   balanced_accuracy_mean |   balanced_accuracy_std |   f1_mean |    f1_std |   sensitivity_mean |   sensitivity_std |   specificity_mean |   specificity_std |
|:----------|:--------------|:---------------------|-----------:|----------:|-------------------------:|------------------------:|----------:|----------:|-------------------:|------------------:|-------------------:|------------------:|
| brainiac  | logreg        | True                 |   0.709502 | 0.115336  |                 0.687783 |               0.0872435 | 0.650129  | 0.0835314 |          0.646154  |         0.0688021 |           0.729412 |         0.158932  |
| brainiac  | svm_linear    | False                |   0.692308 | 0.106551  |                 0.623077 |               0.0899958 | 0.513073  | 0.13952   |          0.446154  |         0.183651  |           0.8      |         0.206302  |
| brainiac  | svm_rbf       | False                |   0.598643 | 0.13348   |                 0.528507 |               0.0392389 | 0.133333  | 0.188562  |          0.0923077 |         0.137604  |           0.964706 |         0.07892   |
| brainiac  | lightgbm      | False                |   0.58733  | 0.0835819 |                 0.505882 |               0.0833673 | 0.400911  | 0.144087  |          0.4       |         0.19911   |           0.611765 |         0.158932  |
| brainiac  | xgboost       | False                |   0.572851 | 0.0427596 |                 0.515837 |               0.0737987 | 0.415611  | 0.0703624 |          0.384615  |         0.0942111 |           0.647059 |         0.155632  |
| brainiac  | random_forest | False                |   0.549321 | 0.0711753 |                 0.475113 |               0.086581  | 0.350946  | 0.152892  |          0.338462  |         0.193075  |           0.611765 |         0.121979  |
| radiomics | svm_rbf       | True                 |   0.639819 | 0.111646  |                 0.565611 |               0.0460895 | 0.380781  | 0.117972  |          0.307692  |         0.153846  |           0.823529 |         0.186016  |
| radiomics | logreg        | False                |   0.61267  | 0.0818617 |                 0.574208 |               0.110228  | 0.473125  | 0.156697  |          0.430769  |         0.159511  |           0.717647 |         0.0644379 |
| radiomics | random_forest | False                |   0.58733  | 0.138966  |                 0.547059 |               0.0852464 | 0.442629  | 0.110044  |          0.4       |         0.114095  |           0.694118 |         0.105227  |
| radiomics | xgboost       | False                |   0.557466 | 0.121474  |                 0.528054 |               0.0680917 | 0.382271  | 0.153573  |          0.338462  |         0.149951  |           0.717647 |         0.113149  |
| radiomics | lightgbm      | False                |   0.528507 | 0.0843743 |                 0.488688 |               0.0706267 | 0.375438  | 0.124111  |          0.353846  |         0.139738  |           0.623529 |         0.0670691 |
| radiomics | svm_linear    | False                |   0.478733 | 0.153259  |                 0.519005 |               0.0267983 | 0.0952381 | 0.146772  |          0.0615385 |         0.100295  |           0.976471 |         0.0526134 |
| volumetry | logreg        | True                 |   0.632579 | 0.0860562 |                 0.574661 |               0.11961   | 0.442242  | 0.181625  |          0.384615  |         0.196116  |           0.764706 |         0.0720438 |
| volumetry | random_forest | False                |   0.531674 | 0.0693652 |                 0.548416 |               0.0812687 | 0.467503  | 0.113206  |          0.461538  |         0.172005  |           0.635294 |         0.0766965 |
| volumetry | lightgbm      | False                |   0.507692 | 0.052589  |                 0.523529 |               0.0636427 | 0.416876  | 0.126567  |          0.4       |         0.157645  |           0.647059 |         0.144088  |
| volumetry | xgboost       | False                |   0.50181  | 0.0362838 |                 0.461086 |               0.026258  | 0.376203  | 0.0428472 |          0.369231  |         0.0643585 |           0.552941 |         0.0670691 |
| volumetry | svm_rbf       | False                |   0.417195 | 0.109227  |                 0.5      |               0         | 0         | 0         |          0         |         0         |           1        |         0         |
| volumetry | svm_linear    | False                |   0.411765 | 0.126089  |                 0.501357 |               0.0204623 | 0.0898246 | 0.138887  |          0.0615385 |         0.100295  |           0.941176 |         0.0720438 |

## Limitacoes
- Dataset: OASIS-1 (nao OpenNeuro - ver PROGRESS.md para justificativa da mudanca).
- Rotulo binario definido por CDR (Clinical Dementia Rating), nao por biomarcador molecular (ex. PET amiloide/tau) - reflete diagnostico clinico de demencia, nao confirmacao patologica de AD.
- Confundimento de idade entre grupos (CN media ~69 anos vs AD media ~76.8 anos) nao corrigido estatisticamente neste pipeline.
- Teste de DeLong aplicado a predicoes pooled de validacao cruzada (nao a um unico modelo fixo) - aproximacao pragmatica amplamente usada, mas tecnicamente as predicoes nao sao i.i.d. de um unico classificador.
- N moderado (235 sujeitos, 135 CN / 100 AD) - intervalos de confianca amplos esperados.
- O 'melhor classificador' de cada metodo foi selecionado pela mesma AUC de CV que e depois reportada e comparada via DeLong (selection-on-the-test-metric). Isso se aplica igualmente aos 3 metodos; como a conclusao principal e 'nenhuma diferenca significativa', esse viés de selecao tende a favorecer a deteccao de diferencas, nao a escondê-las - ou seja, joga contra, nao a favor, da conclusao de nao-superioridade do BrainIAC.
- QC de pre-processamento (ver `results/qc_preprocessing.csv` e `src/10_qc_preprocessing.py`): fracao de voxels nao-zero (mascara cerebral pos skull-strip) tem distribuicao estreita e consistente entre os 235 sujeitos (media 0.230, dp 0.021, min 0.185, max 0.296, mesmo shape em todos) - sem evidencia de skull-strips falhos que pudessem explicar artificialmente o desempenho do BrainIAC/radiomics por inputs corrompidos (a volumetria usa um pipeline de processamento totalmente independente do OASIS, entao essa checagem descarta uma assimetria de qualidade de dados entre os metodos).
