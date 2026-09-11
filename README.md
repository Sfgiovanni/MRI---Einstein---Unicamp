# BrainIAC + ML para classificação binária AD vs CN

Pipeline de pesquisa reproduzível que usa o modelo fundacional **BrainIAC** (Tak et al.,
Nature Neuroscience 2026) como extrator de features de MRI estrutural T1w, treina
classificadores de ML clássicos sobre essas features para classificação binária
**Alzheimer (AD) vs. controle cognitivamente normal (CN)**, e compara com baselines de
radiomics e volumetria clássica.



## Datasets

### OASIS-1 (primeiro dataset)
- Fonte: https://sites.wustl.edu/oasisbrains/home/oasis-1/
- Mapeamento binário: **CN = CDR==0** (n=135), **AD = CDR>=0.5** (n=100), sujeitos sem
  CDR avaliado (jovens/meia-idade) excluídos. Ver detalhes e limitações (confundimento
  de idade, Δ~7.7 anos) em `PROGRESS.md`.

### OASIS-2 (segundo dataset)

- Fonte: https://sites.wustl.edu/oasisbrains/home/oasis-2/ (longitudinal, 150 sujeitos,
  1 scan/sujeito mantido = sessão baseline `_MR1`)
## Setup

```bash
# 1. Ambiente conda (inclui `versioneer`, necessário no passo 2)
conda env create -f environment.yml
conda activate brainiac-ad

# 2. pyradiomics precisa ser instalado À PARTE, depois do resto do ambiente, com
#    --no-build-isolation: seu setup.py legado importa numpy/versioneer no momento
#    do build sem declarar isso como build-dependency, o que quebra com o
#    isolamento de build padrão do pip se numpy/versioneer não estiverem já visíveis.
pip install --no-build-isolation pyradiomics

# 3. Clone do BrainIAC + patch necessário
git clone https://github.com/AIM-KannLab/BrainIAC.git third_party/BrainIAC
# O diretório HD_BET do repo upstream não tem __init__.py, o que impede
# `from HD_BET.hd_bet import hd_bet` de funcionar como pacote local. Sem este
# patch, a Etapa 3 (pré-processamento) falha com ModuleNotFoundError.
touch third_party/BrainIAC/src/preprocessing/HD_BET/__init__.py
```

Pesos do BrainIAC: baixe o checkpoint `BrainIAC.ckpt` (362MB, encoder ViT-B) pelo link do Dropbox: https://www.dropbox.com/scl/fo/i51xt63roognvt7vuslbl/AG99uZljziHss5zJz4HiFis?rlkey=9w55le6tslwxlfz6c0viylmjb&st=b9cnvwh8&e=1&dl=0 
e coloque em `checkpoints/BrainIAC.ckpt`.

```bash
# 4. SHAP (teste extra 2) - so precisa no env brainiac-ad, nao tem dependencia especial
conda activate brainiac-ad
pip install shap

# 5. SynthSeg (teste extra 3, hipocampo) - env SEPARADO: o codigo e de 2020-2022 e
#    depende de TensorFlow 2.2.0/Keras 2.3.1 standalone (nao tf.keras), incompativel
#    com o env principal (torch/numpy modernos). cudatoolkit/cudnn instalados via conda
#    dentro do proprio env (nao precisa de CUDA 10.1 no sistema).
conda create -n synthseg python=3.8 -y
conda activate synthseg
git clone https://github.com/BBillot/SynthSeg.git third_party/SynthSeg
pip install -r third_party/SynthSeg/requirements_python3.8.txt
conda install -c conda-forge cudatoolkit=10.1 cudnn=7.6.5 -y
# Pesos (synthseg_1.0.h5) ja vem no clone do repo (third_party/SynthSeg/models/) -
# sem download separado nem cadastro.
conda activate brainiac-ad
```

## Pipeline (executar em ordem, a partir da raiz do projeto)

Todos os scripts aceitam `--dataset {oasis1,oasis2}` (default `oasis1`) em vez de
duplicar lógica por dataset - os caminhos de entrada/saída são derivados automaticamente
(`data/{dataset}_...`, `features/{dataset}_...`, `results` para oasis1 ou
`results_{dataset}` para os demais).

```bash
# OASIS-1 (dataset principal)
python src/01_prepare_dataset.py --dataset oasis1
python src/02_extract_and_convert.py --dataset oasis1
bash src/03_run_brainiac_preprocessing.sh oasis1
python src/04_extract_brainiac_features.py --dataset oasis1
python src/05_train_classifiers_brainiac.py --dataset oasis1
python src/06_radiomics_baseline.py --dataset oasis1
python src/07_train_classifiers_radiomics.py --dataset oasis1
python src/08_volumetry_baseline.py --dataset oasis1
python src/09_evaluate_compare.py --results_dir results --figures_dir figures
python src/10_qc_preprocessing.py --dataset oasis1

# OASIS-2 (segundo dataset - mesmos scripts, so troca --dataset)
python src/01_prepare_dataset.py --dataset oasis2
python src/02_extract_and_convert.py --dataset oasis2
bash src/03_run_brainiac_preprocessing.sh oasis2
python src/04_extract_brainiac_features.py --dataset oasis2
python src/05_train_classifiers_brainiac.py --dataset oasis2
python src/06_radiomics_baseline.py --dataset oasis2
python src/07_train_classifiers_radiomics.py --dataset oasis2
python src/08_volumetry_baseline.py --dataset oasis2
python src/09_evaluate_compare.py --results_dir results_oasis2 --figures_dir figures_oasis2
python src/10_qc_preprocessing.py --dataset oasis2

# Reprocessamento do OASIS-1 a partir do mpr-1 bruto (nivel de entrada pareado com o
# OASIS-2), usado apenas no lado OASIS-1 do experimento cross-dataset - ver diagnostico
# de domain-shift em "Validacao cruzada entre coortes" abaixo
python src/02_extract_and_convert.py --dataset oasis1_mpr1 --labels_csv data/oasis1_labels.csv --archives_dir data/oasis1_raw/discs --output_dir data/oasis1_mpr1_raw/nifti
bash src/03_run_brainiac_preprocessing.sh oasis1_mpr1
python src/04_extract_brainiac_features.py --dataset oasis1_mpr1 --labels_csv data/oasis1_labels.csv
python src/06_radiomics_baseline.py --dataset oasis1_mpr1 --labels_csv data/oasis1_labels.csv
# Intra-dominio no MESMO nivel de entrada (mpr-1), usado so para o calculo correto do
# gap de generalizacao (nao mistura "mudanca de coorte" com "mudanca de pipeline") -
# reaproveita os folds do oasis1 canonico (mesmos sujeitos)
python src/05_train_classifiers_brainiac.py --dataset oasis1_mpr1 --folds_csv data/oasis1_folds.csv --output_dir results_oasis1_mpr1
python src/07_train_classifiers_radiomics.py --dataset oasis1_mpr1 --folds_csv data/oasis1_folds.csv --output_dir results_oasis1_mpr1

# Experimento principal: validacao cruzada entre coortes + comparacao final
python src/11_cross_dataset_validation.py
python src/12_final_comparison.py
python src/13_domain_shift_diagnostic.py
```

1. `01_prepare_dataset.py` — parse demográficos OASIS-1, define rótulo binário, cria
   5-fold CV estratificado por sujeito (`data/oasis1_labels.csv`, `data/oasis1_folds.csv`).
2. `02_extract_and_convert.py` — extrai o volume nativo médio (`SUBJ_111`, Analyze format)
   de cada sujeito dos discs OASIS-1 e converte para NIfTI.
3. `03_run_brainiac_preprocessing.sh` — roda o pipeline oficial do BrainIAC (registro
   rígido + N4 + HD-BET skull-strip) sobre os NIfTIs convertidos.
4. `04_extract_brainiac_features.py` — extrai embeddings ViT (768-d) do BrainIAC.
5. `05_train_classifiers_brainiac.py` — treina LogReg/SVM/RF/XGBoost/LightGBM com CV
   k=5 por sujeito sobre os embeddings.
6. `06_radiomics_baseline.py` + `07_train_classifiers_radiomics.py` — baseline de
   radiomics (PyRadiomics) com o mesmo protocolo de CV.
7. `08_volumetry_baseline.py` — baseline de volumetria clássica (eTIV/nWBV/ASF oficiais
   do OASIS-1) com o mesmo protocolo de CV.
8. `09_evaluate_compare.py` — agrega métricas, roda teste de DeLong, gera figuras
   (`figures/`) e `results/summary.md` / `results/summary.csv`.
9. `10_qc_preprocessing.py` — QC pós-hoc: distribuição da fração de voxels não-zero
   (máscara de cérebro) em todos os sujeitos pré-processados, para detectar
   skull-strips falhos/parciais que corromperiam silenciosamente as features de
   BrainIAC/radiomics sem afetar a volumetria (que vem do pipeline independente do
   OASIS) — checagem importante para não confundir "modelo não ganhou" com
   "pré-processamento quebrou os inputs do modelo". Resultado: distribuição estreita
   (média 0.230, dp 0.021, min 0.185, max 0.296) e mesmo shape em todos os 235
   sujeitos — sem evidência de falha sistemática. Salvo em `results/qc_preprocessing.csv`.
   Rodado também para o OASIS-2 (150/150 sujeitos, distribuição quase idêntica) em
   `results_oasis2/qc_preprocessing.csv`.
10. `11_cross_dataset_validation.py` — **experimento principal da extensão**: treina em
    TODOS os sujeitos de um dataset, testa em TODOS os sujeitos do outro, nos 2 sentidos,
    para cada método (mesmos `CLASSIFIER_GRIDS`/seed de `common_cv.py`). Salva
    `results_cross/metrics_cross_dataset.csv`, `generalization_gap.csv` e predições em
    `results_cross/predictions/`.
11. `12_final_comparison.py` — tabela consolidada (intra-OASIS1, intra-OASIS2, e as 2
    direções cross-dataset, por método), figuras (`figures/cross_dataset_auc_comparison.png`,
    `figures/cross_dataset_roc.png`) e `results_cross/summary.md` com a leitura interpretativa
    completa e as ressalvas obrigatórias.
12. `13_domain_shift_diagnostic.py` — diagnóstico crítico: treina um classificador para
    prever apenas de qual dataset (OASIS-1 ou OASIS-2) um sujeito veio, ignorando o rótulo
    AD/CN. Um AUC alto indica que o espaço de features é dominado por um efeito de lote,
    limitando quanto do AUC cross-dataset reflete sinal de doença de fato transferido.
    Motivou o reprocessamento do OASIS-1 a partir do `mpr-1` bruto (`--dataset oasis1_mpr1`
    em `02`/`03`/`04`/`06`) para testar (e, como o diagnóstico mostrou, refutar) a hipótese
    de que a assimetria `SUBJ_111` vs `mpr-1` fosse a causa do efeito de lote. Salva
    `results_cross/domain_shift_diagnostic.csv`. Ver "Validação cruzada entre coortes"
    abaixo e `PROGRESS_dataset2.md` para o relato completo.

## Rodando o pipeline com outro dataset

Todo script (`01`-`13`, `20`-`31`) aceita `--dataset <nome>` e deriva os caminhos
automaticamente (`data/<nome>_...`, `features/<nome>_...`, `results_<nome>/` — exceto
`oasis1`, que por legado usa `results/` sem sufixo). Nenhum script tem os nomes
`oasis1`/`oasis2` fixos no código além dos passos 1-2 (parsing de demográficos e
extração dos arquivos brutos), que são específicos do formato OASIS por natureza.

**Contrato de entrada** — para plugar um dataset novo (`$DATASET` = qualquer nome), o
pipeline a partir do passo 03 só precisa de 3 arquivos, no formato abaixo:

| Arquivo | Colunas / conteúdo obrigatório |
|---|---|
| `data/${DATASET}_raw/nifti/{subject_id}.nii.gz` | 1 scan T1w nativo (com crânio) por sujeito |
| `data/${DATASET}_labels.csv` | `subject_id`, `label` (0=CN/1=AD); `eTIV` também é obrigatório se for rodar o baseline de volumetria e o teste de volume hipocampal (normalização) |
| `data/${DATASET}_folds.csv` | `subject_id`, `label`, `fold` — gerar com `common_cv.make_stratified_subject_folds()` (é isso que `01_prepare_dataset.py` chama por baixo) |

Se o seu dataset já tem demográficos num `.xlsx` parecido com OASIS, adicione uma função
`prepare_<nome>()` a `DATASET_PREPARERS` em `01_prepare_dataset.py` (mesmo padrão de
`prepare_oasis1`/`prepare_oasis2`) e uma entrada a `DATASET_PATTERNS` em
`02_extract_and_convert.py` se os dados brutos vierem em `.tar.gz`. Caso contrário, mais
simples é escrever um script próprio (fora deste repo ou como `01_prepare_dataset_<nome>.py`)
que produza os 3 arquivos da tabela acima — a partir daí todo o resto funciona sem
alteração de código.

```bash
export DATASET=meudataset   # troque pelo nome do seu dataset

# --- Pipeline base: BrainIAC vs radiomics vs volumetria (se os 3 arquivos acima existem) ---
bash src/03_run_brainiac_preprocessing.sh $DATASET
python src/04_extract_brainiac_features.py --dataset $DATASET
python src/05_train_classifiers_brainiac.py --dataset $DATASET
python src/06_radiomics_baseline.py --dataset $DATASET
python src/07_train_classifiers_radiomics.py --dataset $DATASET
python src/08_volumetry_baseline.py --dataset $DATASET
python src/09_evaluate_compare.py --results_dir results_$DATASET --figures_dir figures_$DATASET
python src/10_qc_preprocessing.py --dataset $DATASET

# --- Testes extra: fusão (1A/1B) + SHAP (2) - reusam as features do bloco acima ---
python src/20_prepare_fusion_features.py --dataset $DATASET
python src/21_train_fusion_early.py --dataset $DATASET
python src/22_train_fusion_stacking.py --dataset $DATASET
python src/23_shap_feature_selection.py --dataset $DATASET --feature_set radiomics
python src/23_shap_feature_selection.py --dataset $DATASET --feature_set fusion
python src/23_shap_feature_selection.py --dataset $DATASET --feature_set brainiac

# --- Teste extra: hipocampo (3A/3B) - precisa do env `synthseg` (ver Setup) ---
# ANTES de rodar: verifique a orientação do seu NIfTI bruto (ver nota em src/24) -
# a correção aplicada é especifica do OASIS-1, outros datasets normalmente não precisam.
python src/24_prepare_hippocampus_input.py --dataset $DATASET
conda activate synthseg  # ou: conda run -n synthseg
# --crop 224 224 224 e um chute seguro pro tamanho de cabeca adulta numa GPU de ~11GB
# (usamos 160x224x224 pro OASIS-1/2, ~256x256x160 nativo); se der OOM, reduza - mas
# nunca menor que a extensao real do cerebro na sua imagem (cheque com nibabel: bounding
# box dos voxels > threshold de fundo, e some uma margem).
python third_party/SynthSeg/scripts/commands/SynthSeg_predict.py \
  --i data/${DATASET}_hippo_input --o results_hippocampus_$DATASET/segmentations \
  --vol results_hippocampus_$DATASET/synthseg_volumes.csv --v1 --threads 4 --crop 224 224 224
conda activate brainiac-ad
python src/25_extract_hippocampus_features.py --dataset $DATASET
python src/26_train_classifiers_hippocampus.py --dataset $DATASET
python src/27_extract_brainiac_hippo_roi.py --dataset $DATASET
python src/04_extract_brainiac_features.py --dataset $DATASET \
  --processed_dir data/${DATASET}_brainiac_hippo_input \
  --output_csv features/${DATASET}_brainiac_hippo_features_raw.csv \
  --output_parquet features/${DATASET}_brainiac_hippo_features.parquet
python src/28_train_classifiers_brainiac_hippo.py --dataset $DATASET

# --- Consolidação: DeLong + Holm-Bonferroni vs o melhor baseline (auto-detectado) ---
python src/30_consolidate_dataset.py --dataset $DATASET

# --- Comparar 2+ datasets lado a lado (roda 30 para cada um antes) ---
python src/31_consolidate_overall.py --datasets oasis1 oasis2 $DATASET
```

O "melhor método base" usado como referência do DeLong em `src/30`/`src/31` é
auto-detectado (maior AUC média de CV entre BrainIAC/radiomics/volumetria daquele
dataset, ver `stats_utils.load_baseline_predictions`) — não precisa configurar nada.

## Resultados

### OASIS-1 

| Método | Melhor classificador | AUC (média CV ± dp) |
|---|---|---|
| BrainIAC (embeddings ViT congelados) | svm_linear | 0.750 ± 0.046 |
| Radiomics (PyRadiomics) | svm_rbf | 0.748 ± 0.012 |
| Volumetria clássica (eTIV/nWBV/ASF) | logreg | 0.784 ± 0.039 |


### OASIS-2 

| Método | Melhor classificador | AUC (média CV ± dp) |
|---|---|---|
| BrainIAC | logreg | 0.710 ± 0.115 |
| Radiomics | svm_rbf | 0.640 ± 0.112 |
| Volumetria | logreg | 0.633 ± 0.086 |

### Testes extra: fusão, seleção de features (SHAP), segmentação hipocampal


| Método | AUC OASIS-1 | AUC OASIS-2 | 
|---|---|---|
| **Baseline** | **0.748** (volumetria) | **0.710** (BrainIAC) | 
| Volume hipocampal (SynthSeg) | 0,807 | 0,723 | 
| Fusão late (stacking) | 0,763 | 0,705 | 
| Fusão early (concat) | 0,752 | 0,691 | 
| SHAP + fusão | 0,769 | 0,626 | 
| SHAP + BrainIAC (768d) | 0,754 | 0,642 | 
| SHAP + radiomics | 0,720 | 0,602 | 
| BrainIAC no ROI do hipocampo *(exploratório/OOD)* | 0,634 | 0,567 | 



### Bonn (epilepsia pediátrica: displasia cortical focal vs controles)

Terceiro dataset, usado para testar o mesmo pipeline fora do contexto AD/CN: 170 sujeitos
(85 DCF / 85 controles saudáveis, 3–13 anos), T1w em BIDS, CV 5-fold estratificado por
sujeito. Sem a volumetria clássica (eTIV/nWBV/ASF vêm da planilha do OASIS e não existem
aqui), então os baselines são só BrainIAC e radiomics. Resultados em `results_bonn/` e
`results_extra_bonn/`.

| Método | Melhor classificador | AUC (média CV ± dp) |
|---|---|---|
| BrainIAC (embeddings ViT congelados) | svm_linear | 0.804 ± 0.076 |
| Radiomics (PyRadiomics) | logreg | 0.791 ± 0.038 |

Testes extra (AUC das predições pooled de CV, teste de DeLong vs o melhor baseline;
Holm-Bonferroni na família primária):

| Método | AUC Bonn | Δ vs baseline 
|---|---|---|---|
| **Baseline** | **0,791** (radiomics) | — |
| Fusão early (concat) | 0,873 | +0,082 
| SHAP + fusão | 0,837 | +0,047 |
| Fusão late (stacking) | 0,817 | +0,026 
| SHAP + radiomics | 0,797 | +0,006 |
| SHAP + BrainIAC (768d) | 0,777 | −0,014 |
| BrainIAC no ROI do hipocampo *(exploratório/OOD)* | 0,740 | −0,050 | 
| Volume hipocampal (SynthSeg) | 0,558 | −0,233 

A fusão early é a única diferença significativa após correção. O volume hipocampal é
próximo do acaso — esperado, já que a DCF é uma lesão cortical, não hipocampal — e o
BrainIAC no ROI do hipocampo herda a mesma limitação, além de ser out-of-distribution
para o encoder.

## Estrutura do projeto

```
data/            dados brutos e processados de ambos os datasets (não versionado)
checkpoints/     pesos do BrainIAC e modelos treinados
features/        embeddings/features extraídas (parquet), prefixadas por dataset
results/         OASIS-1: métricas, predições por fold, modelos, summary
results_oasis2/  OASIS-2: idem, mesma estrutura
results_cross/   validação cruzada entre coortes: métricas, predições, summary final
results_fusion_{oasis1,oasis2}/       testes extra: fusão early/late
results_shap_{oasis1,oasis2}/         testes extra: seleção SHAP
results_hippocampus_{oasis1,oasis2}/  testes extra: segmentação/volume hipocampal
results_extra_{oasis1,oasis2}/        consolidado por dataset dos testes extra
results_extra/                        consolidado cross-dataset dos testes extra
results_bonn/, results_*_bonn/        Bonn (DCF vs controles): mesma estrutura por dataset
figures/         figuras comparativas (OASIS-1 + cross-dataset + testes extra)
figures_oasis2/  figuras comparativas do OASIS-2
logs/            stdout/stderr bruto de cada etapa
src/             scripts numerados por etapa, parametrizados por --dataset
third_party/     clones do BrainIAC e SynthSeg (submódulos lógicos, não git submodule)
```

## Reprodutibilidade

Seed fixa (`SEED=42`) para numpy/torch/sklearn em todos os scripts. Splits de CV
salvos em `data/{dataset}_folds.csv` e reutilizados por todos os métodos (BrainIAC,
radiomics, volumetria) para garantir comparação pareada nos mesmos sujeitos/folds
dentro de cada dataset. No cross-dataset, o mesmo checkpoint/pré-processamento/transform
é usado para os dois datasets - condição necessária para a comparação ser válida (ver
`PROGRESS_dataset2.md`, seção de verificação final).
