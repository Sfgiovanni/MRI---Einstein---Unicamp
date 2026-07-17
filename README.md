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

Pesos do BrainIAC: baixe o zip completo de checkpoints pelo link do Dropbox no README
de `third_party/BrainIAC` (~7.42GB, contém todos os modelos de downstream tasks — não
há como baixar seletivamente um único arquivo, o Dropbox sempre serve a pasta inteira),
extraia apenas `BrainIAC.ckpt` (362MB, encoder ViT-B) e coloque em `checkpoints/BrainIAC.ckpt`.

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

Todos os métodos caem em relação ao OASIS-1 (N menor, folds mais ruidosos), mas a
**volumetria cai muito mais** (0.784→0.633) que o BrainIAC (0.750→0.710). Investigação:
a separação de `nWBV` entre grupos (a feature mais discriminativa) é Δ=0.047 no OASIS-1
mas só Δ=0.020 no OASIS-2 — **achado que sugere que o forte desempenho da volumetria no
OASIS-1 é parcialmente um artefato do confundimento de idade** (Δ~7.7 anos lá vs
Δ~-0.87 anos aqui), não puramente sinal de doença. Ver `PROGRESS_dataset2.md` para a
verificação completa de que isso não é um bug. Sem teste de DeLong significativo entre
métodos (N pequeno, ver `results_oasis2/summary.md`).

### Validação cruzada entre coortes (experimento principal)

**Diagnóstico de domain-shift (verificação crítica adicional):** antes de interpretar
qualquer gap de generalização, testamos se um classificador consegue prever de qual
dataset um sujeito veio (ignorando o rótulo AD/CN). BrainIAC e radiomics separam
OASIS-1 e OASIS-2 com AUC de dataset-membership de **1.000** — um efeito de lote domina
completamente esses espaços de features. A hipótese inicial era que isso viesse de uma
assimetria de pré-processamento (OASIS-1 usava `SUBJ_111`, já processado pelo pipeline
do próprio OASIS; OASIS-2 usava `mpr-1` bruto). **Reprocessamos o OASIS-1 inteiro a
partir do `mpr-1` bruto** (mesmo nível de entrada do OASIS-2) para testar isso — e a
separabilidade continuou em ~1.0, **refutando** essa hipótese: o efeito de lote parece
ser intrínseco às duas coortes (scanner/protocolo/era de aquisição diferentes), não um
artefato do nosso pipeline. A volumetria (eTIV/nWBV/ASF), que não passa pelo nosso
pré-processamento de imagem, tem separabilidade bem menor (AUC=0.58) e por isso é mais
diretamente interpretável. Ver `results_cross/summary.md` e `PROGRESS_dataset2.md` para
o relato completo, incluindo a rodada original (superseded) que não pareava o nível de
entrada.

Tabela abaixo já usa o OASIS-1 reprocessado (`mpr-1`) para os lados cross-dataset de
BrainIAC/radiomics; volumetria e os AUCs intra-domínio exibidos são inalterados (canônicos,
via `SUBJ_111`):

| Método | Intra OASIS-1 | Intra OASIS-2 | OASIS1→OASIS2 | OASIS2→OASIS1 | Gap médio |
|---|---|---|---|---|---|
| BrainIAC | 0.750 | 0.710 | 0.639 | 0.487 | +0.146 |
| Radiomics | 0.748 | 0.640 | 0.616 | 0.717 | +0.020 |
| Volumetria | 0.784 | 0.633 | 0.650 | 0.784 | **-0.009** |

**Nota metodológica sobre o gap:** o gap OASIS1→OASIS2 de BrainIAC/radiomics NÃO usa o
AUC intra-domínio canônico (`SUBJ_111`) mostrado na tabela — isso misturaria "mudança de
coorte" com "mudança de pipeline de entrada" (o lado OASIS-1 do treino cross-dataset usa
`mpr-1`, não `SUBJ_111`). Em vez disso, usa um AUC intra-domínio recalculado nos mesmos
sujeitos/folds do OASIS-1 mas com `mpr-1` (0.707 para BrainIAC, 0.734 para radiomics) —
sem essa correção, o gap do BrainIAC estaria superestimado em quase 2x (0.115 vs o valor
correto 0.069). Ver `results_cross/summary.md` para o detalhamento completo.

**Leitura honesta, com a ressalva do domain-shift no centro — ver `results_cross/summary.md`
para a discussão completa:**
- **Nenhum método mostra evidência clara e isolável de generalização "genuína" de sinal
  de doença entre coortes.** BrainIAC e radiomics têm seus AUCs cross-dataset
  contaminados por um efeito de lote forte e irredutível (dataset-membership AUC~1.0
  mesmo após parear o pré-processamento); volumetria tem seu resultado dominado pelo
  confundidor de idade (gap com sinais opostos entre direções: +0.135/-0.152 — assinatura
  clássica de exploração de confundidor, não de robustez).
- **A hipótese original do experimento** ("BrainIAC empata intra-domínio mas degrada
  menos cross-dataset, evidenciando robustez de foundation model") **não encontra
  suporte claro nestes dados** — o desenho com 2 coortes da mesma instituição não isola
  bem essa pergunta, por conta do efeito de lote intrínseco descoberto. Uma leitura
  anterior (baseada no OASIS-1 não reprocessado) havia enfatizado a consistência de
  sinal do gap do BrainIAC como evidência qualitativa a favor da hipótese; essa leitura
  foi revisada para baixo após o diagnóstico de domain-shift, pois o espaço de features
  onde o gap é medido está dominado por um efeito não relacionado à doença.
- Um teste de mudança de site/scanner mais limpo exigiria MIRIAD ou OASIS-3 (acesso
  credenciado indisponível no momento).

**Ressalvas obrigatórias** (ver `results_cross/summary.md` para a lista completa):
OASIS-2 é do mesmo site/scanner que o OASIS-1 (não testa mudança real de site/scanner);
efeito de lote entre coortes não explicado pela assimetria de pré-processamento (ver
diagnóstico acima); os números intra-domínio do OASIS-1 usam `SUBJ_111` enquanto o lado
cross-dataset usa `mpr-1` reprocessado, então não são estritamente comparáveis em termos
de pipeline de entrada; confundimento de idade assimétrico entre os datasets; N
moderado nos dois lados.

## Estrutura do projeto

```
data/            dados brutos e processados de ambos os datasets (não versionado)
checkpoints/     pesos do BrainIAC e modelos treinados
features/        embeddings/features extraídas (parquet), prefixadas por dataset
results/         OASIS-1: métricas, predições por fold, modelos, summary
results_oasis2/  OASIS-2: idem, mesma estrutura
results_cross/   validação cruzada entre coortes: métricas, predições, summary final
figures/         figuras comparativas (OASIS-1 + cross-dataset)
figures_oasis2/  figuras comparativas do OASIS-2
logs/            stdout/stderr bruto de cada etapa
src/             scripts numerados por etapa, parametrizados por --dataset
third_party/     clone do repositório BrainIAC (submódulo lógico, não git submodule)
```

## Reprodutibilidade

Seed fixa (`SEED=42`) para numpy/torch/sklearn em todos os scripts. Splits de CV
salvos em `data/{dataset}_folds.csv` e reutilizados por todos os métodos (BrainIAC,
radiomics, volumetria) para garantir comparação pareada nos mesmos sujeitos/folds
dentro de cada dataset. No cross-dataset, o mesmo checkpoint/pré-processamento/transform
é usado para os dois datasets - condição necessária para a comparação ser válida (ver
`PROGRESS_dataset2.md`, seção de verificação final).
