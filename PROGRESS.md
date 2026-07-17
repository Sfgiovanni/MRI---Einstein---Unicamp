# PROGRESS — BrainIAC + ML pipeline for AD vs CN classification

## Status geral
- **Etapa atual:** ✅ PIPELINE COMPLETO, incluindo revisão de qualidade pós-hoc. BrainIAC+ML, radiomics e volumetria treinados e comparados (DeLong + figuras + summary.md). Verificação de leakage/consistência de folds entre os 3 métodos: OK. QC de pré-processamento em todos os 235 sujeitos: OK (sem skull-strips falhos). Gaps de reprodutibilidade encontrados numa revisão externa (advisor) foram corrigidos: `versioneer` faltando em `environment.yml`, e patch do `HD_BET/__init__.py` agora documentado explicitamente no `README.md` como passo de setup (antes só existia no clone local, não seria reproduzido do zero). Baseline opcional de 3D CNN do zero: **decisão do usuário (2026-07-10T02:31Z) foi pular**.
- **% concluído do pipeline total:** 100% do escopo obrigatório + QC extra de robustez.
- **Última atualização:** 2026-07-10T02:38Z
- **Tempo total decorrido:** ~2026-07-09T19:52Z a 2026-07-10T02:31Z (~6h40min de conversa, mas boa parte foi pesquisa de dataset/decisões + tempo de espera de downloads/instalação em background; tempo de computação real: ~15-20min setup ambiente, ~30min download OASIS-1, ~5min conversão NIfTI, ~1h52min pré-processamento BrainIAC (235 sujeitos), poucos minutos extração de features BrainIAC, ~34min extração radiomics, poucos minutos treino de classificadores e avaliação final).

## Resultado final (2026-07-10T02:30Z)

| Método | Melhor classificador | AUC (média CV ± dp) | AUC (pooled, usado no DeLong) |
|---|---|---|---|
| BrainIAC (embeddings ViT) | svm_linear | 0.750 ± 0.046 | 0.751 |
| Radiomics (PyRadiomics) | svm_rbf | 0.748 ± 0.012 | 0.731 |
| Volumetria (eTIV/nWBV/ASF) | logreg | 0.784 ± 0.039 | 0.776 |

**Teste de DeLong (BrainIAC vs cada baseline, predições pooled de CV, pareadas por sujeito):**
- BrainIAC vs Radiomics: diferença = +0.020, p = 0.565 (não significativo)
- BrainIAC vs Volumetria: diferença = -0.024, p = 0.411 (não significativo)

**Conclusão factual:** neste dataset (OASIS-1, N=235, split CDR≥0.5 vs CDR=0) e protocolo, **BrainIAC (linear probe sobre embeddings congelados) não superou estatisticamente nenhum dos dois baselines**. O baseline de volumetria clássica (apenas 3 features: eTIV/nWBV/ASF) teve o maior AUC pontual, mas a diferença não é estatisticamente significativa dado o N moderado. Nenhum resultado foi ajustado ou descartado para favorecer uma narrativa - números batem exatamente com `results/predictions/*.csv` e `results/metrics_*.csv`.

## Verificação de integridade (2026-07-10T02:30Z)
- Confirmado programaticamente: os 3 métodos (brainiac/radiomics/volumetria) usam **exatamente os mesmos 235 sujeitos**, com **fold assignments idênticos** a `data/oasis1_folds.csv`, e labels consistentes em todas as predições salvas — condição necessária para a comparação pareada (DeLong) ser válida. Sem sujeitos duplicados em nenhum conjunto de predições.

## Revisão externa (advisor) e correções (2026-07-10T02:31-02:38Z)
Antes de declarar o pipeline concluído, pedi uma revisão crítica externa. Achados e correções:
1. **Reprodutibilidade quebrada num clone limpo (bloqueante, corrigido):** `pip install --no-build-isolation pyradiomics` falharia num ambiente novo porque `versioneer` (dependência de build do setup.py legado do pyradiomics) não estava listado em nenhum lugar — só foi instalado manualmente nesta sessão. Corrigido: adicionado `versioneer` a `environment.yml`.
2. **Patch não documentado (bloqueante, corrigido):** o `HD_BET/__init__.py` vazio que criei existe apenas no clone local de `third_party/BrainIAC` (não versionado). Um `git clone` limpo do BrainIAC não teria esse arquivo, e a Etapa 3 falharia com `ModuleNotFoundError`. Corrigido: passo explícito de `touch .../__init__.py` adicionado ao `README.md` logo após o clone.
3. **QC de preprocessamento faltando (recomendado, executado):** só havia validado 1 sujeito piloto visualmente; rodei checagem em todos os 235 (`src/10_qc_preprocessing.py` → `results/qc_preprocessing.csv`) para descartar a hipótese de que skull-strips falhos/parciais (silenciosos no HD-BET modo fast) estivessem degradando as features do BrainIAC/radiomics de forma assimétrica em relação à volumetria (que vem de um pipeline totalmente independente do OASIS). **Resultado: distribuição estreita e consistente** (fração de voxels não-zero: média 0.230, dp 0.021, min 0.185, max 0.296, mesmo shape 170x206x162 em todos os 235) — sem evidência de falha sistemática. Isso reforça que a conclusão "BrainIAC não supera os baselines" reflete desempenho real dos métodos, não um artefato de pré-processamento corrompido.
4. **Notas estatísticas menores (documentadas em `results/summary.md`, sem re-execução):** (a) AUC "média de fold" vs AUC "pooled" usada no DeLong são quantidades diferentes do mesmo conjunto de predições — sinalizado explicitamente para não parecer erro de digitação; (b) seleção do "melhor classificador" por método usa a mesma métrica depois comparada via DeLong (selection-on-the-test-metric) — mencionado como limitação, mas o viés joga *contra* a conclusão de não-superioridade do BrainIAC (não a favor), então não compromete a conclusão principal.

## Validações concluídas (2026-07-09T20:45-20:48Z)
- Ambiente conda `brainiac-ad` criado com sucesso (torch 2.3.1+cu121, monai 1.3.2, pytorch-lightning 2.3.3, etc.). `pyradiomics` precisou de instalação separada com `pip install --no-build-isolation` + `versioneer` como dependência de build (setup.py legado importa numpy/versioneer no build sem declarar como build-requirement) — funcionou, `pyradiomics==3.0.1` instalado.
- **Forward pass do BrainIAC validado**: carregado `checkpoints/BrainIAC.ckpt`, volume dummy (1,1,96,96,96) → saída (1, 768) em `cuda:0`. Confirma embedding_dim=768 conforme esperado.
- GPU: 3.94GB livres de 11.34GB no momento do teste (segue compartilhada com outro processo).
- **Download OASIS-1 completo**: 12/12 discs, 16GB total, ~2min45s/disco em média (20:17Z-20:47Z, ~30min).
- Adicionado `third_party/BrainIAC/src/preprocessing/HD_BET/__init__.py` (vazio) — o diretório HD_BET vem sem `__init__.py` no repo original, o que impediria `from HD_BET.hd_bet import hd_bet` de funcionar como pacote; corrigido para tornar importável localmente (parâmetros do modelo HD-BET em modo "fast" já vêm inclusos em `hd-bet_params/0.model`, sem necessidade de download extra).

## Bugs encontrados e corrigidos (2026-07-09T23:5X Z)
- `01_prepare_dataset.py` rodado com sucesso: 135 CN / 100 AD / 235 total, folds 5x estratificados por sujeito (27/20 por fold), salvos em `data/oasis1_labels.csv` e `data/oasis1_folds.csv`.
- **Bug em `02_extract_and_convert.py`**: `subject_id` nos labels não tem sufixo `_MR` (ex. `OAS1_0001`), mas as pastas dentro dos `.tar.gz` do OASIS-1 usam `OAS1_0001_MR1`/`_MR2`. Primeira execução converteu 0/235 sujeitos. Corrigido: comparar pelo `base_id` (removendo `_MR{n}`) e **filtrar explicitamente `_MR1`** (ignorar `_MR2`, presente em 20 sujeitos com sessão de reteste) para evitar pegar 4 arquivos (2 sessões x hdr/img) em vez de 2. Re-executado com sucesso: **235/235 sujeitos convertidos** para NIfTI (`data/oasis1_raw/nifti/`, 1.6GB), ~4m40s de execução real (20:50:14 a 20:54:52 horário local -03:00 = 23:50-23:54Z).
- **Teste piloto de pré-processamento (5 sujeitos)**: rodou sem erros. Registro rígido (SimpleITK, N4+Euler3D) ~13-24s/imagem; HD-BET (skull-strip, modo fast) rápido. **Tempo total: 1m34s para 5 sujeitos (~19s/sujeito)**. Output validado (shape 170x206x162, ~21% voxels não-zero, intensidades 128-3390, afim plausível) — sem sinais de corrupção.
- **Bug encontrado no teste piloto**: HD-BET preserva o sufixo `_0000` no nome do arquivo de saída (exigido como sufixo de entrada), gerando `OAS1_0001_0000.nii.gz` em vez de `OAS1_0001.nii.gz`. Corrigido em `03_run_brainiac_preprocessing.sh`: passo de rename após o preprocessing para remover o sufixo `_0000`, compatibilizando com o nome esperado por `get_brainiac_features.py`/`BrainAgeDataset` (`{pat_id}.nii.gz`).
- **Pré-processamento completo dos 235 sujeitos concluído**: iniciado 2026-07-09T23:58:03Z, concluído 2026-07-10T01:50:30Z — **1h52min reais (~28.6s/sujeito em média)**, um pouco mais lento que o piloto (19s/sujeito), provavelmente por contenção sustentada de GPU/CPU compartilhada ao longo da execução completa. Verificado: 235/235 arquivos gerados em `data/oasis1_processed/`, renomeados corretamente (sem sufixo `_0000` residual).

## ETA — final (pipeline concluído em 2026-07-10T02:31Z, nada mais pendente)
Gargalos reais, do maior para o menor: (1) pré-processamento BrainIAC (1h52min, 235 sujeitos), (2) download dos 12 discs OASIS-1 (~30min, 16GB), (3) extração de radiomics (~34min), (4) setup do ambiente conda (~20min, torch+CUDA). Extração de features BrainIAC e treino de classificadores foram rápidos (~1min cada). Nenhuma etapa restante.

## Estimativa de término (ETA)
Ainda não é possível estimar com confiança — faltam dados de: (a) tamanho do dataset OpenNeuro escolhido, (b) tempo de download, (c) tempo de pré-processamento por scan (N4 + registro + skull-strip), (d) tempo de extração de features por scan na GPU (compartilhada, ver nota abaixo).
Regra: ETA será recalculado a cada etapa concluída, com base no tempo real gasto por unidade (por scan / por fold), não em suposições.

**Gargalos esperados, em ordem provável de impacto:**
1. Pré-processamento (N4 bias correction + registro MNI + skull-strip) — tipicamente a etapa mais lenta por scan em CPU.
2. Download do dataset OpenNeuro (depende do tamanho escolhido).
3. Extração de features na GPU — **GPU é compartilhada com outro processo** (ver Problemas & Decisões), então o tempo real pode ser maior que o nominal.
4. Treino dos classificadores clássicos — deve ser rápido (segundos a minutos por fold).

## Checklist de etapas

| # | Etapa | Estado | Início | Fim | Duração |
|---|-------|--------|--------|-----|---------|
| 0 | Setup ambiente + hardware check | ✅ | 2026-07-09T19:52Z | 2026-07-09T19:53Z | ~1min |
| 1 | Clone BrainIAC + download pesos + validação forward pass | ✅ | 2026-07-09T19:53Z | 2026-07-09T20:48Z | ~55min (dominado pelo download do zip de 7.4GB) |
| 2 | Escolha e download dataset (OASIS-1, no lugar de OpenNeuro) | ✅ | 2026-07-09T20:10Z | 2026-07-09T20:47Z | ~37min (pesquisa OpenNeuro + decisão usuário + download 16GB) |
| 3 | Pré-processamento (registro + N4 + HD-BET) | ✅ | 2026-07-09T23:58Z | 2026-07-10T01:50Z | 1h52min (235 sujeitos, ~28.6s/sujeito) |
| 4 | Extração de features (BrainIAC encoder) | ✅ | 2026-07-10T01:51Z | 2026-07-10T01:51Z | ~1min |
| 5 | Classificadores de ML sobre features BrainIAC (CV k=5 por sujeito) | ✅ | 2026-07-10T01:52Z | 2026-07-10T01:52Z | ~1min |
| 6 | Baseline: Radiomics (PyRadiomics) + classificadores | ✅ | 2026-07-10T01:54Z | 2026-07-10T02:29Z | 35min (34min extração + 1min treino) |
| 7 | Baseline: Volumetria/morfometria (eTIV/nWBV/ASF oficiais do OASIS-1) | ✅ | 2026-07-10T01:54Z | 2026-07-10T01:54Z | <1min (features já existiam nos demográficos) |
| 8 | Baseline opcional: 3D CNN do zero | ❌ pulado (decisão do usuário 2026-07-10T02:31Z — GPU compartilhada com pouca VRAM + N=235 tornaria o baseline fraco/instável) | — | — | — |
| 9 | Avaliação, comparação estatística (DeLong), figuras | ✅ | 2026-07-10T02:29Z | 2026-07-10T02:29Z | <1min |
| 10 | Entregáveis finais (README, summary.md/csv) | ✅ | 2026-07-10T02:31Z | 2026-07-10T02:31Z | poucos min |

## Log cronológico
- `2026-07-09T19:52Z` — Iniciado trabalho. Diretório de projeto: `/home/franco/MRI` (não era repositório git).
- `2026-07-09T19:52Z` — `nvidia-smi`: 1x NVIDIA GeForce RTX 2080 Ti, 11264 MiB VRAM total. **ATENÇÃO:** já há um processo `python` (PID 357269) usando 6904 MiB e GPU-Util 99% no momento da checagem — GPU é compartilhada com outra carga de trabalho, não dedicada. VRAM livre estimada: ~4.3 GB.
- `2026-07-09T19:52Z` — CPU: 24 cores. RAM: 15 GiB total, ~12 GiB disponível (uso considerado modesto para pipelines de neuroimagem tipo FreeSurfer `recon-all`, que tipicamente pede ~4-8GB por instância — deve funcionar mas evitar paralelismo agressivo).
- `2026-07-09T19:52Z` — Disco: 333 GB livres em `/home` (montado em `/dev/sdb1`, 880G total). Suficiente para um dataset de porte moderado (dezenas de sujeitos T1w) + derivativos.
- `2026-07-09T19:52Z` — Ferramentas ausentes detectadas: `datalad`, `docker`, FSL (`flirt`), ANTs (`antsRegistration`, `N4BiasFieldCorrection`), FreeSurfer (`recon-all`), `nvcc` standalone. Presentes: `git` 2.43.0, `python3` 3.13.12, `pip`, `conda` (miniconda3).
- `2026-07-09T19:52Z` — Criada estrutura de diretórios: `data/ checkpoints/ features/ results/ figures/ logs/ src/ third_party/`. `git init` executado, branch renomeada para `main`.
- `2026-07-09T19:53Z` — Iniciando Etapa 1: clonar BrainIAC e ler README para entender setup oficial, formato de pré-processamento e método de download de pesos.

## Problemas & decisões
- **GPU compartilhada:** a RTX 2080 Ti já está sendo usada por outro processo (99% util, 6.9GB VRAM) no início do trabalho. Decisão: prosseguir, mas (a) usar batch size = 1 para extração de features (volumes 3D são grandes e a VRAM livre é limitada), (b) monitorar OOM e cair para CPU se necessário, (c) re-checar `nvidia-smi` periodicamente e registrar aqui se isso virar gargalo real.
- **Python 3.13.12 via conda:** é uma versão muito recente; pacotes de neuroimagem/ML mais antigos (PyRadiomics, certas versões de PyTorch/SimpleITK) podem não ter wheels compatíveis. Decisão: criar um ambiente conda **dedicado e isolado** com uma versão de Python mais conservadora (provavelmente 3.10 ou 3.11) para este projeto, em vez de usar o Python 3.13 do sistema/base. Será documentado em `environment.yml`.
- Ferramentas de pré-processamento (FSL/ANTs/FreeSurfer) precisarão ser instaladas via conda-forge/pip ou containers leves — decisão final será tomada após ler o README do BrainIAC para saber exatamente o que ele exige.

## Métricas parciais (2026-07-10T01:55Z)

**BrainIAC (embeddings ViT 768-d) + classificadores, CV 5-fold por sujeito, AUC (média ± dp entre folds):**
| Classificador | AUC |
|---|---|
| logreg | 0.749 ± 0.040 |
| svm_linear | 0.750 ± 0.041 |
| svm_rbf | 0.706 ± 0.056 |
| random_forest | 0.720 ± 0.068 |
| xgboost | 0.699 ± 0.051 |
| lightgbm | 0.710 ± 0.045 |

**Baseline volumetria (eTIV, nWBV, ASF oficiais do OASIS-1) + classificadores, mesmo protocolo:**
| Classificador | AUC |
|---|---|
| logreg | 0.784 ± 0.035 |
| svm_linear | 0.780 ± 0.039 |
| svm_rbf | 0.749 ± 0.042 |
| random_forest | 0.706 ± 0.089 |
| xgboost | 0.700 ± 0.096 |
| lightgbm | 0.723 ± 0.095 |

**Achado preliminar (honesto, não escondido):** o baseline de volumetria clássica (apenas 3 features: eTIV/nWBV/ASF) supera ligeiramente o BrainIAC (0.784 vs 0.749 no melhor classificador de cada). Isso é plausível e consistente com a literatura: nWBV é um biomarcador macroestrutural forte e bem estabelecido para AD, e há confundimento de idade conhecido no nosso split (ver acima) que pode inflar o sinal volumétrico. Será reportado como resultado factual em `results/summary.md`, não ajustado/escondido.

Radiomics (PyRadiomics) ainda em extração (~10s/sujeito, ETA ~35-40min a partir de 01:54Z).

## Achados da Etapa 1 (BrainIAC) — 2026-07-09T19:56Z
- Backbone: ViT-B (MONAI `ViT`), `img_size=(96,96,96)`, `patch_size=16`, `hidden_size=768`, 12 camadas, 12 heads. **Embedding = CLS token, dimensão 768.**
- Pré-processamento oficial (`src/preprocessing/mri_preprocess_3d_simple.py`, usado no `quickstart.ipynb`): (1) registro rígido (Euler3D, SimpleITK, mutual information) para um template (`preprocessing/atlases/temp_head.nii.gz`, resample 1mm isotrópico) + correção de bias N4 (`sitk.N4BiasFieldCorrection`); (2) skull-strip via **HD-BET** (rede U-Net, bundled em `src/preprocessing/HD_BET/`, modo `fast`). Não é MNI152 padrão — é o `temp_head.nii.gz` fornecido no próprio repo.
- Ao extrair features (`get_brainiac_features.py` → `dataset.py: get_validation_transform`): `Resized` para 96×96×96 (trilinear) + `NormalizeIntensityd(nonzero=True, channel_wise=True)`. Sem augmentation na inferência.
- Pesos: **não há download automatizado no repo** — link é uma pasta Dropbox (`checkpoints.zip`, ~7.42 GB, contém TODOS os checkpoints de downstream tasks, não só o encoder BrainIAC). Dropbox serve a pasta inteira como zip independente do parâmetro `dl`; não há como baixar seletivamente um único arquivo antes de baixar o zip completo (limitação do Dropbox, não nossa). Fonte HF `ilex-hub/brainiac.1` mencionada na tarefa ainda não foi verificada (checar após o zip, só se necessário).
- Ação: iniciado download em background do zip completo (`checkpoints/checkpoints_full.zip.part`, PID 442475, `wget -c`), pois é a única forma de obter `BrainIAC.ckpt`.

## Achados da Etapa 2 (dataset OpenNeuro) — 2026-07-09T20:10Z — ⚠️ BLOQUEIO METODOLÓGICO
Busca exaustiva por dataset OpenNeuro com MRI T1w estrutural + rótulo AD vs CN:
- **API GraphQL do OpenNeuro** (`https://openneuro.org/crn/graphql`, introspectada manualmente): campo `advancedSearch(query: DatasetSearchInput)` tem um campo estruturado `diagnosis`. Testado com termos: "Alzheimer's Disease", "Alzheimer", "alzheimer", "Dementia", "dementia", "MCI", "AD", "cognitive impairment" → **apenas "Dementia" retornou 1 resultado**, `ds003800` ("Auditory Gamma Entrainment"), que é um estudo de EEG/potenciais auditivos, não MRI estrutural de coorte AD/CN.
- **Busca por repositório no GitHub org `OpenNeuroDatasets`** (mirror 1:1 de todo dataset público do OpenNeuro) com termos: alzheimer, dementia, "cognitive impairment", amnestic, ADRC, neurodegenerative, "aging cognitive", "memory clinic", APOE, OASIS, AIBL, CDR/"clinical dementia rating" → únicos hits relevantes:
  - `ds007561` — "UCB-J PET and T1W Images for Synapse Project Cohort". `participants.tsv` real (verificado via GitHub raw) mostra **apenas 20 sujeitos**: 1 AD, 5 MCI, 14 Control. A descrição do estudo-mãe (ClinicalTrials/protocolo) menciona N=120 (60 CN/30 MCI/30 AD) mas **isso NÃO está refletido nos dados atualmente publicados no OpenNeuro (versão 1.0.0)** — upload parece parcial/incompleto.
  - `ds004767` — atlas MRI ex-vivo + histologia de lobo temporal medial — não é coorte clínica in-vivo com grupos AD/CN.
  - `ds007522` — "Precision Aging Network" (>1000 sujeitos) — verificado `participants.tsv`: só tem `age, sex, handedness, site`, **sem coluna de diagnóstico** (coorte de envelhecimento saudável, não clínica).
  - `ds004796` (PEARL-Neuro), `ds004504`/`ds006036` (EEG, não MRI), `ds007427` (EEG) — não servem (modalidade errada ou sem T1w+diagnóstico).
- **Conclusão factual:** no momento desta busca (2026-07-09), **não existe no OpenNeuro um dataset de MRI T1w estrutural dedicado a classificação AD vs CN com N razoável (≥30/classe)**. O único com rótulo explícito de diagnóstico (AD/MCI/Control) + T1w é `ds007561`, mas com apenas 1 sujeito AD — estatisticamente inviável para classificação binária com CV.
- **Decisão do usuário (2026-07-09T20:12Z):** perguntado via pergunta estruturada. Usuário escolheu **relaxar a restrição de "deve ser OpenNeuro"** e usar **OASIS-1** (Open Access Series of Imaging Studies, cross-sectional), o benchmark padrão da literatura para classificação AD vs CN, em vez de prosseguir com o dataset OpenNeuro subdimensionado (ds007561, n=1 AD). Isso será registrado explicitamente no `README.md` e no paper como desvio documentado da instrução original.

## Dataset final escolhido: OASIS-1 (cross-sectional) — 2026-07-09T20:15Z
- **Fonte:** https://sites.wustl.edu/oasisbrains/home/oasis-1/ — acesso via URLs de download diretas (`download.nrg.wustl.edu`), funcionam sem autenticação prévia (testado com `wget`, HTTP 200, `Content-Length` correto). O site pede para aceitar um "academic use agreement" via formulário — este projeto é uso acadêmico não comercial, dentro do uso pretendido pelos dados sob licença ODC-by.
- **N total no dataset:** 416 sujeitos, 18-96 anos. **436 linhas na planilha de demográficos** (`oasis_cross-sectional-*.xlsx`, algumas com sessões `MR2` repetidas — checar duplicatas de sujeito antes do split).
- **Distribuição de CDR (Clinical Dementia Rating)** entre os sujeitos clinicamente avaliados (subset de idosos, os demais são jovens/meia-idade não avaliados clinicamente, CDR=`None`, n=201):
  - CDR=0 (sem demência): **n=135**
  - CDR=0.5 (demência muito leve): n=70
  - CDR=1 (demência leve): n=28
  - CDR=2 (demência moderada): n=2
  - CDR=None (não avaliado, jovens/meia-idade): n=201 — **excluídos do binário**, não usados como CN para evitar confundimento de idade.
- **Mapeamento binário definido (regra explícita):** seguindo a própria convenção do artigo original do OASIS-1 (Marcus et al. 2007), que define "demented" = CDR>0:
  - **CN (label=0) = CDR == 0** → n=135
  - **AD (label=1) = CDR >= 0.5** (0.5, 1 ou 2 — muito leve a moderada) → n=100 (70+28+2)
  - Nenhum sujeito excluído por ambiguidade de estágio, pois CDR>0 já é diagnóstico clínico formal de demência (não há categoria "MCI" separada nesta versão do OASIS; CDR 0.5 é classificado pelos próprios autores como demência muito leve, não como comprometimento cognitivo leve não-demencial).
  - **Total usado: 235 sujeitos** (135 CN + 100 AD), satisfaz a meta de ≥30/classe com folga.
- **⚠️ Confundimento de idade (documentar como limitação):** grupo CN (CDR=0) tem idade média 69.1 anos (min 33, max 94); grupo AD (CDR≥0.5) tem idade média 76.8 anos (min 62, max 96) — diferença de ~7.7 anos entre médias. Isso é um confundidor conhecido e documentado na literatura sobre OASIS-1 (idade correlaciona com atrofia cerebral independente de AD). Será reportado explicitamente nas limitações do `results/summary.md`; não faremos correção formal (ex.: age-matching ou regressão de idade) neste pipeline por ora, mas é uma ameaça real à validade interna que deve ser mencionada no paper.
- **Baseline de volumetria/morfometria — decisão de custo:** a planilha de demográficos do OASIS-1 já inclui **eTIV** (volume intracraniano estimado), **nWBV** (volume cerebral total normalizado) e **ASF** (fator de escala do atlas) — medidas volumétricas macroestruturais computadas oficialmente pelo grupo do OASIS (via pipeline baseado em SPM, não FreeSurfer, mas conceitualmente equivalente e amplamente usado como biomarcador de referência para AD na literatura, ex. Fotenos et al. 2005). **Decisão:** usar essas 3 features oficiais como baseline de "volumetria clássica" em vez de rodar FreeSurfer `recon-all` (muito lento nesta CPU, seria preciso ~1-3h/sujeito × 235 sujeitos) ou baixar os discs FreeSurfer-processed oficiais do OASIS (11 discs × ~9.5GB ≈ 104GB, ~2.9h de download). Isso segue a orientação explícita da tarefa de usar alternativa mais barata quando o custo é alto, documentando a escolha. Caso sobre tempo depois, podemos enriquecer com volumes regionais de 1-2 discs FreeSurfer como extensão.
- **Download em andamento:** todos os 12 discs de MRI raw (`oasis_cross-sectional_disc1..12.tar.gz`, ~1.3-1.5GB cada, ~16-18GB total) sendo baixados sequencialmente em background (`logs/download_oasis_discs.log`, PID 443980) — precisamos da maioria/todos os discs pois os 235 sujeitos de interesse (CDR definido) parecem distribuídos ao longo de todos os discs, não concentrados. Vamos filtrar para os 235 sujeitos relevantes após extração, descartando os demais (jovens/meia-idade sem CDR) do processamento pesado (pré-processamento BrainIAC, radiomics), embora os arquivos baixados sejam mantidos em disco (custo de armazenamento desprezível face aos 333GB livres).
