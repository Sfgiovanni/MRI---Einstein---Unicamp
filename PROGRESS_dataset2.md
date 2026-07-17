# PROGRESS_dataset2 — Extensão: segundo dataset + validação cruzada entre coortes

## Status geral
- **Etapa atual:** ✅ PIPELINE COMPLETO (com correção pós-verificação). Replicação intra-domínio no OASIS-2 e validação cruzada entre coortes concluídas. Um diagnóstico adicional de domain-shift (não previsto no pedido original, sugerido em revisão) revelou que os resultados cross-dataset da primeira rodada estavam potencialmente confundidos por uma assimetria de pré-processamento entre os dois lados do OASIS-1 usado; OASIS-1 foi reprocessado a partir do dado bruto (mpr-1) para corrigir isso, e a validação cruzada + comparação final foram recalculadas. Ver seção "Correção pós-verificação" abaixo para o relato completo. Etapa opcional 7 (pooled+ComBat) não executada (fora do essencial pedido).
- **% concluído do pipeline total:** 100% do escopo obrigatório (etapas 1-6 e 8 do pedido original), incluindo a correção pós-verificação
- **Última atualização:** 2026-07-10T11:53Z

## Decisão do dataset 2 (2026-07-10T03:06-03:12Z) — ⚠️ mudança em relação ao pedido original
O pedido original pedia MIRIAD (com OASIS-3 como alternativa). Pesquisei o acesso a ambos antes de comprometer tempo:
- **MIRIAD**: hospedado pelo UCL Dementia Research Centre. O site institucional (`ucl.ac.uk/.../miriad`) retorna **HTTP 403** para acesso automatizado (bot-block), e o texto da própria página/NITRC confirma: *"Yes, registration is required. Users must register and download data from the site to proceed."* Isso é diferente do OASIS-1, cujo "acordo acadêmico" era só um formulário de honra sem barreira real — aqui parece haver de fato criação de conta com possível aprovação humana pelo UCL DRC (contato `drc-miriad@ucl.ac.uk`). Não é algo que eu consiga completar autonomamente.
- **OASIS-3**: documentação oficial da própria WashU confirma processo formal: assinar o termo de uso em oasis-brains.org → **espera de ~1 semana** por um convite por email ao XNAT Central → só então criar conta e baixar. Prazo determinístico mas incompatível com continuar agora.
- **OASIS-2 (não pedido originalmente, descoberto durante a pesquisa)**: mesma família do OASIS-1, mesmo mecanismo de download direto (`download.nrg.wustl.edu`, sem barreira real de autenticação — confirmado, mesmo padrão que funcionou para o OASIS-1). 150 sujeitos, longitudinal (2+ visitas/sujeito), **mesmo schema exato de rótulo/planilha do OASIS-1** (colunas `CDR, eTIV, nWBV, ASF` idênticas) — elimina a necessidade de harmonizar volumetria com SynthSeg/FastSurfer (Etapa 4 opção b do pedido original vira desnecessária, já temos volumetria diretamente comparável).
- **Trade-off explícito, apresentado ao usuário:** OASIS-2 é da MESMA instituição/scanner que o OASIS-1 (Washington University) — o teste cross-dataset mede generalização entre **coortes distintas, mesmo site/scanner**, não mudança real de site/scanner (a alegação mais forte de "foundation model generaliza mesmo com domain shift de aquisição"). MIRIAD (site diferente, UCL) daria um teste mais forte dessa alegação, mas exige ação do usuário + espera indefinida.
- **Decisão do usuário (2026-07-10T03:12Z): usar OASIS-2 agora.** Essa limitação (mesmo site/scanner) será reportada de forma explícita e proeminente em todas as conclusões sobre o experimento de validação cruzada — a interpretação do "gap de generalização" deve ser lida como "generalização entre coortes" e não "robustez a mudança de scanner/site".

## Verificação de não-sobreposição de sujeitos OASIS-1 x OASIS-2
IDs usam prefixos de namespace disjuntos por design (`OAS1_XXXX` vs `OAS2_XXXX`), e a documentação oficial do WashU/OASIS descreve OASIS-1 (cross-sectional) e OASIS-2 (longitudinal) como coortes de sujeitos distintas recrutadas para propósitos diferentes. Não há mecanismo de overlap de ID possível dado o prefixo diferente. Nenhum sujeito do OASIS-2 aparece na planilha do OASIS-1 (`data/oasis1_labels.csv`) e vice-versa — confirmado por inspeção dos prefixos.

## Dataset 2 final: OASIS-2 (longitudinal) — parsing de rótulos (2026-07-10T03:10Z)
- Fonte: https://sites.wustl.edu/oasisbrains/home/oasis-2/, planilha `oasis_longitudinal_demographics` (373 sessões, 150 sujeitos únicos).
- **Seleção de 1 scan/sujeito:** mantida apenas a sessão baseline (`MRI ID` termina em `_MR1`), igual à regra usada no OASIS-1 (mesmo raciocínio: 1 imagem por sujeito, sem vazamento de sessões repetidas). 150 sujeitos após filtro (1 linha por sujeito, confirmado).
- **Mapeamento binário (idêntico à regra do OASIS-1, aplicado ao CDR do baseline):** CN (label=0) = CDR==0 (n=85); AD (label=1) = CDR>=0.5 (n=65). Coluna `Group` do OASIS-2 (`Nondemented/Demented/Converted`) NÃO foi usada diretamente para o rótulo — usamos só o CDR bruto, para manter a regra 100% idêntica à do OASIS-1 e não introduzir um critério de rotulagem diferente entre datasets.
  - Nuance documentada: 13 sujeitos rotulados `Converted` (nondemented no baseline, T mas que desenvolveram demência em visitas futuras) têm CDR=0 no baseline e por isso entram no grupo CN pela regra estrita. Isso é ruído de rótulo potencial (podem ser pré-clínicos), mas é o preço de manter a regra idêntica ao OASIS-1 — mantido por consistência metodológica, não por conveniência.
- **Idade por grupo (comparar com o confundidor do OASIS-1, Δ~7.7 anos):** CN media=75.8 anos (min 60, max 93); AD media=75.0 anos (min 61, max 96). **Δ(AD-CN) = -0.87 anos — OASIS-2 é bem mais balanceado em idade que o OASIS-1.** Achado relevante: se o BrainIAC "ganhar" mais no OASIS-2 que no OASIS-1, isso é consistente com o sinal de idade estar inflando o desempenho da volumetria no OASIS-1 (hipótese a ser discutida nas conclusões, não afirmada sem evidência adicional).
- Sem valores faltantes de eTIV/nWBV/ASF nos 150 sujeitos de baseline.
- Todos os subject IDs confirmados com prefixo `OAS2_`, sem overlap possível com OASIS-1.

## Log cronológico
- `2026-07-10T03:06Z` — Retomado o projeto para a extensão de dataset 2. GPU agora livre (0% util, 49MiB usados). Ambiente conda `brainiac-ad` presente.
- `2026-07-10T03:06Z` — Pesquisa de acesso ao MIRIAD: página institucional retorna HTTP 403 para wget/WebFetch; conteúdo indica registro de conta exigido, não documentado como automatizável.
- `2026-07-10T03:08Z` — Pesquisa de acesso ao OASIS-3: confirmado processo de ~1 semana de espera por convite ao XNAT Central após assinar termo de uso.
- `2026-07-10T03:09Z` — Testado `download.nrg.wustl.edu/data/OAS2_RAW_PART1.tar.gz`: HTTP 200, mesmo padrão de acesso direto do OASIS-1. Baixada a planilha de demográficos do OASIS-2 e confirmado schema idêntico ao OASIS-1 (CDR/eTIV/nWBV/ASF).
- `2026-07-10T03:12Z` — Pergunta estruturada feita ao usuário com 3 opções (OASIS-2 agora / registrar MIRIAD e aguardar / assinar termo OASIS-3 e aguardar ~1 semana). **Usuário escolheu OASIS-2.**
- `2026-07-10T03:12Z` — Iniciado download em background dos 2 arquivos brutos do OASIS-2 (`OAS2_RAW_PART1.tar.gz` ~10GB, `OAS2_RAW_PART2.tar.gz` ~8GB, total ~18GB).
- `2026-07-10T03:13-03:21Z` — Generalizados `src/01_prepare_dataset.py` e `src/02_extract_and_convert.py` para aceitar `--dataset {oasis1,oasis2}` em vez de duplicar a lógica (conforme pedido). Lógica de folds (StratifiedKFold) extraída para `common_cv.make_stratified_subject_folds`, reutilizada por ambos. **Regressão verificada**: rodar com `--dataset oasis1` reproduz `data/oasis1_labels.csv`/`data/oasis1_folds.csv` byte-a-byte idênticos aos já existentes.
- `2026-07-10T03:14Z` — Rodado `01_prepare_dataset.py --dataset oasis2`: 85 CN / 65 AD / 150 total, folds 5x estratificados (17/13 por fold).
- `2026-07-10T03:18Z` — **Achado estrutural importante:** o arquivo bruto do OASIS-2 (`OAS2_RAW_PARTn.tar.gz`) só contém a pasta `RAW/` (repetições individuais `mpr-1..4.nifti.{hdr,img}`), SEM o equivalente a `PROCESSED/MPRAGE/SUBJ_111` do OASIS-1 (a média motion-corrected oficial). Decisão: usar `mpr-1` (primeira repetição bruta) como volume representativo por sujeito, em vez de inventar um pipeline de média não validado. **Assimetria documentada**: o input do OASIS-1 passou por correção de movimento+média oficial do OASIS antes do pré-processamento do BrainIAC; o do OASIS-2 não. O pré-processamento do BrainIAC em si (registro+N4+HD-BET) é idêntico para os dois.
- `2026-07-10T03:20-03:22Z` — Renomeados `features/brainiac_features.parquet` → `features/oasis1_brainiac_features.parquet` (e radiomics/raw analogamente) para consistência de nomenclatura com o padrão `{dataset}_` usado em `data/`. **Apenas renomeação de arquivo, sem recomputação** — conteúdo idêntico. Generalizados `src/04` a `src/10` para aceitar `--dataset`, com defaults que preservam exatamente os caminhos/nomes originais do OASIS-1 quando `--dataset oasis1` (default). **Regressão verificada em todos:** `05_train_classifiers_brainiac.py`, `07_train_classifiers_radiomics.py`, `08_volumetry_baseline.py`, `09_evaluate_compare.py`, `10_qc_preprocessing.py` rodados novamente com `--dataset oasis1` e comparados byte-a-byte com os resultados já existentes — **todos idênticos**.
- `2026-07-10T03:23-03:27Z` — Escritos `src/11_cross_dataset_validation.py` (treino no dataset completo A, teste no dataset completo B, nos 2 sentidos, para cada método, usando os mesmos `CLASSIFIER_GRIDS`/seed de `common_cv.py`) e `src/12_final_comparison.py` (tabela consolidada intra-domínio + cross-dataset, figuras, `results_cross/summary.md`) — ainda não executados, aguardando features do OASIS-2 (que dependem do download em andamento + pré-processamento).

## Objetivo (recapitulando)
1. Replicação intra-domínio do pipeline OASIS-1 (BrainIAC+ML vs radiomics vs volumetria) num segundo dataset.
2. **Experimento principal:** validação cruzada entre coortes (treina num dataset, testa no outro, nos dois sentidos) para medir generalização — hipótese: BrainIAC degrada menos que os baselines fora do domínio.

## ETA (recalculado em 2026-07-10T04:17Z) — ⚠️ pré-processamento muito mais lento que o esperado
Etapas 1-3 (pesquisa de acesso, download, conversão) concluídas em ~44min, dentro do esperado.
**Etapa 4 (pré-processamento) está rodando MUITO mais devagar que no OASIS-1:** o registro rígido (SimpleITK) está levando **~108s/imagem em média** (10 imagens em 18min01s), contra ~13-24s/imagem no OASIS-1 — **cerca de 5-8x mais lento**. Nesse ritmo, só a etapa de registro para os 150 sujeitos do OASIS-2 levaria **~4.5 horas**, mais o tempo de HD-BET (skull-strip) depois, ainda não medido nesta rodada.
- **Causa provável:** `load average` da máquina em 15.4/17.7/13.8 no momento da checagem (vs. carga bem mais baixa durante a etapa correspondente do OASIS-1) — parece ser uma máquina mais disputada agora, não uma característica dos dados do OASIS-2 em si (as dimensões da imagem são até ligeiramente menores: OASIS-2 256x256x128 @ 1x1x1.25mm vs OASIS-1 256x256x160 @ 1x1x1mm). Não há outro processo pesado visível de outro usuário no `ps aux` além do nosso próprio script (que já é multi-thread, ~1628% CPU) — então a causa exata da maior contenção não está 100% confirmada, mas o fato mensurável é: **está mais lento, e não vou fingir que não está**.
- **Decisão:** deixar rodando (o processo está progredindo normalmente, só mais devagar), sem interromper, e reportar o tempo real gasto ao final. Vou monitorar em background e atualizar este arquivo assim que a etapa terminar (ou a cada verificação relevante).
- **Gargalo dominante agora confirmado: pré-processamento BrainIAC**, não mais o download.

## Checklist de etapas

| # | Etapa | Estado | Início | Fim | Duração |
|---|-------|--------|--------|-----|---------|
| 0 | Setup / checagem hardware+ambiente | ✅ | 2026-07-10T03:06Z | 2026-07-10T03:06Z | ~1min |
| 1 | Pesquisa de acesso: MIRIAD vs OASIS-3 vs OASIS-2, decisão do dataset | ✅ | 2026-07-10T03:06Z | 2026-07-10T03:12Z | ~6min |
| 2 | Aquisição do segundo dataset (download 2 partes 18GB + conversão NIfTI) | ✅ | 2026-07-10T03:12Z | 2026-07-10T03:56Z | ~44min (download ~31min + conversão ~9min) |
| 3 | Harmonização de rótulo + confundidores (idade) | ✅ (feita junto com a Etapa 1, ver acima) | 2026-07-10T03:10Z | 2026-07-10T03:10Z | ~2min |
| 4 | Pré-processamento BrainIAC idêntico + QC | ✅ | 2026-07-10T03:56Z | 2026-07-10T08:41Z | 4h45min (bem mais lento que o OASIS-1, ver nota de ETA acima — confirmado, não foi estimativa errada) |
| 5 | Extração de features (BrainIAC + radiomics; volumetria já pronta via eTIV/nWBV/ASF nativos) | ✅ | 2026-07-10T08:42Z | 2026-07-10T09:05Z | ~23min (BrainIAC ~1min + radiomics ~22.5min) |
| 6 | Replicação intra-domínio (CV 5-fold no dataset 2) | ✅ | 2026-07-10T08:42Z | 2026-07-10T08:44Z | ~2min |
| 7 | Validação cruzada entre coortes (treino/teste nos 2 sentidos) | ✅ | 2026-07-10T09:06Z | 2026-07-10T09:07Z | ~1min |
| 8 | (Opcional) Pooled + ComBat | ⬜ não executada (fora do escopo essencial; ver nota) | — | — | — |
| 9 | Comparação final consolidada + figuras + conclusões | ✅ | 2026-07-10T09:07Z | 2026-07-10T09:10Z | ~3min |

## Log cronológico
- `2026-07-10T03:06Z` — Retomado o projeto para a extensão de dataset 2. GPU agora livre (0% util, 49MiB usados — diferente da sessão anterior, que estava com outro processo competindo). Ambiente conda `brainiac-ad` presente. Estrutura do OASIS-1 (`PROGRESS.md`, `src/01-10`, `results/`, `features/`) intacta e será reutilizada/parametrizada, não duplicada.
- `2026-07-10T03:06Z` — Iniciando pesquisa de acesso ao MIRIAD (opção primária pedida) antes de comprometer tempo — precisamos saber se o download é automatizável (como o OASIS-1 acabou sendo, apesar do "acordo acadêmico") ou se exige aprovação manual humana (bloqueio real, como quase aconteceu com o OpenNeuro).

## Problemas & decisões
(nenhum ainda — em andamento)

## Métricas parciais

### QC de pré-processamento OASIS-2 (2026-07-10T08:41Z)
n=150, fração de voxels não-zero: média=0.2329, dp=0.0247, min=0.1862, max=0.3077, 1 outlier (>3dp), 0 suspeitos, mesmo shape (170,206,162) em todos. **Distribuição quase idêntica à do OASIS-1** (média 0.2303, dp 0.0214) — pré-processamento consistente entre os dois datasets, sem evidência de falha sistemática.

### Intra-domínio OASIS-2 — resultados parciais (2026-07-10T08:44Z)
**BrainIAC:** melhor = logreg, AUC 0.710±0.103 (vs 0.750±0.046 no OASIS-1 — mais baixo e com dp bem maior, esperado dado N menor).

**Volumetria (eTIV/nWBV/ASF):** melhor = logreg, AUC 0.633±0.077 — **queda grande** vs 0.784±0.039 no OASIS-1. Alguns classificadores (svm_linear 0.412, svm_rbf 0.417) ficaram **abaixo do acaso**.
- **Verificação de que não é bug:** comparei as médias de eTIV/nWBV/ASF por grupo nos dois datasets. `nWBV` (a feature mais discriminativa) tem separação CN-AD de **Δ=0.047** no OASIS-1 (0.769 vs 0.722) mas só **Δ=0.020** no OASIS-2 (0.745 vs 0.725) — menos da metade. `eTIV`/`ASF` não mostram separação de grupo real em nenhum dos dois datasets (diferenças de médias muito menores que o desvio-padrão).
- **Interpretação (achado relevante para o paper):** isso é consistente com a hipótese já registrada acima — o forte desempenho da volumetria no OASIS-1 provavelmente é parcialmente um artefato do confundimento de idade (Δ~7.7 anos), que infla a separação de nWBV entre grupos via atrofia relacionada à idade, não só à doença. No OASIS-2 (bem pareado por idade, Δ~-0.87 anos), esse atalho não está disponível, e a volumetria perde a maior parte do seu poder discriminativo. Classificadores com AUC<0.5 em folds pequenos (~30 sujeitos/fold, 3 features fracas) são consistentes com ausência de sinal real, não com erro de código.
- Isso já é um resultado factual importante mesmo antes do experimento principal (cross-dataset): sugere que a comparação BrainIAC vs volumetria do OASIS-1 sozinho pode ter sido enganosa, e reforça a necessidade do teste cross-dataset para separar "sinal real de doença" de "atalho de confundidor".

## Experimento principal: validação cruzada entre coortes (2026-07-10T09:06-09:10Z) — ⚠️ NÚMEROS SUPERSEDIDOS, ver "Correção pós-verificação" abaixo
**Esta primeira rodada usava OASIS-1 via SUBJ_111 (média motion-corrected oficial do OASIS) contra OASIS-2 via mpr-1 (repetição bruta), uma assimetria de nível de pré-processamento entre os dois lados do experimento cross-dataset. Um diagnóstico de domain-shift feito em revisão (ver seção de correção) mostrou que essa assimetria era uma fonte real de confundimento, e o experimento foi refeito com OASIS-1 reprocessado a partir do mpr-1 bruto. A tabela e leitura abaixo ficam registradas por completude histórica, mas os números corretos/finais estão na seção "Correção pós-verificação".**

**Verificação prévia:** colunas de features de radiomics idênticas entre OASIS-1 e OASIS-2 (107/107, mesma ordem) — comparação 1:1 válida. Sem overlap de subject_id entre datasets (confirmado). Predições cross-dataset verificadas: nenhuma duplicata, nenhum sujeito de treino aparecendo no teste (impossível por construção, coortes disjuntas).

| Método | Intra OASIS-1 | Intra OASIS-2 | OASIS1→OASIS2 | OASIS2→OASIS1 | Gap OASIS1→2 | Gap OASIS2→1 | Gap médio |
|---|---|---|---|---|---|---|---|
| BrainIAC | 0.750 (svm_linear) | 0.710 (logreg) | 0.524 | 0.415 | +0.225 | +0.295 | **+0.260** |
| Radiomics | 0.748 (svm_rbf) | 0.640 (svm_rbf) | 0.406 | 0.652 | +0.342 | **-0.012** | +0.165 |
| Volumetria | 0.784 (logreg) | 0.633 (logreg) | 0.650 | 0.784 | +0.135 | **-0.152** | **-0.009** |

**Leitura honesta (não força uma narrativa de vencedor único):**
- **BrainIAC é o único método que degrada nas DUAS direções** (gap +0.225 e +0.295, mesma ordem de grandeza, sem inversão de sinal) — padrão de degradação consistente/interpretável como perda real (mas parcial) de sinal transferível.
- **Radiomics e volumetria têm gap médio menor (até negativo), mas isso NÃO é evidência de robustez** — é uma assinatura de exploração de confundidor: as duas direções têm sinais opostos (radiomics +0.342 vs -0.012; volumetria +0.135 vs -0.152). Treinar no OASIS-1 (confundido por idade) e testar no OASIS-2 (pareado) perde a vantagem que a idade dava; treinar no OASIS-2 (sem esse atalho) e testar no OASIS-1 (onde o confundidor de idade ajuda a bater com o rótulo) ganha desempenho artificialmente. A média dessas duas distorções que se cancelam não significa "boa generalização".
- **Mas, em termos de AUC absoluto no cross-dataset, o BrainIAC não vence**: fica perto do acaso (~0.41-0.52) nas duas direções, pior que radiomics/volumetria na maioria das direções cross-dataset em valor absoluto. A hipótese do enunciado ("BrainIAC empata intra-domínio mas degrada menos") é **parcialmente confirmada apenas no sentido qualitativo/padrão de degradação** (mais consistente, sem sinal de exploração de confundidor), **não no sentido quantitativo de AUC absoluto pós-transferência** (onde o BrainIAC não supera os baselines).
- Isso é reportado como achado factual e nuançado, não como "BrainIAC venceu" nem "BrainIAC perdeu" — os dados sustentam uma leitura mais matizada, documentada de forma completa em `results_cross/summary.md`.

**Figuras geradas:** `figures/cross_dataset_auc_comparison.png` (barras: intra-OASIS1, intra-OASIS2, e as 2 direções cross-dataset, por método) e `figures/cross_dataset_roc.png` (curvas ROC das 2 direções, um painel por método) — inspecionadas visualmente, consistentes com a tabela acima (BrainIAC com curvas próximas à diagonal do acaso nas 2 direções; radiomics/volumetria com uma direção bem acima e outra bem abaixo/próxima da diagonal).

## Nota sobre Etapa 7 (opcional): Pooled + ComBat
Não executada nesta rodada — o enunciado a marcava como opcional ("se sobrar tempo"), e o experimento principal (Etapa 6) já entregou o resultado central pedido.

**Atualização pós-diagnóstico de domain-shift:** o ComBat deixou de ser um "extra opcional genérico" e passou a ser o experimento natural que resolveria a ambiguidade central documentada acima (AUC de dataset-membership ~1.0 para BrainIAC/radiomics mesmo após parear pré-processamento) — harmonizar estatisticamente o efeito de lote entre coortes e então rodar o cross-dataset de novo, para ver se sinal de doença sobrevive à remoção do efeito de lote. Isso é a forma mais direta de tentar separar "sem sinal transferível" de "sinal mascarado por batch effect". Ainda não executado (decisão do usuário se quer essa extensão); script sugerido: `src/14_pooled_combat.py`.

## Verificação final (2026-07-10T09:10Z) — ver também a rodada de verificação adicional abaixo
- ✅ Pré-processamento e transform de extração idênticos ao OASIS-1 (mesmo checkpoint, mesmo `mri_preprocess_3d_simple.py`, mesmo `get_validation_transform` via `get_brainiac_features.py` reutilizado sem modificação).
- ✅ Sem overlap de sujeitos entre treino e teste no cross-dataset (coortes disjuntas por construção, IDs com prefixos diferentes, confirmado programaticamente).
- ✅ Números da tabela final conferem com os arquivos de predição salvos em `results_cross/predictions/` (mesma fonte usada para gerar a tabela).
- ✅ QC de pré-processamento rodado nos 150 sujeitos do OASIS-2 (não só piloto) — distribuição consistente com OASIS-1, sem skull-strips falhos.

## Correção pós-verificação: diagnóstico de domain-shift e reprocessamento do OASIS-1 (2026-07-10T06:20-11:53Z)

**Origem:** em uma revisão adicional após a conclusão do pipeline acima, foi levantado um ponto crítico não coberto pelo desenho original do experimento: um cross-dataset AUC próximo do acaso (como o do BrainIAC, ~0.41-0.52) é **indistinguível** de "o modelo não tem sinal transferível" quando não sabemos quanto do sinal é dominado por um efeito de lote (batch effect) entre os dois datasets. Além disso, o experimento usava uma assimetria de pré-processamento entre os dois lados (OASIS-1 = SUBJ_111, já processado pelo próprio pipeline do OASIS com correção de movimento/média entre repetições/N4; OASIS-2 = mpr-1, repetição bruta sem esse pré-processamento) que estava **confundida exatamente com a variável de dataset** que o experimento tenta isolar.

### Diagnóstico de dataset-membership (2026-07-10T06:20Z)
Testamos: treinar um classificador (LogReg, 5-fold CV, AUC) para prever apenas de qual dataset um sujeito veio (ignorando o rótulo AD/CN), usando cada tipo de feature:

| Feature | AUC de dataset-membership |
|---|---|
| BrainIAC (embeddings, oasis1 SUBJ_111 vs oasis2) | **1.000** |
| Radiomics (oasis1 SUBJ_111 vs oasis2) | **1.000** |
| Volumetria (eTIV/nWBV/ASF, oasis1 vs oasis2) | 0.580 |

BrainIAC e radiomics separam os dois datasets com AUC praticamente perfeito — um efeito de lote domina completamente esses espaços de features, o que significa que os AUCs cross-dataset da primeira rodada (~0.41-0.65) podiam refletir majoritariamente esse efeito, não sinal de doença transferido. A volumetria, que vem de estatísticas computadas pelo próprio OASIS (nunca passa pelo nosso pré-processamento de imagem), tem separabilidade bem menor.

### Decisão (apresentada ao usuário via pergunta explícita, 2026-07-10T06:22Z)
Duas opções foram apresentadas: (a) reprocessar o OASIS-1 inteiro a partir do mpr-1 bruto (mesmo nível de entrada do OASIS-2) para testar se a assimetria de pré-processamento era a causa, ou (b) manter os resultados e apenas documentar a limitação. **Usuário escolheu (a).** Sistema estava ocioso (load average 0.06) no momento, permitindo reprocessamento na taxa rápida original (~6-10s/imagem) em vez da taxa anômala vista no pré-processamento original do OASIS-2 (~108s/imagem).

### Reprocessamento (2026-07-10T06:24-08:14Z)
- Confirmado que os arquivos brutos `RAW/*_mpr-1_anon.{img,hdr}` (repetição única, sem processamento OASIS) existem também no OASIS-1 (não só no OASIS-2) — dentro dos mesmos `.tar.gz` de disco já baixados.
- Adicionado o dataset `oasis1_mpr1` a `src/02_extract_and_convert.py` (`DATASET_PATTERNS`), reaproveitando a mesma lógica de extração de tar.gz.
- `02_extract_and_convert.py --dataset oasis1_mpr1 --labels_csv data/oasis1_labels.csv --archives_dir data/oasis1_raw/discs --output_dir data/oasis1_mpr1_raw/nifti`: **235/235 convertidos** (~1min).
- `03_run_brainiac_preprocessing.sh oasis1_mpr1` (registro rígido + N4 + HD-BET, idêntico ao pipeline usado para oasis1/oasis2): **235/235 processados com sucesso**, ~1h54min real (06:24-08:14Z; mais lento que a estimativa inicial de ~1-1.5h devido a variabilidade de carga do sistema observada em tempo real, mesma característica já vista no pré-processamento do OASIS-2 original — não um erro).
- `04_extract_brainiac_features.py --dataset oasis1_mpr1 --labels_csv data/oasis1_labels.csv`: 235/235, ~16s.
- `06_radiomics_baseline.py --dataset oasis1_mpr1 --labels_csv data/oasis1_labels.csv`: 235/235, 107 features, ~34min.
- Novo script `src/13_domain_shift_diagnostic.py` criado para reprodutibilidade do diagnóstico, salvando `results_cross/domain_shift_diagnostic.csv`.

### Resultado do reprocessamento: hipótese REFUTADA
Repetindo o diagnóstico de dataset-membership com `oasis1_mpr1` (nível de entrada pareado) vs `oasis2`:

| Feature | AUC de dataset-membership (nível de entrada pareado) |
|---|---|
| BrainIAC | **1.000** (sem mudança) |
| Radiomics | **0.9997** (sem mudança prática) |

**A separabilidade continuou em ~1.0 mesmo após parear o nível de entrada.** Isso refuta a hipótese de que a assimetria de pré-processamento (SUBJ_111 vs mpr-1) fosse a causa principal do efeito de lote — o efeito parece ser intrínseco às duas coortes (provavelmente diferenças de scanner/protocolo/era de aquisição entre OASIS-1, coletado ~2007, e OASIS-2, coletado em anos posteriores pela mesma instituição), não um artefato introduzido pela nossa escolha de pipeline. **Este é um resultado negativo importante e foi documentado como tal, não descartado** — reprocessar não "consertou" o problema, mas descartou uma explicação específica e mais tratável, deixando uma explicação menos tratável (diferença intrínseca de aquisição entre coortes/eras).

### `src/11_cross_dataset_validation.py` e `src/12_final_comparison.py` recalculados com `oasis1_mpr1`
Mesmo sem eliminar o efeito de lote, o AUC absoluto cross-dataset mudou de forma não-trivial ao usar o OASIS-1 com nível de entrada pareado (volumetria não muda, pois não depende do nosso pipeline de imagem).

**Segunda correção (mesma revisão): o gap não pode misturar "mudança de coorte" com "mudança de pipeline".** O cálculo inicial do gap comparava o AUC intra-domínio canônico do OASIS-1 (`SUBJ_111`, 0.750 para BrainIAC) contra o AUC cross-dataset (que usa `mpr-1` do lado OASIS-1) — dois números calculados com pipelines de entrada diferentes. Corrigido: recalculamos o AUC intra-domínio do OASIS-1 usando os MESMOS sujeitos/folds mas com `mpr-1` (`05_train_classifiers_brainiac.py --dataset oasis1_mpr1 --folds_csv data/oasis1_folds.csv --output_dir results_oasis1_mpr1`, e o equivalente com `07` para radiomics), obtendo 0.707 (logreg) para BrainIAC e 0.734 (svm_rbf) para radiomics — ambos mais baixos que os canônicos (0.750, 0.748), porque `mpr-1` é mais ruidoso que a média motion-corrected oficial. Sem essa correção, o gap OASIS1→OASIS2 do BrainIAC estava superestimado em quase 2x (0.750-0.635=0.115 vs o valor correto 0.707-0.639=0.069). `11_cross_dataset_validation.py` e `12_final_comparison.py` foram atualizados (`GAP_INTRA_DOMAIN_OVERRIDE`/`GAP_INTRA_SOURCE`) para usar esse AUC pareado apenas no cálculo do gap OASIS1→OASIS2, mantendo o AUC canônico `SUBJ_111` como o número exibido na coluna "Intra OASIS-1" da tabela (não há mistura de pipelines na direção OASIS2→OASIS1, que não foi reprocessada).

| Método | Intra OASIS-1 (exibido, SUBJ_111) | Intra OASIS-1 (usado no gap, mpr-1) | Intra OASIS-2 | OASIS1→OASIS2 | OASIS2→OASIS1 | Gap OASIS1→2 | Gap OASIS2→1 | Gap médio |
|---|---|---|---|---|---|---|---|---|
| BrainIAC | 0.750 (svm_linear) | 0.707 (logreg) | 0.710 (logreg) | 0.639 | 0.487 | +0.069 | +0.222 | +0.146 |
| Radiomics | 0.748 (svm_rbf) | 0.734 (svm_rbf) | 0.640 (svm_rbf) | 0.616 | 0.717 | +0.118 | -0.077 | +0.020 |
| Volumetria | 0.784 (logreg) | 0.784 (logreg, sem correção) | 0.633 (logreg) | 0.650 | 0.784 | +0.135 | -0.152 | -0.009 |

**Leitura final, honesta, com a ressalva do domain-shift no centro** (texto completo em `results_cross/summary.md`):
- Nenhum método mostra evidência clara e isolável de generalização "genuína" de sinal de doença entre coortes, dado que BrainIAC e radiomics têm seus AUCs cross-dataset contaminados por um efeito de lote forte e irredutível (AUC de dataset-membership ~1.0 mesmo após parear pré-processamento), e volumetria tem seu resultado dominado pelo confundidor de idade (assinatura de sinais opostos entre direções, +0.135/-0.152).
- A hipótese original do experimento ("BrainIAC empata intra-domínio mas degrada menos cross-dataset, evidenciando robustez de foundation model") **não encontra suporte claro nestes dados** — o desenho com 2 coortes da mesma instituição (ainda que sujeitos distintos) não isola bem essa pergunta, por conta do efeito de lote intrínseco descoberto.
- Isso é uma correção material em relação à leitura da primeira rodada (que enfatizava "BrainIAC degrada de forma mais consistente/interpretável" como um sinal qualitativo a favor da hipótese) — essa leitura foi **revisada para baixo** após o diagnóstico de domain-shift: a consistência de sinal do gap do BrainIAC não pode mais ser lida como evidência de generalização genuína, porque o próprio espaço de features onde o gap é medido está dominado por um efeito de lote não relacionado à doença. Além disso, o gap OASIS1→OASIS2 do BrainIAC, uma vez corrigido para não misturar pipelines, é bem menor (+0.069) do que a direção OASIS2→OASIS1 (+0.222) — a assimetria entre direções que antes parecia "consistente" (mesmo sinal, magnitude parecida) na verdade tem magnitudes bem diferentes, o que enfraquece ainda mais a leitura de "degradação genuína e uniforme".
- Um teste de mudança de site/scanner mais limpo exigiria MIRIAD ou OASIS-3 (acesso credenciado indisponível no momento).

### Arquivos afetados por esta correção
- **Novos:** `data/oasis1_mpr1_raw/nifti/*.nii.gz` (235), `data/oasis1_mpr1_processed/*.nii.gz` (235), `features/oasis1_mpr1_brainiac_features.parquet`, `features/oasis1_mpr1_radiomics_features.parquet`, `src/13_domain_shift_diagnostic.py`, `results_cross/domain_shift_diagnostic.csv`, `results_oasis1_mpr1/metrics_brainiac.csv`, `results_oasis1_mpr1/metrics_radiomics.csv` (AUC intra-domínio nos mesmos folds do OASIS-1, mas com pipeline `mpr-1` pareado — usado apenas para o cálculo correto do gap, não como resultado intra-domínio publicável por si só).
- **Modificados:** `src/02_extract_and_convert.py` (novo padrão `oasis1_mpr1`), `src/11_cross_dataset_validation.py` (usa `oasis1_mpr1` para brainiac/radiomics cross-dataset via `CROSS_FEATURE_DATASET`; gap usa `results_oasis1_mpr1` via `GAP_INTRA_DOMAIN_OVERRIDE`, mantendo volumetria e AUCs intra-domínio canônicos inalterados), `src/12_final_comparison.py` (`GAP_INTRA_SOURCE`, texto interpretativo reescrito, nova seção de diagnóstico de domain-shift e nota metodológica sobre o gap no `summary.md`), `results_cross/metrics_cross_dataset.csv`, `results_cross/generalization_gap.csv`, `results_cross/final_comparison.csv`, `results_cross/summary.md`, `figures/cross_dataset_auc_comparison.png`, `figures/cross_dataset_roc.png`.
- **Não afetados:** resultados intra-domínio do OASIS-1 (`results/`, usa SUBJ_111, publicados na sessão anterior) e do OASIS-2 (`results_oasis2/`) — permanecem exatamente como estavam.

### Verificação de integridade pós-correção (2026-07-10T11:53Z)
- ✅ 0 overlap de subject_id entre OASIS-1 e OASIS-2 (reconfirmado).
- ✅ Arquivos de predição cross-dataset com nível de entrada pareado: `oasis1_to_oasis2` tem 150 sujeitos de teste (todos em `oasis2_labels.csv`), `oasis2_to_oasis1` tem 235 (todos em `oasis1_labels.csv`), sem duplicatas.
- ✅ Números do `results_cross/summary.md` conferem com os CSVs de predição salvos.
- ✅ Diagnóstico de domain-shift reproduzível via `src/13_domain_shift_diagnostic.py` → `results_cross/domain_shift_diagnostic.csv`.
