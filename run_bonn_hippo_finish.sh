#!/bin/bash
# Retoma o pipeline hipocampal do Bonn após a falha (OOM) do SynthSeg no sub-00001:
# segmenta só o sujeito faltante, mescla o volume no CSV e roda os passos restantes.
set -euo pipefail

cd "$(dirname "$0")"
BRAINIAC_PY=/home/franco/miniconda3/envs/brainiac-ad/bin/python
CONDA=/home/franco/miniconda3/bin/conda
SEG_DIR=results_hippocampus_bonn/segmentations
VOL_CSV=results_hippocampus_bonn/synthseg_volumes.csv
MISSING=sub-00001

if [ ! -f "$SEG_DIR/${MISSING}_synthseg.nii.gz" ]; then
  "$CONDA" run -n synthseg python third_party/SynthSeg/scripts/commands/SynthSeg_predict.py \
    --i data/bonn_hippo_input/${MISSING}.nii.gz \
    --o "$SEG_DIR/${MISSING}_synthseg.nii.gz" \
    --vol results_hippocampus_bonn/synthseg_volumes_${MISSING}.csv \
    --v1 --threads 4 --crop 192 208 208
fi

"$BRAINIAC_PY" - <<PY
import pandas as pd
main = pd.read_csv("$VOL_CSV")
if "$MISSING" not in set(main["subject"]):
    extra = pd.read_csv("results_hippocampus_bonn/synthseg_volumes_${MISSING}.csv")
    main = pd.concat([main, extra]).sort_values("subject").reset_index(drop=True)
    main.to_csv("$VOL_CSV", index=False)
print(f"[bonn] synthseg_volumes.csv: {len(main)} subjects")
PY

"$BRAINIAC_PY" src/25_prepare_bonn_etiv_proxy.py
"$BRAINIAC_PY" src/25_extract_hippocampus_features.py \
  --dataset bonn --labels_csv data/bonn_labels_hippo.csv
"$BRAINIAC_PY" src/26_train_classifiers_hippocampus.py --dataset bonn
"$BRAINIAC_PY" src/27_extract_brainiac_hippo_roi.py --dataset bonn
"$BRAINIAC_PY" src/04_extract_brainiac_features.py --dataset bonn \
  --processed_dir data/bonn_brainiac_hippo_input \
  --output_csv features/bonn_brainiac_hippo_features_raw.csv \
  --output_parquet features/bonn_brainiac_hippo_features.parquet
"$BRAINIAC_PY" src/28_train_classifiers_brainiac_hippo.py --dataset bonn
"$BRAINIAC_PY" src/30_consolidate_dataset.py --dataset bonn
echo "[bonn] hippocampus pipeline finished"
