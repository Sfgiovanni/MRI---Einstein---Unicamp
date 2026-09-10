#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
BRAINIAC_PY=/home/franco/miniconda3/envs/brainiac-ad/bin/python
CONDA=/home/franco/miniconda3/bin/conda

"$BRAINIAC_PY" src/24_prepare_hippocampus_input.py --dataset bonn
"$CONDA" run -n synthseg python third_party/SynthSeg/scripts/commands/SynthSeg_predict.py \
  --i data/bonn_hippo_input \
  --o results_hippocampus_bonn/segmentations \
  --vol results_hippocampus_bonn/synthseg_volumes.csv \
  --v1 --threads 4 --crop 192 208 208
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
