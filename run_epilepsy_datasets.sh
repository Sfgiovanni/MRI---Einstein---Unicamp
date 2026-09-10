#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

run_dataset() {
  local dataset="$1"
  local expected="$2"
  local processed="data/${dataset}_processed"
  local results="results_${dataset}"
  local figures="figures_${dataset}"
  local processed_count

  if [ -f "$results/qc_preprocessing.csv" ] && [ -f "$results/summary.csv" ]; then
    echo "Skipping completed dataset: $dataset"
    return
  fi

  mkdir -p "$processed"
  processed_count=$(find "$processed" -maxdepth 1 -name '*.nii.gz' 2>/dev/null | wc -l)
  if [ "$processed_count" -ne "$expected" ]; then
    python src/03_run_brainiac_preprocessing_parallel.py "$dataset" \
      --workers 4 --threads-per-worker 6
  fi

  python src/04_extract_brainiac_features.py --dataset "$dataset"
  python src/05_train_classifiers_brainiac.py --dataset "$dataset"
  python src/06_radiomics_baseline.py --dataset "$dataset"
  python src/07_train_classifiers_radiomics.py --dataset "$dataset"
  python src/09_evaluate_compare.py --results_dir "$results" --figures_dir "$figures"
  python src/10_qc_preprocessing.py --dataset "$dataset"
}

# These epilepsy datasets do not publish the OASIS-specific eTIV/nWBV/ASF
# measures, so the OASIS volumetry baseline (step 08) is intentionally omitted.
run_dataset bonn 170
run_dataset ideas 427
