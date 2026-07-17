#!/bin/bash
# Run BrainIAC's official preprocessing (rigid registration to its template +
# N4 bias correction + HD-BET skull-stripping) on the converted NIfTI volumes for the
# given dataset. Same template/registration/skull-strip code path for every dataset -
# this identity is what makes the cross-dataset comparison valid.
set -e
DATASET="${1:-oasis1}"
cd "$(dirname "$0")/.."

BRAINIAC_SRC=third_party/BrainIAC/src
INPUT_DIR="data/${DATASET}_raw/nifti"
OUTPUT_DIR="data/${DATASET}_processed"

mkdir -p "$OUTPUT_DIR"

python "$BRAINIAC_SRC/preprocessing/mri_preprocess_3d_simple.py" \
  --temp_img "$BRAINIAC_SRC/preprocessing/atlases/temp_head.nii.gz" \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR"

# HD-BET preserves the "_0000" suffix (required on input) in its output filenames;
# strip it so files are named "{subject_id}.nii.gz", matching what
# get_brainiac_features.py's BrainAgeDataset expects.
cd "$OUTPUT_DIR"
for f in *_0000.nii.gz; do
  [ -e "$f" ] || continue
  mv "$f" "${f/_0000.nii.gz/.nii.gz}"
done
