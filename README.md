# BrainIAC + ML for Binary AD vs CN Classification

Reproducible research pipeline using the **BrainIAC** foundation model (Tak et al., Nature Neuroscience 2026) as a feature extractor for structural T1w MRI, training classical ML classifiers on these features for binary **Alzheimer's disease (AD) vs. cognitively normal control (CN)** classification, and comparing them against radiomics and classical volumetry baselines.

## Datasets

### OASIS-1 (first dataset)

* Source: https://sites.wustl.edu/oasisbrains/home/oasis-1/
* Binary mapping: **CN = CDR==0** (n=135), **AD = CDR>=0.5** (n=100). Subjects without an assessed CDR score (young/middle-aged participants) are excluded. See `PROGRESS.md` for details and limitations, including age confounding (Δ~7.7 years).

### OASIS-2 (second dataset)

* Source: https://sites.wustl.edu/oasisbrains/home/oasis-2/ (longitudinal, 150 subjects, 1 scan/subject retained = baseline session `_MR1`)

## Setup

```bash
# 1. Conda environment (includes `versioneer`, required in step 2)
conda env create -f environment.yml
conda activate brainiac-ad

# 2. pyradiomics must be installed SEPARATELY, after the rest of the environment,
#    using --no-build-isolation. Its legacy setup.py imports numpy/versioneer at
#    build time without declaring them as build dependencies, which breaks with
#    pip's default build isolation when numpy/versioneer are not already available.
pip install --no-build-isolation pyradiomics

# 3. Clone BrainIAC + required patch
git clone https://github.com/AIM-KannLab/BrainIAC.git third_party/BrainIAC

# The upstream repository's HD_BET directory does not contain __init__.py, which
# prevents `from HD_BET.hd_bet import hd_bet` from working as a local package.
# Without this patch, Step 3 (preprocessing) fails with ModuleNotFoundError.
touch third_party/BrainIAC/src/preprocessing/HD_BET/__init__.py
```

BrainIAC weights: download the `BrainIAC.ckpt` checkpoint (362 MB, ViT-B encoder) from the Dropbox link:

https://www.dropbox.com/scl/fo/i51xt63roognvt7vuslbl/AG99uZljziHss5zJz4HiFis?rlkey=9w55le6tslwxlfz6c0viylmjb&st=b9cnvwh8&e=1&dl=0

and place it at `checkpoints/BrainIAC.ckpt`.

```bash
# 4. SHAP (extra test 2) - only required in the brainiac-ad environment;
#    no special dependencies are needed
conda activate brainiac-ad
pip install shap

# 5. SynthSeg (extra test 3, hippocampus) - SEPARATE environment:
#    the code is from 2020-2022 and depends on TensorFlow 2.2.0/Keras 2.3.1
#    standalone (not tf.keras), which is incompatible with the main environment
#    (modern torch/numpy). cudatoolkit/cudnn are installed through conda inside
#    the environment itself (CUDA 10.1 does not need to be installed system-wide).
conda create -n synthseg python=3.8 -y
conda activate synthseg
git clone https://github.com/BBillot/SynthSeg.git third_party/SynthSeg
pip install -r third_party/SynthSeg/requirements_python3.8.txt
conda install -c conda-forge cudatoolkit=10.1 cudnn=7.6.5 -y

# Weights (synthseg_1.0.h5) are already included in the cloned repository
# (third_party/SynthSeg/models/) - no separate download or registration required.
conda activate brainiac-ad
```

## Pipeline

Run the following commands in order from the project root.

All scripts accept `--dataset {oasis1,oasis2}` (default: `oasis1`) instead of duplicating logic for each dataset. Input/output paths are derived automatically (`data/{dataset}_...`, `features/{dataset}_...`, `results` for OASIS-1, or `results_{dataset}` for the others).

```bash
# OASIS-1 (main dataset)
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

# OASIS-2 (second dataset - same scripts, only --dataset changes)
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

# Reprocessing OASIS-1 starting from raw mpr-1
# (input level matched to OASIS-2), used only on the OASIS-1 side of the
# cross-dataset experiment - see the domain-shift diagnostic in
# "Cross-cohort validation" below
python src/02_extract_and_convert.py --dataset oasis1_mpr1 --labels_csv data/oasis1_labels.csv --archives_dir data/oasis1_raw/discs --output_dir data/oasis1_mpr1_raw/nifti
bash src/03_run_brainiac_preprocessing.sh oasis1_mpr1
python src/04_extract_brainiac_features.py --dataset oasis1_mpr1 --labels_csv data/oasis1_labels.csv
python src/06_radiomics_baseline.py --dataset oasis1_mpr1 --labels_csv data/oasis1_labels.csv

# In-domain evaluation at the SAME input level (mpr-1), used only for the
# correct computation of the generalization gap (avoids mixing "cohort shift"
# with "pipeline shift") - reuses the canonical OASIS-1 folds (same subjects)
python src/05_train_classifiers_brainiac.py --dataset oasis1_mpr1 --folds_csv data/oasis1_folds.csv --output_dir results_oasis1_mpr1
python src/07_train_classifiers_radiomics.py --dataset oasis1_mpr1 --folds_csv data/oasis1_folds.csv --output_dir results_oasis1_mpr1

# Main experiment: cross-cohort validation + final comparison
python src/11_cross_dataset_validation.py
python src/12_final_comparison.py
python src/13_domain_shift_diagnostic.py
```

1. `01_prepare_dataset.py` — parses OASIS-1 demographic data, defines the binary label, and creates subject-level stratified 5-fold CV splits (`data/oasis1_labels.csv`, `data/oasis1_folds.csv`).

2. `02_extract_and_convert.py` — extracts the native averaged volume (`SUBJ_111`, Analyze format) for each subject from the OASIS-1 discs and converts it to NIfTI.

3. `03_run_brainiac_preprocessing.sh` — runs the official BrainIAC preprocessing pipeline (rigid registration + N4 + HD-BET skull stripping) on the converted NIfTIs.

4. `04_extract_brainiac_features.py` — extracts BrainIAC ViT embeddings (768-d).

5. `05_train_classifiers_brainiac.py` — trains LogReg/SVM/RF/XGBoost/LightGBM using subject-level k=5 CV on the embeddings.

6. `06_radiomics_baseline.py` + `07_train_classifiers_radiomics.py` — radiomics baseline (PyRadiomics) using the same CV protocol.

7. `08_volumetry_baseline.py` — classical volumetry baseline using the official OASIS-1 eTIV/nWBV/ASF variables with the same CV protocol.

8. `09_evaluate_compare.py` — aggregates metrics, runs the DeLong test, generates figures (`figures/`), and creates `results/summary.md` / `results/summary.csv`.

9. `10_qc_preprocessing.py` — post-hoc QC: evaluates the distribution of the fraction of non-zero voxels (brain mask) across all preprocessed subjects to detect failed/partial skull stripping that could silently corrupt BrainIAC/radiomics features without affecting volumetry, which comes from the independent OASIS pipeline. This check avoids confusing "the model did not outperform the baseline" with "preprocessing corrupted the model inputs."

   Result: narrow distribution (mean 0.230, SD 0.021, min 0.185, max 0.296) and identical shape across all 235 subjects, with no evidence of systematic failure. Saved to `results/qc_preprocessing.csv`.

   The same analysis was also run for OASIS-2 (150/150 subjects, nearly identical distribution) and saved to `results_oasis2/qc_preprocessing.csv`.

10. `11_cross_dataset_validation.py` — **main experiment of the extension**: trains on ALL subjects from one dataset and tests on ALL subjects from the other dataset, in both directions, for each method using the same `CLASSIFIER_GRIDS`/seed from `common_cv.py`. Saves `results_cross/metrics_cross_dataset.csv`, `generalization_gap.csv`, and predictions under `results_cross/predictions/`.

11. `12_final_comparison.py` — consolidated table containing intra-OASIS1, intra-OASIS2, and both cross-dataset directions for each method; figures (`figures/cross_dataset_auc_comparison.png`, `figures/cross_dataset_roc.png`); and `results_cross/summary.md` containing the complete interpretation and required caveats.

12. `13_domain_shift_diagnostic.py` — critical diagnostic: trains a classifier to predict only which dataset (OASIS-1 or OASIS-2) a subject came from while ignoring the AD/CN label. A high AUC indicates that the feature space is dominated by a batch effect, limiting how much of the cross-dataset AUC reflects genuinely transferable disease signal.

    This motivated reprocessing OASIS-1 from raw `mpr-1` (`--dataset oasis1_mpr1` in steps `02`/`03`/`04`/`06`) to test — and, as shown by the diagnostic, reject — the hypothesis that the `SUBJ_111` vs. `mpr-1` asymmetry caused the batch effect.

    Results are saved to `results_cross/domain_shift_diagnostic.csv`. See "Cross-cohort validation" below and `PROGRESS_dataset2.md` for the complete report.

## Running the Pipeline with Another Dataset

Every script (`01`-`13`, `20`-`31`) accepts `--dataset <name>` and automatically derives the corresponding paths (`data/<name>_...`, `features/<name>_...`, `results_<name>/` — except `oasis1`, which for legacy reasons uses `results/` without a suffix).

No script contains hard-coded `oasis1`/`oasis2` names except steps 1-2 (demographic parsing and raw-file extraction), which are inherently specific to the OASIS data format.

**Input contract** — to plug in a new dataset (`$DATASET` = any name), the pipeline from step 03 onward requires only the following three inputs:

| File                                            | Required columns / content                                                                                                                         |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data/${DATASET}_raw/nifti/{subject_id}.nii.gz` | 1 native T1w scan (with skull) per subject                                                                                                         |
| `data/${DATASET}_labels.csv`                    | `subject_id`, `label` (0=CN/1=AD); `eTIV` is also required when running the volumetry baseline and hippocampal volume test (normalization)         |
| `data/${DATASET}_folds.csv`                     | `subject_id`, `label`, `fold` — generate using `common_cv.make_stratified_subject_folds()` (this is what `01_prepare_dataset.py` calls internally) |

If your dataset already contains demographic information in an `.xlsx` file similar to OASIS, add a `prepare_<name>()` function to `DATASET_PREPARERS` in `01_prepare_dataset.py`, following the same pattern as `prepare_oasis1`/`prepare_oasis2`, and add an entry to `DATASET_PATTERNS` in `02_extract_and_convert.py` if the raw data are distributed as `.tar.gz`.

Otherwise, the simplest approach is to write a custom script outside this repository, or as `01_prepare_dataset_<name>.py`, that produces the three files listed in the table above. From that point onward, the rest of the pipeline works without code changes.

```bash
export DATASET=mydataset   # replace with your dataset name

# --- Base pipeline: BrainIAC vs radiomics vs volumetry
#     (if the 3 files above exist) ---
bash src/03_run_brainiac_preprocessing.sh $DATASET
python src/04_extract_brainiac_features.py --dataset $DATASET
python src/05_train_classifiers_brainiac.py --dataset $DATASET
python src/06_radiomics_baseline.py --dataset $DATASET
python src/07_train_classifiers_radiomics.py --dataset $DATASET
python src/08_volumetry_baseline.py --dataset $DATASET
python src/09_evaluate_compare.py --results_dir results_$DATASET --figures_dir figures_$DATASET
python src/10_qc_preprocessing.py --dataset $DATASET

# --- Extra tests: fusion (1A/1B) + SHAP (2)
#     reuse features from the block above ---
python src/20_prepare_fusion_features.py --dataset $DATASET
python src/21_train_fusion_early.py --dataset $DATASET
python src/22_train_fusion_stacking.py --dataset $DATASET
python src/23_shap_feature_selection.py --dataset $DATASET --feature_set radiomics
python src/23_shap_feature_selection.py --dataset $DATASET --feature_set fusion
python src/23_shap_feature_selection.py --dataset $DATASET --feature_set brainiac

# --- Extra test: hippocampus (3A/3B)
#     requires the `synthseg` environment (see Setup) ---
# BEFORE running: check the orientation of your raw NIfTI (see note in src/24).
# The correction applied there is specific to OASIS-1; other datasets usually
# do not require it.
python src/24_prepare_hippocampus_input.py --dataset $DATASET
conda activate synthseg  # or: conda run -n synthseg

# --crop 224 224 224 is a safe starting point for an adult head on a ~11GB GPU
# (we used 160x224x224 for OASIS-1/2, with ~256x256x160 native images).
# If OOM occurs, reduce it - but never below the actual brain extent in your image.
# Check with nibabel: bounding box of voxels > background threshold, plus a margin.
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

# --- Consolidation: DeLong + Holm-Bonferroni vs the best baseline
#     (automatically detected) ---
python src/30_consolidate_dataset.py --dataset $DATASET

# --- Compare 2+ datasets side by side
#     (run 30 for each dataset first) ---
python src/31_consolidate_overall.py --datasets oasis1 oasis2 $DATASET
```

The "best base method" used as the DeLong reference in `src/30`/`src/31` is automatically detected as the method with the highest mean CV AUC among BrainIAC/radiomics/volumetry for that dataset (see `stats_utils.load_baseline_predictions`). No manual configuration is required.

## Results

### OASIS-1

| Method                              | Best classifier | AUC (mean CV ± SD) |
| ----------------------------------- | --------------- | ------------------ |
| BrainIAC (frozen ViT embeddings)    | svm_linear      | 0.750 ± 0.046      |
| Radiomics (PyRadiomics)             | svm_rbf         | 0.748 ± 0.012      |
| Classical volumetry (eTIV/nWBV/ASF) | logreg          | 0.784 ± 0.039      |

### OASIS-2

| Method    | Best classifier | AUC (mean CV ± SD) |
| --------- | --------------- | ------------------ |
| BrainIAC  | logreg          | 0.710 ± 0.115      |
| Radiomics | svm_rbf         | 0.640 ± 0.112      |
| Volumetry | logreg          | 0.633 ± 0.086      |

### Extra Tests: Fusion, Feature Selection (SHAP), and Hippocampal Segmentation

| Method                                          | OASIS-1 AUC           | OASIS-2 AUC          |
| ----------------------------------------------- | --------------------- | -------------------- |
| **Baseline**                                    | **0.748** (volumetry) | **0.710** (BrainIAC) |
| Hippocampal volume (SynthSeg)                   | 0.807                 | 0.723                |
| Late fusion (stacking)                          | 0.763                 | 0.705                |
| Early fusion (concatenation)                    | 0.752                 | 0.691                |
| SHAP + fusion                                   | 0.769                 | 0.626                |
| SHAP + BrainIAC (768d)                          | 0.754                 | 0.642                |
| SHAP + radiomics                                | 0.720                 | 0.602                |
| BrainIAC on hippocampal ROI *(exploratory/OOD)* | 0.634                 | 0.567                |

### Bonn (Pediatric Epilepsy: Focal Cortical Dysplasia vs. Controls)

Third dataset, used to test the same pipeline outside the AD/CN setting: 170 subjects (85 FCD / 85 healthy controls, ages 3–13), T1w images in BIDS format, with subject-level stratified 5-fold CV.

Classical volumetry is not included because eTIV/nWBV/ASF are obtained from the OASIS spreadsheet and are unavailable here. Therefore, the baseline methods are BrainIAC and radiomics only.

Results are stored in `results_bonn/` and `results_extra_bonn/`.

| Method                           | Best classifier | AUC (mean CV ± SD) |
| -------------------------------- | --------------- | ------------------ |
| BrainIAC (frozen ViT embeddings) | svm_linear      | 0.804 ± 0.076      |
| Radiomics (PyRadiomics)          | logreg          | 0.791 ± 0.038      |

Extra tests (AUC from pooled CV predictions, DeLong test vs. the best baseline; Holm-Bonferroni correction within the primary family):

| Method                                          |              Bonn AUC | Δ vs. baseline |
| ----------------------------------------------- | --------------------: | -------------: |
| **Baseline**                                    | **0.791** (radiomics) |              — |
| Early fusion (concatenation)                    |                 0.873 |         +0.082 |
| SHAP + fusion                                   |                 0.837 |         +0.047 |
| Late fusion (stacking)                          |                 0.817 |         +0.026 |
| SHAP + radiomics                                |                 0.797 |         +0.006 |
| SHAP + BrainIAC (768d)                          |                 0.777 |         −0.014 |
| BrainIAC on hippocampal ROI *(exploratory/OOD)* |                 0.740 |         −0.050 |
| Hippocampal volume (SynthSeg)                   |                 0.558 |         −0.233 |

Early fusion is the only statistically significant difference after correction. Hippocampal volume is close to chance level, as expected because FCD is a cortical lesion rather than a hippocampal lesion. BrainIAC applied to the hippocampal ROI inherits the same limitation and is additionally out-of-distribution for the encoder.

## Project Structure

```text
data/            raw and processed data from all datasets (not versioned)
checkpoints/     BrainIAC weights and trained models
features/        extracted embeddings/features (parquet), prefixed by dataset
results/         OASIS-1: metrics, per-fold predictions, models, summary
results_oasis2/  OASIS-2: same structure
results_cross/   cross-cohort validation: metrics, predictions, final summary
results_fusion_{oasis1,oasis2}/       extra tests: early/late fusion
results_shap_{oasis1,oasis2}/         extra tests: SHAP selection
results_hippocampus_{oasis1,oasis2}/  extra tests: hippocampal segmentation/volume
results_extra_{oasis1,oasis2}/        per-dataset consolidated extra tests
results_extra/                        cross-dataset consolidated extra tests
results_bonn/, results_*_bonn/        Bonn (FCD vs controls): same per-dataset structure
figures/         comparative figures (OASIS-1 + cross-dataset + extra tests)
figures_oasis2/  OASIS-2 comparative figures
logs/            raw stdout/stderr from each step
src/             scripts numbered by pipeline step, parameterized by --dataset
third_party/     BrainIAC and SynthSeg clones (logical submodules, not Git submodules)
```

## Reproducibility

A fixed seed (`SEED=42`) is used for numpy/torch/sklearn in all scripts.

CV splits are saved to `data/{dataset}_folds.csv` and reused across all methods (BrainIAC, radiomics, volumetry) to guarantee paired comparisons using the same subjects/folds within each dataset.

For cross-dataset experiments, the same checkpoint, preprocessing pipeline, and transforms are used for both datasets — a necessary condition for a valid comparison. See the final verification section in `PROGRESS_dataset2.md`.
