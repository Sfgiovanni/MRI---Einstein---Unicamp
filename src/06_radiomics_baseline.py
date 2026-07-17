"""
Baseline 1/2 - Radiomics features (PyRadiomics) extracted from the same
BrainIAC-preprocessed (registered, N4-corrected, skull-stripped) T1w volumes,
using the nonzero brain region (post skull-strip) as the region of interest,
since HD-BET does not save a separate mask by default. Same classifier suite
and CV protocol as the BrainIAC-embedding pipeline for a fair comparison.
"""
import argparse
import os

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from tqdm import tqdm


def build_extractor():
    settings = {
        "binWidth": 25,
        "resampledPixelSpacing": None,  # already resampled to 1mm iso by BrainIAC preprocessing
        "interpolator": "sitkBSpline",
        "normalize": True,
        "normalizeScale": 100,
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.enableAllFeatures()
    return extractor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="oasis1")
    parser.add_argument("--labels_csv", default=None)
    parser.add_argument("--processed_dir", default=None)
    parser.add_argument("--output_parquet", default=None)
    args = parser.parse_args()

    labels_csv = args.labels_csv or f"data/{args.dataset}_labels.csv"
    processed_dir = args.processed_dir or f"data/{args.dataset}_processed"
    output_parquet = args.output_parquet or f"features/{args.dataset}_radiomics_features.parquet"

    labels = pd.read_csv(labels_csv)
    available = {f.replace(".nii.gz", "") for f in os.listdir(processed_dir) if f.endswith(".nii.gz")}
    labels = labels[labels["subject_id"].isin(available)].copy()
    print(f"Extracting radiomics for {len(labels)} subjects")

    extractor = build_extractor()
    rows = []
    failed = []
    for _, row in tqdm(labels.iterrows(), total=len(labels)):
        sid = row["subject_id"]
        img_path = os.path.join(processed_dir, f"{sid}.nii.gz")
        try:
            image = sitk.ReadImage(img_path)
            arr = sitk.GetArrayFromImage(image)
            mask_arr = (arr > 0).astype(np.uint8)
            mask = sitk.GetImageFromArray(mask_arr)
            mask.CopyInformation(image)

            if mask_arr.sum() < 100:
                raise ValueError("brain mask nearly empty after skull-stripping")

            result = extractor.execute(image, mask)
            feats = {k: v for k, v in result.items() if not k.startswith("diagnostics_")}
            feats = {k: float(v) for k, v in feats.items()}
            feats["subject_id"] = sid
            feats["label"] = row["label"]
            rows.append(feats)
        except Exception as e:
            print(f"FAILED {sid}: {e}")
            failed.append(sid)

    df = pd.DataFrame(rows)
    cols = ["subject_id", "label"] + [c for c in df.columns if c not in ("subject_id", "label")]
    df = df[cols]
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    print(f"Saved {df.shape[0]} subjects x {df.shape[1] - 2} radiomic features to {output_parquet}")
    if failed:
        print(f"FAILED subjects ({len(failed)}): {failed}")


if __name__ == "__main__":
    main()
