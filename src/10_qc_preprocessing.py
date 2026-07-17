"""
QC check on BrainIAC-preprocessed volumes: distribution of nonzero (brain mask)
voxel fraction across all subjects, to catch silently failed/partial skull-strips
(HD-BET fast mode can occasionally over/under-strip individual subjects) before
trusting downstream BrainIAC/radiomics feature quality.
"""
import argparse
import glob
import os
import statistics

import nibabel as nib
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="oasis1")
    parser.add_argument("--processed_dir", default=None)
    parser.add_argument("--output_csv", default=None)
    args = parser.parse_args()

    processed_dir = args.processed_dir or f"data/{args.dataset}_processed"
    output_dir = "results" if args.dataset == "oasis1" else f"results_{args.dataset}"
    output_csv = args.output_csv or f"{output_dir}/qc_preprocessing.csv"

    rows = []
    for f in sorted(glob.glob(os.path.join(processed_dir, "*.nii.gz"))):
        sid = os.path.basename(f).replace(".nii.gz", "")
        img = nib.load(f)
        data = img.get_fdata()
        rows.append({
            "subject_id": sid,
            "nonzero_fraction": (data != 0).mean(),
            "nonzero_voxels": int((data != 0).sum()),
            "shape": str(data.shape),
        })

    df = pd.DataFrame(rows)
    mean_f, std_f = df["nonzero_fraction"].mean(), df["nonzero_fraction"].std()
    df["outlier_3std"] = (df["nonzero_fraction"] - mean_f).abs() > 3 * std_f
    df["suspicious"] = (df["nonzero_fraction"] < 0.05) | (df["nonzero_fraction"] > 0.5)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"n={len(df)}")
    print(f"nonzero fraction: mean={mean_f:.4f} std={std_f:.4f} min={df['nonzero_fraction'].min():.4f} max={df['nonzero_fraction'].max():.4f}")
    print(f"outliers (>3 std): {df['outlier_3std'].sum()}")
    print(f"suspicious (<0.05 or >0.5, likely failed skull-strip): {df['suspicious'].sum()}")
    print(f"distinct shapes: {df['shape'].unique()}")
    print(f"Saved per-subject QC to {output_csv}")


if __name__ == "__main__":
    main()
