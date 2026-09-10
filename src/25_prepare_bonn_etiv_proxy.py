"""Add a SynthSeg-based normalization volume to Bonn labels as an eTIV proxy."""
import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volumes-csv", default="results_hippocampus_bonn/synthseg_volumes.csv")
    parser.add_argument("--labels-csv", default="data/bonn_labels.csv")
    parser.add_argument("--output-csv", default="data/bonn_labels_hippo.csv")
    args = parser.parse_args()

    volumes = pd.read_csv(args.volumes_csv)
    volumes["subject_id"] = volumes["subject"].str.replace(r"\.nii(\.gz)?$", "", regex=True)
    region_cols = [column for column in volumes.columns if column not in {"subject", "subject_id"}]
    volumes["eTIV"] = volumes[region_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)

    labels = pd.read_csv(args.labels_csv)
    output = labels.merge(volumes[["subject_id", "eTIV"]], on="subject_id", how="left", validate="one_to_one")
    if output["eTIV"].isna().any() or (output["eTIV"] <= 0).any():
        raise RuntimeError("Missing or invalid SynthSeg normalization volumes")
    output.to_csv(args.output_csv, index=False)
    print(f"Saved {len(output)} Bonn labels with SynthSeg segmented-volume eTIV proxy to {args.output_csv}")


if __name__ == "__main__":
    main()
