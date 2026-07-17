"""
Build the pat_id,label CSV expected by BrainIAC's get_brainiac_features.py and run it
to extract the 768-d ViT CLS-token embedding for every preprocessed subject in the
given dataset. Same checkpoint/encoder/transform for every dataset - required for the
cross-dataset comparison to be valid.
"""
import argparse
import os
import subprocess
import sys

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="oasis1")
    parser.add_argument("--labels_csv", default=None)
    parser.add_argument("--processed_dir", default=None)
    parser.add_argument("--checkpoint", default="checkpoints/BrainIAC.ckpt")
    parser.add_argument("--brainiac_src", default="third_party/BrainIAC/src")
    parser.add_argument("--tmp_csv", default=None)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--output_parquet", default=None)
    args = parser.parse_args()

    labels_csv = args.labels_csv or f"data/{args.dataset}_labels.csv"
    processed_dir = args.processed_dir or f"data/{args.dataset}_processed"
    tmp_csv = args.tmp_csv or f"data/{args.dataset}_brainiac_input.csv"
    output_csv = args.output_csv or f"features/{args.dataset}_brainiac_features_raw.csv"
    output_parquet = args.output_parquet or f"features/{args.dataset}_brainiac_features.parquet"

    labels = pd.read_csv(labels_csv)

    available = {
        f.replace(".nii.gz", "") for f in os.listdir(processed_dir) if f.endswith(".nii.gz")
    }
    labels_available = labels[labels["subject_id"].isin(available)].copy()
    print(f"{len(labels_available)}/{len(labels)} subjects have a preprocessed volume available.")

    out = labels_available[["subject_id", "label"]].rename(columns={"subject_id": "pat_id"})
    os.makedirs(os.path.dirname(tmp_csv), exist_ok=True)
    out.to_csv(tmp_csv, index=False)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    cmd = [
        sys.executable,
        "get_brainiac_features.py",
        "--checkpoint", os.path.abspath(args.checkpoint),
        "--input_csv", os.path.abspath(tmp_csv),
        "--output_csv", os.path.abspath(output_csv),
        "--root_dir", os.path.abspath(processed_dir),
        "--batch_size", "1",
        "--num_workers", "1",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=args.brainiac_src, check=True)

    # get_brainiac_features.py does not preserve subject_id in its output CSV; DataLoader
    # with shuffle=False (default SequentialSampler) guarantees row order == input CSV order
    # regardless of num_workers, so we reattach subject_id positionally and sanity-check
    # the label round-trips correctly.
    raw = pd.read_csv(output_csv)
    assert len(raw) == len(out), f"row count mismatch: {len(raw)} features vs {len(out)} input rows"
    mismatched = (raw["GroundTruthClassLabel"].astype(int) != out["label"].values).sum()
    assert mismatched == 0, f"{mismatched} rows have a label mismatch after positional join - ordering assumption broken"

    feature_cols = [c for c in raw.columns if c.startswith("Feature_")]
    emb_df = raw[feature_cols].copy()
    emb_df.columns = [f"emb_{i}" for i in range(len(feature_cols))]
    final = pd.concat([
        pd.DataFrame({"subject_id": out["pat_id"].values, "label": out["label"].values}),
        emb_df.reset_index(drop=True),
    ], axis=1)

    final.to_parquet(output_parquet, index=False)
    print(f"Saved {final.shape[0]} subjects x {len(feature_cols)} embedding dims to {output_parquet}")


if __name__ == "__main__":
    main()
