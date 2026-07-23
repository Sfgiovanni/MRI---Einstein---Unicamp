"""
Teste 1 - Fusao BrainIAC (768-d) + radiomics (107-d): early-fusion feature table.
Simple column concat on subject_id, no scaling/selection here - all of that happens
per-fold inside the classifiers (src/21_train_fusion_early.py) to avoid leakage.
"""
import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="oasis1")
    parser.add_argument("--brainiac_parquet", default=None)
    parser.add_argument("--radiomics_parquet", default=None)
    parser.add_argument("--output_parquet", default=None)
    args = parser.parse_args()

    brainiac_parquet = args.brainiac_parquet or f"features/{args.dataset}_brainiac_features.parquet"
    radiomics_parquet = args.radiomics_parquet or f"features/{args.dataset}_radiomics_features.parquet"
    output_parquet = args.output_parquet or f"features/{args.dataset}_fusion_features.parquet"

    b = pd.read_parquet(brainiac_parquet)
    r = pd.read_parquet(radiomics_parquet)

    b_cols = [c for c in b.columns if c not in ("subject_id", "label")]
    r_cols = [c for c in r.columns if c not in ("subject_id", "label")]
    overlap = set(b_cols) & set(r_cols)
    assert not overlap, f"unexpected overlapping feature names between brainiac/radiomics: {overlap}"

    merged = b.merge(r, on=["subject_id", "label"], how="inner", validate="one_to_one")
    assert len(merged) == len(b) == len(r), (
        f"subject mismatch fusing brainiac (n={len(b)}) and radiomics (n={len(r)}) for {args.dataset}: "
        f"got {len(merged)} after inner join"
    )

    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    merged.to_parquet(output_parquet, index=False)
    print(f"Fusion features for {args.dataset}: {merged.shape[0]} subjects x "
          f"{len(b_cols)} (brainiac) + {len(r_cols)} (radiomics) = {len(b_cols) + len(r_cols)} dims "
          f"-> {output_parquet}")


if __name__ == "__main__":
    main()
