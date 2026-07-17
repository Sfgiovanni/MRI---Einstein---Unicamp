"""
Parse demographics for a given dataset (OASIS-1 or OASIS-2), define binary AD vs CN
mapping, and create fixed stratified 5-fold subject-level CV splits.

Binary mapping (identical rule applied to both datasets - see PROGRESS.md /
PROGRESS_dataset2.md for the justification of using the same CDR-based rule):
  CN (label=0) = CDR == 0
  AD (label=1) = CDR >= 0.5 (very mild to moderate dementia)
  Subjects with CDR == None (never clinically assessed) are excluded.
  Only one scan (baseline / MR1 session) is kept per subject to avoid duplicate-scan leakage.
"""
import argparse
import os

import openpyxl
import pandas as pd

from common_cv import make_stratified_subject_folds

SEED = 42


def load_xlsx(xlsx_path):
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    header = rows[0]
    return pd.DataFrame(rows[1:], columns=header)


def prepare_oasis1(demographics_path):
    df = load_xlsx(demographics_path)
    df["base_id"] = df["ID"].apply(lambda x: x.rsplit("_MR", 1)[0])
    df["session"] = df["ID"].apply(lambda x: x.rsplit("_MR", 1)[1])
    df = df[df["session"] == "1"].copy()
    assert df["base_id"].is_unique, "Duplicate subjects after MR1 filtering"

    assessed = df[df["CDR"].notna()].copy()
    assessed["label"] = (assessed["CDR"] >= 0.5).astype(int)
    out = assessed[["base_id", "label", "CDR", "Age", "M/F", "MMSE", "eTIV", "nWBV", "ASF"]].copy()
    out = out.rename(columns={"base_id": "subject_id"})

    print(f"CN (CDR=0): {(out['label'] == 0).sum()}")
    print(f"AD (CDR>=0.5): {(out['label'] == 1).sum()}")
    print(f"Excluded (CDR=None, unassessed): {df['CDR'].isna().sum()}")
    return out


def prepare_oasis2(demographics_path):
    df = load_xlsx(demographics_path)
    # Keep only the baseline (first) session per subject - OASIS-2 is longitudinal
    # (2+ visits/subject); using only MR1 keeps exactly 1 image/subject, matching OASIS-1.
    df = df[df["MRI ID"].str.endswith("_MR1")].copy()
    df = df.rename(columns={"Subject ID": "subject_id"})
    assert df["subject_id"].is_unique, "Duplicate subjects after MR1 filtering"

    assessed = df[df["CDR"].notna()].copy()
    assessed["label"] = (assessed["CDR"] >= 0.5).astype(int)
    # Note: label derived purely from CDR (not the 'Group' column) to keep the mapping
    # rule identical to OASIS-1. This means the 13 baseline-CDR=0 "Converted" subjects
    # (nondemented at baseline, later progressed to dementia) are counted as CN here -
    # documented as a known label-noise nuance in PROGRESS_dataset2.md, not filtered out,
    # to avoid an ad-hoc exclusion criterion that OASIS-1 doesn't have.
    out = assessed[["subject_id", "label", "CDR", "Age", "M/F", "MMSE", "eTIV", "nWBV", "ASF"]].copy()

    print(f"CN (CDR=0): {(out['label'] == 0).sum()}")
    print(f"AD (CDR>=0.5): {(out['label'] == 1).sum()}")
    print(f"Excluded (CDR=None, unassessed): {df['CDR'].isna().sum()}")
    return out


DATASET_PREPARERS = {
    "oasis1": prepare_oasis1,
    "oasis2": prepare_oasis2,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASET_PREPARERS.keys()), default="oasis1")
    parser.add_argument("--demographics", default=None)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--output_folds", default=None)
    parser.add_argument("--n_folds", type=int, default=5)
    args = parser.parse_args()

    demographics = args.demographics or f"data/{args.dataset}_raw/demographics.xlsx"
    output_csv = args.output_csv or f"data/{args.dataset}_labels.csv"
    output_folds = args.output_folds or f"data/{args.dataset}_folds.csv"

    out = DATASET_PREPARERS[args.dataset](demographics)
    out.to_csv(output_csv, index=False)
    print(f"Saved labels for {len(out)} subjects to {output_csv}")

    out = make_stratified_subject_folds(out, n_folds=args.n_folds, seed=SEED)
    fold_counts = out.groupby(["fold", "label"]).size().unstack()
    print("Per-fold class counts:")
    print(fold_counts)

    out[["subject_id", "label", "fold"]].to_csv(output_folds, index=False)
    print(f"Saved {args.n_folds}-fold subject-level splits to {output_folds}")


if __name__ == "__main__":
    main()
