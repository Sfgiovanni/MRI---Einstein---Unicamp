"""Prepare binary epilepsy datasets distributed as BIDS for this pipeline.

The script selects one T1w scan per subject, links it into the pipeline's flat
input directory, writes labels, and creates fixed subject-level CV folds.
"""
import argparse
from pathlib import Path

import pandas as pd

from common_cv import make_stratified_subject_folds


def normalize_subject_id(value):
    value = str(value).strip().lstrip("\ufeff")
    return value if value.startswith("sub-") else f"sub-{value}"


def find_t1w_by_subject(bids_dir):
    scans = {}
    for path in sorted(bids_dir.glob("sub-*/**/*T1w.nii.gz")):
        subject_id = next(part for part in path.parts if part.startswith("sub-"))
        scans.setdefault(subject_id, []).append(path)
    return scans


def choose_one_t1w(paths):
    # Prefer a scan directly under anat; otherwise choose the lexically first
    # session/run so the selection remains deterministic and leakage-free.
    return sorted(paths, key=lambda p: ("ses-" in str(p), str(p)))[0]


def prepare_bonn(participants_path):
    df = pd.read_csv(participants_path, sep="\t", encoding="utf-8-sig")
    df.columns = [column.lstrip("\ufeff") for column in df.columns]
    df["subject_id"] = df["participant_id"].map(normalize_subject_id)
    group = df["group"].astype(str).str.lower()
    if not group.isin(["hc", "fcd"]).all():
        raise ValueError(f"Unexpected Bonn groups: {sorted(group.unique())}")
    df["label"] = group.map({"hc": 0, "fcd": 1})
    keep = ["subject_id", "label"]
    for column in ["group", "sex", "age_scan", "split"]:
        if column in df:
            keep.append(column)
    return df[keep]


def find_column(df, candidates):
    normalized = {
        "".join(ch for ch in column.lower() if ch.isalnum()): column
        for column in df.columns
    }
    for candidate in candidates:
        key = "".join(ch for ch in candidate.lower() if ch.isalnum())
        if key in normalized:
            return normalized[key]
    raise ValueError(f"None of the expected columns {candidates} occur in {list(df.columns)}")


def prepare_ideas(metadata_path):
    suffix = metadata_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(metadata_path)
    else:
        delimiter = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(metadata_path, sep=delimiter, encoding="utf-8-sig")

    id_col = find_column(df, ["participant_id", "subject_id", "patient_id", "ID"])
    outcome_col = find_column(
        df,
        [
            "12-month ILAE outcome",
            "12 month ILAE outcome",
            "ILAE 1 year",
            "ILAE_Year1",
            "ILAE_1year",
            "ILAE_12m",
        ],
    )
    outcome = pd.to_numeric(df[outcome_col], errors="coerce")
    valid = outcome.notna()
    out = pd.DataFrame(
        {
            "subject_id": df[id_col].map(normalize_subject_id),
            # ILAE 1 is seizure-free (favourable); ILAE 2-6 is unfavourable.
            "label": (outcome > 1).where(valid).astype("Int64"),
            "ilae_12m": outcome,
        }
    ).dropna(subset=["label"])
    out["label"] = out["label"].astype(int)
    if out.groupby("subject_id")["label"].nunique().gt(1).any():
        raise RuntimeError("Conflicting outcome labels for a duplicated IDEAS subject")
    return out.drop_duplicates(subset=["subject_id"]) 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["bonn", "ideas"], required=True)
    parser.add_argument("--bids-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    labels = prepare_bonn(args.metadata) if args.dataset == "bonn" else prepare_ideas(args.metadata)
    scans = find_t1w_by_subject(args.bids_dir)
    labels = labels[labels["subject_id"].isin(scans)].copy()
    if labels.empty or labels["label"].nunique() != 2:
        raise RuntimeError("Need subjects with T1w scans in both binary outcome classes")
    if labels["subject_id"].duplicated().any():
        raise RuntimeError("Duplicate subject identifiers in metadata")

    input_dir = Path(f"data/{args.dataset}_raw/nifti")
    input_dir.mkdir(parents=True, exist_ok=True)
    for subject_id in labels["subject_id"]:
        source = choose_one_t1w(scans[subject_id]).resolve()
        destination = input_dir / f"{subject_id}.nii.gz"
        if destination.is_symlink() and destination.resolve() == source:
            continue
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Refusing to replace existing input: {destination}")
        destination.symlink_to(source)

    labels.to_csv(f"data/{args.dataset}_labels.csv", index=False)
    folds = make_stratified_subject_folds(labels, n_folds=args.n_folds)
    folds[["subject_id", "label", "fold"]].to_csv(
        f"data/{args.dataset}_folds.csv", index=False
    )
    print(f"Prepared {len(labels)} subjects for {args.dataset}")
    print(labels["label"].value_counts().sort_index().rename("count"))
    print(folds.groupby(["fold", "label"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
