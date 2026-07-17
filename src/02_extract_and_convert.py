"""
Extract each subject's representative native-space T1w volume from the dataset's raw
archive(s) and convert to NIfTI (.nii.gz), one file per subject, ready for BrainIAC's
own registration+skull-strip preprocessing pipeline. Deliberately does NOT use any
atlas-registered/skull-stripped derivative provided by the dataset, since BrainIAC
expects native-space head-included input and performs its own registration/skull-strip.

Dataset-specific choice of "representative volume" (documented, not identical by
necessity - see PROGRESS.md / PROGRESS_dataset2.md):
  - oasis1: PROCESSED/MPRAGE/SUBJ_111/*_sbj_111.{hdr,img} - OASIS's own motion-corrected,
    gain-field-corrected AVERAGE across repeated scans, native space, head included.
  - oasis2: RAW/mpr-1.nifti.{hdr,img} - the raw archive for OASIS-2 only ships individual
    repeated scans (mpr-1..4), no pre-averaged equivalent to OASIS-1's SUBJ_111. We use
    the first raw repetition (mpr-1) rather than invent an unvalidated custom averaging
    step across repetitions. This is a documented asymmetry: OASIS-1's input had OASIS's
    own motion-correction-averaging applied before BrainIAC preprocessing, OASIS-2's
    input did not. BrainIAC's own registration+N4+skull-strip pipeline is applied
    identically afterward in both cases.
"""
import argparse
import glob
import os
import tarfile
import tempfile

import nibabel as nib
import pandas as pd

DATASET_PATTERNS = {
    "oasis1": {
        "session_suffix": "_MR1",
        "match": lambda name: "/PROCESSED/MPRAGE/SUBJ_111/" in name
        and (name.endswith("_sbj_111.img") or name.endswith("_sbj_111.hdr")),
        "expected_files": 2,
    },
    "oasis2": {
        "session_suffix": "_MR1",
        "match": lambda name: name.endswith("/RAW/mpr-1.nifti.img") or name.endswith("/RAW/mpr-1.nifti.hdr"),
        "expected_files": 2,
    },
    "oasis1_mpr1": {
        # Input-level-matched variant of OASIS-1, used ONLY for the cross-dataset
        # experiment: raw first repetition (mpr-1), same input level as oasis2's
        # RAW/mpr-1.nifti, instead of OASIS-1's own SUBJ_111 (which had already been
        # through OASIS's motion-correction-averaging + N4 + atlas-resampling before
        # ever reaching BrainIAC's preprocessing). See PROGRESS_dataset2.md for the
        # domain-shift diagnostic (dataset-membership AUC=1.0 on SUBJ_111-based
        # features) that motivated this re-run.
        "session_suffix": "_MR1",
        "match": lambda name: "/RAW/" in name
        and (name.endswith("_mpr-1_anon.img") or name.endswith("_mpr-1_anon.hdr")),
        "expected_files": 2,
    },
}


def find_and_convert(archive_path, wanted_subjects, output_dir, done, pattern):
    with tarfile.open(archive_path, "r:gz") as tf:
        members = tf.getmembers()
        by_subject = {}
        for m in members:
            parts = m.name.split("/")
            if len(parts) < 2:
                continue
            subj_dir = parts[1]  # e.g. "OAS1_0001_MR1" or "OAS2_0001_MR1"
            if not subj_dir.endswith(pattern["session_suffix"]):
                continue  # skip repeat-scan sessions (MR2+) - one scan per subject
            subj = subj_dir.rsplit("_MR", 1)[0]
            if subj not in wanted_subjects or subj in done:
                continue
            if pattern["match"](m.name):
                by_subject.setdefault(subj, []).append(m)

        for subj, subj_members in by_subject.items():
            if len(subj_members) != pattern["expected_files"]:
                print(f"WARNING: {subj} has {len(subj_members)} matching files in {archive_path}, "
                      f"expected {pattern['expected_files']}. Skipping.")
                continue
            with tempfile.TemporaryDirectory() as tmp:
                for m in subj_members:
                    fname = os.path.basename(m.name)
                    with tf.extractfile(m) as src, open(os.path.join(tmp, fname), "wb") as dst:
                        dst.write(src.read())
                img_path = glob.glob(os.path.join(tmp, "*.img"))[0]
                img = nib.load(img_path)
                out_path = os.path.join(output_dir, f"{subj}.nii.gz")
                nib.save(img, out_path)
                done.add(subj)
                print(f"Converted {subj} -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASET_PATTERNS.keys()), default="oasis1")
    parser.add_argument("--labels_csv", default=None)
    parser.add_argument("--archives_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    labels_csv = args.labels_csv or f"data/{args.dataset}_labels.csv"
    archives_dir = args.archives_dir or f"data/{args.dataset}_raw"
    output_dir = args.output_dir or f"data/{args.dataset}_raw/nifti"
    pattern = DATASET_PATTERNS[args.dataset]

    os.makedirs(output_dir, exist_ok=True)
    labels = pd.read_csv(labels_csv)
    wanted = set(labels["subject_id"])
    print(f"Need to find {len(wanted)} subjects across archives in {archives_dir}")

    done = set()
    for f in sorted(glob.glob(os.path.join(archives_dir, "*.tar.gz"))):
        remaining = wanted - done
        if not remaining:
            break
        print(f"Scanning {f} ({len(remaining)} subjects still needed)...")
        find_and_convert(f, remaining, output_dir, done, pattern)

    missing = wanted - done
    print(f"\nDone. Converted {len(done)}/{len(wanted)} subjects.")
    if missing:
        print(f"MISSING ({len(missing)}): {sorted(missing)}")


if __name__ == "__main__":
    main()
