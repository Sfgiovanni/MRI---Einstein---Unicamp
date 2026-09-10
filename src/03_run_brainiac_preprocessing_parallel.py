"""Parallel registration wrapper for BrainIAC's unchanged preprocessing steps."""
import argparse
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BRAINIAC_PREPROCESSING = ROOT / "third_party/BrainIAC/src/preprocessing"
sys.path.insert(0, str(BRAINIAC_PREPROCESSING))


def register_chunk(args):
    chunk_dir, output_dir, template, threads = args
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(threads)
    from mri_preprocess_3d_simple import registration

    return registration(str(chunk_dir), str(output_dir), str(template))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads-per-worker", type=int, default=8)
    args = parser.parse_args()

    input_dir = ROOT / f"data/{args.dataset}_raw/nifti"
    output_dir = ROOT / f"data/{args.dataset}_processed"
    registered_dir = output_dir / "temp_registered"
    chunks_root = output_dir / "registration_chunks"
    template = BRAINIAC_PREPROCESSING / "atlases/temp_head.nii.gz"
    output_dir.mkdir(parents=True, exist_ok=True)
    registered_dir.mkdir(parents=True, exist_ok=True)
    # Chunk directories contain only generated symlinks. Always rebuild them so
    # retries with a different worker count cannot retain duplicate assignments.
    if chunks_root.exists():
        shutil.rmtree(chunks_root)
    chunks_root.mkdir(parents=True)

    inputs = sorted(input_dir.glob("*.nii.gz"))
    completed = {path.name.removesuffix("_0000.nii.gz") for path in registered_dir.glob("*_0000.nii.gz")}
    pending = [path for path in inputs if path.name.removesuffix(".nii.gz") not in completed]
    chunks = [[] for _ in range(args.workers)]
    for index, path in enumerate(pending):
        chunks[index % args.workers].append(path)

    jobs = []
    for index, paths in enumerate(chunks):
        chunk_dir = chunks_root / f"chunk_{index}"
        chunk_dir.mkdir(exist_ok=True)
        for source in paths:
            destination = chunk_dir / source.name
            if not destination.exists():
                destination.symlink_to(source.resolve())
        if paths:
            jobs.append((chunk_dir, registered_dir, template, args.threads_per_worker))

    print(f"Already registered: {len(completed)}; pending: {len(pending)}; workers: {len(jobs)}")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        if not all(executor.map(register_chunk, jobs)):
            raise RuntimeError("At least one registration worker produced no output")

    if len(list(registered_dir.glob("*_0000.nii.gz"))) != len(inputs):
        raise RuntimeError("Registration output count does not match input count")

    from mri_preprocess_3d_simple import brain_extraction
    import torch

    brain_extraction(str(registered_dir), str(output_dir), "0" if torch.cuda.is_available() else "cpu")
    shutil.rmtree(chunks_root)
    shutil.rmtree(registered_dir)
    for path in output_dir.glob("*_0000.nii.gz"):
        path.rename(path.with_name(path.name.replace("_0000.nii.gz", ".nii.gz")))


if __name__ == "__main__":
    main()
