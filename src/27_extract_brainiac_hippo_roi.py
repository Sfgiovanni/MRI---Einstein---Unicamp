"""
Teste 3B (exploratorio) - Recorta a regiao do hipocampo (bounding box da mascara SynthSeg
de 3A, labels 17=Left-Hippocampus/53=Right-Hippocampus, + margem) do MESMO volume nativo
usado na segmentacao (ja com orientacao corrigida para OASIS-1 - ver src/24), e extrai
embeddings BrainIAC (768-d) desse recorte usando o transform oficial (Resized 96^3 +
NormalizeIntensity), reaproveitando src/04_extract_brainiac_features.py sem modificacao,
so apontando para este novo processed_dir.

Escolha metodologica documentada: o crop usa o espaco NATIVO (mesmo da segmentacao),
nao o espaco `data/{dataset}_processed/` (registrado ao template BrainIAC/skull-stripped).
Motivo: descobrimos que `data/oasis1_processed/` herda o defeito de orientacao do bruto
do OASIS-1 (o registro rigido, partindo de uma inicializacao com eixos trocados, nao
corrige uma permutacao de 90 graus - ver PROGRESS_extra.md) - usar esse espaco para o
crop arriscaria recortar tecido errado. O espaco nativo (corrigido) e o unico verificado
como anatomicamente correto nos dois datasets.

AVISO (nao escondido): o encoder BrainIAC foi treinado em volumes de cerebro inteiro
registrados a um template canonico de 96^3; um recorte pequeno do hipocampo em espaco
nativo (nao registrado) e out-of-distribution para ele em dois sentidos (campo de visao
muito menor E sem registro). Resultado tratado como exploratorio, nao comparado
diretamente 1:1 com o BrainIAC-768d whole-brain do pipeline base.
"""
import argparse
import os

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to
from tqdm import tqdm

HIPPO_LABELS = {17, 53}  # Left-Hippocampus, Right-Hippocampus (FreeSurfer aseg LUT / SynthSeg)
MARGIN_VOX = 10


def crop_subject(native_path, seg_path, out_path):
    native_img = nib.load(native_path)
    native = np.squeeze(np.asarray(native_img.dataobj))
    if native.ndim != 3:
        raise ValueError(f"expected 3D after squeeze, got {native.shape} for {native_path}")
    native_img = nib.Nifti1Image(native, native_img.affine, native_img.header)

    # SynthSeg always outputs at 1mm isotropic - resample the label mask (nearest-neighbor,
    # order=0, to keep it a discrete label map) back onto the native image's own grid
    # (same axis order/orientation, just different spacing/shape for OASIS-2's 1.25mm z-axis).
    seg_img_native_grid = resample_from_to(nib.load(seg_path), (native.shape, native_img.affine), order=0)
    seg = np.round(np.asarray(seg_img_native_grid.dataobj)).astype(int)
    mask = np.isin(seg, list(HIPPO_LABELS))
    if not mask.any():
        return False

    nz = np.argwhere(mask)
    lo = np.maximum(nz.min(axis=0) - MARGIN_VOX, 0)
    hi = np.minimum(nz.max(axis=0) + MARGIN_VOX + 1, native.shape)

    crop = native[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    crop_affine = native_img.affine.copy()
    crop_affine[:3, 3] = native_img.affine[:3, :3] @ lo + native_img.affine[:3, 3]

    crop_img = nib.Nifti1Image(crop.astype(np.float32), crop_affine)
    crop_img.header.set_qform(crop_affine, code=1)
    crop_img.header.set_sform(crop_affine, code=1)
    nib.save(crop_img, out_path)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--native_dir", default=None)
    parser.add_argument("--seg_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    native_dir = args.native_dir or f"data/{args.dataset}_hippo_input"
    seg_dir = args.seg_dir or f"results_hippocampus_{args.dataset}/segmentations"
    output_dir = args.output_dir or f"data/{args.dataset}_brainiac_hippo_input"
    os.makedirs(output_dir, exist_ok=True)

    native_files = sorted(f for f in os.listdir(native_dir) if f.endswith(".nii.gz"))
    failed = []
    for fname in tqdm(native_files, desc=f"cropping hippocampus ROI [{args.dataset}]"):
        native_path = os.path.join(native_dir, fname)
        seg_fname = fname.replace(".nii.gz", "_synthseg.nii.gz")
        seg_path = os.path.join(seg_dir, seg_fname)
        out_path = os.path.join(output_dir, fname)
        if not os.path.exists(seg_path):
            failed.append(fname)
            continue
        ok = crop_subject(native_path, seg_path, out_path)
        if not ok:
            failed.append(fname)

    print(f"[{args.dataset}] cropped {len(native_files) - len(failed)}/{len(native_files)} subjects to {output_dir}")
    if failed:
        print(f"FAILED (missing seg or empty hippocampus mask): {failed}")


if __name__ == "__main__":
    main()
