"""
Teste 3 - Prepara o input nativo (skull-on) para segmentacao hipocampal via SynthSeg.

OASIS-1 (`data/oasis1_raw/nifti/*.nii.gz`, convertido de PROCESSED/MPRAGE/SUBJ_111 em
ANALYZE 7.5): esses arquivos NAO tem qform/sform validos (ANALYZE 7.5 nao padroniza
orientacao) - nibabel cai para um affine default que ROTULA ERRADO a identidade
anatomica dos eixos do array (confirmado via nib.aff2axcodes + inspecao visual manual,
ver PROGRESS_extra.md). Isso quebra a suposicao espacial de redes treinadas em
orientacao canonica (FastSurfer E SynthSeg, mesmo o SynthSeg sendo robusto a
orientacao/contraste POR RANDOMIZACAO DE DOMINIO - nao a um header que MENTE sobre a
ordem dos eixos). Aplicamos a correcao determinada empiricamente (transpose para
[L-R, A-P, S-I] + affine valido) SOMENTE para esta pipeline de segmentacao - nao altera
`data/oasis1_raw/`, `data/oasis1_processed/` nem nenhum arquivo usado nos resultados ja
publicados (baseline BrainIAC/radiomics/volumetria).

OASIS-2 (`data/oasis2_raw/nifti/*.nii.gz`, de RAW/mpr-1.nifti): ja tem qform_code=1
valido, aff2axcodes bate com a identidade real dos eixos (verificado empiricamente) -
usado como esta, sem nenhuma correcao (link simbolico, sem duplicar dados).
"""
import argparse
import os

import nibabel as nib
import numpy as np

# established empirically for OASIS-1 raw (PROCESSED/MPRAGE/SUBJ_111 via ANALYZE 7.5):
# array axis0=A-P (coronal slices), axis1=S-I (axial slices), axis2=L-R (sagittal slices).
# Reorder to standard [L-R, A-P, S-I] = old axes (2, 0, 1).
OASIS1_TRANSPOSE = (2, 0, 1)


def fix_oasis1(in_path, out_path):
    img = nib.load(in_path)
    data = np.squeeze(np.asarray(img.dataobj))
    assert data.ndim == 3, f"expected 3D after squeeze, got shape {data.shape} for {in_path}"

    pixdim = np.asarray(img.header["pixdim"][1:4], dtype=float)
    fixed_data = np.transpose(data, OASIS1_TRANSPOSE)
    fixed_pixdim = pixdim[list(OASIS1_TRANSPOSE)]

    affine = np.array([
        [-fixed_pixdim[0], 0., 0., fixed_data.shape[0] * fixed_pixdim[0] / 2],
        [0., fixed_pixdim[1], 0., -fixed_data.shape[1] * fixed_pixdim[1] / 2],
        [0., 0., fixed_pixdim[2], -fixed_data.shape[2] * fixed_pixdim[2] / 2],
        [0., 0., 0., 1.],
    ])
    new_img = nib.Nifti1Image(fixed_data.astype(np.float32), affine)
    new_img.header.set_qform(affine, code=1)
    new_img.header.set_sform(affine, code=1)
    nib.save(new_img, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["oasis1", "oasis2"])
    parser.add_argument("--raw_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    raw_dir = args.raw_dir or f"data/{args.dataset}_raw/nifti"
    output_dir = args.output_dir or f"data/{args.dataset}_hippo_input"
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(raw_dir) if f.endswith(".nii.gz"))
    print(f"[{args.dataset}] {len(files)} subjects in {raw_dir}")

    for i, fname in enumerate(files):
        in_path = os.path.join(raw_dir, fname)
        out_path = os.path.join(output_dir, fname)
        if args.dataset == "oasis1":
            fix_oasis1(in_path, out_path)
        else:
            if os.path.lexists(out_path):
                os.remove(out_path)
            os.symlink(os.path.abspath(in_path), out_path)
        if (i + 1) % 50 == 0 or (i + 1) == len(files):
            print(f"  {i + 1}/{len(files)}")

    print(f"[{args.dataset}] hippocampus segmentation input ready at {output_dir}")


if __name__ == "__main__":
    main()
