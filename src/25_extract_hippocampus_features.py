"""
Teste 3A - Constroi features/{dataset}_hippocampus_features.parquet a partir do CSV de
volumes do SynthSeg (src/24 + batch SynthSeg_predict.py --vol). Volume hipocampal
esquerdo, direito e total, cada um normalizado pelo eTIV oficial (mesma coluna usada na
volumetria classica, ja presente em data/{dataset}_labels.csv) - nao pelo "total
segmentation volume" do proprio SynthSeg, para manter a normalizacao 1:1 comparavel com
o baseline de volumetria classica do mesmo dataset.
QC: distribuicao de volumes por grupo (CN vs AD) + outliers > 3 sigma marcados para
revisao (nao removidos automaticamente).
"""
import argparse
import os

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["oasis1", "oasis2"])
    parser.add_argument("--volumes_csv", default=None)
    parser.add_argument("--labels_csv", default=None)
    parser.add_argument("--output_parquet", default=None)
    parser.add_argument("--qc_csv", default=None)
    args = parser.parse_args()

    volumes_csv = args.volumes_csv or f"results_hippocampus_{args.dataset}/synthseg_volumes.csv"
    labels_csv = args.labels_csv or f"data/{args.dataset}_labels.csv"
    output_parquet = args.output_parquet or f"features/{args.dataset}_hippocampus_features.parquet"
    qc_csv = args.qc_csv or f"results_hippocampus_{args.dataset}/hippocampus_qc.csv"

    vol = pd.read_csv(volumes_csv)
    vol["subject_id"] = vol["subject"].str.replace(r"\.nii(\.gz)?$", "", regex=True)
    vol["hippo_left"] = vol["left hippocampus"]
    vol["hippo_right"] = vol["right hippocampus"]
    vol["hippo_total"] = vol["hippo_left"] + vol["hippo_right"]

    labels = pd.read_csv(labels_csv)
    assert "eTIV" in labels.columns, f"{labels_csv} missing eTIV column needed for normalization"

    df = labels[["subject_id", "label", "eTIV"]].merge(
        vol[["subject_id", "hippo_left", "hippo_right", "hippo_total"]], on="subject_id", how="inner"
    )
    print(f"[{args.dataset}] {len(df)}/{len(labels)} subjects with hippocampus volumes matched to labels")
    missing = set(labels["subject_id"]) - set(df["subject_id"])
    if missing:
        print(f"WARNING: {len(missing)} subjects missing hippocampus volumes: {sorted(missing)[:10]}...")

    df["hippo_left_norm"] = df["hippo_left"] / df["eTIV"]
    df["hippo_right_norm"] = df["hippo_right"] / df["eTIV"]
    df["hippo_total_norm"] = df["hippo_total"] / df["eTIV"]

    feature_cols = ["hippo_left_norm", "hippo_right_norm", "hippo_total_norm"]
    out = df[["subject_id", "label"] + feature_cols].copy()
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    out.to_parquet(output_parquet, index=False)
    print(f"Saved {out.shape[0]} subjects x {len(feature_cols)} hippocampus features to {output_parquet}")

    # ---- QC: distribution by group + 3-sigma outliers ----
    qc_rows = []
    for label_name, g in df.groupby("label"):
        for col in ["hippo_left", "hippo_right", "hippo_total", "hippo_total_norm"]:
            qc_rows.append({
                "label": label_name, "feature": col,
                "mean": g[col].mean(), "std": g[col].std(),
                "min": g[col].min(), "max": g[col].max(), "n": len(g),
            })
    qc_df = pd.DataFrame(qc_rows)

    mu, sigma = df["hippo_total_norm"].mean(), df["hippo_total_norm"].std()
    df["outlier_3sigma"] = (df["hippo_total_norm"] - mu).abs() > 3 * sigma
    outliers = df[df["outlier_3sigma"]][["subject_id", "label", "hippo_total_norm"]]

    os.makedirs(os.path.dirname(qc_csv), exist_ok=True)
    qc_df.to_csv(qc_csv, index=False)
    outliers.to_csv(qc_csv.replace(".csv", "_outliers.csv"), index=False)

    print(f"\nQC by group:\n{qc_df.to_string(index=False)}")
    print(f"\n3-sigma outliers (hippo_total_norm): {len(outliers)}")
    if len(outliers):
        print(outliers.to_string(index=False))


if __name__ == "__main__":
    main()
