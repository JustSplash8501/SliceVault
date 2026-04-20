from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.sparse import csc_matrix
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from splicevault.classify import predict_samples, save_model, train_classifier

BASE = Path("data/raw/recount3/gtex_v8/files")
META_PATH = Path("data/raw/recount3/gtex_v8/recount3_raw_project_files_with_default_annotation.csv")
OUT = Path("data/processed/realworld_multitissue_noleak")
OUT.mkdir(parents=True, exist_ok=True)

# Keep features manageable while preserving tissue signal and runtime.
TOP_PER_TISSUE = 200


def read_ids(path: Path) -> list[str]:
    with gzip.open(path, "rt") as fh:
        rows = [line.strip() for line in fh if line.strip()]
    if rows and rows[0] == "rail_id":
        rows = rows[1:]
    return rows


def read_rr(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt") as fh:
        rr = pd.read_csv(fh, sep="\t")
    rr["coord"] = (
        rr["chromosome"].astype(str)
        + ":"
        + rr["start"].astype(str)
        + "-"
        + rr["end"].astype(str)
        + ":"
        + rr["strand"].astype(str)
    )
    rr["is_main_chr"] = rr["chromosome"].astype(str).str.startswith("chr")
    return rr


def row_variance_sparse(m: csc_matrix) -> np.ndarray:
    mean = np.asarray(m.mean(axis=1)).ravel()
    sq_mean = np.asarray(m.power(2).mean(axis=1)).ravel()
    return sq_mean - np.square(mean)


def detect_complete_tissues(base: Path) -> list[str]:
    out: list[str] = []
    for tissue_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
        tissue = tissue_dir.name
        mm = tissue_dir / f"gtex.junctions.{tissue}.ALL.MM.gz"
        rr = tissue_dir / f"gtex.junctions.{tissue}.ALL.RR.gz"
        jid = tissue_dir / f"gtex.junctions.{tissue}.ALL.ID.gz"
        if mm.exists() and rr.exists() and jid.exists():
            out.append(tissue)
    return out


def load_tissue_sample_names(tissue: str) -> list[str]:
    id_path = BASE / tissue / f"gtex.junctions.{tissue}.ALL.ID.gz"
    ids = read_ids(id_path)
    return [f"{tissue}:{rid}" for rid in ids]


def main() -> None:
    tissues = detect_complete_tissues(BASE)
    if len(tissues) < 3:
        raise RuntimeError("Need at least 3 complete tissues to build multiclass splits")

    # Optional: sort by tissue sample count descending using metadata to keep deterministic ordering.
    meta = pd.read_csv(META_PATH)
    meta = meta[
        (meta["organism"] == "human")
        & (meta["file_source"] == "gtex")
        & (meta["project_type"] == "data_sources")
    ][["project", "n_samples"]]
    n_map = dict(zip(meta["project"], meta["n_samples"], strict=False))
    tissues = sorted(tissues, key=lambda t: (-int(n_map.get(t, 0)), t))

    print("Using tissues:", ", ".join(tissues))

    # Build global sample list for splitting.
    all_samples: list[str] = []
    all_labels: list[str] = []
    for tissue in tissues:
        samples = load_tissue_sample_names(tissue)
        all_samples.extend(samples)
        all_labels.extend([tissue] * len(samples))

    all_samples_arr = np.array(all_samples)
    all_labels_arr = np.array(all_labels)

    # Stratified train/val/test split (70/15/15).
    idx = np.arange(len(all_samples_arr))
    train_idx, temp_idx = train_test_split(
        idx,
        test_size=0.30,
        stratify=all_labels_arr,
        random_state=42,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        stratify=all_labels_arr[temp_idx],
        random_state=42,
    )

    train_set = set(all_samples_arr[train_idx].tolist())
    val_set = set(all_samples_arr[val_idx].tolist())
    test_set = set(all_samples_arr[test_idx].tolist())

    print(
        f"Split sizes -> train={len(train_set)} val={len(val_set)} test={len(test_set)}"
    )

    # Feature selection using TRAIN samples only.
    selected_coords: set[str] = set()
    for tissue in tissues:
        mm_path = BASE / tissue / f"gtex.junctions.{tissue}.ALL.MM.gz"
        rr_path = BASE / tissue / f"gtex.junctions.{tissue}.ALL.RR.gz"

        rr = read_rr(rr_path)
        with gzip.open(mm_path, "rb") as fh:
            mat = mmread(fh).tocsc()

        sample_names = load_tissue_sample_names(tissue)
        train_cols = [i for i, s in enumerate(sample_names) if s in train_set]
        if not train_cols:
            continue

        train_mat = mat[:, train_cols]
        var = row_variance_sparse(train_mat)
        keep_idx = np.where(rr["is_main_chr"].to_numpy())[0]
        if len(keep_idx) == 0:
            continue

        keep_var = var[keep_idx]
        k = min(TOP_PER_TISSUE, len(keep_idx))
        top_local = keep_idx[np.argsort(keep_var)[-k:]]
        selected_coords.update(rr.iloc[top_local]["coord"].tolist())

        print(f"{tissue}: selected {k} train-variance features")

    selected_coords = sorted(selected_coords)
    coord_to_col = {c: i for i, c in enumerate(selected_coords)}
    print(f"Total selected features: {len(selected_coords)}")

    # Build sample x feature matrix.
    rows: list[np.ndarray] = []
    row_names: list[str] = []
    row_labels: list[str] = []

    for tissue in tissues:
        mm_path = BASE / tissue / f"gtex.junctions.{tissue}.ALL.MM.gz"
        rr_path = BASE / tissue / f"gtex.junctions.{tissue}.ALL.RR.gz"

        rr = read_rr(rr_path)
        sample_names = load_tissue_sample_names(tissue)

        with gzip.open(mm_path, "rb") as fh:
            mat = mmread(fh).tocsc()

        rr_index = {coord: i for i, coord in enumerate(rr["coord"].tolist())}
        tissue_dense = np.zeros((len(sample_names), len(selected_coords)), dtype=np.float32)
        present = [(rr_index[c], feat_col) for c, feat_col in coord_to_col.items() if c in rr_index]
        if present:
            row_idx = [r for r, _ in present]
            feat_col = [c for _, c in present]
            # Bulk extraction is much faster than per-row sparse getrow() loops.
            block = mat[row_idx, :].toarray().astype(np.float32)  # shape: rows x samples
            tissue_dense[:, feat_col] = block.T

        # Per-sample CPM-like normalization + log1p transform.
        libsize = tissue_dense.sum(axis=1, keepdims=True)
        libsize[libsize == 0] = 1.0
        tissue_dense = np.log1p((tissue_dense / libsize) * 1e6)

        for i, sample in enumerate(sample_names):
            rows.append(tissue_dense[i])
            row_names.append(sample)
            row_labels.append(tissue)

    x_df = pd.DataFrame(np.vstack(rows), index=row_names, columns=selected_coords)
    y_s = pd.Series(row_labels, index=row_names, name="label")

    # Persist dataset + splits.
    x_df.to_parquet(OUT / "junction_features.parquet")
    y_s.to_csv(OUT / "junction_labels.csv", header=True)

    train_samples = sorted(train_set)
    val_samples = sorted(val_set)
    test_samples = sorted(test_set)

    splits = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }
    (OUT / "splits.json").write_text(json.dumps(splits, indent=2))

    # Train model on train split only.
    train_matrix = x_df.loc[train_samples].T
    train_labels = y_s.loc[train_samples]

    clf = train_classifier(train_matrix, train_labels, model_type="logreg")
    save_model(
        clf,
        "models/splicevault_realworld_multitissue_model.joblib",
        "models/splicevault_realworld_multitissue_model.h5",
    )

    def evaluate(samples: list[str]) -> dict:
        mat = x_df.loc[samples].T
        pred_df, _ = predict_samples(clf, mat)
        pred = pred_df.set_index("sample")["predicted_class"].reindex(samples)
        true = y_s.loc[samples]
        return {
            "n": int(len(samples)),
            "accuracy": float(accuracy_score(true, pred)),
            "macro_f1": float(f1_score(true, pred, average="macro")),
            "report": classification_report(true, pred, output_dict=True),
        }

    metrics = {
        "tissues": tissues,
        "features": int(len(selected_coords)),
        "samples": int(len(x_df)),
        "class_counts": y_s.value_counts().to_dict(),
        "train": evaluate(train_samples),
        "val": evaluate(val_samples),
        "test": evaluate(test_samples),
    }

    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(json.dumps({"train": metrics["train"], "val": metrics["val"], "test": metrics["test"]}, indent=2))
    print("Saved model: models/splicevault_realworld_multitissue_model.joblib")
    print(f"Saved splits/metrics under: {OUT}")


if __name__ == "__main__":
    main()
