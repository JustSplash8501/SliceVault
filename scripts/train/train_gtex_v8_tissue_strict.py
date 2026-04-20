from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.sparse import csc_matrix
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from splicevault.classify import predict_samples, save_model, train_classifier

PROJECTS = ["BLADDER", "CERVIX_UTERI", "KIDNEY"]
TOP_PER_TISSUE = 400
BASE = Path("data/raw/recount3/gtex_v8/files")
OUT = Path("data/processed/gtex_v8_tissue_strict")
OUT.mkdir(parents=True, exist_ok=True)


def read_ids(path: Path) -> list[str]:
    with gzip.open(path, "rt") as fh:
        lines = [line.strip() for line in fh if line.strip()]
    if lines and lines[0] == "rail_id":
        lines = lines[1:]
    return lines


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


# Load per-tissue matrices and metadata
raw: dict[str, dict] = {}
all_samples: list[str] = []
all_labels: list[str] = []
for project in PROJECTS:
    mm_path = BASE / project / f"gtex.junctions.{project}.ALL.MM.gz"
    rr_path = BASE / project / f"gtex.junctions.{project}.ALL.RR.gz"
    id_path = BASE / project / f"gtex.junctions.{project}.ALL.ID.gz"

    with gzip.open(mm_path, "rb") as fh:
        mat = mmread(fh).tocsc()
    rr = read_rr(rr_path)
    ids = read_ids(id_path)

    sample_names = [f"{project}:{rid}" for rid in ids]
    raw[project] = {"mat": mat, "rr": rr, "ids": ids, "sample_names": sample_names}

    all_samples.extend(sample_names)
    all_labels.extend([project] * len(sample_names))

# Stratified split across all samples
all_samples_arr = np.array(all_samples)
all_labels_arr = np.array(all_labels)
idx = np.arange(len(all_samples_arr))
train_idx, temp_idx = train_test_split(idx, test_size=0.30, stratify=all_labels_arr, random_state=42)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.50, stratify=all_labels_arr[temp_idx], random_state=42
)

train_set = set(all_samples_arr[train_idx].tolist())
val_set = set(all_samples_arr[val_idx].tolist())
test_set = set(all_samples_arr[test_idx].tolist())

# Feature selection using TRAIN samples only
selected_coords: set[str] = set()
for project in PROJECTS:
    mat: csc_matrix = raw[project]["mat"]
    rr: pd.DataFrame = raw[project]["rr"]
    sample_names: list[str] = raw[project]["sample_names"]

    local_train_cols = [i for i, s in enumerate(sample_names) if s in train_set]
    train_mat = mat[:, local_train_cols]

    var = row_variance_sparse(train_mat)
    keep_idx = np.where(rr["is_main_chr"].to_numpy())[0]
    keep_var = var[keep_idx]
    top_local = keep_idx[np.argsort(keep_var)[-TOP_PER_TISSUE:]]
    selected_coords.update(rr.iloc[top_local]["coord"].tolist())

selected_coords = sorted(selected_coords)
coord_to_col = {c: i for i, c in enumerate(selected_coords)}

# Build full sample x feature matrix using selected coords
rows: list[np.ndarray] = []
row_names: list[str] = []
row_labels: list[str] = []

for project in PROJECTS:
    mat: csc_matrix = raw[project]["mat"]
    rr: pd.DataFrame = raw[project]["rr"]
    sample_names: list[str] = raw[project]["sample_names"]

    rr_index = {coord: i for i, coord in enumerate(rr["coord"].tolist())}
    tissue_dense = np.zeros((len(sample_names), len(selected_coords)), dtype=np.float32)

    for coord, col_idx in coord_to_col.items():
        r = rr_index.get(coord)
        if r is None:
            continue
        tissue_dense[:, col_idx] = mat.getrow(r).toarray().ravel().astype(np.float32)

    libsize = tissue_dense.sum(axis=1, keepdims=True)
    libsize[libsize == 0] = 1.0
    tissue_dense = np.log1p((tissue_dense / libsize) * 1e6)

    for i, s in enumerate(sample_names):
        rows.append(tissue_dense[i])
        row_names.append(s)
        row_labels.append(project)

x_df = pd.DataFrame(np.vstack(rows), index=row_names, columns=selected_coords)
y_s = pd.Series(row_labels, index=row_names, name="label")
x_df.to_parquet(OUT / "junction_features.parquet")
y_s.to_csv(OUT / "junction_labels.csv", header=True)

train_samples = sorted(train_set)
val_samples = sorted(val_set)
test_samples = sorted(test_set)

train_matrix = x_df.loc[train_samples].T
train_labels = y_s.loc[train_samples]

clf = train_classifier(train_matrix, train_labels, model_type="logreg")
save_model(clf, "models/splicevault_gtex_v8_tissue_model_strict.joblib", "models/splicevault_gtex_v8_tissue_model_strict.h5")


def eval_split(samples: list[str]) -> dict[str, float | int]:
    m = x_df.loc[samples].T
    pred_df, _ = predict_samples(clf, m)
    pred = pred_df.set_index("sample")["predicted_class"].reindex(samples)
    true = y_s.loc[samples]
    return {
        "n": int(len(samples)),
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro")),
    }

metrics = {
    "features": len(selected_coords),
    "class_counts": y_s.value_counts().to_dict(),
    "train": eval_split(train_samples),
    "val": eval_split(val_samples),
    "test": eval_split(test_samples),
}

# Permutation sanity check (train-label shuffle)
perm_rng = np.random.default_rng(42)
perm_labels = pd.Series(perm_rng.permutation(train_labels.values), index=train_labels.index)
perm_clf = train_classifier(train_matrix, perm_labels, model_type="logreg")
perm_pred_df, _ = predict_samples(perm_clf, x_df.loc[test_samples].T)
perm_pred = perm_pred_df.set_index("sample")["predicted_class"].reindex(test_samples)
perm_true = y_s.loc[test_samples]
metrics["permutation_test_accuracy"] = float(accuracy_score(perm_true, perm_pred))

(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
(OUT / "splits.json").write_text(
    json.dumps({"train": train_samples, "val": val_samples, "test": test_samples}, indent=2)
)
print(json.dumps(metrics, indent=2))
