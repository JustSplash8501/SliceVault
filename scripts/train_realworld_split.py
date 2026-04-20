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

PROJECTS = ["BLADDER", "CERVIX_UTERI", "KIDNEY"]
TOP_PER_TISSUE = 400
BASE = Path("data/raw/recount3/gtex_v8/files")
OUT = Path("data/processed/realworld")
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
    var = sq_mean - np.square(mean)
    return var


print("Selecting high-variance junction features per tissue...")
selected_coords: set[str] = set()

for project in PROJECTS:
    mm_path = BASE / project / f"gtex.junctions.{project}.ALL.MM.gz"
    rr_path = BASE / project / f"gtex.junctions.{project}.ALL.RR.gz"

    rr = read_rr(rr_path)
    with gzip.open(mm_path, "rb") as fh:
        mat = mmread(fh).tocsc()

    var = row_variance_sparse(mat)
    keep_mask = rr["is_main_chr"].to_numpy()
    candidate_idx = np.where(keep_mask)[0]
    candidate_var = var[candidate_idx]

    top_local = candidate_idx[np.argsort(candidate_var)[-TOP_PER_TISSUE:]]
    coords = rr.iloc[top_local]["coord"].tolist()
    selected_coords.update(coords)
    print(f"  {project}: selected {len(coords)} top-variance coordinates")

selected_coords = sorted(selected_coords)
coord_to_col = {c: i for i, c in enumerate(selected_coords)}
print(f"Total unique selected features: {len(selected_coords)}")

print("Building combined sample-feature matrix...")
rows: list[np.ndarray] = []
sample_names: list[str] = []
labels: list[str] = []

for project in PROJECTS:
    mm_path = BASE / project / f"gtex.junctions.{project}.ALL.MM.gz"
    rr_path = BASE / project / f"gtex.junctions.{project}.ALL.RR.gz"
    id_path = BASE / project / f"gtex.junctions.{project}.ALL.ID.gz"

    rr = read_rr(rr_path)
    ids = read_ids(id_path)

    with gzip.open(mm_path, "rb") as fh:
        mat = mmread(fh).tocsc()

    rr_index = {coord: i for i, coord in enumerate(rr["coord"].tolist())}

    tissue_dense = np.zeros((len(ids), len(selected_coords)), dtype=np.float32)
    for coord, col_idx in coord_to_col.items():
        row_idx = rr_index.get(coord)
        if row_idx is None:
            continue
        vals = mat.getrow(row_idx).toarray().ravel().astype(np.float32)
        tissue_dense[:, col_idx] = vals

    # Library-size normalize + log1p
    libsize = tissue_dense.sum(axis=1, keepdims=True)
    libsize[libsize == 0] = 1.0
    tissue_dense = np.log1p((tissue_dense / libsize) * 1e6)

    for i, rid in enumerate(ids):
        rows.append(tissue_dense[i])
        sample_names.append(f"{project}:{rid}")
        labels.append(project)

x = np.vstack(rows)
y = np.array(labels)

x_df = pd.DataFrame(x, index=sample_names, columns=selected_coords)
labels_s = pd.Series(y, index=sample_names, name="label")

x_df.to_parquet(OUT / "junction_features.parquet")
labels_s.to_csv(OUT / "junction_labels.csv", header=True)

print("Creating stratified train/val/test splits...")
all_idx = np.arange(len(sample_names))
train_idx, temp_idx = train_test_split(
    all_idx,
    test_size=0.30,
    stratify=y,
    random_state=42,
)
val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    stratify=y[temp_idx],
    random_state=42,
)

train_samples = [sample_names[i] for i in train_idx]
val_samples = [sample_names[i] for i in val_idx]
test_samples = [sample_names[i] for i in test_idx]

splits = {
    "train": train_samples,
    "val": val_samples,
    "test": test_samples,
}
(Path(OUT / "splits.json")).write_text(json.dumps(splits, indent=2))

train_matrix = x_df.loc[train_samples].T
train_labels = labels_s.loc[train_samples]

print("Training classifier on train split only...")
clf = train_classifier(train_matrix, train_labels, model_type="logreg")
save_model(clf, "models/splicevault_realworld_model.joblib", "models/splicevault_realworld_model.h5")


def eval_split(name: str, split_samples: list[str]) -> dict:
    split_matrix = x_df.loc[split_samples].T
    pred_df, _ = predict_samples(clf, split_matrix)
    pred = pred_df.set_index("sample")["predicted_class"].reindex(split_samples)
    true = labels_s.loc[split_samples]
    return {
        "n": int(len(split_samples)),
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro")),
        "report": classification_report(true, pred, output_dict=True),
    }

metrics = {
    "features": len(selected_coords),
    "samples": len(sample_names),
    "class_counts": labels_s.value_counts().to_dict(),
    "train": eval_split("train", train_samples),
    "val": eval_split("val", val_samples),
    "test": eval_split("test", test_samples),
}

(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps({k: metrics[k] for k in ["train", "val", "test"]}, indent=2))
print("Saved metrics to", OUT / "metrics.json")
print("Saved model to models/splicevault_realworld_model.joblib")
