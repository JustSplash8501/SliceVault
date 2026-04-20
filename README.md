# SpliceVault

SpliceVault is a Python toolkit for tissue and disease classification using **alternative splicing patterns** (junction/PSI-derived features) from RNA-seq, rather than standard gene-expression-only signatures.

## What This Repo Contains

- Core package: `splicevault/`
- CLI: `splicevault.cli`
- GTEx/recount3 data download scripts: `scripts/data/`
- Leak-aware training pipelines: `scripts/train/`
- Workflow scaffold: `workflow/Snakefile`

## Repository Layout

```text
SpliceVault/
├── splicevault/                 # Python package (PSI, signatures, classify, viz, CLI)
├── scripts/
│   ├── data/                    # Data acquisition/downloading helpers
│   └── train/                   # Train/val/test training pipelines
├── data/
│   ├── raw/                     # External raw data (recount3 GTEx files)
│   └── processed/               # Derived feature matrices, splits, metrics
├── models/                      # Trained model artifacts (.joblib + .h5)
├── tests/                       # Pytest suite
├── notebooks/                   # Example notebook
├── workflow/                    # Snakemake scaffold
└── .github/workflows/ci.yml     # CI
```

## Installation

### Poetry (recommended)

```bash
poetry install
```

### Pip

```bash
pip install -e .
```

## Pipeline: End-to-End (Real GTEx Junction Data)

### 1. Download recount3 metadata table

```bash
curl -L --fail -o data/raw/recount3/gtex_v8/recount3_raw_project_files_with_default_annotation.csv \
  https://raw.githubusercontent.com/LieberInstitute/recount3-docs/master/docs/recount3_raw_project_files_with_default_annotation.csv
```

### 2. Download GTEx junction matrices

Use the downloader to fetch junction assets (`MM`, `RR`, `ID`) for GTEx tissues:

```bash
python3 scripts/data/download_recount3_gtex_v8.py --include jxn_MM,jxn_RR,jxn_ID
```

Notes:
- Downloads are resumable (`curl -C -`).
- Files are stored under `data/raw/recount3/gtex_v8/files/<TISSUE>/`.

### 3. Train with leak-aware train/val/test splits

Run the multi-tissue training pipeline:

```bash
PYTHONPATH=. python3 scripts/train/train_realworld_multitissue_noleak.py
```

What this script does:
- Detects tissues with complete `MM/RR/ID` junction files
- Builds stratified `train/val/test` splits
- Selects high-variance junction features using **train split only**
- Trains classifier (`logreg` backend in `splicevault.classify`)
- Evaluates on train/val/test holdouts
- Saves model + split/metrics artifacts

## Outputs

After training, artifacts are written to:

- Model bundle: `models/splicevault_realworld_multitissue_model.joblib`
- HDF5 metadata: `models/splicevault_realworld_multitissue_model.h5`
- Feature matrix: `data/processed/realworld_multitissue_noleak/junction_features.parquet`
- Labels: `data/processed/realworld_multitissue_noleak/junction_labels.csv`
- Splits: `data/processed/realworld_multitissue_noleak/splits.json`
- Metrics: `data/processed/realworld_multitissue_noleak/metrics.json`

## Use a Trained Model

Classify from a junction file via CLI:

```bash
splicevault classify \
  --junctions path/to/sample.junc \
  --model models/splicevault_realworld_multitissue_model.joblib
```

Train from an existing matrix/labels pair:

```bash
splicevault train \
  --matrix path/to/signature_matrix.parquet \
  --labels path/to/labels.csv \
  --out models/custom_model.joblib
```

## Reproducibility and Data Hygiene

- `.gitignore` excludes large raw datasets and model artifacts.
- Training scripts use deterministic random seeds for split stability.
- Split-first + train-only feature selection is used to reduce leakage risk.

## Development

```bash
python3 -m ruff check .
python3 -m pytest -q
```

## Current Roadmap

- Expand to more tissues and harder out-of-distribution evaluation
- Add donor-aware and study-aware holdout protocols
- Integrate long-read isoform evidence for v2 signatures
