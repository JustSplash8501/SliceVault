"""Typer CLI for SpliceVault."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from loguru import logger

from splicevault.classify import (
    load_model,
    predict_samples,
    save_model,
    train_classifier,
)
from splicevault.psi import compute_psi_matrix

app = typer.Typer(add_completion=False, help="SpliceVault CLI")


@app.command("build-psi")
def build_psi(
    junctions: list[Path] = typer.Option(..., "--junctions", exists=True, readable=True),
    out: Path = typer.Option(..., "--out", help="Output PSI parquet"),
    min_reads: int = typer.Option(10, "--min-reads", help="Minimum reads for PSI"),
    samples: str | None = typer.Option(
        None,
        "--samples",
        help="Optional comma-separated sample names matching junction order",
    ),
) -> None:
    """Build a PSI matrix from one or more junction files."""
    sample_names = [s.strip() for s in samples.split(",")] if samples else None
    psi = compute_psi_matrix(
        junction_files=junctions,
        sample_names=sample_names,
        min_reads=min_reads,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    psi.to_parquet(out)
    typer.echo(f"PSI matrix saved to {out}")


@app.command()
def train(
    matrix: Path = typer.Option(
        ...,
        "--matrix",
        exists=True,
        readable=True,
        help="Parquet PSI matrix",
    ),
    labels: Path = typer.Option(
        ...,
        "--labels",
        exists=True,
        readable=True,
        help="CSV labels table",
    ),
    out: Path = typer.Option(..., "--out", help="Output model file path"),
    model_type: str = typer.Option("lightgbm", "--model-type", help="lightgbm or logreg"),
    hdf5_out: Path | None = typer.Option(
        None,
        "--hdf5-out",
        help="Optional HDF5 metadata output",
    ),
) -> None:
    """Train a classifier from a PSI signature matrix and label file."""
    psi = pd.read_parquet(matrix)

    labels_df = pd.read_csv(labels)
    if labels_df.shape[1] < 2:
        raise typer.BadParameter("Labels CSV must have at least two columns: sample,label")
    sample_col = labels_df.columns[0]
    label_col = labels_df.columns[1]
    y = labels_df.set_index(sample_col)[label_col]

    classifier = train_classifier(psi_signature_matrix=psi, labels=y, model_type=model_type)
    save_model(classifier=classifier, output_path=out, hdf5_path=hdf5_out)
    typer.echo(f"Model saved to {out}")


@app.command()
def classify(
    junctions: Path = typer.Option(
        ...,
        "--junctions",
        exists=True,
        readable=True,
        help="Input junction file",
    ),
    model: Path = typer.Option(
        ...,
        "--model",
        exists=True,
        readable=True,
        help="Trained model bundle",
    ),
    out: Path | None = typer.Option(None, "--out", help="Optional predictions CSV path"),
    sample_name: str | None = typer.Option(None, "--sample", help="Optional sample id override"),
    min_reads: int = typer.Option(10, "--min-reads", help="Minimum reads for PSI"),
    top_n: int = typer.Option(10, "--top-n", help="Top driving splice events per sample"),
) -> None:
    """Classify a sample from a junction file and trained model."""
    clf = load_model(model)
    psi = compute_psi_matrix(
        [junctions],
        sample_names=[sample_name] if sample_name else None,
        min_reads=min_reads,
    )
    pred_df, drivers = predict_samples(classifier=clf, psi_matrix=psi, top_n=top_n)

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(out, index=False)
        logger.info("Wrote predictions to {}", out)

    typer.echo(pred_df.to_string(index=False))
    for sample, table in drivers.items():
        typer.echo(f"\nTop driving events for {sample}:")
        typer.echo(table.to_string(index=False))


if __name__ == "__main__":
    app()
