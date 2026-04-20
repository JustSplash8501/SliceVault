"""Splicing signature construction utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


def filter_high_variance_events(
    psi_matrix: pd.DataFrame,
    min_variance: float = 0.02,
    min_non_na_frac: float = 0.5,
) -> pd.DataFrame:
    """Filter PSI matrix to high-variance, sufficiently observed events.

    Parameters
    ----------
    psi_matrix
        PSI matrix with events as rows and samples as columns.
    min_variance
        Minimum variance threshold for retained events.
    min_non_na_frac
        Minimum non-missing fraction of samples per event.

    Returns
    -------
    pandas.DataFrame
        Filtered PSI matrix.
    """
    if psi_matrix.empty:
        return psi_matrix.copy()

    non_na_frac = psi_matrix.notna().mean(axis=1)
    variances = psi_matrix.var(axis=1, skipna=True)
    keep = (non_na_frac >= min_non_na_frac) & (variances >= min_variance)
    filtered = psi_matrix.loc[keep]

    logger.info(
        "Filtered PSI events: {} -> {} using min_variance={} min_non_na_frac={}",
        psi_matrix.shape[0],
        filtered.shape[0],
        min_variance,
        min_non_na_frac,
    )
    return filtered


def compute_eta_squared(psi_values: pd.Series, labels: pd.Series) -> float:
    """Estimate tissue-specificity using one-way ANOVA eta-squared.

    Parameters
    ----------
    psi_values
        PSI vector across samples for one event.
    labels
        Condition labels aligned to sample names.

    Returns
    -------
    float
        Eta-squared effect size in [0, 1].
    """
    aligned = pd.concat([psi_values.rename("psi"), labels.rename("label")], axis=1).dropna()
    if aligned.empty or aligned["label"].nunique() < 2:
        return 0.0

    grand_mean = aligned["psi"].mean()
    ss_total = ((aligned["psi"] - grand_mean) ** 2).sum()
    if np.isclose(ss_total, 0.0):
        return 0.0

    grouped = aligned.groupby("label")["psi"]
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for _, g in grouped)
    return float(ss_between / ss_total)


def build_signature_matrix(
    psi_matrix: pd.DataFrame,
    labels: pd.Series,
    top_n: int = 500,
    min_eta_sq: float = 0.15,
    min_variance: float = 0.02,
    min_non_na_frac: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a tissue/disease splicing signature matrix.

    Parameters
    ----------
    psi_matrix
        PSI matrix with events as rows and samples as columns.
    labels
        Sample labels indexed by sample name.
    top_n
        Maximum number of events retained in the final signature matrix.
    min_eta_sq
        Minimum eta-squared tissue-specificity threshold.
    min_variance
        Minimum PSI variance threshold.
    min_non_na_frac
        Minimum observed sample fraction per event.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Signature PSI matrix and accompanying per-event statistics.
    """
    if psi_matrix.empty:
        empty_stats = pd.DataFrame(columns=["variance", "eta_squared"])
        return psi_matrix.copy(), empty_stats  # pragma: no cover

    labels = labels.reindex(psi_matrix.columns)
    if labels.isna().any():
        missing = labels[labels.isna()].index.tolist()
        raise ValueError(f"Missing labels for samples: {missing}")

    filtered = filter_high_variance_events(
        psi_matrix=psi_matrix,
        min_variance=min_variance,
        min_non_na_frac=min_non_na_frac,
    )

    variances = filtered.var(axis=1, skipna=True)
    eta_sq = filtered.apply(lambda row: compute_eta_squared(row, labels), axis=1)
    stats = pd.DataFrame({"variance": variances, "eta_squared": eta_sq})

    selected = stats.loc[stats["eta_squared"] >= min_eta_sq].sort_values(
        by=["eta_squared", "variance"], ascending=False
    )
    if top_n > 0:
        selected = selected.head(top_n)

    signature = filtered.loc[selected.index]
    logger.info(
        "Constructed signature matrix with {} events across {} samples",
        signature.shape[0],
        signature.shape[1],
    )
    return signature, selected


def save_signature_matrix(matrix: pd.DataFrame, output_path: str | Path) -> None:
    """Serialize signature matrix to Parquet.

    Parameters
    ----------
    matrix
        Signature matrix to persist.
    output_path
        Destination parquet file path.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(out)
    logger.info("Saved signature matrix to {}", out)
