"""PSI computation utilities from junction files."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

INCLUSION_LABELS = {"inc", "incl", "inclusion", "include"}
SKIPPING_LABELS = {"skip", "skipping", "exclusion", "exc"}


def _infer_kind_and_event(junction_name: str) -> tuple[str, str]:
    """Infer junction kind and event id from a junction name.

    Parameters
    ----------
    junction_name
        Junction identifier. Recommended format: ``EVENT_ID|inc`` or
        ``EVENT_ID|skip``.

    Returns
    -------
    tuple[str, str]
        Event id and kind ("inc" or "skip").
    """
    parts = junction_name.split("|")
    if len(parts) >= 2:
        kind_token = parts[-1].strip().lower()
        event_id = "|".join(parts[:-1]).strip()
        if kind_token in INCLUSION_LABELS:
            return event_id, "inc"
        if kind_token in SKIPPING_LABELS:
            return event_id, "skip"

    # Fallback: treat unknown as inclusion-like singleton event.
    return junction_name.strip(), "inc"


def parse_junction_file(
    path: str | Path,
    sample_name: str | None = None,
    min_reads: int = 10,
) -> pd.DataFrame:
    """Parse a regtools-style junction file and compute PSI per event.

    The parser supports tab-delimited files with at least BED-like columns.
    Junction names should encode event type using ``EVENT|inc`` and
    ``EVENT|skip``.

    Parameters
    ----------
    path
        Path to a junction file.
    sample_name
        Optional sample identifier. Defaults to filename stem.
    min_reads
        Minimum total reads (inclusion + skipping) needed to report PSI.

    Returns
    -------
    pandas.DataFrame
        One-column PSI matrix indexed by event id.
    """
    file_path = Path(path)
    if sample_name is None:
        sample_name = file_path.stem

    logger.info("Parsing junction file {} for sample {}", file_path, sample_name)
    df = pd.read_csv(file_path, sep="\t", header=None, comment="#")
    if df.shape[1] < 6:
        raise ValueError("Junction file must have at least 6 tab-delimited columns")

    names = df.iloc[:, 3].astype(str)
    counts = pd.to_numeric(df.iloc[:, 4], errors="coerce").fillna(0.0)

    parsed = names.map(_infer_kind_and_event)
    work = pd.DataFrame(
        {
            "event_id": parsed.map(lambda t: t[0]),
            "kind": parsed.map(lambda t: t[1]),
            "count": counts,
        }
    )

    grouped = work.groupby(["event_id", "kind"], as_index=False)["count"].sum()
    pivoted = grouped.pivot(index="event_id", columns="kind", values="count").fillna(0.0)
    if "inc" not in pivoted.columns:
        pivoted["inc"] = 0.0
    if "skip" not in pivoted.columns:
        pivoted["skip"] = 0.0

    total = pivoted["inc"] + pivoted["skip"]
    psi = pivoted["inc"] / total.replace(0, np.nan)
    psi = psi.where(total >= min_reads, np.nan)

    out = pd.DataFrame({sample_name: psi})
    out.index.name = "event_id"
    logger.info("Computed PSI for {} events from {}", out.shape[0], sample_name)
    return out


def compute_psi_matrix(
    junction_files: Iterable[str | Path],
    sample_names: Iterable[str] | None = None,
    min_reads: int = 10,
) -> pd.DataFrame:
    """Compute PSI matrix across samples from junction files.

    Parameters
    ----------
    junction_files
        Iterable of junction file paths.
    sample_names
        Optional sample names aligned with junction files.
    min_reads
        Minimum reads required per event/sample PSI value.

    Returns
    -------
    pandas.DataFrame
        PSI matrix with rows as events and columns as samples.
    """
    files = list(junction_files)
    if not files:
        raise ValueError("No junction files provided")

    provided_names = list(sample_names) if sample_names is not None else []
    if provided_names and len(provided_names) != len(files):
        raise ValueError("sample_names length must match junction_files length")

    matrices: list[pd.DataFrame] = []
    for idx, path in enumerate(files):
        name = provided_names[idx] if provided_names else None
        matrices.append(parse_junction_file(path=path, sample_name=name, min_reads=min_reads))

    psi_matrix = pd.concat(matrices, axis=1).sort_index()
    psi_matrix = psi_matrix.loc[~psi_matrix.index.duplicated(keep="first")]
    logger.info("Built PSI matrix with shape {}", psi_matrix.shape)
    return psi_matrix
