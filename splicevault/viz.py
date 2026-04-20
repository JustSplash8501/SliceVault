"""Visualization utilities for SpliceVault."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

try:
    import umap
except ImportError:  # pragma: no cover - optional import
    umap = None


def plot_umap(
    signature_matrix: pd.DataFrame,
    labels: pd.Series,
    random_state: int = 42,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    figsize: tuple[int, int] = (8, 6),
) -> plt.Figure:
    """Generate a UMAP projection from a splicing signature matrix.

    Parameters
    ----------
    signature_matrix
        Event x sample matrix.
    labels
        Sample labels indexed by sample id.
    random_state
        Random seed.
    n_neighbors
        UMAP neighbor parameter.
    min_dist
        UMAP minimum distance parameter.
    figsize
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
        Rendered figure handle.
    """
    if umap is None:
        raise ImportError("umap-learn is required for UMAP plotting")

    x = signature_matrix.T
    y = labels.reindex(x.index)
    if y.isna().any():
        raise ValueError("Missing labels for one or more samples")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(x.fillna(x.median()))

    fig, ax = plt.subplots(figsize=figsize)
    label_values = y.astype(str)
    for klass in sorted(label_values.unique()):
        mask = label_values == klass
        ax.scatter(embedding[mask, 0], embedding[mask, 1], label=klass, alpha=0.85)

    ax.set_title("Splicing Signature UMAP")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(frameon=False, title="Class")
    fig.tight_layout()

    logger.info("Generated UMAP plot for {} samples", x.shape[0])
    return fig


def plot_sashimi_arc(
    event_id: str,
    junction_counts: pd.DataFrame,
    groups: pd.Series,
    group_order: Sequence[str] | None = None,
    figsize: tuple[int, int] = (9, 4),
) -> plt.Figure:
    """Draw a minimal sashimi-style arc plot for one event.

    Parameters
    ----------
    event_id
        Event identifier to display in title.
    junction_counts
        DataFrame with samples in rows and junction labels in columns.
    groups
        Group labels indexed by sample name.
    group_order
        Optional explicit group plotting order.
    figsize
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
        Rendered arc plot.
    """
    aligned_groups = groups.reindex(junction_counts.index)
    if aligned_groups.isna().any():
        raise ValueError("Missing group labels for one or more samples")

    order = list(group_order) if group_order is not None else sorted(aligned_groups.unique())
    group_means = {}
    for grp in order:
        group_means[grp] = junction_counts.loc[aligned_groups == grp].mean(
            axis=0, numeric_only=True
        )

    fig, ax = plt.subplots(figsize=figsize)
    x_coords = np.arange(len(junction_counts.columns))

    for idx, grp in enumerate(order):
        means = group_means[grp].to_numpy(dtype=float)
        baseline = idx * (np.nanmax(means) + 1.0)
        for j in range(len(means) - 1):
            height = max(means[j], means[j + 1]) * 0.35
            center = (x_coords[j] + x_coords[j + 1]) / 2
            width = abs(x_coords[j + 1] - x_coords[j])
            theta = np.linspace(0, np.pi, 50)
            arc_x = center + (width / 2) * np.cos(theta)
            arc_y = baseline + height * np.sin(theta)
            ax.plot(arc_x, arc_y, linewidth=1.5)
        ax.text(x_coords[-1] + 0.2, baseline, str(grp), va="center")

    ax.set_xticks(x_coords)
    ax.set_xticklabels(junction_counts.columns, rotation=30, ha="right")
    ax.set_title(f"Sashimi-style arcs: {event_id}")
    ax.set_ylabel("Relative junction support")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    logger.info("Generated sashimi-style arc plot for event {}", event_id)
    return fig
