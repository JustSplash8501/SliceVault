from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def synthetic_junction_text() -> str:
    # chrom start end name score strand
    return "\n".join(
        [
            "chr1\t100\t200\tEVT1|inc\t30\t+",
            "chr1\t100\t300\tEVT1|skip\t10\t+",
            "chr2\t400\t500\tEVT2|inc\t5\t-",
            "chr2\t400\t700\tEVT2|skip\t5\t-",
            "chr3\t800\t900\tEVT3|inc\t1\t+",
            "chr3\t800\t950\tEVT3|skip\t1\t+",
        ]
    )


@pytest.fixture
def small_psi_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "S1": [0.9, 0.1, 0.6],
            "S2": [0.85, 0.15, 0.55],
            "S3": [0.2, 0.8, 0.4],
            "S4": [0.15, 0.75, 0.45],
        },
        index=["GENE1|chr1:100-200", "GENE2|chr2:300-400", "GENE3|chr3:500-600"],
    )


@pytest.fixture
def small_labels() -> pd.Series:
    return pd.Series({"S1": "tissue_a", "S2": "tissue_a", "S3": "tissue_b", "S4": "tissue_b"})
