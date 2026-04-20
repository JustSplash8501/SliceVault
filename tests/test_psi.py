from __future__ import annotations

import numpy as np

from splicevault.psi import compute_psi_matrix, parse_junction_file


def test_parse_junction_file_basic(tmp_path, synthetic_junction_text):
    path = tmp_path / "sample1.junc"
    path.write_text(synthetic_junction_text)

    psi = parse_junction_file(path, sample_name="S1", min_reads=10)

    assert "S1" in psi.columns
    assert np.isclose(float(psi.loc["EVT1", "S1"]), 0.75)
    assert np.isclose(float(psi.loc["EVT2", "S1"]), 0.5)
    assert np.isnan(psi.loc["EVT3", "S1"])


def test_compute_psi_matrix_single_sample(tmp_path, synthetic_junction_text):
    path = tmp_path / "single.junc"
    path.write_text(synthetic_junction_text)

    mat = compute_psi_matrix([path], sample_names=["S1"], min_reads=10)

    assert mat.shape[1] == 1
    assert "S1" in mat.columns


def test_compute_psi_matrix_raises_on_empty():
    try:
        compute_psi_matrix([])
    except ValueError as exc:
        assert "No junction files" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_all_na_event_across_samples(tmp_path):
    text = "\n".join(
        [
            "chr1\t1\t2\tEVT_LOW|inc\t1\t+",
            "chr1\t1\t3\tEVT_LOW|skip\t1\t+",
        ]
    )
    p1 = tmp_path / "s1.junc"
    p2 = tmp_path / "s2.junc"
    p1.write_text(text)
    p2.write_text(text)

    mat = compute_psi_matrix([p1, p2], sample_names=["S1", "S2"], min_reads=10)
    assert mat.loc["EVT_LOW"].isna().all()
