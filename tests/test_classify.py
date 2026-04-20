from __future__ import annotations

import numpy as np
import pandas as pd

from splicevault.classify import benchmark, predict_samples, train_classifier


def test_train_and_predict_logreg(small_psi_matrix, small_labels):
    clf = train_classifier(small_psi_matrix, small_labels, model_type="logreg")
    pred, drivers = predict_samples(clf, small_psi_matrix, top_n=2)

    assert pred.shape[0] == small_psi_matrix.shape[1]
    assert set(pred["predicted_class"]).issubset(set(small_labels.unique()))
    assert all(table.shape[0] == 2 for table in drivers.values())


def test_unseen_events_are_ignored(small_psi_matrix, small_labels):
    clf = train_classifier(small_psi_matrix, small_labels, model_type="logreg")
    with_extra = small_psi_matrix.copy()
    with_extra.loc["UNSEEN|chr9:1-2"] = [0.1, 0.2, 0.3, 0.4]

    pred, _ = predict_samples(clf, with_extra)
    assert pred.shape[0] == 4


def test_benchmark_returns_modalities(small_psi_matrix, small_labels):
    expr = pd.DataFrame(
        {
            "S1": [10.0, 9.5, 2.0],
            "S2": [9.8, 9.2, 2.1],
            "S3": [2.2, 1.8, 8.9],
            "S4": [2.0, 1.9, 9.1],
        },
        index=["G1", "G2", "G3"],
    )

    out = benchmark(small_psi_matrix, expr, small_labels)

    assert set(out["modality"]) == {"splicing", "expression"}
    assert np.all((out["accuracy"] >= 0.0) & (out["accuracy"] <= 1.0))
