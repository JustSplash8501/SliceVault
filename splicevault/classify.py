"""Model training and inference for splicing signature classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover - optional in some environments
    lgb = None


@dataclass
class SpliceClassifier:
    """Container for trained splice classifier artifacts."""

    model: Any
    features: list[str]
    classes: list[str]
    model_type: str


def _prepare_training_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """Transpose events x samples matrix into samples x features."""
    if matrix.empty:
        raise ValueError("Training matrix is empty")
    return matrix.T


def train_classifier(
    psi_signature_matrix: pd.DataFrame,
    labels: pd.Series,
    model_type: str = "lightgbm",
    random_state: int = 42,
) -> SpliceClassifier:
    """Train a classifier on PSI signatures.

    Parameters
    ----------
    psi_signature_matrix
        PSI signature matrix with events as rows and samples as columns.
    labels
        Sample labels indexed by sample names.
    model_type
        Model type: ``lightgbm`` or ``logreg``.
    random_state
        Random seed for reproducible training.

    Returns
    -------
    SpliceClassifier
        Trained classifier wrapper.
    """
    x = _prepare_training_matrix(psi_signature_matrix)
    y = labels.reindex(x.index)
    if y.isna().any():
        missing = y[y.isna()].index.tolist()
        raise ValueError(f"Missing labels for samples: {missing}")

    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    if model_type == "lightgbm":
        if lgb is None:
            raise ImportError("lightgbm is not installed; use model_type='logreg' or install it")
        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            random_state=random_state,
            n_jobs=-1,
        )
        pipeline: Pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", model),
            ]
        )
    elif model_type == "logreg":
        model = LogisticRegression(max_iter=300, random_state=random_state)
        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )
    else:
        raise ValueError("model_type must be 'lightgbm' or 'logreg'")

    pipeline.fit(x, y_enc)
    logger.info(
        "Trained {} classifier with {} samples and {} events",
        model_type,
        x.shape[0],
        x.shape[1],
    )

    return SpliceClassifier(
        model={"pipeline": pipeline, "label_encoder": label_encoder},
        features=list(x.columns),
        classes=list(label_encoder.classes_),
        model_type=model_type,
    )


def save_model(
    classifier: SpliceClassifier,
    output_path: str | Path,
    hdf5_path: str | Path | None = None,
) -> None:
    """Save model artifacts to disk.

    Parameters
    ----------
    classifier
        Trained classifier object.
    output_path
        Destination path for serialized model bundle.
    hdf5_path
        Optional path for metadata mirrored in HDF5 format.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, out)
    logger.info("Saved classifier bundle to {}", out)

    if hdf5_path is not None:
        h5_out = Path(hdf5_path)
        h5_out.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(h5_out, "w") as hf:
            hf.create_dataset("features", data=np.array(classifier.features, dtype="S"))
            hf.create_dataset("classes", data=np.array(classifier.classes, dtype="S"))
            hf.attrs["model_type"] = classifier.model_type
        logger.info("Saved classifier metadata to {}", h5_out)


def load_model(path: str | Path) -> SpliceClassifier:
    """Load a serialized classifier."""
    clf = joblib.load(path)
    if not isinstance(clf, SpliceClassifier):
        raise TypeError("Loaded object is not a SpliceClassifier")
    return clf


def _extract_event_metadata(event_id: str) -> dict[str, str]:
    """Infer gene/coordinate metadata from an event id string."""
    tokens = event_id.split("|")
    gene = tokens[0] if tokens else event_id
    coordinates = tokens[1] if len(tokens) > 1 else event_id
    return {"event_id": event_id, "gene": gene, "coordinates": coordinates}


def top_driving_events(
    classifier: SpliceClassifier,
    predicted_class: str,
    top_n: int = 10,
    feature_meta: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return top model-driving splice events for a predicted class.

    Parameters
    ----------
    classifier
        Fitted classifier.
    predicted_class
        Target class label.
    top_n
        Number of events to return.
    feature_meta
        Optional metadata indexed by event id.

    Returns
    -------
    pandas.DataFrame
        Ranked event table with importance scores and metadata.
    """
    if predicted_class not in classifier.classes:
        raise ValueError(f"Unknown predicted class: {predicted_class}")

    pipeline = classifier.model["pipeline"]
    model = pipeline.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        scores = pd.Series(model.feature_importances_, index=classifier.features, dtype=float)
    elif hasattr(model, "coef_"):
        class_idx = classifier.classes.index(predicted_class)
        coef = model.coef_
        if coef.ndim == 1:
            raw = coef
        elif coef.shape[0] == 1:
            # Binary logistic regression stores one coefficient vector.
            raw = coef[0]
        else:
            raw = coef[class_idx]
        scores = pd.Series(np.abs(raw), index=classifier.features, dtype=float)
    else:
        scores = pd.Series(0.0, index=classifier.features, dtype=float)

    top = scores.sort_values(ascending=False).head(top_n).rename("importance").to_frame()
    if feature_meta is not None and not feature_meta.empty:
        top = top.join(feature_meta, how="left")
    else:
        meta = pd.DataFrame(
            [_extract_event_metadata(eid) for eid in top.index]
        ).set_index("event_id")
        top = top.join(meta)

    return top.reset_index(names="event_id")


def predict_samples(
    classifier: SpliceClassifier,
    psi_matrix: pd.DataFrame,
    top_n: int = 10,
    feature_meta: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Predict classes and confidence scores for samples.

    Parameters
    ----------
    classifier
        Fitted classifier.
    psi_matrix
        PSI matrix with events as rows and samples as columns.
    top_n
        Number of top driving events to include per sample.
    feature_meta
        Optional feature annotation table.

    Returns
    -------
    tuple[pandas.DataFrame, dict[str, pandas.DataFrame]]
        Predictions table and per-sample driver-event tables.
    """
    x = psi_matrix.reindex(classifier.features).T
    unseen_events = set(psi_matrix.index) - set(classifier.features)
    if unseen_events:
        logger.warning("Ignoring {} unseen events at inference", len(unseen_events))

    pipeline = classifier.model["pipeline"]
    label_encoder: LabelEncoder = classifier.model["label_encoder"]

    probs = pipeline.predict_proba(x)
    pred_idx = probs.argmax(axis=1)
    pred_classes = label_encoder.inverse_transform(pred_idx)
    confidences = probs.max(axis=1)

    pred_df = pd.DataFrame(
        {
            "sample": x.index,
            "predicted_class": pred_classes,
            "confidence": confidences,
        }
    )

    drivers: dict[str, pd.DataFrame] = {}
    for sample, klass in zip(x.index, pred_classes, strict=True):
        drivers[sample] = top_driving_events(
            classifier=classifier,
            predicted_class=str(klass),
            top_n=top_n,
            feature_meta=feature_meta,
        )
    return pred_df, drivers


def benchmark(
    psi_matrix: pd.DataFrame,
    expression_matrix: pd.DataFrame,
    labels: pd.Series,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare splicing- and expression-based classification performance.

    Parameters
    ----------
    psi_matrix
        Splicing signature matrix (events x samples).
    expression_matrix
        Gene expression matrix (genes x samples) on the same sample set.
    labels
        Sample class labels indexed by sample name.
    random_state
        Random seed.

    Returns
    -------
    pandas.DataFrame
        Accuracy summary for both data modalities.
    """
    sample_order = [s for s in psi_matrix.columns if s in expression_matrix.columns]
    if len(sample_order) < 2:
        raise ValueError("Need at least 2 shared samples for benchmarking")

    y = labels.reindex(sample_order)
    if y.isna().any():
        raise ValueError("Labels missing for benchmark samples")

    y_enc = LabelEncoder().fit_transform(y)
    class_counts = pd.Series(y_enc).value_counts()
    max_splits = int(class_counts.min())
    if max_splits < 2:
        raise ValueError("Each class needs at least 2 samples for stratified benchmarking")
    cv = StratifiedKFold(n_splits=min(5, max_splits), shuffle=True, random_state=random_state)

    def _score(mat: pd.DataFrame) -> float:
        x = mat.reindex(columns=sample_order).T
        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=300, random_state=random_state)),
            ]
        )
        scores = cross_val_score(pipeline, x, y_enc, cv=cv, scoring="accuracy")
        return float(np.mean(scores))

    splice_acc = _score(psi_matrix)
    expr_acc = _score(expression_matrix)

    out = pd.DataFrame(
        {
            "modality": ["splicing", "expression"],
            "accuracy": [splice_acc, expr_acc],
        }
    )
    logger.info("Benchmark complete: splicing={:.3f}, expression={:.3f}", splice_acc, expr_acc)
    return out


def evaluate_holdout(
    classifier: SpliceClassifier,
    psi_matrix: pd.DataFrame,
    true_labels: pd.Series,
) -> float:
    """Evaluate accuracy on a holdout PSI matrix.

    Parameters
    ----------
    classifier
        Trained classifier.
    psi_matrix
        Holdout PSI matrix (events x samples).
    true_labels
        True labels indexed by sample name.

    Returns
    -------
    float
        Accuracy score.
    """
    pred_df, _ = predict_samples(classifier, psi_matrix)
    pred = pred_df.set_index("sample")["predicted_class"]
    y_true = true_labels.reindex(pred.index)
    if y_true.isna().any():
        raise ValueError("Missing true labels for holdout samples")
    return float(accuracy_score(y_true, pred))
