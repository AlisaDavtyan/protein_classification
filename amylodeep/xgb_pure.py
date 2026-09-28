"""Evaluate the XGBoost head from its JSON, without libxgboost.

Why this exists. The XGBoost head is where AmyloDeep crashes on macOS arm64:
three PyTorch models load, then the process dies inside libxgboost.dylib with
SIGSEGV -- no exception, nothing to catch. The version recorded in the artefact
(2.1.4) matches the installed package, so it is not a cross-version rejection; the
fault is in the native library itself, in a process that has already loaded
PyTorch's own OpenMP runtime.

The head is 10 trees of 15 nodes over 1280 features, all splits numerical, no
categorical features, objective ``binary:logistic``. That is a few hundred
comparisons per sample. Loading a 100 MB compiled library with its own threading
runtime to do that arithmetic is the reason the process dies, and the arithmetic
itself is unambiguous:

    margin = logit(base_score) + sum over trees of the reached leaf's weight
    p      = sigmoid(margin)

XGBoost stores ``base_score`` for ``binary:logistic`` in PROBABILITY space and
converts it with ``ProbToMargin`` -- the logit -- before boosting, which is why the
logit appears above rather than the raw value. Traversal follows XGBoost's own
rule: at an internal node, ``feature < split_condition`` goes left, otherwise
right; a missing value follows ``default_left``. A leaf is marked by
``left_children[i] == -1`` and its weight is in ``split_conditions[i]``.

This is a re-implementation of someone else's model format, so it is opt-in
(``--xgboost-backend pure``) and carries :func:`compare_with_native`, which
reports the maximum absolute difference against the real library on any machine
where that library loads. Run it once before trusting these numbers in a result:
agreement to ~1e-6 is what a correct implementation gives, since the artefact
stores float32.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class UnsupportedModel(RuntimeError):
    """The JSON uses a feature this evaluator deliberately does not implement."""


@dataclass
class _Tree:
    left: np.ndarray
    right: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    default_left: np.ndarray

    def leaf_values(self, X: np.ndarray) -> np.ndarray:
        """Walk every row to its leaf. Vectorised over samples, looped over depth."""
        node = np.zeros(len(X), dtype=np.int32)
        active = self.left[node] != -1
        # A complete tree of N nodes is at most N levels deep; the bound makes a
        # malformed tree terminate instead of spinning.
        for _ in range(len(self.left) + 1):
            if not active.any():
                break
            rows = np.nonzero(active)[0]
            current = node[rows]
            values = X[rows, self.feature[current]]
            go_left = values < self.threshold[current]
            missing = np.isnan(values)
            if missing.any():
                go_left = np.where(missing, self.default_left[current] == 1, go_left)
            node[rows] = np.where(go_left, self.left[current], self.right[current])
            active[rows] = self.left[node[rows]] != -1
        if active.any():
            raise UnsupportedModel(
                f"{active.sum()} sample(s) did not reach a leaf; the tree's "
                f"left/right children are inconsistent."
            )
        # For a leaf, the weight is stored in split_conditions.
        return self.threshold[node]


class PureXGBoostBinaryClassifier:
    """The ``binary:logistic`` booster, read from JSON and evaluated in numpy."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        document = json.loads(self.path.read_text())
        learner = document["learner"]

        objective = learner["objective"]["name"]
        if objective != "binary:logistic":
            raise UnsupportedModel(
                f"{self.path.name}: objective is {objective!r}. This evaluator "
                f"implements binary:logistic only -- the margin-to-probability "
                f"transform differs per objective, and guessing it would be worse "
                f"than refusing."
            )

        params = learner["learner_model_param"]
        self.n_features = int(params["num_feature"])
        if int(params.get("num_class", 0) or 0) > 1:
            raise UnsupportedModel(
                f"{self.path.name}: num_class={params['num_class']}, i.e. multiclass. "
                f"Not implemented."
            )
        base_score = float(params["base_score"])
        # PROBABILITY space -> margin, as XGBoost's LogisticRegression does.
        self.base_margin = math.log(base_score / (1.0 - base_score))
        self.xgboost_version = document.get("version")

        model = learner["gradient_booster"]["model"]
        self.trees: list[_Tree] = []
        for index, raw in enumerate(model["trees"]):
            if raw.get("categories"):
                raise UnsupportedModel(
                    f"{self.path.name}: tree {index} has categorical splits, which "
                    f"this evaluator does not implement."
                )
            if any(int(t) != 0 for t in raw.get("split_type", [])):
                raise UnsupportedModel(
                    f"{self.path.name}: tree {index} has a non-numerical split type."
                )
            self.trees.append(
                _Tree(
                    left=np.asarray(raw["left_children"], dtype=np.int32),
                    right=np.asarray(raw["right_children"], dtype=np.int32),
                    feature=np.asarray(raw["split_indices"], dtype=np.int32),
                    threshold=np.asarray(raw["split_conditions"], dtype=np.float64),
                    default_left=np.asarray(raw["default_left"], dtype=np.int32),
                )
            )
        if not self.trees:
            raise UnsupportedModel(f"{self.path.name}: no trees in the model.")

    def margin(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"expected a 2-D feature matrix, got shape {X.shape}")
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"{self.path.name} expects {self.n_features} features, got "
                f"{X.shape[1]}. The XGBoost head reads mean-pooled ESM2-650M "
                f"embeddings, so a mismatch means the wrong features were passed."
            )
        total = np.full(len(X), self.base_margin, dtype=np.float64)
        for tree in self.trees:
            total += tree.leaf_values(X)
        return total

    def predict(self, X) -> np.ndarray:
        """P(class = 1), matching ``Booster.predict`` for this objective."""
        margins = self.margin(X)
        # Numerically stable logistic.
        return np.where(
            margins >= 0,
            1.0 / (1.0 + np.exp(-margins)),
            np.exp(margins) / (1.0 + np.exp(margins)),
        )

    def predict_proba(self, X) -> np.ndarray:
        """Two-column form, so this can stand in for an XGBClassifier."""
        positive = self.predict(X)
        return np.column_stack([1.0 - positive, positive])

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"PureXGBoostBinaryClassifier(trees={len(self.trees)}, "
            f"features={self.n_features}, written_by={self.xgboost_version})"
        )


def compare_with_native(path: str | Path, *, n_samples: int = 64, seed: int = 0) -> dict:
    """Max absolute difference against the real library, on random features.

    Run this once on a machine where libxgboost loads. This module re-implements
    another project's model format from its serialised form; that is worth doing to
    avoid a crashing dependency, and it is not worth trusting unchecked.
    """
    import xgboost

    pure = PureXGBoostBinaryClassifier(path)
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, pure.n_features)).astype(np.float32)

    booster = xgboost.Booster()
    booster.load_model(str(path))
    native = booster.predict(xgboost.DMatrix(X))
    ours = pure.predict(X)
    difference = float(np.max(np.abs(native - ours)))
    return {
        "max_abs_difference": difference,
        "agrees": difference < 1e-5,
        "xgboost_version": xgboost.__version__,
        "model_written_by": pure.xgboost_version,
        "n_samples": n_samples,
    }
