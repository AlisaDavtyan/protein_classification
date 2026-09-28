"""Public helpers. Model loading lives in :mod:`amylodeep.loading`.

This module used to hold the loader itself, with a module-level cache that had no
key, so two different model directories in one process silently shared whichever
was loaded first. It now delegates, and every entry point takes ``model_dir``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from .loading import load_models_and_calibrators
from .model_store import describe, download, is_downloaded, resolve_model_dir


def load_token():
    """Kept for backward compatibility; the models are public, so None."""
    return None


def predict_ensemble_rolling(
    sequence: str,
    window_size: int = 6,
    *,
    aggregate: str = "mean",
    model_dir: str | Path | None = None,
    allow_download: bool = True,
    verbose: bool = False,
    xgboost_backend: str = "native",
):
    """Rolling-window prediction over one sequence, with per-residue output.

    The default window is 6 rather than 10. At 10 the projection onto residues
    overshoots a 6-residue APR by about 4 residues even with the conservative
    aggregation; at 6 it recovers it exactly. 5-6 is also the range the amyloid
    predictors that resolve residues are built around, so it is the honest default
    for a per-residue claim. Pass ``window_size=10`` to reproduce earlier results.

    Returns the dict documented on
    :meth:`EnsembleRollingWindowPredictor.rolling_window_prediction`, which now
    carries ``residue_probs`` and ``coverage_depth`` alongside ``position_probs``.
    """
    from .ensemble_predictor import EnsembleRollingWindowPredictor

    models, calibrators, tokenizer = load_models_and_calibrators(
        model_dir,
        allow_download=allow_download,
        verbose=verbose,
        xgboost_backend=xgboost_backend,
    )
    predictor = EnsembleRollingWindowPredictor(
        models,
        calibrators,
        tokenizer,
        model_dir=model_dir,
        allow_download=allow_download,
    )
    return predictor.rolling_window_prediction(sequence, window_size, aggregate=aggregate)


def ensure_models_downloaded(model_dir=None, **_ignored):
    """Deprecated alias for :func:`amylodeep.model_store.download`."""
    warnings.warn(
        "ensure_models_downloaded() is deprecated; use "
        "amylodeep.model_store.download(model_dir) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return download(model_dir)


__all__ = [
    "predict_ensemble_rolling",
    "load_models_and_calibrators",
    "ensure_models_downloaded",
    "is_downloaded",
    "describe",
    "download",
    "resolve_model_dir",
]
