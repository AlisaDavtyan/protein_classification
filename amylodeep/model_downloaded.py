"""Deprecated. Superseded by :mod:`amylodeep.model_store`.

Kept so existing imports keep working. The original implementation asked
``try_to_load_from_cache`` about files that ``snapshot_download(local_dir=...)``
had written as plain files, so ``is_downloaded()`` always answered False and the
whole repository was re-fetched on every import. ``model_store`` decides presence
by looking at the files instead, and takes the directory as a parameter.
"""

from __future__ import annotations

import warnings

from .model_store import DEFAULT_MODEL_DIR, REPO_ID, download, is_downloaded

CACHE_DIR = str(DEFAULT_MODEL_DIR)

#: Historic list of subfolders. Unused by the original code; kept for callers
#: that imported it.
FILES = [
    "esm2_150M", "esm2_650M", "isotonic_650M_NN", "isotonic_XGBoost",
    "platt_unirep", "svm", "unirep", "xgb",
]


def ensure_models_downloaded(model_dir=None):
    warnings.warn(
        "amylodeep.model_downloaded is deprecated; use "
        "amylodeep.model_store.download(model_dir).",
        DeprecationWarning,
        stacklevel=2,
    )
    return download(model_dir)


__all__ = ["CACHE_DIR", "FILES", "REPO_ID", "ensure_models_downloaded", "is_downloaded"]
