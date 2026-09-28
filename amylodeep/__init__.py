"""AmyloDeep: amyloid propensity from sequence, with a pLM ensemble.

An ensemble of five heads over protein-language-model representations -- ESM2-150M
fine-tuned, a classifier over mean-pooled ESM2-650M embeddings, a classifier over
UniRep embeddings, and an SVM and an XGBoost over the same ESM2-650M features --
averaged after per-head calibration, and applied along a rolling window.

Importing this package is cheap and has no side effects. Two things changed to
make that true, and both were causing real trouble:

* It no longer downloads at import. The previous version called
  ``ensure_models_downloaded()`` from module scope, so ``import amylodeep`` could
  start a multi-gigabyte fetch -- and since the presence check could never
  succeed, it did so on every import.
* ``torch``, ``transformers`` and ``jax_unirep`` are imported on first use, not at
  import time. ``amylodeep --check``, ``--help`` and the projection logic now work
  on a machine where the native stack does not load, which is exactly the machine
  on which you need to ask what is installed and where the weights are.
"""

from __future__ import annotations

__version__ = "0.4.0"
__author__ = "Alisa Davtyan"
__email__ = "alisadavtyan7@gmail.com"

# Light, dependency-free surface: paths, inventory, and the window -> residue
# projection. Safe to import anywhere.
from .model_store import (
    ARTEFACTS,
    ENV_VAR,
    REPO_ID,
    describe,
    download,
    is_downloaded,
    resolve_model_dir,
)
from .projection import project_to_residues, window_rows, window_statistics

#: Attribute name -> module it lives in. Resolved on first access so that the
#: heavy dependencies load only when something actually needs a model.
_LAZY = {
    "predict_ensemble_rolling": ".utils",
    "load_models_and_calibrators": ".loading",
    "EnsembleRollingWindowPredictor": ".ensemble_predictor",
    "ESMClassifier": ".esm_classifier",
    "ESMClassifierConfig": ".esm_classifier",
    "UniRepClassifier": ".unirep_model",
    "UniRepClassifierConfig": ".unirep_model",
}


def __getattr__(name: str):
    """PEP 562 lazy import for everything that needs torch."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "predict_ensemble_rolling",
    "load_models_and_calibrators",
    "EnsembleRollingWindowPredictor",
    "ESMClassifier",
    "ESMClassifierConfig",
    "UniRepClassifier",
    "UniRepClassifierConfig",
    "project_to_residues",
    "window_statistics",
    "window_rows",
    "ARTEFACTS",
    "ENV_VAR",
    "REPO_ID",
    "describe",
    "download",
    "is_downloaded",
    "resolve_model_dir",
    "__version__",
]
