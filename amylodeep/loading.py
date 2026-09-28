"""Load each AmyloDeep artefact separately, and name the one that fails.

The previous loader wrapped five models and three calibrators in a single
``warnings.catch_warnings()`` block with no per-artefact handling, so any fault
inside it produced one message for eight possible causes -- or, when the fault
was in compiled code, no message at all and a bare signal.

Each artefact is now loaded through :func:`_step`, which records what it was
about to do before it does it. A Python exception is re-raised with that context
attached; a native crash leaves the context in the progress log, which is printed
to stderr, so the last line before the silence is the artefact that killed it.

The pickled artefacts get their own explanation because they are the likeliest
crash site: ``scikit-learn`` pickles are explicitly not portable across versions,
unpickling an ``SVC`` restores libsvm structures, and AmyloDeep pins only
``scikit-learn>=1.0`` with no upper bound. XGBoost is exempt -- it is loaded from
JSON via ``load_model``, which is version-stable by design.
"""

from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path
from typing import Any, Callable

from .model_store import ARTEFACTS, describe, esm_base_source, require, resolve_model_dir


def _versions() -> str:
    """Versions of the libraries that decide whether an artefact loads."""
    parts = []
    for module_name in ("torch", "transformers", "sklearn", "xgboost", "numpy", "jax"):
        try:
            module = __import__(module_name)
            parts.append(f"{module_name} {getattr(module, '__version__', '?')}")
        except Exception:
            parts.append(f"{module_name} (not importable)")
    return ", ".join(parts)


def _step(label: str, path: Any, fn: Callable[[], Any], *, verbose: bool) -> Any:
    """Run one load, announcing it first so a crash leaves a breadcrumb."""
    if verbose:
        print(f"AmyloDeep: loading {label} from {path}", file=sys.stderr, flush=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load AmyloDeep artefact {label!r} from {path}.\n"
            f"  {type(exc).__name__}: {exc}\n"
            f"  environment: {_versions()}"
        ) from exc


def _unpickle(path: Path, label: str) -> Any:
    try:
        with open(path, "rb") as handle:
            return pickle.load(handle)
    except Exception as exc:
        raise RuntimeError(
            f"{label} could not be unpickled from {path} "
            f"({path.stat().st_size} bytes).\n"
            f"  {type(exc).__name__}: {exc}\n"
            f"scikit-learn pickles are not portable between versions, and this "
            f"package pins only 'scikit-learn>=1.0'. Installing the version the "
            f"artefact was written with is the fix; there is no way to read it "
            f"with an incompatible one. Current: {_versions()}"
        ) from exc


def _load_xgboost(path: Path, *, backend: str = "native"):
    """Load the booster, checking the file's own version before native code runs.

    This is where AmyloDeep segfaulted, and the sequence matters. ``load_model``
    hands the path to XGBoost's C++ parser, and a model written by a different
    MAJOR version is not reliably rejected there -- on macOS arm64 it terminates
    the process with SIGSEGV instead of raising, so there is no exception to catch
    and nothing to print. The version is therefore read from the JSON in Python,
    where a mismatch is a message rather than a crash.

    The file records its own writer: ``{"version": [2, 1, 4]}`` at the top level.
    That is a fact about the artefact, not a guess about the environment.

    A bare ``Booster`` is used rather than ``XGBClassifier``. The sklearn wrapper
    additionally parses the ``scikit_learn`` attribute embedded in the JSON, which
    is one more cross-version surface for no benefit here: the objective is
    ``binary:logistic``, so ``Booster.predict`` returns P(class=1) directly --
    exactly what ``predict_proba(X)[:, 1]`` would give, and identical numerically.
    """
    import json

    # The file is validated BEFORE xgboost is imported, so a truncated artefact is
    # reported without loading a native library to find out.
    try:
        recorded = json.loads(path.read_text())
    except Exception as exc:
        raise RuntimeError(
            f"{path} is not readable as JSON ({type(exc).__name__}: {exc}), so it "
            f"is not an XGBoost model file. Re-download it: a truncated file or an "
            f"LFS pointer handed to XGBoost's native parser can crash the process "
            f"rather than raise."
        ) from exc

    if backend == "pure":
        # No libxgboost in the process at all. See amylodeep/xgb_pure.py for the
        # arithmetic and for compare_with_native(), which should be run once on a
        # machine where the real library loads.
        from .xgb_pure import PureXGBoostBinaryClassifier

        return PureXGBoostBinaryClassifier(path)

    if backend != "native":
        raise ValueError(
            f"xgboost_backend must be 'native', 'pure' or 'skip', got {backend!r}"
        )

    import xgboost

    check_xgboost_version(recorded.get("version"), xgboost.__version__, name=path.name)

    booster = xgboost.Booster()
    booster.load_model(str(path))
    return booster


def check_xgboost_version(written_by, installed: str, *, name: str = "the model") -> None:
    """Refuse a cross-major XGBoost artefact before native code sees it.

    ``written_by`` is the ``version`` field XGBoost records in its own JSON, e.g.
    ``[2, 1, 4]``. Comparing it against the installed version is the whole check,
    and it has to happen in Python: a cross-major file is not reliably rejected by
    XGBoost's C++ parser, and on macOS arm64 it terminates the process with SIGSEGV
    instead of raising -- no exception, no message, nothing in a log.

    Pure, so the rule is testable without xgboost installed.
    """
    if not isinstance(written_by, (list, tuple)) or not written_by:
        return
    file_version = ".".join(str(part) for part in written_by)
    installed_parts = str(installed).split(".")
    if int(written_by[0]) != int(installed_parts[0]):
        raise RuntimeError(
            f"{name} was written by XGBoost {file_version}, and XGBoost "
            f"{installed} is installed -- a different MAJOR version.\n"
            f"\n"
            f"Loading it is not safe to attempt: XGBoost's model parser is native "
            f"code, and a cross-major file is not reliably rejected there. On "
            f"macOS arm64 it terminates the process with SIGSEGV, which is why "
            f"this failure produced no message at all.\n"
            f"\n"
            f"Fix by matching the artefact:\n"
            f"    python -m pip install 'xgboost=={file_version}'\n"
            f"\n"
            f"AmyloDeep pins only 'xgboost>=1.7.0', so a fresh environment "
            f"resolves to whatever is current, which is how the two drifted apart."
        )
    if len(written_by) > 1 and len(installed_parts) > 1:
        if str(installed_parts[1]) != str(written_by[1]):
            warnings.warn(
                f"{name} was written by XGBoost {file_version}; {installed} is "
                f"installed. Same major version, so it should load, but pin "
                f"'xgboost=={file_version}' if predictions look wrong.",
                RuntimeWarning,
                stacklevel=2,
            )


_cache: dict[tuple[str, bool, str], tuple[dict, dict, Any]] = {}


def load_models_and_calibrators(
    model_dir: str | Path | None = None,
    *,
    allow_download: bool = True,
    verbose: bool = False,
    use_cache: bool = True,
    xgboost_backend: str = "native",
):
    """Return ``(models, calibrators, tokenizer)`` for one model directory.

    ``model_dir`` selects where artefacts are read from and, when missing and
    ``allow_download`` is set, written to. Caching is keyed on that directory, so
    two directories in one process no longer silently share the first one's
    models -- the previous module-level cache had no key at all.
    """
    root = resolve_model_dir(model_dir)
    key = (str(root), allow_download, xgboost_backend)
    if use_cache and key in _cache:
        return _cache[key]

    absent = [a.name for a in ARTEFACTS if not a.is_present(root)]
    if absent and not allow_download:
        raise FileNotFoundError(
            f"{len(absent)} AmyloDeep artefact(s) missing and downloading is "
            f"disabled: {', '.join(absent)}.\n{describe(root)}"
        )

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from .esm_classifier import ESMClassifier
    from .unirep_model import UniRepClassifier

    models: dict[str, Any] = {}
    calibrators: dict[str, Any] = {}

    esm150 = require("esm2_150M", root, allow_download=allow_download)
    models["esm2_150M"] = _step(
        "esm2_150M (classifier)", esm150,
        lambda: AutoModelForSequenceClassification.from_pretrained(esm150),
        verbose=verbose,
    )
    tokenizer = _step(
        "esm2_150M (tokenizer)", esm150,
        lambda: AutoTokenizer.from_pretrained(esm150),
        verbose=verbose,
    )

    unirep = require("unirep", root, allow_download=allow_download)
    models["unirep"] = _step(
        "unirep (classifier head)", unirep,
        lambda: UniRepClassifier.from_pretrained(unirep), verbose=verbose,
    )

    esm650 = require("esm2_650M", root, allow_download=allow_download)
    models["esm2_650M"] = _step(
        "esm2_650M (classifier head)", esm650,
        lambda: ESMClassifier.from_pretrained(esm650), verbose=verbose,
    )

    svm_path = require("svm", root, allow_download=allow_download)
    models["svm"] = _step(
        "svm", svm_path, lambda: _unpickle(svm_path, "The SVM model"), verbose=verbose
    )

    xgb_path = require("xgboost", root, allow_download=allow_download)
    if xgboost_backend == "skip":
        if verbose:
            print(
                "AmyloDeep: SKIPPING the xgboost head (--xgboost-backend skip)",
                file=sys.stderr,
                flush=True,
            )
        warnings.warn(
            "The XGBoost head is disabled, so the ensemble averages 4 models "
            "instead of 5. Probabilities are NOT comparable with a full run and "
            "must not be mixed with one in the same analysis.",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        models["xgboost"] = _step(
            f"xgboost ({xgboost_backend})",
            xgb_path,
            lambda: _load_xgboost(xgb_path, backend=xgboost_backend),
            verbose=verbose,
        )

    for name, label in (
        ("platt_unirep", "Platt calibrator (UniRep)"),
        ("isotonic_650M_NN", "isotonic calibrator (ESM2-650M)"),
        ("isotonic_XGBoost", "isotonic calibrator (XGBoost)"),
    ):
        path = require(name, root, allow_download=allow_download)
        calibrators[name] = _step(
            name, path, lambda p=path, l=label: _unpickle(p, l), verbose=verbose
        )

    if verbose:
        print("AmyloDeep: all artefacts loaded", file=sys.stderr, flush=True)

    result = (models, calibrators, tokenizer)
    if use_cache:
        _cache[key] = result
    return result


def encoder_source(
    model_dir: str | Path | None = None, *, allow_download: bool = True
) -> str:
    """Passed to ``from_pretrained`` for the frozen ESM2-650M encoder."""
    return esm_base_source(model_dir, allow_download=allow_download)
