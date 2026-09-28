"""Locate, download and validate AmyloDeep's model artefacts.

Replaces the implicit caching in :mod:`amylodeep.model_downloaded`, which had
three problems that together made a failure impossible to read:

* **The location was not configurable.** Everything resolved to
  ``~/.amylodeep_models``, so a shared or read-only install, a scratch volume or
  an air-gapped machine had nowhere to put the weights.
* **``is_downloaded()`` could never return True.** It asked
  ``try_to_load_from_cache(repo_id, name)``, which looks in the Hugging Face hub
  cache, while ``snapshot_download(local_dir=..., local_dir_use_symlinks=False)``
  writes flat real files into the target directory. The cache lookup therefore
  missed files that were sitting on disk, and the whole repository was fetched
  again on every import. ``get_model_file()`` then used ``cache_dir=`` on the same
  directory -- a second, incompatible layout in one folder -- so the per-file
  downloads did not reuse the snapshot either.
* **Nothing said which artefact failed.** Five models and three calibrators were
  loaded in sequence with one ``try`` around the lot, so a fault in any of them
  surfaced as a single opaque message, or as a signal with no message at all.

Presence is now decided by looking at the files, each artefact is loaded and
reported individually, and the directory is a parameter.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ID = "AlisaDavtyan/amylodeep-models"

#: The public ESM2 encoder used as a frozen feature extractor. It is NOT part of
#: the AmyloDeep repository and is fetched from its own Hugging Face repo, which
#: is why a reader looking for an "ESM download" in this package does not find
#: one: it arrives through ``from_pretrained`` on a hub id.
ESM_BASE_REPO = "facebook/esm2_t33_650M_UR50D"

#: Environment override, checked when no directory is passed explicitly.
ENV_VAR = "AMYLODEEP_MODEL_DIR"

DEFAULT_MODEL_DIR = Path("~/.amylodeep_models").expanduser()


@dataclass(frozen=True)
class Artefact:
    """One model or calibrator on disk.

    ``filename`` is None for artefacts that are a *directory* of Hugging Face
    files (config.json + weights), and a file name for the pickled and JSON
    models.
    """

    name: str
    subfolder: str
    filename: str | None
    kind: str
    note: str

    def local_dir(self, model_dir: Path) -> Path:
        return Path(model_dir) / self.subfolder

    def local_path(self, model_dir: Path) -> Path:
        base = self.local_dir(model_dir)
        return base if self.filename is None else base / self.filename

    def is_present(self, model_dir: Path) -> bool:
        target = self.local_path(model_dir)
        if self.filename is None:
            # A transformers directory needs a config and at least one weight
            # file; a directory that holds only a config is a half-finished
            # download and must not be reported as present.
            if not target.is_dir() or not (target / "config.json").is_file():
                return False
            weights = list(target.glob("*.safetensors")) + list(target.glob("*.bin"))
            return bool(weights)
        if not (target.is_file() and target.stat().st_size > 0):
            return False
        if target.suffix == ".json":
            # A JSON artefact is handed to a NATIVE parser (XGBoost's), where a
            # truncated file or an LFS pointer can crash the process instead of
            # raising. Non-zero length is not enough: it must parse.
            import json

            try:
                json.loads(target.read_text())
            except Exception:
                return False
        return True


#: Every artefact AmyloDeep loads, in the order it loads them. The order is the
#: diagnostic: when a run dies without a message, the last artefact reported is
#: where it died.
ARTEFACTS: tuple[Artefact, ...] = (
    Artefact(
        "esm2_150M", "esm2_150M", None, "transformers",
        "ESM2-150M fine-tuned for sequence classification, with its tokenizer",
    ),
    Artefact(
        "unirep", "unirep", None, "transformers",
        "classifier head over UniRep's 1900-d embedding",
    ),
    Artefact(
        "esm2_650M", "esm2_650M", None, "transformers",
        "classifier head over mean-pooled ESM2-650M embeddings",
    ),
    Artefact(
        "svm", "svm", "svm_model.pkl", "pickle",
        "scikit-learn SVM over mean-pooled ESM2-650M embeddings",
    ),
    Artefact(
        "xgboost", "xgb", "xgb_model.json", "xgboost_json",
        "XGBoost over mean-pooled ESM2-650M embeddings",
    ),
    Artefact(
        "platt_unirep", "platt_unirep", "platt_unirep.pkl", "pickle",
        "Platt calibrator for the UniRep head",
    ),
    Artefact(
        "isotonic_650M_NN", "isotonic_650M_NN", "isotonic_650M_NN.pkl", "pickle",
        "isotonic calibrator for the ESM2-650M head",
    ),
    Artefact(
        "isotonic_XGBoost", "isotonic_XGBoost", "isotonic_XGBoost.pkl", "pickle",
        "isotonic calibrator for the XGBoost head",
    ),
)

BY_NAME = {artefact.name: artefact for artefact in ARTEFACTS}


def resolve_model_dir(model_dir: str | Path | None = None) -> Path:
    """Where the artefacts live, by explicit argument, environment, then default.

    Returned without being created: a caller that only wants to *report* on a
    directory should not bring it into existence as a side effect.
    """
    if model_dir is not None:
        return Path(model_dir).expanduser().resolve()
    from_env = os.environ.get(ENV_VAR)
    if from_env:
        return Path(from_env).expanduser().resolve()
    return DEFAULT_MODEL_DIR


def missing(model_dir: str | Path | None = None) -> list[Artefact]:
    root = resolve_model_dir(model_dir)
    return [artefact for artefact in ARTEFACTS if not artefact.is_present(root)]


def is_downloaded(model_dir: str | Path | None = None) -> bool:
    """True when every artefact is on disk, decided by inspecting the files."""
    return not missing(model_dir)


def describe(model_dir: str | Path | None = None) -> str:
    """A present/missing inventory with sizes, for an error message or --check."""
    root = resolve_model_dir(model_dir)
    lines = [f"AmyloDeep model directory: {root}"]
    if not root.exists():
        lines.append("  (does not exist)")
    for artefact in ARTEFACTS:
        target = artefact.local_path(root)
        if artefact.is_present(root):
            if artefact.filename is None:
                size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
            else:
                size = target.stat().st_size
            lines.append(f"  [present] {artefact.name:18s} {size / 1e6:8.1f} MB  {artefact.note}")
        else:
            lines.append(f"  [MISSING] {artefact.name:18s}           {artefact.note}")

    # Reported last and separately because it is a different repository and by far
    # the largest item; a directory that has all eight artefacts and not this one
    # still cannot run offline.
    local = encoder_dir(root)
    if encoder_is_present(root):
        size = sum(f.stat().st_size for f in local.rglob("*") if f.is_file())
        lines.append(
            f"  [present] {'encoder':18s} {size / 1e6:8.1f} MB  "
            f"frozen {ESM_BASE_REPO} (feature extractor)"
        )
    else:
        lines.append(
            f"  [MISSING] {'encoder':18s}           frozen {ESM_BASE_REPO} "
            f"(~2.6 GB; without it a run falls back to the HF cache)"
        )
    return "\n".join(lines)


def download(
    model_dir: str | Path | None = None,
    *,
    only: list[str] | None = None,
    quiet: bool = False,
) -> Path:
    """Fetch the missing artefacts into ``model_dir``.

    Only what is missing is fetched, and into the flat layout ``is_present``
    checks, so a second call is a no-op rather than a second download.
    """
    root = resolve_model_dir(model_dir)
    wanted = [a for a in ARTEFACTS if only is None or a.name in set(only)]
    absent = [a for a in wanted if not a.is_present(root)]
    if not absent:
        return root

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError(
            "huggingface_hub is required to download AmyloDeep's models. "
            f"Install it, or place the files under {root} yourself "
            f"(see amylodeep.model_store.describe()).\n{exc}"
        ) from exc

    root.mkdir(parents=True, exist_ok=True)
    patterns = [f"{a.subfolder}/*" for a in absent]
    if not quiet:
        names = ", ".join(a.name for a in absent)
        print(f"AmyloDeep: downloading {len(absent)} artefact(s) into {root}: {names}")
    try:
        snapshot_download(
            repo_id=REPO_ID,
            local_dir=str(root),
            allow_patterns=patterns,
            ignore_patterns=["*.lock"],
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not download AmyloDeep models from {REPO_ID} into {root}.\n"
            f"  reason: {type(exc).__name__}: {exc}\n"
            f"If this machine has no access to huggingface.co, copy the "
            f"directory from a connected machine and point --model-dir at it, or "
            f"set {ENV_VAR}. Expected contents:\n{describe(root)}"
        ) from exc

    still_absent = [a.name for a in absent if not a.is_present(root)]
    if still_absent:
        raise RuntimeError(
            f"Download finished but these artefacts are still missing or empty: "
            f"{', '.join(still_absent)}.\n{describe(root)}"
        )
    return root


def require(
    name: str,
    model_dir: str | Path | None = None,
    *,
    allow_download: bool = True,
) -> Path:
    """Path to one artefact, downloading it if permitted and absent."""
    artefact = BY_NAME[name]
    root = resolve_model_dir(model_dir)
    if artefact.is_present(root):
        return artefact.local_path(root)
    if not allow_download:
        raise FileNotFoundError(
            f"AmyloDeep artefact {name!r} ({artefact.note}) is not at "
            f"{artefact.local_path(root)} and downloading is disabled. "
            f"Either drop the file there, point --model-dir at a directory that "
            f"has it, or drop --no-download."
        )
    download(root, only=[name])
    return artefact.local_path(root)


#: Directory name the frozen encoder is kept under, inside the model directory.
ENCODER_DIRNAME = "esm2_t33_650M_UR50D"


def encoder_dir(model_dir: str | Path | None = None) -> Path:
    return resolve_model_dir(model_dir) / ENCODER_DIRNAME


def encoder_is_present(model_dir: str | Path | None = None) -> bool:
    """A config plus real weights, same standard as the other model directories."""
    local = encoder_dir(model_dir)
    if not (local / "config.json").is_file():
        return False
    return bool(list(local.glob("*.safetensors")) + list(local.glob("*.bin")))


def download_encoder(
    model_dir: str | Path | None = None, *, quiet: bool = False
) -> Path:
    """Fetch the frozen ESM2-650M encoder INTO the model directory.

    Previously this was the one piece the model directory could not hold. The
    encoder was reached by ``from_pretrained("facebook/esm2_t33_650M_UR50D")``,
    which downloads into the Hugging Face cache under ``~/.cache/huggingface`` --
    so ``--model-dir`` did not actually make a run self-contained, and the largest
    single download (~2.6 GB) landed somewhere the caller had not chosen and could
    not point at. It is now fetched here like everything else.
    """
    local = encoder_dir(model_dir)
    if encoder_is_present(local.parent):
        return local

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError(
            f"huggingface_hub is required to download the ESM2 encoder. Install "
            f"it, or place the files in {local} yourself.\n{exc}"
        ) from exc

    local.mkdir(parents=True, exist_ok=True)
    if not quiet:
        print(
            f"AmyloDeep: downloading the frozen encoder {ESM_BASE_REPO} "
            f"(~2.6 GB) into {local}"
        )
    # ONE copy of the weights and the tokenizer. facebook/esm2_t33_650M_UR50D
    # carries model.safetensors AND pytorch_model.bin -- the same 2.6 GB model in
    # two formats -- plus tf_model.h5. Allowing every weight extension downloads
    # 5+ GB to load one of them. safetensors is preferred and is what transformers
    # picks, so it is fetched alone; the .bin fallback below covers a repo that
    # has no safetensors copy.
    small = ["*.json", "*.txt", "*.model"]
    try:
        snapshot_download(
            repo_id=ESM_BASE_REPO,
            local_dir=str(local),
            allow_patterns=small + ["*.safetensors"],
            ignore_patterns=["*.h5", "*.msgpack", "*.lock", "rust_model.ot"],
        )
        if not list(local.glob("*.safetensors")):
            if not quiet:
                print("AmyloDeep: no safetensors in that repo, fetching .bin weights")
            snapshot_download(
                repo_id=ESM_BASE_REPO,
                local_dir=str(local),
                allow_patterns=small + ["*.bin"],
                ignore_patterns=["*.h5", "*.msgpack", "*.lock", "rust_model.ot"],
            )
    except Exception as exc:
        raise RuntimeError(
            f"Could not download {ESM_BASE_REPO} into {local}.\n"
            f"  reason: {type(exc).__name__}: {exc}\n"
            f"Alternative, on a machine with access:\n"
            f"  huggingface-cli download {ESM_BASE_REPO} --local-dir {local}\n"
            f"then copy that directory across. It is NOT part of the AmyloDeep "
            f"model repository, which is why it is easy to miss."
        ) from exc

    if not encoder_is_present(local.parent):
        raise RuntimeError(
            f"Encoder download finished but {local} still has no config.json and "
            f"weights. Contents: "
            f"{sorted(f.name for f in local.glob('*'))[:12]}"
        )
    return local


def esm_base_source(
    model_dir: str | Path | None = None,
    *,
    allow_download: bool = True,
) -> str:
    """Where to load the frozen ESM2-650M encoder from.

    A copy inside the model directory always wins, so a run is reproducible from
    one path. When it is absent and downloading is permitted it is fetched THERE
    rather than into the Hugging Face cache, which is what makes ``--model-dir``
    mean what it says.
    """
    root = resolve_model_dir(model_dir)
    if encoder_is_present(root):
        return str(encoder_dir(root))
    if not allow_download:
        raise FileNotFoundError(
            f"The frozen ESM2 encoder is not at {encoder_dir(root)} and "
            f"downloading is disabled. Fetch it once with:\n"
            f"  amylodeep --model-dir {root} --download\n"
            f"or:\n"
            f"  huggingface-cli download {ESM_BASE_REPO} "
            f"--local-dir {encoder_dir(root)}\n"
            f"It is ~2.6 GB and is NOT part of the AmyloDeep model repository, "
            f"which is why it is easy to miss."
        )
    return str(download_encoder(root))


def download_all(
    model_dir: str | Path | None = None, *, quiet: bool = False
) -> Path:
    """Fetch everything a run needs: the eight artefacts and the encoder.

    The one call that makes a directory self-sufficient, so it can be populated
    once and then used with ``--no-download`` -- or copied to a machine with no
    network at all.
    """
    root = resolve_model_dir(model_dir)
    download(root, quiet=quiet)
    download_encoder(root, quiet=quiet)
    return root
