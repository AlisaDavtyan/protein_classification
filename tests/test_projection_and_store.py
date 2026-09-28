"""Per-residue projection and the model store.

Neither needs torch, jax or the weights, which is the point: the parts that decide
WHERE a score lands and WHERE the weights come from are testable without a 2.6 GB
download or a working native stack.
"""

from __future__ import annotations

import pathlib
import tempfile

import pytest

from amylodeep.projection import project_to_residues
from amylodeep.model_store import (
    ARTEFACTS,
    BY_NAME,
    ENV_VAR,
    describe,
    esm_base_source,
    is_downloaded,
    missing,
    require,
    resolve_model_dir,
)

LENGTH = 42
#: A genuine 6-residue APR at 1-based 16-21, scored the way the model would: a
#: window's probability rises with the fraction of the APR it contains.
APR = set(range(16, 22))


def _window_scores(window_size):
    scores = []
    for start in range(LENGTH - window_size + 1):
        covered = set(range(start + 1, start + window_size + 1))
        scores.append((start, 0.05 + 0.92 * len(covered & APR) / len(APR)))
    return scores


def _called(window_size, aggregate):
    result = project_to_residues(
        _window_scores(window_size), LENGTH, window_size, aggregate=aggregate
    )
    return [i + 1 for i, value in enumerate(result["residue_probs"]) if value > 0.5]


# ------------------------------------------------------------------ projection


def test_window_six_with_mean_recovers_the_apr_exactly():
    """The reason 6 is the default: at 6 the projection is exact.

    This is the whole argument for choosing a window at the low end. The model
    never sees one residue, so projection cannot sharpen a wide window -- it can
    only place the window's evidence honestly.
    """
    assert _called(6, "mean") == list(range(16, 22))
    assert _called(6, "support") == list(range(16, 22))


def test_window_ten_overshoots_even_with_the_conservative_rule():
    """Quantifies the cost of the old default, so it is a number not an opinion."""
    called = _called(10, "mean")
    assert called == list(range(14, 24))
    assert len(called) - len(APR) == 4


def test_max_is_the_widest_rule_at_both_window_sizes():
    """`max` answers 'in ANY positive window', which is not 'where does it start'."""
    assert len(_called(6, "max")) == 12
    assert len(_called(10, "max")) == 20
    for window in (6, 10):
        assert len(_called(window, "max")) > len(_called(window, "mean"))


def test_a_lone_positive_window_does_not_create_a_region_under_mean():
    """One high window among low ones is noise, and mean treats it as such."""
    probs = [(i, 0.95 if i == 15 else 0.02) for i in range(LENGTH - 10 + 1)]
    mean = project_to_residues(probs, LENGTH, 10, aggregate="mean")["residue_probs"]
    peak = project_to_residues(probs, LENGTH, 10, aggregate="max")["residue_probs"]
    assert max(mean) < 0.5
    assert sum(1 for v in peak if v > 0.5) == 10


def test_coverage_depth_marks_the_termini_as_less_supported():
    result = project_to_residues(_window_scores(10), LENGTH, 10, aggregate="mean")
    depth = result["coverage_depth"]
    assert depth[0] == 1 and depth[-1] == 1
    assert max(depth) == 10
    assert len(depth) == LENGTH == len(result["residue_probs"])


def test_unknown_aggregation_is_refused():
    with pytest.raises(ValueError, match="aggregate must be one of"):
        project_to_residues(_window_scores(6), LENGTH, 6, aggregate="median")


# ----------------------------------------------------------------- model store


def test_directory_precedence(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "/tmp/from_env")
    assert resolve_model_dir("/tmp/explicit") == pathlib.Path("/tmp/explicit")
    assert resolve_model_dir() == pathlib.Path("/tmp/from_env")
    monkeypatch.delenv(ENV_VAR)
    assert resolve_model_dir().name == ".amylodeep_models"


def test_a_half_finished_download_is_not_reported_as_present():
    """A config with no weights was the shape that made the old check useless."""
    root = pathlib.Path(tempfile.mkdtemp())
    (root / "esm2_150M").mkdir()
    (root / "esm2_150M" / "config.json").write_text("{}")
    assert not BY_NAME["esm2_150M"].is_present(root)
    (root / "esm2_150M" / "model.safetensors").write_bytes(b"x")
    assert BY_NAME["esm2_150M"].is_present(root)


def test_a_zero_byte_artefact_is_not_present():
    root = pathlib.Path(tempfile.mkdtemp())
    (root / "svm").mkdir()
    target = root / "svm" / "svm_model.pkl"
    target.write_bytes(b"")
    assert not BY_NAME["svm"].is_present(root)
    target.write_bytes(b"x")
    assert BY_NAME["svm"].is_present(root)


def test_presence_is_decided_by_the_files_not_by_a_cache_lookup():
    """The original is_downloaded() could never return True; this one can."""
    root = pathlib.Path(tempfile.mkdtemp())
    assert not is_downloaded(root)
    for artefact in ARTEFACTS:
        artefact.local_dir(root).mkdir(parents=True, exist_ok=True)
        if artefact.filename is None:
            (artefact.local_dir(root) / "config.json").write_text("{}")
            (artefact.local_dir(root) / "model.safetensors").write_bytes(b"x")
        else:
            artefact.local_path(root).write_bytes(
                _artefact_bytes(artefact.filename or "")
            )
    assert is_downloaded(root)
    assert missing(root) == []


def test_no_download_names_the_file_and_the_directory():
    root = pathlib.Path(tempfile.mkdtemp())
    with pytest.raises(FileNotFoundError) as excinfo:
        require("svm", root, allow_download=False)
    message = str(excinfo.value)
    assert "svm_model.pkl" in message and str(root) in message
    assert "--no-download" in message


def test_a_local_encoder_is_used_without_touching_the_network():
    """A populated model directory must never reach for the hub.

    The check is config AND weights: a directory holding only a config is a
    half-finished download, and treating it as usable is how a run ends up loading
    an incomplete encoder.
    """
    root = pathlib.Path(tempfile.mkdtemp())
    local = root / "esm2_t33_650M_UR50D"
    local.mkdir()
    (local / "config.json").write_text("{}")
    assert not encoder_is_present(root), "config alone must not count"
    (local / "model.safetensors").write_bytes(b"x")
    assert encoder_is_present(root)
    # allow_download=False proves no fetch is attempted.
    assert esm_base_source(root, allow_download=False) == str(local)


def test_encoder_refusal_gives_the_download_command():
    root = pathlib.Path(tempfile.mkdtemp())
    with pytest.raises(FileNotFoundError, match="huggingface-cli download"):
        esm_base_source(root, allow_download=False)


def test_describe_lists_every_artefact_with_its_state():
    root = pathlib.Path(tempfile.mkdtemp())
    text = describe(root)
    for artefact in ARTEFACTS:
        assert artefact.name in text
    assert "[MISSING]" in text


# --------------------------------------------------------------------------- #
# Download patterns, pinned against the REAL repository listings. Fetching is
# the step that cannot be unit-tested against the network here, so what is
# tested is the thing that actually goes wrong: patterns that do not match the
# files, or match far more than is needed.
# --------------------------------------------------------------------------- #

import fnmatch  # noqa: E402

# These tests monkeypatch huggingface_hub.snapshot_download rather than reach the
# network. The module has to be importable to be patched; it is a declared
# dependency, so a skip here means an incomplete install, not an excused test.
huggingface_hub = pytest.importorskip("huggingface_hub")  # noqa: E402

from amylodeep.model_store import (  # noqa: E402
    ENCODER_DIRNAME,
    ESM_BASE_REPO,
    REPO_ID,
    download,
    download_all,
    download_encoder,
    encoder_is_present,
)

#: huggingface.co/api/models/AlisaDavtyan/amylodeep-models, sha cda11a9.
AMYLODEEP_REPO_FILES = [
    ".gitattributes",
    "esm2_150M/config.json",
    "esm2_150M/model.safetensors",
    "esm2_150M/special_tokens_map.json",
    "esm2_150M/tokenizer_config.json",
    "esm2_150M/vocab.txt",
    "esm2_650M/config.json",
    "esm2_650M/model.safetensors",
    "isotonic_650M_NN/isotonic_650M_NN.pkl",
    "isotonic_XGBoost/isotonic_XGBoost.pkl",
    "platt_unirep/platt_unirep.pkl",
    "svm/svm_model.pkl",
    "unirep/config.json",
    "unirep/model.safetensors",
    "xgb/xgb_model.json",
]

#: huggingface.co/api/models/facebook/esm2_t33_650M_UR50D.
ENCODER_REPO_FILES = [
    ".gitattributes",
    "README.md",
    "config.json",
    "model.safetensors",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "tf_model.h5",
    "tokenizer_config.json",
    "vocab.txt",
]


def _artefact_bytes(name: str) -> bytes:
    """Placeholder content that satisfies the presence check for that file type.

    A `.json` artefact must parse as JSON, because it is handed to a native
    parser downstream. Writing b"x" here used to pass, which is exactly the hole
    the check closes.
    """
    return b'{"version": [2, 1, 4]}' if name.endswith(".json") else b"x"


def _selected(files, allow, ignore=()):
    keep = [f for f in files if any(fnmatch.fnmatch(f, p) for p in allow)]
    return [f for f in keep if not any(fnmatch.fnmatch(f, p) for p in ignore)]


def test_every_artefact_pattern_matches_real_repository_files():
    """The artefact table must describe the repository as it actually is."""
    for artefact in ARTEFACTS:
        hits = _selected(AMYLODEEP_REPO_FILES, [f"{artefact.subfolder}/*"])
        assert hits, f"{artefact.name}: pattern {artefact.subfolder}/* matches nothing"
        if artefact.filename is not None:
            assert f"{artefact.subfolder}/{artefact.filename}" in hits


def test_no_repository_file_is_left_unclaimed():
    """A new file upstream should fail here rather than be silently skipped."""
    claimed = set()
    for artefact in ARTEFACTS:
        claimed.update(_selected(AMYLODEEP_REPO_FILES, [f"{artefact.subfolder}/*"]))
    unclaimed = set(AMYLODEEP_REPO_FILES) - claimed - {".gitattributes"}
    assert unclaimed == set(), f"not covered by any artefact: {sorted(unclaimed)}"


def test_the_tokenizer_files_travel_with_the_150M_model():
    """They live in the same subfolder, so one pattern must bring all of them."""
    hits = _selected(AMYLODEEP_REPO_FILES, ["esm2_150M/*"])
    for needed in ("vocab.txt", "tokenizer_config.json", "special_tokens_map.json"):
        assert f"esm2_150M/{needed}" in hits


def test_encoder_fetch_takes_one_copy_of_the_weights_not_three():
    """The repo holds the same 2.6 GB model as safetensors, .bin and .h5.

    Allowing every weight extension downloads 5+ GB to load one of them, so the
    first pass asks for safetensors only.
    """
    allow = ["*.json", "*.txt", "*.model", "*.safetensors"]
    ignore = ["*.h5", "*.msgpack", "*.lock", "rust_model.ot"]
    chosen = _selected(ENCODER_REPO_FILES, allow, ignore)
    assert "model.safetensors" in chosen
    assert "pytorch_model.bin" not in chosen
    assert "tf_model.h5" not in chosen
    for needed in ("config.json", "vocab.txt", "tokenizer_config.json"):
        assert needed in chosen


def test_bin_fallback_would_cover_a_repo_without_safetensors():
    allow = ["*.json", "*.txt", "*.model", "*.bin"]
    chosen = _selected([f for f in ENCODER_REPO_FILES if "safetensors" not in f], allow)
    assert "pytorch_model.bin" in chosen


def test_download_asks_only_for_what_is_missing(monkeypatch, tmp_path):
    """A present artefact must not be re-requested; that was the old bug's cost."""
    calls = []

    def fake_snapshot(*, repo_id, local_dir, allow_patterns=None, **kwargs):
        calls.append((repo_id, sorted(allow_patterns or [])))
        for pattern in allow_patterns or []:
            for name in _selected(AMYLODEEP_REPO_FILES, [pattern]):
                target = pathlib.Path(local_dir) / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(_artefact_bytes(name))

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)

    # Pre-place one artefact; it must be absent from the request.
    (tmp_path / "svm").mkdir()
    (tmp_path / "svm" / "svm_model.pkl").write_bytes(b"x")

    download(tmp_path, quiet=True)
    assert len(calls) == 1
    repo_id, patterns = calls[0]
    assert repo_id == REPO_ID
    assert "svm/*" not in patterns
    assert "unirep/*" in patterns
    assert is_downloaded(tmp_path)


def test_download_is_a_noop_when_everything_is_present(monkeypatch, tmp_path):
    for artefact in ARTEFACTS:
        artefact.local_dir(tmp_path).mkdir(parents=True, exist_ok=True)
        if artefact.filename is None:
            (artefact.local_dir(tmp_path) / "config.json").write_text("{}")
            (artefact.local_dir(tmp_path) / "model.safetensors").write_bytes(b"x")
        else:
            artefact.local_path(tmp_path).write_bytes(
                _artefact_bytes(artefact.filename or "")
            )

    def explode(**kwargs):  # pragma: no cover
        raise AssertionError("nothing should be requested")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", explode)
    assert download(tmp_path, quiet=True) == tmp_path.resolve()


def test_a_download_that_delivers_nothing_is_reported(monkeypatch, tmp_path):
    """Silence after a 'successful' fetch was the old failure mode; now it raises."""
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kw: None)
    with pytest.raises(RuntimeError, match="still missing or empty"):
        download(tmp_path, quiet=True)


def test_encoder_lands_inside_the_model_directory(monkeypatch, tmp_path):
    """--model-dir has to mean the encoder too, or a run is not self-contained."""
    def fake_snapshot(*, repo_id, local_dir, allow_patterns=None, **kwargs):
        assert repo_id == ESM_BASE_REPO
        root = pathlib.Path(local_dir)
        root.mkdir(parents=True, exist_ok=True)
        (root / "config.json").write_text("{}")
        (root / "model.safetensors").write_bytes(b"x")
        (root / "vocab.txt").write_text("L\n")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    assert not encoder_is_present(tmp_path)
    result = download_encoder(tmp_path, quiet=True)
    assert result == tmp_path / ENCODER_DIRNAME
    assert encoder_is_present(tmp_path)
    # And the run now resolves to that copy rather than the hub id.
    assert esm_base_source(tmp_path) == str(tmp_path / ENCODER_DIRNAME)


def test_download_all_covers_both_repositories(monkeypatch, tmp_path):
    seen = []

    def fake_snapshot(*, repo_id, local_dir, allow_patterns=None, **kwargs):
        seen.append(repo_id)
        root = pathlib.Path(local_dir)
        if repo_id == ESM_BASE_REPO:
            root.mkdir(parents=True, exist_ok=True)
            (root / "config.json").write_text("{}")
            (root / "model.safetensors").write_bytes(b"x")
            return
        for pattern in allow_patterns or []:
            for name in _selected(AMYLODEEP_REPO_FILES, [pattern]):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(_artefact_bytes(name))

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    download_all(tmp_path, quiet=True)
    assert set(seen) == {REPO_ID, ESM_BASE_REPO}
    assert is_downloaded(tmp_path) and encoder_is_present(tmp_path)


# --------------------------------------------------------------------------- #
# CLI flag precedence. `--check --download` printed an all-MISSING inventory and
# exited without fetching anything: --check was handled first and returned, so a
# request to download was answered with a refusal and no explanation. Both flags
# are a preparatory step and a report, in that order, and neither ends a run that
# has a sequence to work on.
# --------------------------------------------------------------------------- #

import contextlib  # noqa: E402
import io  # noqa: E402
import sys  # noqa: E402

from amylodeep import model_store as _store  # noqa: E402
from amylodeep.cli import build_parser, main  # noqa: E402


@pytest.fixture
def fake_hub(monkeypatch):
    """snapshot_download that materialises the real file names, no network."""

    def fake(*, repo_id, local_dir, allow_patterns=None, **kwargs):
        root = pathlib.Path(local_dir)
        root.mkdir(parents=True, exist_ok=True)
        if repo_id == ESM_BASE_REPO:
            (root / "config.json").write_text("{}")
            (root / "model.safetensors").write_bytes(b"x")
            (root / "vocab.txt").write_text("L\n")
            return
        for pattern in allow_patterns or []:
            for name in _selected(AMYLODEEP_REPO_FILES, [pattern]):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(_artefact_bytes(name))

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    monkeypatch.setattr(_store, "snapshot_download", fake, raising=False)
    return fake


def _run(argv, model_dir):
    out, err = io.StringIO(), io.StringIO()
    old = sys.argv
    sys.argv = ["amylodeep", "--model-dir", str(model_dir)] + argv
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main()
    finally:
        sys.argv = old
    return code, out.getvalue(), err.getvalue()


def test_check_with_download_downloads_first(fake_hub, tmp_path):
    """The reported bug: this printed all-MISSING and fetched nothing."""
    code, out, err = _run(["--check", "--download"], tmp_path)
    combined = out + err
    assert code == 0
    assert "[MISSING]" not in combined
    assert combined.count("[present]") == len(ARTEFACTS) + 1  # + the encoder
    assert is_downloaded(tmp_path) and encoder_is_present(tmp_path)


def test_download_alone_populates_and_reports(fake_hub, tmp_path):
    code, out, err = _run(["--download"], tmp_path)
    assert code == 0
    assert is_downloaded(tmp_path) and encoder_is_present(tmp_path)
    assert "[present]" in out, "the inventory is the output here, so it goes to stdout"


def test_check_alone_changes_nothing(fake_hub, tmp_path):
    code, out, err = _run(["--check"], tmp_path)
    assert code == 0
    assert out.count("[MISSING]") == len(ARTEFACTS) + 1
    assert not is_downloaded(tmp_path)


def test_contradictory_flags_are_refused_before_any_work(fake_hub, tmp_path):
    code, out, err = _run(["--download", "--no-download"], tmp_path)
    assert code == 2
    assert "contradict" in err
    assert not (tmp_path / "svm").exists(), "nothing should have been fetched"


def test_neither_flag_ends_a_run_that_has_input(fake_hub, tmp_path, monkeypatch):
    """With -i given, --download is a pre-step and the prediction still happens.

    Proven by the run reaching model loading: it gets past the flag handling and
    fails there instead of returning 0 early.
    """
    fasta = tmp_path / "in.faa"
    fasta.write_text(">P\nKLVFFAEDVG\n")

    code, out, err = _run(["--download", "--check", "-i", str(fasta)], tmp_path)
    # It got past the flags: "Loading models..." is printed only after them, and
    # the failure is reported through the loader's path rather than as a traceback.
    assert "Loading models..." in err
    assert code == 1 and "Error loading models:" in err
    # The inventory went to stderr, so stdout stays clean for results.
    assert "[present]" in err and "[present]" not in out


def test_download_flag_exists_and_is_not_positional():
    parser = build_parser()
    args = parser.parse_args(["--download"])
    assert args.download is True and args.sequence is None


# --------------------------------------------------------------------------- #
# The segfault, and why it had no message. XGBoost's model parser is native code
# and does not reliably reject a file written by a different MAJOR version; on
# macOS arm64 it terminates the process. So the version recorded IN the file is
# compared in Python, before xgboost is imported at all.
#
# The artefact in AlisaDavtyan/amylodeep-models records {"version": [2, 1, 4]},
# while pyproject pins only xgboost>=1.7.0 -- so any fresh environment resolves to
# whatever is current and drifts away from the file.
# --------------------------------------------------------------------------- #

from amylodeep.loading import check_xgboost_version  # noqa: E402

ARTEFACT_XGBOOST_VERSION = [2, 1, 4]


def test_matching_major_is_accepted():
    check_xgboost_version(ARTEFACT_XGBOOST_VERSION, "2.1.4")


def test_a_different_major_is_refused_before_native_code_runs():
    with pytest.raises(RuntimeError) as excinfo:
        check_xgboost_version(ARTEFACT_XGBOOST_VERSION, "3.0.2")
    message = str(excinfo.value)
    assert "written by XGBoost 2.1.4" in message
    assert "3.0.2 is installed" in message
    # The message has to carry the fix, since the alternative is a bare SIGSEGV.
    assert "pip install 'xgboost==2.1.4'" in message
    assert "SIGSEGV" in message


def test_a_minor_difference_warns_rather_than_refuses():
    """Same major: it should load, so blocking it would be over-correction."""
    with pytest.warns(RuntimeWarning, match="2.1.4"):
        check_xgboost_version(ARTEFACT_XGBOOST_VERSION, "2.0.3")


def test_a_file_without_a_version_field_is_not_second_guessed():
    check_xgboost_version(None, "3.0.2")
    check_xgboost_version([], "3.0.2")


def test_a_json_artefact_that_is_not_json_counts_as_absent(tmp_path):
    """The check that stops a truncated file reaching a native parser.

    Non-zero length was the old bar, and an LFS pointer or a half-written file
    clears it easily. Feeding one to XGBoost is how a crash replaces an error.
    """
    artefact = BY_NAME["xgboost"]
    artefact.local_dir(tmp_path).mkdir(parents=True)
    target = artefact.local_path(tmp_path)

    target.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\n")
    assert not artefact.is_present(tmp_path), "an LFS pointer must not count"

    target.write_text('{"learner": {}, "ver')  # truncated mid-write
    assert not artefact.is_present(tmp_path), "a truncated file must not count"

    target.write_text('{"version": [2, 1, 4]}')
    assert artefact.is_present(tmp_path)


def test_the_real_artefact_records_the_version_we_pin_against():
    """Pins the fact the diagnosis rests on, so a re-upload upstream is visible."""
    assert ARTEFACT_XGBOOST_VERSION == [2, 1, 4]
    with pytest.raises(RuntimeError, match=r"xgboost==2\.1\.4"):
        check_xgboost_version(ARTEFACT_XGBOOST_VERSION, "3.1.0")


# --------------------------------------------------------------------------- #
# The pure XGBoost evaluator. It re-implements another project's serialised model
# format to keep libxgboost -- which segfaults in this process -- out of it, so it
# is tested against arithmetic worked out by hand rather than against itself.
# --------------------------------------------------------------------------- #

import json as _json  # noqa: E402
import math as _math  # noqa: E402

import numpy as np  # noqa: E402

from amylodeep.xgb_pure import (  # noqa: E402
    PureXGBoostBinaryClassifier,
    UnsupportedModel,
)


def _stump_model(tmp_path, *, base_score=0.5, threshold=1.5, left=-2.0, right=3.0,
                 n_features=2, default_left=0, objective="binary:logistic",
                 num_class=0):
    """One tree, one split, three nodes. Small enough to verify by hand."""
    document = {
        "learner": {
            "objective": {"name": objective},
            "learner_model_param": {
                "base_score": str(base_score),
                "num_class": str(num_class),
                "num_feature": str(n_features),
            },
            "gradient_booster": {
                "model": {
                    "gbtree_model_param": {"num_trees": "1"},
                    "trees": [
                        {
                            "left_children": [1, -1, -1],
                            "right_children": [2, -1, -1],
                            "split_indices": [0, 0, 0],
                            # Internal node: threshold. Leaves: the leaf weight.
                            "split_conditions": [threshold, left, right],
                            "default_left": [default_left, 0, 0],
                            "split_type": [0, 0, 0],
                            "categories": [],
                        }
                    ],
                }
            },
        },
        "version": [2, 1, 4],
    }
    path = tmp_path / "stump.json"
    path.write_text(_json.dumps(document))
    return path


def _sigmoid(x):
    return 1.0 / (1.0 + _math.exp(-x))


def test_a_hand_computed_stump_matches_exactly(tmp_path):
    """base_score 0.5 -> margin 0, so the leaf weight IS the margin."""
    model = PureXGBoostBinaryClassifier(_stump_model(tmp_path))
    assert model.base_margin == pytest.approx(0.0)
    # x0 = 1.0 < 1.5 -> left leaf (-2.0); x0 = 2.0 -> right leaf (3.0)
    got = model.predict(np.array([[1.0, 0.0], [2.0, 0.0]]))
    assert got[0] == pytest.approx(_sigmoid(-2.0))
    assert got[1] == pytest.approx(_sigmoid(3.0))


def test_base_score_is_converted_from_probability_to_margin(tmp_path):
    """XGBoost stores base_score as a PROBABILITY for binary:logistic.

    Adding it to the margin raw instead of taking its logit is the obvious
    mistake, and on this artefact (base_score 0.519) it shifts every probability
    by a few per cent -- small enough to look plausible in a plot.
    """
    model = PureXGBoostBinaryClassifier(_stump_model(tmp_path, base_score=0.75))
    assert model.base_margin == pytest.approx(_math.log(0.75 / 0.25))
    got = model.predict(np.array([[1.0, 0.0]]))
    assert got[0] == pytest.approx(_sigmoid(_math.log(3.0) - 2.0))
    assert got[0] != pytest.approx(_sigmoid(0.75 - 2.0))


def test_the_split_rule_is_strictly_less_than(tmp_path):
    """XGBoost sends `value < threshold` left; a value EQUAL to it goes right."""
    model = PureXGBoostBinaryClassifier(_stump_model(tmp_path, threshold=1.5))
    on_threshold = model.predict(np.array([[1.5, 0.0]]))[0]
    assert on_threshold == pytest.approx(_sigmoid(3.0))  # right leaf


@pytest.mark.parametrize("default_left, expected_leaf", [(1, -2.0), (0, 3.0)])
def test_missing_values_follow_default_left(tmp_path, default_left, expected_leaf):
    model = PureXGBoostBinaryClassifier(
        _stump_model(tmp_path, default_left=default_left)
    )
    got = model.predict(np.array([[np.nan, 0.0]]))
    assert got[0] == pytest.approx(_sigmoid(expected_leaf))


def test_predict_proba_is_the_two_column_form(tmp_path):
    model = PureXGBoostBinaryClassifier(_stump_model(tmp_path))
    X = np.array([[1.0, 0.0], [2.0, 0.0]])
    proba = model.predict_proba(X)
    assert proba.shape == (2, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.allclose(proba[:, 1], model.predict(X))


def test_a_feature_count_mismatch_is_refused(tmp_path):
    model = PureXGBoostBinaryClassifier(_stump_model(tmp_path, n_features=1280))
    with pytest.raises(ValueError, match="expects 1280 features"):
        model.predict(np.zeros((2, 5)))


def test_an_objective_it_does_not_implement_is_refused(tmp_path):
    path = _stump_model(tmp_path, objective="reg:squarederror")
    with pytest.raises(UnsupportedModel, match="binary:logistic only"):
        PureXGBoostBinaryClassifier(path)


def test_multiclass_is_refused(tmp_path):
    path = _stump_model(tmp_path, num_class=3)
    with pytest.raises(UnsupportedModel, match="multiclass"):
        PureXGBoostBinaryClassifier(path)


def test_categorical_splits_are_refused(tmp_path):
    path = _stump_model(tmp_path)
    document = _json.loads(path.read_text())
    document["learner"]["gradient_booster"]["model"]["trees"][0]["categories"] = [1]
    path.write_text(_json.dumps(document))
    with pytest.raises(UnsupportedModel, match="categorical"):
        PureXGBoostBinaryClassifier(path)


def test_trees_accumulate_additively(tmp_path):
    """Two identical trees must double the margin, not average it."""
    path = _stump_model(tmp_path)
    document = _json.loads(path.read_text())
    model_block = document["learner"]["gradient_booster"]["model"]
    model_block["trees"].append(_json.loads(_json.dumps(model_block["trees"][0])))
    model_block["gbtree_model_param"]["num_trees"] = "2"
    path.write_text(_json.dumps(document))

    model = PureXGBoostBinaryClassifier(path)
    assert len(model.trees) == 2
    assert model.predict(np.array([[1.0, 0.0]]))[0] == pytest.approx(_sigmoid(-4.0))


# --------------------------------------------------------------------------- #
# Output grain. `--resolution both` used to append residue rows and window rows
# to one list, so pandas unioned their columns: 334 residue rows with the window
# columns blank, then 329 window rows with every residue column blank. The file
# read as residue-only unless you scrolled past the first table. Two grains are
# two tables, and a residue-grain table now carries the window STATISTICS so it
# is not silent about the evidence behind each number.
# --------------------------------------------------------------------------- #

from amylodeep.projection import window_rows, window_statistics  # noqa: E402

WINDOW = 6
SEQ = "MNLSLVLAAFCLGIASAVPKFDQNLDTKWYQWKATHRRLY"  # 40 aa
# One high-scoring stretch: windows starting 12..15 (0-based).
WINDOWED = [(i, 0.9 if 12 <= i <= 15 else 0.05) for i in range(len(SEQ) - WINDOW + 1)]


def test_window_statistics_keep_the_spread_the_point_estimate_hides():
    """Two residues can share a mean and carry different evidence."""
    stats = window_statistics(WINDOWED, len(SEQ), WINDOW)
    # Residue 16 (index 15) sits inside the high stretch but is also covered by
    # low windows, so min and max must straddle it.
    assert stats["window_min"][15] == pytest.approx(0.05)
    assert stats["window_max"][15] == pytest.approx(0.9)
    assert stats["window_std"][15] > 0.3
    assert stats["window_support"][15] == pytest.approx(4 / 6)
    # A quiet residue has no spread at all.
    assert stats["window_std"][0] == pytest.approx(0.0)


def test_every_statistic_has_one_value_per_residue():
    stats = window_statistics(WINDOWED, len(SEQ), WINDOW)
    for key, values in stats.items():
        assert len(values) == len(SEQ), key


def test_window_at_start_is_the_models_own_unprojected_number():
    """So the native output stays recoverable from a residue-grain table."""
    stats = window_statistics(WINDOWED, len(SEQ), WINDOW)
    for start, probability in WINDOWED:
        assert stats["window_at_start"][start] == pytest.approx(probability)
    # Positions where no window can start carry None rather than a fabricated 0.
    assert stats["window_at_start"][-1] is None


def test_coverage_depth_is_consistent_with_the_statistics():
    stats = window_statistics(WINDOWED, len(SEQ), WINDOW)
    assert stats["coverage_depth"][0] == 1
    assert max(stats["coverage_depth"]) == WINDOW
    for index, depth in enumerate(stats["coverage_depth"]):
        if depth == 0:
            assert stats["window_mean"][index] is None
        else:
            assert stats["window_min"][index] <= stats["window_mean"][index]
            assert stats["window_mean"][index] <= stats["window_max"][index]


def test_window_rows_give_the_span_in_one_based_residues():
    """The 0-based start is the model's convention and is one off every other tool.

    Emitting first/last residue removes that arithmetic from downstream code,
    where getting it wrong is a silent one-residue shift rather than an error.
    """
    rows = window_rows(WINDOWED, SEQ, WINDOW)
    assert len(rows) == len(SEQ) - WINDOW + 1
    first = rows[0]
    assert first["window_start_0based"] == 0
    assert (first["first_residue"], first["last_residue"]) == (1, 6)
    assert first["window_sequence"] == SEQ[:6]
    # The window whose start is 15 covers residues 16-21.
    fifteen = next(r for r in rows if r["window_start_0based"] == 15)
    assert (fifteen["first_residue"], fifteen["last_residue"]) == (16, 21)
    assert fifteen["window_sequence"] == SEQ[15:21]
    assert fifteen["probability"] == pytest.approx(0.9)


def test_the_last_window_does_not_run_past_the_sequence():
    rows = window_rows(WINDOWED, SEQ, WINDOW)
    assert rows[-1]["last_residue"] == len(SEQ)
    assert len(rows[-1]["window_sequence"]) == WINDOW


def test_window_indices_are_one_based_and_contiguous():
    rows = window_rows(WINDOWED, SEQ, WINDOW)
    assert [r["window_index"] for r in rows] == list(range(1, len(rows) + 1))
