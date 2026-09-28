#!/usr/bin/env python
"""AmyloDeep command line interface."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

from .projection import window_rows, window_statistics


def parse_fasta(fasta_file):
    """Parse a FASTA file and return [(id, sequence)]."""
    sequences = []
    current_id = None
    current_seq = ""

    with open(fasta_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    sequences.append((current_id, current_seq.upper()))
                current_id = line[1:]
                current_seq = ""
            else:
                current_seq += line
        if current_id is not None:
            sequences.append((current_id, current_seq.upper()))

    return sequences


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run amyloid propensity predictions on a sequence or FASTA file."
    )
    parser.add_argument(
        "sequence", nargs="?", default=None,
        help="Amino acid sequence string (positional, optional).",
    )
    parser.add_argument(
        "-i", "--input", default=None,
        help="Path to input FASTA file containing amino acid sequences.",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Path to output file (prints to stdout if not set).",
    )
    parser.add_argument(
        "-w", "--window-size", type=int, default=6,
        help=(
            "Rolling window size (default: 6). The window decides how sharply a "
            "score can be attributed to a residue: at 6 the projection recovers a "
            "6-residue APR exactly, at 10 it overshoots by about 4 residues. Use "
            "10 to reproduce results from version 0.3."
        ),
    )
    parser.add_argument(
        "--format", choices=["csv", "json"], default="csv",
        help="Output format: csv or json (default: csv)",
    )
    parser.add_argument(
        "-r", "--resolution", choices=["residue", "window", "both"], default="residue",
        help=(
            "residue: one row per residue (default). window: one row per window, "
            "as version 0.3 emitted. both: residue rows plus the window columns."
        ),
    )
    parser.add_argument(
        "-a", "--aggregate", choices=["mean", "max", "support"], default="mean",
        help=(
            "How overlapping windows become a per-residue value. mean (default) is "
            "the expected probability over covering windows; max is 'any covering "
            "window fired' and widens regions by up to window_size-1; support is "
            "the fraction of covering windows above 0.5."
        ),
    )
    parser.add_argument(
        "-m", "--model-dir", default=None,
        help=(
            "Directory holding the model and weight files. Defaults to "
            "$AMYLODEEP_MODEL_DIR, else ~/.amylodeep_models. Missing artefacts are "
            "downloaded here unless --no-download is given. Put a copy of "
            "facebook/esm2_t33_650M_UR50D in it as esm2_t33_650M_UR50D/ to pin the "
            "frozen encoder too."
        ),
    )
    parser.add_argument(
        "--no-download", action="store_true",
        help="Never fetch anything; fail naming what is missing and where.",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Report which artefacts are present in the model directory and exit.",
    )
    parser.add_argument(
        "--download", action="store_true",
        help=(
            "Fetch everything into the model directory and exit: the eight trained "
            "artefacts AND the ~2.6 GB frozen encoder. Do this once, then the "
            "directory can be used with --no-download, or copied to a machine with "
            "no network. Needs neither torch nor a working native stack."
        ),
    )
    parser.add_argument(
        "--xgboost-backend", choices=["native", "pure", "skip"], default="native",
        help=(
            "How to evaluate the XGBoost head. native (default) uses libxgboost. "
            "pure reads the JSON and evaluates the 10 trees in numpy, loading no "
            "native library -- use it when libxgboost crashes the process, and run "
            "amylodeep.xgb_pure.compare_with_native() once where it does load. "
            "skip drops the head: the ensemble then averages 4 models, not 5, and "
            "those probabilities are NOT comparable with a full run."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Announce each artefact before loading it, so a crash names its cause.",
    )
    return parser


def main():
    args = build_parser().parse_args()

    from .model_store import describe, download_all, resolve_model_dir

    if args.download and args.no_download:
        print(
            "Error: --download and --no-download contradict each other.",
            file=sys.stderr,
        )
        return 2

    # --download runs BEFORE --check, and neither one ends the run on its own.
    # An earlier version returned as soon as --check was seen, so
    # `--check --download` printed an all-MISSING inventory and exited without
    # fetching anything -- the flags read as a request and answered with a
    # refusal. They are a preparatory step and a report, in that order, and they
    # stop the program only when there is no sequence to work on.
    if args.download:
        root = resolve_model_dir(args.model_dir)
        print(f"Populating {root}", file=sys.stderr)
        try:
            download_all(args.model_dir)
        except Exception as exc:
            print(f"Download failed: {exc}", file=sys.stderr)
            print(f"\n{describe(args.model_dir)}", file=sys.stderr)
            return 1

    if args.check or args.download:
        # stdout when the inventory IS the output, stderr when a prediction
        # follows: without -o the results go to stdout, and an inventory printed
        # there would corrupt the CSV of anyone piping it.
        going_on = bool(args.sequence or args.input)
        print(describe(args.model_dir), file=sys.stderr if going_on else sys.stdout)
        if not going_on:
            return 0

    if not args.sequence and not args.input:
        print(
            "Error: provide a sequence string or -i FASTA file.\n"
            "       Use --check to see where the weights are, or --download to "
            "fetch them.",
            file=sys.stderr,
        )
        return 1

    # Loading comes FIRST, and the predictor class is imported after it. The
    # reverse order meant `from .ensemble_predictor import ...` -- which pulls in
    # torch and jax_unirep -- ran before any of the loader's diagnostics, so a
    # broken native stack surfaced as a bare ModuleNotFoundError traceback rather
    # than through the path that names the artefact, the directory and the
    # installed versions.
    try:
        print("Loading models...", file=sys.stderr)
        from .loading import load_models_and_calibrators

        models, calibrators, tokenizer_1 = load_models_and_calibrators(
            args.model_dir,
            allow_download=not args.no_download,
            verbose=args.verbose,
            xgboost_backend=args.xgboost_backend,
        )
        from .ensemble_predictor import EnsembleRollingWindowPredictor

        predictor = EnsembleRollingWindowPredictor(
            models_dict=models,
            calibrators_dict=calibrators,
            tokenizer=tokenizer_1,
            model_dir=args.model_dir,
            allow_download=not args.no_download,
        )
    except Exception as exc:
        print(f"Error loading models: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"\n{describe(args.model_dir)}", file=sys.stderr)
        return 1

    if args.sequence:
        sequences = [("input_sequence", args.sequence.upper())]
    else:
        if not os.path.exists(args.input):
            print(f"Error: Input file '{args.input}' not found.", file=sys.stderr)
            return 1
        try:
            sequences = parse_fasta(args.input)
        except Exception as exc:
            print(f"Error parsing FASTA file: {exc}", file=sys.stderr)
            return 1
        if not sequences:
            print("Error: No sequences found in FASTA file.", file=sys.stderr)
            return 1

    residue_rows: list[dict] = []
    window_table: list[dict] = []
    for i, (seq_id, sequence) in enumerate(sequences, 1):
        print(f"Processing sequence {i}/{len(sequences)}: {seq_id}", file=sys.stderr)

        if not sequence.replace("X", "").isalpha():
            print(
                f"Warning: Sequence {seq_id} contains invalid characters. Skipping.",
                file=sys.stderr,
            )
            continue

        result = predictor.rolling_window_prediction(
            sequence, args.window_size, aggregate=args.aggregate
        )
        common = {
            "sequence_id": seq_id,
            "sequence_length": result["sequence_length"],
            "window_size": result["window_size"],
            "aggregate": result["aggregate"],
            "heads_used": result["heads_used"],
            "avg_probability": round(float(result["avg_probability"]), 4),
            "max_probability": round(float(result["max_probability"]), 4),
        }

        if args.resolution in ("residue", "both"):
            # Residue rows carry the WINDOW statistics too, so a residue-grain
            # table is not silent about the window evidence behind it. The chosen
            # aggregation gives one number; min/max/support/std say whether the
            # windows covering that residue agreed, which the point estimate
            # cannot. window_at_start is the model's own unprojected output for
            # the window beginning there, so the native numbers are always
            # recoverable from this table alone.
            stats = window_statistics(
                result["position_probs"], result["sequence_length"], result["window_size"]
            )

            def _round(value):
                return None if value is None else round(float(value), 4)

            for index, probability in enumerate(result["residue_probs"], start=1):
                position = index - 1
                residue_rows.append(
                    {
                        **common,
                        "residue_number": index,
                        "residue": sequence[position],
                        "probability": round(float(probability), 4),
                        "coverage_depth": stats["coverage_depth"][position],
                        "window_min": _round(stats["window_min"][position]),
                        "window_mean": _round(stats["window_mean"][position]),
                        "window_max": _round(stats["window_max"][position]),
                        "window_support": _round(stats["window_support"][position]),
                        "window_std": _round(stats["window_std"][position]),
                        "window_at_start": _round(stats["window_at_start"][position]),
                    }
                )

        if args.resolution in ("window", "both"):
            for row in window_rows(
                result["position_probs"], sequence, result["window_size"]
            ):
                window_table.append(
                    {
                        **common,
                        **row,
                        "probability": round(float(row["probability"]), 4),
                    }
                )

    def _serialise(rows):
        if args.format == "json":
            return json.dumps(rows, indent=2)
        import pandas as pd

        return pd.DataFrame(rows).to_csv(index=False)

    # One GRAIN per table. The previous version appended residue rows and window
    # rows to a single list, so pandas unioned their columns: the window rows
    # landed underneath with every residue column blank and vice versa, and the
    # file read as residue-only unless you scrolled past 334 rows. Two grains are
    # two tables.
    tables = []
    if args.resolution in ("residue", "both"):
        tables.append(("residue", residue_rows))
    if args.resolution in ("window", "both"):
        tables.append(("window", window_table))

    if args.output:
        destination = pathlib.Path(args.output)
        for grain, rows in tables:
            if len(tables) == 1:
                target = destination
            else:
                # out.csv -> out.csv and out.windows.csv
                suffix = ".windows" if grain == "window" else ""
                target = destination.with_name(
                    f"{destination.stem}{suffix}{destination.suffix}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(_serialise(rows))
            print(
                f"{grain} table: {len(rows)} rows -> {target}", file=sys.stderr
            )
    else:
        for grain, rows in tables:
            if len(tables) > 1:
                print(f"# {grain} table ({len(rows)} rows)")
            print(_serialise(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
