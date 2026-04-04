#!/usr/bin/env python

import argparse
import os
import sys
from .utils import load_models_and_calibrators
from .ensemble_predictor import EnsembleRollingWindowPredictor

def parse_fasta(fasta_file):
    """
    Parse a FASTA file and return sequences with their IDs.
    """
    sequences = []
    current_id = None
    current_seq = ""

    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    sequences.append((current_id, current_seq.upper()))
                current_id = line[1:]  # Remove '>' character
                current_seq = ""
            else:
                current_seq += line

        # Final sequence
        if current_id is not None:
            sequences.append((current_id, current_seq.upper()))

    return sequences

def main():
    parser = argparse.ArgumentParser(
        description="Run amyloid propensity predictions on a sequence or FASTA file."
    )
    parser.add_argument(
        "sequence",
        nargs="?",
        default=None,
        help="Amino acid sequence string (positional, optional)."
    )
    parser.add_argument(
        "-i", "--input",
        default=None,
        help="Path to input FASTA file containing amino acid sequences."
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Path to output file (prints to stdout if not set)."
    )
    parser.add_argument(
        "-w", "--window-size",
        type=int,
        default=10,
        help="Rolling window size (default: 10)"
    )
    parser.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help="Output format: csv or json (default: csv)"
    )

    args = parser.parse_args()

    # Must provide either sequence string or FASTA file
    if not args.sequence and not args.input:
        print("Error: provide a sequence string or -i FASTA file.", file=sys.stderr)
        sys.exit(1)

    # Load models
    try:
        print("Loading models...", file=sys.stderr)
        from .utils import load_models_and_calibrators
        from .ensemble_predictor import EnsembleRollingWindowPredictor
        models, calibrators, tokenizer_1 = load_models_and_calibrators()
    except Exception as e:
        print(f"Error loading models: {e}", file=sys.stderr)
        sys.exit(1)

    predictor = EnsembleRollingWindowPredictor(
        models_dict=models,
        calibrators_dict=calibrators,
        tokenizer=tokenizer_1
    )

    # Build list of (id, sequence) tuples
    if args.sequence:
        sequences = [("input_sequence", args.sequence.upper())]
    else:
        if not os.path.exists(args.input):
            print(f"Error: Input file '{args.input}' not found.", file=sys.stderr)
            sys.exit(1)
        try:
            sequences = parse_fasta(args.input)
            if not sequences:
                print("Error: No sequences found in FASTA file.", file=sys.stderr)
                sys.exit(1)
        except Exception as e:
            print(f"Error parsing FASTA file: {e}", file=sys.stderr)
            sys.exit(1)

    # Run predictions
    results = []
    for i, (seq_id, sequence) in enumerate(sequences, 1):
        print(f"Processing sequence {i}/{len(sequences)}: {seq_id}", file=sys.stderr)

        if not sequence.replace('X', '').isalpha():
            print(f"Warning: Sequence {seq_id} contains invalid characters. Skipping.", file=sys.stderr)
            continue

        result = predictor.rolling_window_prediction(sequence, args.window_size)

        for position, probability in result['position_probs']:
            results.append({
                'sequence_id': seq_id,
                'position': position,
                'probability': round(float(probability), 4),
                'sequence_length': result['sequence_length'],
                'avg_probability': round(float(result['avg_probability']), 4),
                'max_probability': round(float(result['max_probability']), 4)
            })

    # Output results
    if args.format == "json":
        output_str = json.dumps(results, indent=2)
    else:
        import pandas as pd
        output_str = pd.DataFrame(results).to_csv(index=False)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_str)
        print(f"Results saved to {args.output}", file=sys.stderr)
    else:
        print(output_str)

# Allow running as `python -m amylodeep.cli`
if __name__ == "__main__":
    main()
