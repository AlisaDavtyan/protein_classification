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
    """
    CLI entry point for AmyloDeeP predictions.
    """
    parser = argparse.ArgumentParser(
        description="Run amyloid propensity predictions on a FASTA file."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to input FASTA file containing amino acid sequences."
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path to output CSV file for writing predictions."
    )
    parser.add_argument(
        "-w", "--window-size",
        type=int,
        default=6,
        help="Rolling window size (default: 6)"
    )

    args = parser.parse_args()


    # Check input file
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)

    # Load models, calibrators, and tokenizer
    try:
        print("Loading models and calibrators...", file=sys.stderr)
        models, calibrators, tokenizer_1 = load_models_and_calibrators()
    except Exception as e:
        print(f"Error loading models: {e}", file=sys.stderr)
        sys.exit(1)

    # Initialize ensemble predictor
    predictor = EnsembleRollingWindowPredictor(
        models_dict=models,
        calibrators_dict=calibrators,
        tokenizer=tokenizer_1
    )

    # Parse FASTA
    try:
        print("Parsing FASTA file...", file=sys.stderr)
        sequences = parse_fasta(args.input)
        if not sequences:
            print("Error: No sequences found in FASTA file.", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(sequences)} sequences.", file=sys.stderr)
    except Exception as e:
        print(f"Error parsing FASTA file: {e}", file=sys.stderr)
        sys.exit(1)

    # Run predictions
    results = []
    try:
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
                    'probability': probability,
                    'sequence_length': result['sequence_length'],
                    'avg_probability': result['avg_probability'],
                    'max_probability': result['max_probability']
                })

    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        sys.exit(1)

    # Write CSV
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"Predictions saved to {args.output}")
        print(f"Total predictions: {len(results)} position-wise results from {len(sequences)} sequences")
    except ImportError:
        print("pandas is required to write CSV outputs. Install via 'pip install pandas'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)

# Allow running as `python -m amylodeep.cli`
if __name__ == "__main__":
    main()
