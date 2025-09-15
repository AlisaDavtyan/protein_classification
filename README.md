# AmyloDeep

**AmyloDeep: pLM-based ensemble model for predicting amyloid propensity from the amino acid sequence**

AmyloDeep is a Python package that uses ensemble model to predict amyloidogenic regions in protein sequences using a rolling window approach. 

## Features

- **Multi-model ensemble**: Combines 5 different models for robust predictions
- **Rolling window analysis**: Analyzes sequences using sliding windows of configurable size
- **Easy-to-use API**: Simple Python interface and command-line tool
- **Web interface**: Light version of tool is available at amylodeep.com

## Installation

### From PyPI (recommended)

```bash
pip install amylodeep
```

### From source

```bash
git clone https://github.com/AlisaDavtyan/protein_classification.git
cd amylodeep
pip install amylodeep
```

## Quick Start

```python
from amylodeep import predict_ensemble_rolling

# Predict amyloid propensity for a protein sequence
sequence = "MKTFFFLLLLFTIGFCYVQFSKLKLENLHFKDNSEGLKNGGLQRQLGLTLKFNSNSLHHTSNL"
window_size = 10
result = predict_ensemble_rolling(sequence, window_size = window_size)

print(f"Average probability: {result['avg_probability']:.4f}")
print(f"Maximum probability: {result['max_probability']:.4f}")

# Access position-wise probabilities
for position, probability in result['position_probs']:
    print(f"Position {position}: {probability:.4f}")
  
# Plot probabilities
import numpy as np
import matplotlib.pyplot as plt
positions, probs = zip(*result['position_probs'])
x = np.arange(0, len(sequence) - window_size + 1)
bar_colors = [
                (0, 0, 1, 0.8) if p > 0.8 else (0, 0, 1, 0.6) if p > 0.5 else (0, 0, 1, 0.2) for p in probs
            ]
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x, probs, color=bar_colors, width=1, edgecolor="black")
ax.set_ylabel("Probability", fontsize=12)
ax.set_xlabel("Residue", fontsize=12)
L = len(sequence)

ax.set_xlim(-1, L - window_size + 1)

if L < 100:
   ax.set_xticks(np.arange(0, L+1, 5))
else:
  step = int(np.ceil(L/5/10) * 10)
  tick_labels = np.arange(0, L+1, step)
  tick_positions = np.minimum(tick_labels, L - window_size)
  ax.set_xticks(tick_positions)
  ax.set_xticklabels([str(t) for t in tick_labels])

ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.7)
ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7)
ax.tick_params(axis='both', labelsize=12)
ax.set_title('Amyloidogenicity probability per window ', fontsize=16)

```

### Command Line Interface

```bash
# Basic prediction
amylodeep "MKTFFFLLLLFTIGFCYVQFSKLKLENLHFKDNSEGLKNGGLQRQLGLTLKFNSNSLHHTSNL"

# With custom window size
amylodeep "SEQUENCE" --window-size 10

# Save results to file
amylodeep "SEQUENCE" --output results.json --format json

# CSV output
amylodeep "SEQUENCE" --output results.csv --format csv
```

## Requirements

- torch>=1.12.0
- transformers>=4.30.0
- huggingface_hub>=0.14.0
- xgboost>=1.7.0
- numpy>=1.20 
- pandas>=1.3
- scikit-learn>=1.0
- jax-unirep>=2.0.0


### Main Functions

#### `predict_ensemble_rolling(sequence, window_size=10)`

Predict amyloid propensity for a protein sequence using rolling window analysis.

**Parameters:**
- `sequence` (str): Protein sequence (amino acid letters)
- `window_size` (int): Size of the rolling window (default: 10)

**Returns:**
Dictionary containing:
- `position_probs`: List of (position, probability) tuples
- `avg_probability`: Average probability across all windows
- `max_probability`: Maximum probability across all windows
- `sequence_length`: Length of the input sequence
- `num_windows`: Number of windows analyzed

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use AmyloDeep in your research, please cite:

```bibtex
@software{amylodeep2025,
  title={AmyloDeep: pLM-based ensemble model for predicting amyloid propensity from the amino acid sequence},
  author={Alisa Davtyan, Anahit Khachatryan, Rafayel Petrosyan},
  year={2025},
  url={https://github.com/AlisaDavtyan/protein_classification}
}
```

## Support

For questions and support:
- Open an issue on GitHub
- Contact: alisadavtyan7@gmail.com , xachatryan96an@gmail.com 
