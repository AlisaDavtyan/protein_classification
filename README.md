# AmyloDeep

**AmyloDeep: pLM-based ensemble model for predicting amyloid propensity from the amino acid sequence**

AmyloDeep is a Python package that uses ensemble model to predict amyloidogenic regions in protein sequences using a rolling window approach. 

## Features

- **Multi-model ensemble**: Combines 5 different models for robust predictions
- **Per-residue resolution**: window scores projected onto individual residues,
  each row reporting the coverage depth and the min/mean/max/support/spread of the
  windows it came from, plus that window's own unprojected score
- **Rolling window analysis**: Analyzes sequences using sliding windows of configurable size
- **Configurable model directory**: read weights from any path, download into it, or run fully offline
- **Easy-to-use API**: Simple Python interface and command-line tool
- **Web interface**: Light version of tool is available at amylodeep.com

## Where the models live

Artefacts are read from, and downloaded into, one directory, resolved in this
order: `--model-dir`, then `$AMYLODEEP_MODEL_DIR`, then `~/.amylodeep_models`.

```bash
amylodeep --check                              # what is present, and where
amylodeep --model-dir /data/amylodeep --check
amylodeep --model-dir /data/amylodeep --no-download SEQUENCE   # fail rather than fetch
```

Two downloads are involved, and only one of them is this project's:

| what | source | size |
|---|---|---|
| the eight trained heads and calibrators | `AlisaDavtyan/amylodeep-models` | small |
| the frozen ESM2-650M encoder | `facebook/esm2_t33_650M_UR50D` | ~2.6 GB |

To pin the encoder or run offline, put a copy in the model directory as
`esm2_t33_650M_UR50D/` and it is used instead of the hub:

```bash
huggingface-cli download facebook/esm2_t33_650M_UR50D \
  --local-dir /data/amylodeep/esm2_t33_650M_UR50D
```

Importing the package downloads nothing; artefacts are fetched when models load.

## If the XGBoost head crashes the process

On macOS arm64 the run can die with a bare `Segmentation fault` inside
`libxgboost.dylib`, after three PyTorch models have loaded. It is not a corrupt
artefact and not a version rejection: the shipped `xgb_model.json` records
`{"version": [2, 1, 4]}`, and it happens with exactly that version installed.

The head is 10 trees over 1280 features with `binary:logistic`, so it can be
evaluated without the library at all:

```bash
amylodeep --xgboost-backend pure -i proteins.fasta -o out.csv
```

| backend | what it does |
|---|---|
| `native` (default) | `libxgboost` via `xgboost.Booster` |
| `pure` | the 10 trees evaluated in numpy; no native library loaded |
| `skip` | drops the head — the ensemble averages **4** models, not 5 |

`pure` re-implements another project's serialised format, so verify it once on any
machine where the native library does load:

```python
from amylodeep.xgb_pure import compare_with_native
print(compare_with_native("<model-dir>/xgb/xgb_model.json"))
# {'max_abs_difference': ..., 'agrees': True, ...}
```

Agreement to ~1e-6 is what a correct implementation gives, since the artefact
stores float32.

`skip` changes the ensemble and therefore every probability, so it is a diagnostic
rather than a mode to publish from. Every output row carries `heads_used`, so a
4-head run stays identifiable afterwards.

## Output: two grains, two tables

The model scores **windows**. A per-residue number therefore has to be aggregated
from the windows covering that residue, and the two are different grains of
evidence -- so they go in separate tables rather than into one file where half the
columns are blank on every row.

| `--resolution` | writes | one row per |
|---|---|---|
| `residue` (default) | `out.csv` | residue, carrying the window statistics behind it |
| `window` | `out.csv` | window, as version 0.3 emitted |
| `both` | `out.csv` **and** `out.windows.csv` | both of the above, separately |

```bash
amylodeep -i proteins.fasta -w 6 -a mean -o out.csv     # residue grain
amylodeep -i proteins.fasta -r window   -o windows.csv  # window grain
amylodeep -i proteins.fasta -r both     -o out.csv      # out.csv + out.windows.csv
```

Both tables repeat the run's identity on every row -- `sequence_id`,
`sequence_length`, `window_size`, `aggregate`, `heads_used`, `avg_probability`,
`max_probability` -- so either file is interpretable without the other and a
4-head run stays identifiable after the fact.

### The residue table

Each residue row carries the window evidence it was projected from, so a
residue-grain file is not silent about the windows underneath it:

| column | meaning |
|---|---|
| `residue_number`, `residue` | 1-based position and its amino acid |
| `probability` | the projected value, under the `aggregate` rule named in the same row |
| `coverage_depth` | how many windows covered this residue: 1 at the termini, up to `window_size` in the interior, so terminal values rest on a smaller sample |
| `window_min`, `window_max` | range over the covering windows |
| `window_mean`, `window_std` | their centre and spread. `window_std` near 0 means the covering windows agreed; a large one means the residue sits on a boundary |
| `window_support` | fraction of covering windows scoring above 0.5 |
| `window_at_start` | the model's own **unprojected** probability for the window *beginning* at this residue. Empty for the last `window_size - 1` residues, where no window can start |

Two things follow. `window_at_start` makes the projection auditable: the native
per-window numbers are recoverable from the residue table alone, so choosing the
default resolution discards nothing. And the spread columns carry what a point
estimate cannot -- a residue at `probability` 0.35 with `window_min` 0.05 and
`window_max` 0.95 is a boundary residue between a silent and a hot window, not a
moderately amyloidogenic one, and only the spread distinguishes the two.

### The window table

One row per window: `window_index`, `window_start_0based`, the 1-based
`first_residue` and `last_residue` it spans, the `window_sequence` itself, and the
model's `probability` for it. The residue bounds are the reason to prefer this over
the 0.3 output, which reported window starts only: window start 15 covers residues
16-21, an off-by-one better stated in the file than left to the reader.

### Aggregation, and why the window matters more

`-a/--aggregate` chooses the projection rule, and the choice moves where a region
begins:

| rule | reading | effect on boundaries |
|---|---|---|
| `mean` (default) | expected probability over covering windows | contracts onto the segment actually scored high |
| `max` | is the residue in any positive window | widens by up to `window_size - 1` residues |
| `support` | fraction of covering windows above 0.5 | sharpest, discards magnitude |

**The window matters more than the rule.** Projection places window evidence on a
residue axis; it cannot sharpen a wide window. On a synthetic 6-residue APR in a
42-residue sequence:

| window | `mean` | `max` |
|---|---|---|
| 6 | **16-21, exact** | 13-24 (2x too wide) |
| 10 | 14-23 (4 residues too wide) | 9-28 (3x too wide) |

The default window is therefore **6**, not 10. Pass `-w 10` to reproduce results
from version 0.3.

## Installation

### From PyPI (recommended)

```bash
pip install amylodeep
```

### From source

```bash
git clone https://github.com/AlisaDavtyan/protein_classification.git
cd protein_classification          # the repo root, where pyproject.toml is
python -m pip install -e .         # editable: your edits take effect immediately
```

`pip install amylodeep` resolves that name from PyPI wherever it is run, so it
installs the published release and ignores a local checkout entirely. If you have
edited the source and your changes seem to have no effect, that is why:

```bash
python -c "import amylodeep; print(amylodeep.__version__, amylodeep.__file__)"
```

If the path is inside `site-packages` rather than your checkout, the PyPI copy is
shadowing it. `python -m pip uninstall -y amylodeep`, then install with `-e .` as
above. Use `python -m pip` rather than bare `pip` so the install lands in the
interpreter you actually run.

## Quick Start

```python
from amylodeep import predict_ensemble_rolling

# Predict amyloid propensity for a protein sequence
sequence = "MKTFFFLLLLFTIGFCYVQFSKLKLENLHFKDNSEGLKNGGLQRQLGLTLKFNSNSLHHTSNL"
window_size = 10
result = predict_ensemble_rolling(sequence, window_size=window_size)
# result['residue_probs'] holds one probability per residue (index 0 == residue 1)
# result['coverage_depth'] holds how many windows covered each residue

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
# NOTE: this plots one bar per WINDOW, so the x axis is window starts, not
# residues. For a residue axis use result['residue_probs'], which has one
# value per residue and needs no offset correction.

```

### Command Line Interface

```bash
# Per-residue prediction (default: window 6, mean aggregation)
amylodeep "MKTFFFLLLLFTIGFCYVQFSKLKLENLHFKDNSEGLKNGGLQRQLGLTLKFNSNSLHHTSNL"

# A FASTA file, explicit window and aggregation, CSV out
amylodeep -i proteins.fasta -w 6 -a mean -o results.csv

# Window rows only, as version 0.3 emitted them
amylodeep "SEQUENCE" --resolution window -w 10 -o windows.csv

# Both grains: results.csv (one row per residue) + results.windows.csv (one per window)
amylodeep -i proteins.fasta --resolution both -o results.csv

# Where are the weights, and what is missing?
amylodeep --check
amylodeep --model-dir /data/amylodeep --check

# Offline: fail naming what is absent instead of downloading
amylodeep --model-dir /data/amylodeep --no-download -i proteins.fasta

# Name each artefact as it loads (use this if a run dies without a message)
amylodeep --verbose "SEQUENCE"
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

#### `predict_ensemble_rolling(sequence, window_size=6, *, aggregate="mean", model_dir=None, allow_download=True, verbose=False)`

Predict amyloid propensity for a protein sequence using rolling window analysis.

**Parameters:**
- `sequence` (str): Protein sequence (amino acid letters)
- `window_size` (int): Size of the rolling window (default: 6 — see "Per-residue
  output" above for why this is no longer 10)
- `aggregate` (str): `"mean"`, `"max"` or `"support"` — how overlapping windows
  become a per-residue value
- `model_dir` (str | Path | None): where artefacts are read from and downloaded to
- `allow_download` (bool): False raises instead of fetching anything
- `verbose` (bool): announce each artefact as it loads

**Returns:**
Dictionary containing:
- `residue_probs`: One probability per **residue**; index 0 is residue 1
- `coverage_depth`: How many windows covered each residue (lower at the termini,
  so those values rest on a smaller sample)
- `position_probs`: List of (**window start**, probability) tuples, 0-based
- `avg_probability`, `max_probability`: whole-sequence summaries
- `sequence_length`, `num_windows`, `window_size`, `aggregate`

#### `project_to_residues(position_probs, sequence_length, window_size, aggregate="mean")`

The window-to-residue projection on its own, importable without torch. Returns
`{"residue_probs": [...], "coverage_depth": [...]}`.

#### `window_statistics(position_probs, sequence_length, window_size)`

The window evidence behind each residue, without committing to an aggregation rule.
Torch-free. Returns parallel per-residue lists: `window_min`, `window_mean`,
`window_max`, `window_support`, `window_std`, `coverage_depth`, and
`window_at_start` (the unprojected probability of the window *starting* at that
residue, `None` for the last `window_size - 1` positions). `window_mean` is clamped
into `[window_min, window_max]`, so the ordering holds exactly and can be filtered
on. These are the `window_*` columns of the residue table.

#### `window_rows(position_probs, sequence, window_size)`

One dict per window -- `window_index`, `window_start_0based`, `first_residue`,
`last_residue` (both 1-based and inclusive), `window_sequence`, `probability` --
i.e. the window table, with the start-to-residue offset already resolved.
Torch-free.

#### `describe(model_dir=None)` / `is_downloaded(model_dir=None)` / `download(model_dir=None)`

Inventory, presence check and fetch for one model directory.

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
