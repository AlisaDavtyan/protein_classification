"""Turn window scores into per-residue scores.

Deliberately free of torch, jax, transformers and the weights. This module holds
the step that decides WHERE a score lands on the sequence, which is a question
about indices and aggregation rather than about models -- so it can be read,
reasoned about and tested on any machine, including one where the native stack
does not load. That separation is why the projection has tests and the ensemble
does not.
"""

from __future__ import annotations

def project_to_residues(position_probs, sequence_length, window_size, aggregate="mean"):
    """Turn window scores into one value per residue.

    AmyloDeep scores windows, so a residue's value has to come from the windows
    that contain it. Residue *i* (0-based) is covered by the windows starting at
    ``max(0, i - window_size + 1) .. min(i, sequence_length - window_size)``, so
    interior residues are covered by ``window_size`` windows and the first and
    last ``window_size - 1`` by fewer. ``coverage_depth`` reports that, because a
    terminal residue's value is an average over a smaller sample and is not
    directly comparable with one from the middle of the chain.

    **This does not create resolution the model does not have.** The network never
    sees a single residue; it sees a window. Projection redistributes the evidence
    a window carries, it does not sharpen it. What genuinely narrows attribution
    is a smaller window: at ``window_size=6`` each score is attributable to six
    residues rather than ten, which is why 5-6 is the range the amyloid predictors
    that do resolve residues were built around. Treat the projection as the honest
    way to place window evidence on a residue axis, and the window size as the
    thing that decides how sharp that placement can be.

    Aggregation rules, and what each one does to a boundary:

    ``mean`` (default)
        The expected probability over the windows containing the residue. The
        heads are individually calibrated before averaging, so a mean of
        probabilities stays a probability. Boundaries contract toward the segment
        the model actually scored highly, because a residue on the edge of one
        positive window is diluted by the negative windows that also contain it.
    ``max``
        The highest-scoring window containing the residue. One positive window
        paints all of its residues, so a region can exceed the evidence by up to
        ``window_size - 1`` residues -- 9 at the default window of 10. Use it to
        answer "is this residue in any amyloidogenic window", not "where does the
        region begin".
    ``support``
        The FRACTION of covering windows scoring above 0.5, i.e. how consistently
        the windows containing this residue agree. Sharpest boundaries of the
        three and the easiest to read against a consensus panel, at the cost of
        discarding the magnitude.

    Returns ``{'residue_probs': [...], 'coverage_depth': [...]}``, both of length
    ``sequence_length``.
    """
    valid = {"mean", "max", "support"}
    if aggregate not in valid:
        raise ValueError(f"aggregate must be one of {sorted(valid)}, got {aggregate!r}")

    covering = [[] for _ in range(sequence_length)]
    for start, prob in position_probs:
        for offset in range(window_size):
            index = int(start) + offset
            if 0 <= index < sequence_length:
                covering[index].append(float(prob))

    residue_probs = []
    for values in covering:
        if not values:
            # Only reachable if the windows do not span the sequence; recorded as
            # 0.0 with depth 0 rather than silently interpolated.
            residue_probs.append(0.0)
        elif aggregate == "max":
            residue_probs.append(max(values))
        elif aggregate == "support":
            residue_probs.append(sum(1 for v in values if v > 0.5) / len(values))
        else:
            residue_probs.append(sum(values) / len(values))

    return {
        "residue_probs": residue_probs,
        "coverage_depth": [len(values) for values in covering],
    }

def window_statistics(position_probs, sequence_length, window_size):
    """Per-residue summary of the windows covering each residue.

    The chosen aggregation collapses those windows to one number, which throws
    away the thing most worth seeing: whether the windows covering a residue
    AGREE. A residue whose covering windows all score 0.8 and one whose windows
    range 0.1-0.9 get the same mean, and they are not the same evidence.

    So every residue row carries the spread as well as the point estimate --
    min, mean, max, the support fraction and the standard deviation -- and
    ``window_at_start``, the raw probability of the window that BEGINS at this
    residue. That last one is the model's own unprojected output, aligned to the
    position it was computed for, so a reader can always recover the native
    numbers from a residue-grain table.

    Returns a dict of lists, each of length ``sequence_length``.
    """
    covering = [[] for _ in range(sequence_length)]
    at_start = [None] * sequence_length
    for start, prob in position_probs:
        start = int(start)
        value = float(prob)
        if 0 <= start < sequence_length:
            at_start[start] = value
        for offset in range(window_size):
            index = start + offset
            if 0 <= index < sequence_length:
                covering[index].append(value)

    def _std(values, mean):
        if len(values) < 2:
            return 0.0
        return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5

    stats = {
        "window_min": [], "window_mean": [], "window_max": [],
        "window_support": [], "window_std": [], "coverage_depth": [],
        "window_at_start": at_start,
    }
    for values in covering:
        if not values:
            stats["window_min"].append(None)
            stats["window_mean"].append(None)
            stats["window_max"].append(None)
            stats["window_support"].append(None)
            stats["window_std"].append(None)
            stats["coverage_depth"].append(0)
            continue
        low, high = min(values), max(values)
        # Clamped into [min, max]. Summation error can put the mean of six equal
        # values 1e-17 above their maximum, and `window_mean <= window_max` is an
        # invariant a reader may filter on -- worth making exactly true rather
        # than almost true.
        mean = min(max(sum(values) / len(values), low), high)
        stats["window_min"].append(low)
        stats["window_mean"].append(mean)
        stats["window_max"].append(high)
        stats["window_support"].append(sum(1 for v in values if v > 0.5) / len(values))
        stats["window_std"].append(_std(values, mean))
        stats["coverage_depth"].append(len(values))
    return stats


def window_rows(position_probs, sequence, window_size):
    """One record per window, with the span it covers in 1-based residue numbers.

    ``position_probs`` carries 0-based window STARTS, which is the model's own
    convention and is one residue off every other predictor in a panel. Emitting
    the 1-based first and last residue alongside it removes the need for anyone
    downstream to redo that arithmetic -- and getting it wrong is a silent
    one-residue shift, not an error.
    """
    rows = []
    for start, prob in position_probs:
        start = int(start)
        first = start + 1
        last = min(start + window_size, len(sequence))
        rows.append(
            {
                "window_index": len(rows) + 1,
                "window_start_0based": start,
                "first_residue": first,
                "last_residue": last,
                "window_sequence": sequence[start : start + window_size],
                "probability": float(prob),
            }
        )
    return rows
