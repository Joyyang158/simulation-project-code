"""Ch. 2/6 replication-level t intervals; Ch. 7 paired policy differences."""
import numpy as np
from scipy.stats import t


def confidence_interval(values, confidence=.95):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("At least two finite independent replication metrics required")
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    half = float(t.ppf((1 + confidence) / 2, len(values) - 1) * sd / np.sqrt(len(values)))
    return dict(n=len(values), mean=mean, sd=sd, ci_low=mean - half, ci_high=mean + half)


def paired_difference(baseline, alternative):
    if len(baseline) != len(alternative):
        raise ValueError("Paired replications must have equal lengths")
    # Negative = the alternative has a lower metric (good for waiting).
    return confidence_interval(np.asarray(alternative) - np.asarray(baseline))
