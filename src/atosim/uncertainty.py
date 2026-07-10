"""Aggregate results across seeds (and models) into mean +/- spread.

Stability is an acceptance criterion: an estimate whose spread exceeds the
tolerance is flagged ``low_confidence`` rather than reported as a precise number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Estimate:
    mean: float
    std: float
    n: int
    values: list[float] = field(default_factory=list)
    low_confidence: bool = False

    def as_dict(self) -> dict:
        return {
            "mean": self.mean, "std": self.std, "n": self.n,
            "values": self.values, "low_confidence": self.low_confidence,
        }

    def __str__(self) -> str:
        flag = "  [LOW CONFIDENCE]" if self.low_confidence else ""
        return f"{self.mean:.3f} +/- {self.std:.3f} eV (n={self.n}){flag}"


def aggregate(values: list[float], spread_tol: float) -> Estimate:
    """Mean/std of samples; flag low confidence when std exceeds ``spread_tol``."""
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    if arr.size == 0:
        return Estimate(mean=float("nan"), std=float("nan"), n=0, values=[],
                        low_confidence=True)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return Estimate(
        mean=mean, std=std, n=int(arr.size),
        values=[float(v) for v in arr],
        low_confidence=(std > spread_tol) or arr.size < 2,
    )


def rankings_consistent(per_seed_orderings: list[list[str]]) -> bool:
    """True if every seed produced the same ordering of steps/pathways."""
    if not per_seed_orderings:
        return False
    return all(o == per_seed_orderings[0] for o in per_seed_orderings)
