"""A uniform container every Vedic sub-library emits into.

Two kinds of feature are supported, matching the two natural statistical tests:

* **categorical** -- an integer category per event (e.g. Moon's sign 0-11),
  tested with a chi-square goodness-of-fit against the null category
  distribution. ``-1`` marks "not applicable" for that event and is ignored.
* **flag** -- a boolean per event (e.g. "Mars is retrograde"), tested with a
  binomial test against the null success rate.

Each feature also records a ``family`` (``"sign"``, ``"aspect"`` ...), used to
group results and to let the battery include/exclude whole families.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class CatMeta:
    n: int
    names: List[str]
    family: str


@dataclass
class FeatureSet:
    """Categorical + boolean features for ``n`` events, keyed by feature name."""

    cat: Dict[str, np.ndarray] = field(default_factory=dict)
    cat_meta: Dict[str, CatMeta] = field(default_factory=dict)
    flag: Dict[str, np.ndarray] = field(default_factory=dict)
    flag_family: Dict[str, str] = field(default_factory=dict)

    def add_categorical(self, name: str, idx: np.ndarray, n: int,
                        names: List[str], family: str) -> None:
        self.cat[name] = np.asarray(idx, dtype=int)
        self.cat_meta[name] = CatMeta(n=n, names=list(names), family=family)

    def add_flag(self, name: str, values: np.ndarray, family: str) -> None:
        self.flag[name] = np.asarray(values, dtype=bool)
        self.flag_family[name] = family

    def merge(self, other: "FeatureSet") -> "FeatureSet":
        self.cat.update(other.cat)
        self.cat_meta.update(other.cat_meta)
        self.flag.update(other.flag)
        self.flag_family.update(other.flag_family)
        return self

    # -- summaries used by the battery -------------------------------------
    def categorical_counts(self, name: str) -> Tuple[np.ndarray, int]:
        """Return (counts-per-category, n_valid) for one categorical feature."""
        idx = self.cat[name]
        valid = idx[idx >= 0]
        meta = self.cat_meta[name]
        return np.bincount(valid, minlength=meta.n).astype(float), int(valid.size)

    def flag_count(self, name: str) -> Tuple[int, int]:
        """Return (n_true, n_total) for one boolean feature."""
        arr = self.flag[name]
        return int(arr.sum()), int(arr.size)

    @property
    def families(self) -> List[str]:
        fams = {m.family for m in self.cat_meta.values()} | set(self.flag_family.values())
        return sorted(fams)
