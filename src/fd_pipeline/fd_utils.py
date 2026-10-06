"""Shared helpers for FD discovery: encoding, attribute bitmasks, FD checks."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def encode(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Integer-encode every column and drop duplicate rows.

    Duplicate rows never change which FDs hold, and removing them makes
    "X is a key" equivalent to "X has as many classes as there are rows".
    """
    codes = np.empty((len(df), df.shape[1]), dtype=np.int32)
    for j, col in enumerate(df.columns):
        codes[:, j] = pd.factorize(df[col].astype(str), sort=False)[0]
    codes = np.unique(codes, axis=0) if len(codes) else codes
    return codes, [str(c) for c in df.columns]


def bits(mask: int) -> list[int]:
    """Indices of the set bits of an attribute bitmask."""
    out, i = [], 0
    while mask:
        if mask & 1:
            out.append(i)
        mask >>= 1
        i += 1
    return out


def count_classes(codes: np.ndarray, attrs: list[int]) -> int:
    """Number of equivalence classes of the partition induced by `attrs`."""
    if not attrs:
        return 1 if len(codes) else 0
    return len(np.unique(codes[:, attrs], axis=0))


def fd_holds(df: pd.DataFrame, lhs, rhs: str) -> bool:
    """Exact FD check X -> A on a DataFrame (empty LHS = constant column)."""
    lhs = list(lhs)
    if not lhs:
        return df[rhs].nunique(dropna=False) <= 1
    g = df.groupby(lhs, dropna=False)[rhs].nunique(dropna=False)
    return bool((g <= 1).all())


def brute_force_fds(df: pd.DataFrame, max_lhs_size: int | None = None):
    """Reference implementation: enumerate every minimal FD. Slow; tests only."""
    codes, names = encode(df)
    m = len(names)
    limit = m - 1 if max_lhs_size is None else min(max_lhs_size, m - 1)
    found: dict[int, list[frozenset]] = {a: [] for a in range(m)}
    out = []
    for k in range(0, limit + 1):
        for lhs in combinations(range(m), k):
            s = frozenset(lhs)
            for a in range(m):
                if a in s or any(f <= s for f in found[a]):
                    continue
                if count_classes(codes, list(lhs)) == count_classes(codes, list(lhs) + [a]):
                    found[a].append(s)
                    out.append((tuple(names[i] for i in sorted(lhs)), names[a]))
    return out
