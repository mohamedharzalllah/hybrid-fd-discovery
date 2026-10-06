"""FastFD (Wyss et al., 2001): compute the difference sets of all tuple pairs,
then, for every attribute A, enumerate the minimal covers of the difference
sets containing A with a depth-first search. Each minimal cover X gives the
minimal FD  X -> A.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .fd_utils import bits, encode


def _agree_sets(codes: np.ndarray, verbose: bool = False) -> set[int]:
    """Distinct agree sets (as bitmasks) over all pairs of distinct tuples."""
    n, m = codes.shape
    weights = (1 << np.arange(m, dtype=np.int64))
    use_table = m <= 24
    table = np.zeros(1 << m, dtype=bool) if use_table else None
    found: set[int] = set()
    block = max(1, 2_000_000 // max(1, n * m))   # rows compared per numpy step
    for start in range(0, n - 1, block):
        stop = min(start + block, n - 1)
        for i in range(start, stop):
            eq = codes[i + 1:] == codes[i]
            masks = eq @ weights
            if use_table:
                table[masks] = True
            else:
                found.update(np.unique(masks).tolist())
        if verbose and n > 5000 and start % (block * 200) == 0:
            print(f"    agree sets: {stop}/{n} rows", flush=True)
    if use_table:
        found = set(np.flatnonzero(table).tolist())
    return found


def _minimize(sets: set[int]) -> list[int]:
    """Keep only the inclusion-minimal sets."""
    ordered = sorted(sets, key=lambda s: bin(s).count("1"))
    minimal: list[int] = []
    for s in ordered:
        if not any((t & s) == t for t in minimal):
            minimal.append(s)
    return minimal


def _covers(dsets: list[int], m: int, max_size: int | None) -> list[int]:
    """All minimal covers (hitting sets) of `dsets`, FastFD DFS ordering."""
    out: list[int] = []

    def is_minimal(path: int) -> bool:
        for b in bits(path):
            if not any((d & path) == (1 << b) for d in dsets):
                return False
        return True

    def order_of(cur: list[int], allowed: list[int]) -> list[int]:
        cov = {a: sum(1 for d in cur if d >> a & 1) for a in allowed}
        return sorted((a for a in allowed if cov[a] > 0), key=lambda a: (-cov[a], a))

    def dfs(cur: list[int], order: list[int], path: int):
        if not cur:
            if is_minimal(path):
                out.append(path)
            return
        if not order or (max_size is not None and bin(path).count("1") >= max_size):
            return
        for i, a in enumerate(order):
            nxt = [d for d in cur if not (d >> a & 1)]
            dfs(nxt, order_of(nxt, order[i + 1:]), path | (1 << a))

    dfs(dsets, order_of(dsets, list(range(m))), 0)
    return out


def fastfd(df: pd.DataFrame, max_lhs_size: int | None = None, verbose: bool = False):
    codes, names = encode(df)
    n, m = codes.shape
    full = (1 << m) - 1
    if n <= 1:
        return [((), names[a]) for a in range(m)]
    diff_sets = {full & ~ag for ag in _agree_sets(codes, verbose)}
    diff_sets.discard(0)
    results = []
    for a in range(m):
        d_a = {d & ~(1 << a) for d in diff_sets if d >> a & 1}
        if not d_a:                       # A never differs: constant column
            results.append(((), names[a]))
            continue
        if 0 in d_a:                      # two tuples differ only on A
            continue
        for lhs in _covers(_minimize(d_a), m, max_lhs_size):
            results.append((tuple(names[i] for i in bits(lhs)), names[a]))
    return sorted(set(results), key=lambda f: (len(f[0]), f[1], f[0]))
