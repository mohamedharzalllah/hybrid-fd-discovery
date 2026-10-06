"""TANE (Huhtala et al., 1999): level-wise search over the attribute lattice
using partition refinement and C+ (right-hand-side candidate) pruning.

Returns every *minimal*, non-trivial FD  X -> A  (optionally |X| <= max_lhs_size).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .fd_utils import bits, count_classes, encode


def tane(df: pd.DataFrame, max_lhs_size: int | None = None):
    codes, names = encode(df)
    n, m = codes.shape
    full = (1 << m) - 1
    results: list[tuple[int, int]] = []          # (lhs_mask, rhs_index)

    if n <= 1:                                   # every column is constant
        return [((), names[a]) for a in range(m)]

    count_cache: dict[int, int] = {0: 1}

    def count(mask: int) -> int:
        if mask not in count_cache:
            count_cache[mask] = count_classes(codes, bits(mask))
        return count_cache[mask]

    # Level 1: one partition (label array) per attribute
    labels: dict[int, np.ndarray] = {}
    for a in range(m):
        labels[1 << a] = codes[:, a]
        count_cache[1 << a] = int(codes[:, a].max()) + 1
    cplus: dict[int, int] = {0: full}
    level = [1 << a for a in range(m)]
    size = 1

    while level:
        # ---- compute_dependencies -------------------------------------
        for x in level:
            c = full
            for a in bits(x):
                c &= cplus.get(x & ~(1 << a), 0)
            cplus[x] = c
        if max_lhs_size is None or size - 1 <= max_lhs_size:
            for x in level:
                for a in bits(x & cplus[x]):
                    lhs = x & ~(1 << a)
                    if count(lhs) == count(x):
                        results.append((lhs, a))
                        cplus[x] &= ~(1 << a)
                        cplus[x] &= x            # remove every B in R \ X
        # ---- prune ------------------------------------------------------
        kept = []
        for x in level:
            if cplus[x] == 0:
                continue
            if count(x) == n:                    # X is a (super)key
                if max_lhs_size is None or size <= max_lhs_size:
                    for a in bits(cplus[x] & ~x):
                        # X -> A holds (X is a key); keep it only if minimal
                        if all(count(x & ~(1 << b)) != count((x & ~(1 << b)) | (1 << a))
                               for b in bits(x)):
                            results.append((x, a))
                continue
            kept.append(x)
        if max_lhs_size is not None and size >= max_lhs_size + 1:
            break
        # ---- generate_next_level (prefix blocks) ------------------------
        kept_set = set(kept)
        by_prefix: dict[int, list[int]] = {}
        for x in kept:
            hi = x.bit_length() - 1
            by_prefix.setdefault(x & ~(1 << hi), []).append(x)
        nxt, new_labels = [], {}
        for block in by_prefix.values():
            block.sort()
            for i in range(len(block)):
                for j in range(i + 1, len(block)):
                    y = block[i] | block[j]
                    if all((y & ~(1 << b)) in kept_set for b in bits(y)):
                        la, lb = labels[block[i]], labels[block[j]]
                        combo = la.astype(np.int64) * (int(lb.max()) + 1) + lb
                        _, inv = np.unique(combo, return_inverse=True)
                        new_labels[y] = inv.astype(np.int32).ravel()
                        count_cache[y] = int(new_labels[y].max()) + 1
                        nxt.append(y)
        labels = new_labels
        level = nxt
        size += 1

    out = [(tuple(names[i] for i in bits(lhs)), names[a]) for lhs, a in results]
    return sorted(set(out), key=lambda f: (len(f[0]), f[1], f[0]))
