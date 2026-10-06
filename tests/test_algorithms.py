"""
Correctness checks for TANE and FastFD.

    python tests/test_algorithms.py      (or: pytest tests)

1. Both algorithms must return exactly the minimal FDs found by brute force
   on many small random tables.
2. On iris and abalone they must reproduce the Metanome reference FDs given
   with the assignment (data/reference_fds). The reference was produced by
   Metanome reading the first data row as a header, so the test does the same.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fd_pipeline.datasets import DATASETS, DATASETS_DIR  # noqa: E402
from fd_pipeline.fastfd import fastfd  # noqa: E402
from fd_pipeline.fd_utils import brute_force_fds  # noqa: E402
from fd_pipeline.tane import tane  # noqa: E402

REFERENCE_DIR = ROOT / "data" / "reference_fds"


def test_random_tables_match_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(80):
        n, m = int(rng.integers(1, 30)), int(rng.integers(2, 6))
        df = pd.DataFrame(rng.integers(0, int(rng.integers(1, 4)), size=(n, m)),
                          columns=[f"c{i}" for i in range(m)])
        for k in (None, 1, 2):
            expected = set(brute_force_fds(df, k))
            assert set(tane(df, k)) == expected, (df, k)
            assert set(fastfd(df, k)) == expected, (df, k)


def _reference(name):
    text = (REFERENCE_DIR / f"{name}_fds").read_text().split("# RESULTS")[1]
    return {(frozenset(int(x) for x in l.split("->")[0].split(",")), int(l.split("->")[1]))
            for l in text.split() if "->" in l}


def test_metanome_reference():
    for name in ("iris", "abalone"):
        path = DATASETS_DIR / DATASETS[name].files[0]
        df = pd.read_csv(path, header=0, dtype=str, keep_default_na=False)
        index = {c: i + 1 for i, c in enumerate(df.columns)}
        expected = _reference(name)
        for algo in (tane, fastfd):
            got = {(frozenset(index[a] for a in lhs), index[rhs]) for lhs, rhs in algo(df)}
            assert got == expected, f"{algo.__name__} differs from Metanome on {name}"


if __name__ == "__main__":
    test_random_tables_match_brute_force()
    print("OK  TANE and FastFD match brute force on 80 random tables")
    test_metanome_reference()
    print("OK  TANE and FastFD reproduce the Metanome reference FDs (iris, abalone)")
