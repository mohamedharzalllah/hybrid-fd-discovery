"""Dataset registry: where each UCI file lives and what its columns are called.

UCI ``.data`` files have no header row. Reading them with ``header="infer"``
turns the first data row into column names (e.g. ``0.455 -> M``), which hides
the meaning of the attributes from the LLM stages. The names below come from
each dataset's ``.names`` file.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"


@dataclass
class DatasetSpec:
    name: str
    files: list[str]                  # relative to DATASETS_DIR (concatenated)
    columns: list[str]
    skip_lines: int = 0               # lines to skip at the top of every file
    strip_suffix: dict[str, str] = field(default_factory=dict)  # col -> suffix to remove


DATASETS: dict[str, DatasetSpec] = {
    "iris": DatasetSpec(
        "iris", ["iris/iris.data"],
        ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]),
    "balance-scale": DatasetSpec(
        "balance-scale", ["balance+scale/balance-scale.data"],
        ["class", "left_weight", "left_distance", "right_weight", "right_distance"]),
    "abalone": DatasetSpec(
        "abalone", ["abalone/abalone.data"],
        ["sex", "length", "diameter", "height", "whole_weight", "shucked_weight",
         "viscera_weight", "shell_weight", "rings"]),
    "nursery": DatasetSpec(
        "nursery", ["nursery/nursery.data"],
        ["parents", "has_nurs", "form", "children", "housing", "finance",
         "social", "health", "class"]),
    "breast-cancer-wisconsin": DatasetSpec(
        "breast-cancer-wisconsin",
        ["breast+cancer+wisconsin+original/breast-cancer-wisconsin.data"],
        ["sample_code_number", "clump_thickness", "uniformity_cell_size",
         "uniformity_cell_shape", "marginal_adhesion", "single_epithelial_cell_size",
         "bare_nuclei", "bland_chromatin", "normal_nucleoli", "mitoses", "class"]),
    "bridges": DatasetSpec(
        "bridges", ["pittsburgh+bridges/bridges.data.version1"],
        ["identif", "river", "location", "erected", "purpose", "length", "lanes",
         "clear_g", "t_or_d", "material", "span", "rel_l", "type"]),
    "echocardiogram": DatasetSpec(
        "echocardiogram", ["echocardiogram/echocardiogram.data"],
        ["survival", "still_alive", "age_at_heart_attack", "pericardial_effusion",
         "fractional_shortening", "epss", "lvdd", "wall_motion_score",
         "wall_motion_index", "mult", "name", "group", "alive_at_1"]),
    "adult": DatasetSpec(
        "adult", ["adult/adult.data", "adult/adult.test"],
        ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
         "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
         "hours_per_week", "native_country", "income"],
        strip_suffix={"income": "."}),
}

# Datasets listed in the assignment whose data files are NOT in the ZIP:
#   chess   -> only kr-vs-kkn.l (a Lisp generator) is provided, no data table
#   hepatitis, horse -> reference FDs only, no data
MISSING = ["chess", "hepatitis", "horse"]


def _read_rows(path: Path, ncols: int) -> tuple[list[list[str]], int]:
    """Read a comma-separated file, repairing rows with one stray empty field."""
    rows, fixed = [], 0
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for raw in csv.reader(f, skipinitialspace=True):
            if not raw or all(not v.strip() for v in raw) or raw[0].startswith("|"):
                continue
            row = [v.strip() for v in raw]
            if len(row) == ncols + 1 and row[0] == "":
                row, fixed = row[1:], fixed + 1      # e.g. echocardiogram line 50
            elif len(row) == ncols + 1 and row[-1] == "":
                row, fixed = row[:-1], fixed + 1
            if len(row) != ncols:
                raise ValueError(f"{path.name}: expected {ncols} fields, got {len(row)}: {raw}")
            rows.append(row)
    return rows, fixed


def load_dataset(name: str, datasets_dir: Path | None = None) -> pd.DataFrame:
    """Load a registered dataset as strings ('?' kept as an ordinary value)."""
    spec = DATASETS[name]
    base = Path(datasets_dir) if datasets_dir else DATASETS_DIR
    rows = []
    for rel in spec.files:
        part, fixed = _read_rows(base / rel, len(spec.columns))
        if fixed:
            print(f"  [info] {rel}: repaired {fixed} row(s) with a stray empty field")
        rows.extend(part)
    df = pd.DataFrame(rows, columns=spec.columns, dtype=str)
    for col, suffix in spec.strip_suffix.items():
        df[col] = df[col].str.removesuffix(suffix)
    return df


def load_csv(path: Path, has_header: bool = True, sep: str = ",") -> pd.DataFrame:
    """Load any other CSV file (header row required unless has_header=False)."""
    df = pd.read_csv(path, sep=sep, header=0 if has_header else None, dtype=str,
                     keep_default_na=False, skipinitialspace=True)
    if not has_header:
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
    df.columns = [str(c).strip() for c in df.columns]
    return df
