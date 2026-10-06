"""
Task Set 4 Part 1 — end-to-end FD pipeline

dataset -> FD discovery (TANE / FastFD) -> parsed FDs + statistics
        -> pipeline-ready files for the LLM / verification stages
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from .fastfd import fastfd
from .fd_parser import FDParser
from .fd_pipeline_prep import FDPipelinePrep
from .tane import tane

ALGORITHMS = {"TANE": tane, "FASTFD": fastfd}


def write_fds(fds, out_path: Path) -> None:
    """Write FDs in the Metanome-like text format read by FDParser."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for lhs, rhs in fds:
            f.write(f"[{', '.join(lhs)}] -> {rhs}\n")


def column_profile(df: pd.DataFrame) -> dict[str, float]:
    """Distinct-value ratio of every column (1.0 = unique, i.e. an ID)."""
    n = max(len(df), 1)
    return {c: df[c].nunique(dropna=False) / n for c in df.columns}


class DatasetProcessor:
    """Runs one algorithm on one or more datasets and writes all outputs."""

    def __init__(self, output_base_dir, algorithm="TANE", max_lhs_size=None):
        if algorithm not in ALGORITHMS:
            raise ValueError(f"algorithm must be one of {list(ALGORITHMS)}")
        self.output_base_dir = Path(output_base_dir) / algorithm
        self.algorithm = algorithm
        self.max_lhs_size = max_lhs_size
        self.raw_fd_dir = self.output_base_dir / "raw_results"
        self.raw_fd_dir.mkdir(parents=True, exist_ok=True)

    def process_dataset(self, name: str, df: pd.DataFrame) -> dict:
        print("\n" + "=" * 70)
        print(f"{self.algorithm} | {name}  ({df.shape[0]} rows x {df.shape[1]} cols)")
        print("=" * 70)

        dataset_out = self.output_base_dir / name
        parsed_dir = dataset_out / "parsed"
        pipeline_dir = dataset_out / "pipeline_ready"
        parsed_dir.mkdir(parents=True, exist_ok=True)
        pipeline_dir.mkdir(parents=True, exist_ok=True)
        raw_fd_file = self.raw_fd_dir / f"{name}_{self.algorithm}_fds.txt"

        results = {
            "dataset": name,
            "algorithm": self.algorithm,
            "rows": int(df.shape[0]),
            "columns": list(df.columns),
            "max_lhs_size": self.max_lhs_size,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "success": False,
        }
        try:
            print(f"[1/3] FD discovery ({self.algorithm})")
            t0 = time.perf_counter()
            fds = ALGORITHMS[self.algorithm](df, max_lhs_size=self.max_lhs_size)
            results["discovery_seconds"] = round(time.perf_counter() - t0, 2)
            write_fds(fds, raw_fd_file)
            print(f"      {len(fds)} minimal FDs in {results['discovery_seconds']} s")

            print("[2/3] Parsing FDs")
            parser = FDParser(column_profile=column_profile(df))
            parsed = parser.parse_metanome_output(raw_fd_file)
            parser.calculate_statistics()
            parser.print_summary()
            parsed_json = parsed_dir / "parsed_fds.json"
            parser.export_to_json(parsed_json)
            results["num_fds"] = len(parsed)
            results["statistics"] = parser.statistics

            print("[3/3] Preparing pipeline-ready outputs")
            prep = FDPipelinePrep(parsed_json)
            summary = prep.export_pipeline_format(pipeline_dir)
            prep.print_pipeline_summary(summary)
            results["pipeline_summary"] = summary
            results["success"] = True
        except Exception as e:  # keep going with the other datasets
            print(f"[ERROR] {name} failed: {e}")
            results["error"] = repr(e)

        with open(dataset_out / "processing_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        return results

    def process_all(self, datasets: dict[str, pd.DataFrame]) -> dict:
        all_results = {name: self.process_dataset(name, df) for name, df in datasets.items()}
        summary_file = self.output_base_dir / f"overall_summary_{self.algorithm.lower()}.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nOverall summary saved to: {summary_file}")
        return all_results
