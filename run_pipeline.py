"""
Run the FD pipeline.

Examples
    python run_pipeline.py                          # all datasets, TANE + FastFD
    python run_pipeline.py --datasets iris abalone  # some datasets
    python run_pipeline.py --algorithms TANE --max-lhs-size 3
    python run_pipeline.py --csv my_table.csv       # any CSV with a header row
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fd_pipeline.datasets import DATASETS, MISSING, load_csv, load_dataset  # noqa: E402
from fd_pipeline.pipeline import ALGORITHMS, DatasetProcessor  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="TANE / FastFD functional dependency pipeline")
    ap.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS),
                    help="registered datasets to process (default: all)")
    ap.add_argument("--csv", type=Path, help="process this CSV file instead")
    ap.add_argument("--no-header", action="store_true", help="the --csv file has no header row")
    ap.add_argument("--algorithms", nargs="+", choices=list(ALGORITHMS),
                    default=list(ALGORITHMS))
    ap.add_argument("--max-lhs-size", type=int, default=None,
                    help="limit LHS size (default: no limit, same as Metanome)")
    ap.add_argument("--output", type=Path, default=ROOT / "output")
    args = ap.parse_args()

    if args.csv:
        data = {args.csv.stem: load_csv(args.csv, has_header=not args.no_header)}
    else:
        data = {name: load_dataset(name) for name in args.datasets}
        print(f"Datasets: {', '.join(data)}")
        print(f"Not runnable (data missing from the assignment ZIP): {', '.join(MISSING)}")

    all_results = {}
    for alg in args.algorithms:
        proc = DatasetProcessor(args.output, algorithm=alg, max_lhs_size=args.max_lhs_size)
        all_results[alg] = proc.process_all(data)

    # Side-by-side summary of both algorithms
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in data:
        row = {"dataset": name, "rows": len(data[name]), "cols": data[name].shape[1]}
        fd_sets = []
        for alg in args.algorithms:
            r = all_results[alg][name]
            row[f"{alg}_fds"] = r.get("num_fds", "ERROR")
            row[f"{alg}_seconds"] = r.get("discovery_seconds", "")
            raw = args.output / alg / "raw_results" / f"{name}_{alg}_fds.txt"
            fd_sets.append(set(raw.read_text(encoding="utf-8").splitlines()) if raw.exists() else None)
        if len(fd_sets) == 2 and None not in fd_sets:
            row["same_fds"] = fd_sets[0] == fd_sets[1]
        rows.append(row)
    summary = args.output / "summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for row in rows:
        print("  " + "  ".join(f"{k}={v}" for k, v in row.items()))
    print(f"\nSaved: {summary}")


if __name__ == "__main__":
    main()
