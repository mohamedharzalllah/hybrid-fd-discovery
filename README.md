# Hybrid FD Discovery — Data Quality lab (M2 IASD, Paris-Dauphine)

Lab assignment *"Beyond Patterns: Meaningful Functional Dependencies with Classical
Algorithms and LLMs"* (K. Belhajjame). This folder contains the **algorithmic part
of the pipeline (Task Set 4, part 1)**: discover minimal functional dependencies
with TANE and FastFD, parse them, compute statistics, flag suspicious FDs and
prepare the files used by the LLM / verification stages.

## Quick start

```bash
pip install -r requirements.txt
python tests/test_algorithms.py        # correctness checks (~5 s)
python run_pipeline.py                 # all datasets, TANE + FastFD (~1.5 min, adult is the slow one)
```

Options:

```bash
python run_pipeline.py --datasets iris abalone        # only some datasets
python run_pipeline.py --algorithms TANE              # only one algorithm
python run_pipeline.py --max-lhs-size 3               # cap the LHS size (default: no cap, like Metanome)
python run_pipeline.py --csv path/to/table.csv        # any CSV with a header row (--no-header otherwise)
```

## Structure

```
.
├── run_pipeline.py            # entry point (CLI)
├── requirements.txt
├── src/fd_pipeline/
│   ├── datasets.py            # dataset registry: files + real column names from the .names files
│   ├── tane.py                # TANE (partitions, C+ pruning, key pruning)
│   ├── fastfd.py              # FastFD (difference sets + minimal covers, DFS)
│   ├── fd_utils.py            # encoding, FD check, brute-force reference (tests)
│   ├── fd_parser.py           # parse FD files, statistics, suspicious FDs (Task Set 1)
│   ├── fd_pipeline_prep.py    # plausible / suspicious split, sample for the LLM (Task Set 2/4)
│   └── pipeline.py            # discovery -> parsing -> pipeline-ready files
├── tests/test_algorithms.py   # brute-force + Metanome reference checks
├── data/
│   ├── datasets/              # UCI datasets from the assignment ZIP
│   └── reference_fds/         # FDs provided with the assignment (Metanome / TANE)
├── docs/                      # assignment slides, task split, report drafts (reports/)
├── output/                    # generated (git-ignored)
│   ├── summary.csv            # #FDs and runtime per dataset and algorithm
│   └── TANE/ , FASTFD/
│       ├── raw_results/<dataset>_<ALG>_fds.txt
│       └── <dataset>/parsed/ , pipeline_ready/ , processing_results.json
└── _old/                      # previous version of the code and its outputs (kept for reference)
```

`pipeline_ready/` per dataset: `all_fds.json`, `categories.json`, `plausible_fds.json`,
`suspicious_fds.json`, `sample_fds.json` (3 plausible + 3 suspicious) and
`sample_for_llm.txt` (prompt for Task Set 2).

## Verification (6 Oct 2026)

| Dataset | Rows × cols | TANE | FastFD | Assignment table |
|---|---|---|---|---|
| iris | 150 × 5 | 4 | 4 | 4 |
| balance-scale | 625 × 5 | 1 | 1 | 1 |
| abalone | 4 177 × 9 | 137 | 137 | 137 |
| nursery | 12 960 × 9 | 1 | 1 | 1 |
| breast-cancer-wisconsin | 699 × 11 | 46 | 46 | 46 |
| bridges | 108 × 13 | 142 | 142 | 142 |
| echocardiogram | 132 × 13 | 527 | 527 | 538 |
| adult | 48 842 × 15 | 66 | 66 | 78 |

* TANE and FastFD return identical FD sets on every dataset, and both equal a
  brute-force enumeration on 80 random tables.
* On iris and abalone (the reference files that can be mapped back to columns)
  both reproduce the Metanome FDs exactly.
* echocardiogram / adult differ slightly from the table, most likely because the data is read
  differently: Metanome used the first data row as a header and did not repair
  echocardiogram line 50; for adult, the `.` at the end of the test-set labels
  (`>50K.`) is removed so that both halves use the same labels. The adult reference
  file in the ZIP is empty, so this cannot be checked FD by FD.
* chess, hepatitis and horse cannot be run: their data files are not in the ZIP
  (the chess folder only contains a Lisp generator, `kr-vs-kkn.l`).

## What was fixed compared to the first version (`_old/`)

1. **Column names** – `.data` files have no header; the first data row was used as
   column names (`[0.455, 0.514, 0.2245] -> 15`) and that row was lost. Columns now
   carry the real attribute names, which the LLM stages need.
2. **TANE** – the pruning kept only the last LHS per RHS, so non-minimal FDs were
   returned (36 of 56 on breast-cancer-wisconsin, 154 k lines on wdbc). Replaced by a
   real TANE.
3. **FastFD** – stopped at the first FD per RHS attribute (1 FD on iris instead of 4).
   Replaced by a real FastFD.
4. **LHS limit** – the default `max_lhs_size=3` silently dropped FDs (abalone: 18 of 137).
   Default is now no limit.
5. **echocardiogram** crashed (14 fields on line 50); the stray field is now repaired.
6. **Parser** – the regex cut names with dots (`0.101` became `0`); parsing is now
   name-agnostic. Reference files in Metanome index and JSON formats are supported.
7. **Pipeline** – no more subprocess / working-directory dependence; the paths used
   `FDs_labAssignement` (wrong case, breaks outside Windows); the overall summary was
   overwritten for each dataset; extra files (`wdbc`, `wpbc`, `adult.test`,
   `bezdekIris`…) were processed as separate datasets.
8. **Suspicious FDs** – ID columns are detected from the data (≥ 90 % distinct
   values, e.g. `sample_code_number`, `identif`); one "large LHS" threshold (> 3)
   is used everywhere; the LLM sample picks varied FDs instead of the first ones.

## Not in this folder yet

Task Sets 1–3 notebooks and the Task Set 4 part 2 components (LLM semantic filter,
verification module, evaluation metrics). See `docs/project_division.txt`.
