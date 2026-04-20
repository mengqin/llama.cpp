# PQ/TQ Rollback Baseline Bundle

This bundle captures the rollback-to-baseline validation run after reverting independent codebooks and D-aware QJL changes back to the original unified-codebook semantics.

Structure:
- `01_report`: generated `report.json`, `report.html`, and CSV summaries
- `02_raw`: raw outputs for function, bench/PPL, AIME, and needle phases
- `03_scripts`: suite and report scripts used for this run

Run inputs:
- Models are passed by argument to `03_scripts/00_run_pqtq_suite.py`
- PPL text file is passed by `--wiki-test-raw`
- Needle uses OpenCompass dataset id `opencompass/needlebench` with `COMPASS_DATA_CACHE` set in the shell

Phase layout:
- `02_raw/01_function_raw_cli`
- `02_raw/02_bench_ppl`
- `02_raw/03_aime`
- `02_raw/04_needle`

Notes:
- Function, bench, PPL, and needle are the main rollback checks.
- AIME is included, but single-run pass/fail is treated as lower-confidence.
