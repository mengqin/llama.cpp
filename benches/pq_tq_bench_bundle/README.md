# PQ/TQ Full Report

This Report packages one self-contained snapshot of the latest PQ/TQ report, the raw data directories behind that report, and parameterized scripts for rerunning each report section.

Click the link to view the report:
[report.html](https://mengqin.github.io/llama.cpp/pq_tq_bench_report/report.html)

## Layout

- `01_report/`
  - Final report artifacts: `report.json`, `report.html`, `function_summary.csv`, `bench.csv`, `ppl.csv`, `aime.csv`, `needle.csv`
- `02_raw/01_function_raw_cli/`
  - Latest function raw/no-jinja results used by the current report
- `02_raw/02_bench_ppl/`
  - Performance (`bench`) and perplexity (`ppl`) raw outputs used by the current report
- `02_raw/03_aime/`
  - AIME raw outputs used by the current report
- `02_raw/04_needle/`
  - NeedleBench raw outputs used by the current report
- `03_scripts/`
  - Parameterized rerun scripts. Model and dataset paths are passed on the command line instead of being hardcoded in the script body.

## Scripts

- `03_scripts/00_run_pqtq_suite.py`
  - General parameterized runner for `function`, `bench`, `ppl`, `aime`, `needle`
- `03_scripts/01_run_function_matrix.py`
  - Function matrix wrapper
- `03_scripts/02_run_performance_matrix.py`
  - Performance matrix wrapper
- `03_scripts/03_run_ppl_matrix.py`
  - Perplexity matrix wrapper
- `03_scripts/04_run_aime_matrix.py`
  - AIME matrix wrapper
- `03_scripts/05_run_needle_matrix.py`
  - NeedleBench matrix wrapper
- `03_scripts/06_build_report.py`
  - Rebuild `report.json` + CSV files from raw result directories and optionally copy an HTML template

## Minimal usage examples

Function matrix:

```powershell
python 03_scripts/01_run_function_matrix.py `
  --outdir temp\run_function_out `
  --model-d64 gpt-oss-20b-UD-Q4_K_XL.gguf `
  --model-d128 Qwen3-4B-Thinking-2507-UD-Q4_K_XL.gguf `
  --model-d256 Qwen3.5-4B-UD-Q4_K_XL.gguf
```

PPL matrix:

```powershell
python 03_scripts/03_run_ppl_matrix.py `
  --outdir temp\run_ppl_out `
  --wiki-test-raw wiki.test.raw `
  --model-d64 gpt-oss-20b-UD-Q4_K_XL.gguf `
  --model-d128 Qwen3-4B-Thinking-2507-UD-Q4_K_XL.gguf `
  --model-d256 Qwen3.5-4B-UD-Q4_K_XL.gguf
```

Needle matrix:

```powershell
python 03_scripts/05_run_needle_matrix.py `
  --outdir temp\run_needle_out `
  --needle-dataset-path opencompass/needlebench `
  --model-d64 gpt-oss-20b-UD-Q4_K_XL.gguf `
  --model-d128 Qwen3-4B-Thinking-2507-UD-Q4_K_XL.gguf `
  --model-d256 Qwen3.5-4B-UD-Q4_K_XL.gguf
```

Rebuild report:

```powershell
python 03_scripts/06_build_report.py `
  --function-dir 02_raw\01_function_raw_cli `
  --bench-ppl-dir 02_raw\02_bench_ppl `
  --aime-dir 02_raw\03_aime `
  --needle-dir 02_raw\04_needle `
  --outdir report_rebuilt `
  --html-template 01_report\report.html
```

## Notes

- Raw log filenames inside each copied result directory are preserved from the original run. This keeps the report linkage intact.
- The current function conclusion is based on the raw CLI path:
  - `llama-cli --no-jinja -st --simple-io`
  - `gpt-oss`: score the extracted `final` channel
  - `Qwen3`: append `/no_think` and use `n_predict=512`
  - `Qwen3.5`: strip `<think>...</think>` before answer matching
