#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def function_summary(function_results: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in function_results:
        grouped.setdefault((row["dim"], row["type"]), []).append(row)
    out = []
    for (dim, typ), rows in sorted(grouped.items()):
        pass_count = sum(1 for row in rows if row["pass"])
        out.append({
            "dim": dim,
            "type": typ,
            "pass_count": pass_count,
            "total": len(rows),
            "all_pass": pass_count == len(rows),
            "failed_questions": [row["question_id"] for row in rows if not row["pass"]],
        })
    return out


def sort_key(row: dict) -> tuple:
    type_order = {
        "f16": 0,
        "q8_0": 1,
        "q4_0": 2,
        "pq2": 3,
        "tq2": 4,
        "pq3": 5,
        "tq3": 6,
        "pq4": 7,
        "tq4": 8,
    }
    dim_order = {"D64": 0, "D128": 1, "D256": 2}
    needle_order = {"32K": 0, "128K": 1}
    return (
        dim_order.get(row.get("dim", ""), 99),
        needle_order.get(row.get("needle_size", ""), 99),
        type_order.get(row.get("type", ""), 99),
    )


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    lines = [",".join(columns)]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, list):
                value = ";".join(str(x) for x in value)
            text = str(value)
            if any(ch in text for ch in [",", "\"", "\n"]):
                text = "\"" + text.replace("\"", "\"\"") + "\""
            values.append(text)
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--function-dir", required=True)
    parser.add_argument("--bench-ppl-dir", required=True)
    parser.add_argument("--aime-dir", required=True)
    parser.add_argument("--needle-dir", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    function_results = load_json(Path(args.function_dir) / "function_results.json")
    bench_results = load_json(Path(args.bench_ppl_dir) / "bench_results.json")
    ppl_results = load_json(Path(args.bench_ppl_dir) / "ppl_results.json")
    aime_results = load_json(Path(args.aime_dir) / "aime_results.json")
    needle_results = load_json(Path(args.needle_dir) / "needle_results.json")

    report = {
        "function": function_summary(function_results),
        "bench": sorted(bench_results, key=sort_key),
        "ppl": sorted(ppl_results, key=sort_key),
        "aime": sorted(aime_results, key=sort_key),
        "needle": sorted(needle_results, key=sort_key),
        "source_dirs": {
            "function": str(Path(args.function_dir)),
            "bench_ppl": str(Path(args.bench_ppl_dir)),
            "aime": str(Path(args.aime_dir)),
            "needle": str(Path(args.needle_dir)),
        },
    }
    dump_json(outdir / "report.json", report)

    write_csv(
        outdir / "function_summary.csv",
        report["function"],
        ["dim", "type", "pass_count", "total", "all_pass", "failed_questions"],
    )
    write_csv(
        outdir / "bench.csv",
        report["bench"],
        [
            "dim", "type", "exit_code",
            "pp512_tps", "pp512_tps_vs_q8_pct",
            "tg128_tps", "tg128_tps_vs_q8_pct",
            "pg32768_256_tps", "pg32768_256_tps_vs_q8_pct",
        ],
    )
    write_csv(
        outdir / "ppl.csv",
        report["ppl"],
        ["dim", "type", "exit_code", "ppl", "ppl_vs_q8_pct"],
    )
    write_csv(
        outdir / "aime.csv",
        report["aime"],
        ["dim", "type", "exit_code", "final_answer", "pass", "coherence", "token_count", "token_count_vs_q8_pct"],
    )
    write_csv(
        outdir / "needle.csv",
        report["needle"],
        ["dim", "needle_size", "type", "exit_code", "needle_score", "needle_score_vs_q8_pct", "needle_score_en", "needle_score_zh"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
