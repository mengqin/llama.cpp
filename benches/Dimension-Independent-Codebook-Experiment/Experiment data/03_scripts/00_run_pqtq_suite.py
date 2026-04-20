#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import shlex
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
BASE_RUNNER_PATH = SCRIPT_DIR / "_run_full_pqtq_suite_base.py"
DEFAULT_REPO_ROOT = SCRIPT_DIR.parent.parent.parent


def load_base_module():
    spec = importlib.util.spec_from_file_location("pqtq_suite_base", BASE_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load base runner: {BASE_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_models(base, args) -> list:
    all_models = {
        "D64": base.ModelConfig(
            dim="D64",
            model_path=args.model_d64,
            sampling_args=shlex.split(args.sampling_d64),
            func_ctx=args.func_ctx_d64,
            aime_ctx=args.aime_ctx_d64,
        ),
        "D128": base.ModelConfig(
            dim="D128",
            model_path=args.model_d128,
            sampling_args=shlex.split(args.sampling_d128),
            func_ctx=args.func_ctx_d128,
            aime_ctx=args.aime_ctx_d128,
        ),
        "D256": base.ModelConfig(
            dim="D256",
            model_path=args.model_d256,
            sampling_args=shlex.split(args.sampling_d256),
            func_ctx=args.func_ctx_d256,
            aime_ctx=args.aime_ctx_d256,
        ),
    }
    return [all_models[dim] for dim in args.dims]


def install_needle_dataset_override(base, args) -> None:
    def generate_needle_samples(length_cfg):
        from opencompass.datasets.needlebench.origin import NeedleBenchOriginDataset

        samples = []
        dataset_specs = [
            ("English", "PaulGrahamEssays.jsonl", length_cfg.en_length_buffer),
        ]
        for language, file_name, length_buffer in dataset_specs:
            for depth in base.NEEDLE_DEPTHS:
                dataset = NeedleBenchOriginDataset.load(
                    path=args.needle_dataset_path,
                    length=length_cfg.ctx_size,
                    depth=depth,
                    tokenizer_model=args.needle_tokenizer_model,
                    file_list=[file_name],
                    num_repeats_per_file=1,
                    length_buffer=length_buffer,
                    guide=True,
                    language=language,
                    needle_file_name="needles.jsonl",
                )
                if len(dataset) != 1:
                    raise RuntimeError(
                        f"unexpected dataset size for {length_cfg.name} {language} depth={depth}: {len(dataset)}"
                    )
                row = dataset[0]
                samples.append({
                    "language": language,
                    "depth": depth,
                    "prompt": row["prompt"],
                    "answer": row["answer"],
                })
        return samples

    base.generate_needle_samples = generate_needle_samples


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases", nargs="+", choices=["function", "bench", "ppl", "aime", "needle"], required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--cli-mode", choices=["templated", "raw"], default="raw")
    parser.add_argument("--dims", nargs="+", choices=["D64", "D128", "D256"], default=["D64", "D128", "D256"])

    parser.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT))
    parser.add_argument("--build-bin", default=None)

    parser.add_argument("--model-d64", required=True)
    parser.add_argument("--model-d128", required=True)
    parser.add_argument("--model-d256", required=True)

    parser.add_argument("--sampling-d64", default="--temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0")
    parser.add_argument("--sampling-d128", default="--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0")
    parser.add_argument("--sampling-d256", default="--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0")

    parser.add_argument("--func-ctx-d64", type=int, default=4096)
    parser.add_argument("--func-ctx-d128", type=int, default=4096)
    parser.add_argument("--func-ctx-d256", type=int, default=4096)
    parser.add_argument("--aime-ctx-d64", type=int, default=65536)
    parser.add_argument("--aime-ctx-d128", type=int, default=65536)
    parser.add_argument("--aime-ctx-d256", type=int, default=65536)

    parser.add_argument("--wiki-test-raw", default=None)
    parser.add_argument("--needle-dataset-path", default="opencompass/needlebench")
    parser.add_argument("--needle-tokenizer-model", default="gpt-4")

    args = parser.parse_args(argv)
    if "ppl" in args.phases and not args.wiki_test_raw:
        parser.error("--wiki-test-raw is required when running the ppl phase")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base = load_base_module()

    repo_root = Path(args.repo_root).resolve()
    build_bin = Path(args.build_bin).resolve() if args.build_bin else (repo_root / "build" / "bin" / "Release").resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    base.REPO_ROOT = repo_root
    base.BUILD_BIN = build_bin
    base.MODELS = build_models(base, args)
    if args.wiki_test_raw:
        base.WIKI_TEST_RAW = Path(args.wiki_test_raw).resolve()
    install_needle_dataset_override(base, args)

    all_results: dict[str, list[dict]] = {}
    if "function" in args.phases:
        all_results["function"] = base.run_function_phase(outdir, args.cli_mode)
        base.json_dump(outdir / "function_results.json", all_results["function"])
    if "bench" in args.phases:
        all_results["bench"] = base.run_bench_phase(outdir)
        for metric in ("pp512_tps", "tg128_tps", "pg32768_256_tps"):
            base.compute_relatives(all_results["bench"], metric)
        base.json_dump(outdir / "bench_results.json", all_results["bench"])
    if "ppl" in args.phases:
        all_results["ppl"] = base.run_ppl_phase(outdir)
        base.compute_relatives(all_results["ppl"], "ppl")
        base.json_dump(outdir / "ppl_results.json", all_results["ppl"])
    if "aime" in args.phases:
        all_results["aime"] = base.run_aime_phase(outdir, args.cli_mode)
        base.compute_relatives(all_results["aime"], "token_count")
        base.json_dump(outdir / "aime_results.json", all_results["aime"])
    if "needle" in args.phases:
        all_results["needle"] = base.run_needle_phase(outdir)
        base.compute_relatives(all_results["needle"], "needle_score", ("dim", "needle_size"))
        base.json_dump(outdir / "needle_results.json", all_results["needle"])

    summary_path = outdir / "summary.json"
    base.json_dump(summary_path, all_results)
    print(f"SUMMARY={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
