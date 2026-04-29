#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
BASE_RUNNER_PATH = SCRIPT_DIR / "_run_full_pqtq_suite_base.py"
DEFAULT_REPO_ROOT = SCRIPT_DIR.parent.parent.parent


def load_base_module():
    spec = importlib.util.spec_from_file_location("pqtq_weight_matrix_base", BASE_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load base runner: {BASE_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
    parser = argparse.ArgumentParser(description="Run bench-bundle phases for an arbitrary model matrix.")
    parser.add_argument("--models-json", required=True, help="Path to a JSON file describing the model matrix.")
    parser.add_argument("--phases", nargs="+", choices=["function", "bench", "ppl", "aime", "needle"], required=True)
    parser.add_argument("--types", nargs="+", default=None, help="Optional cache-type subset to run, e.g. f16 q8_0 pq4")
    parser.add_argument("--function-n-predict", type=int, default=None, help="Optional override for function-phase n_predict")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--cli-mode", choices=["templated", "raw"], default="raw")

    parser.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT))
    parser.add_argument("--build-bin", default=None)
    parser.add_argument("--wiki-test-raw", default=None)
    parser.add_argument("--needle-dataset-path", default="opencompass/needlebench")
    parser.add_argument("--needle-tokenizer-model", default="gpt-4")

    args = parser.parse_args(argv)
    if "ppl" in args.phases and not args.wiki_test_raw:
        parser.error("--wiki-test-raw is required when running the ppl phase")
    return args


def load_models(base, path: Path) -> list:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    model_items = payload["models"] if isinstance(payload, dict) else payload
    models = []
    for item in model_items:
        name = item["name"]
        model_path = str(Path(item["model_path"]).resolve())
        sampling = item.get("sampling_args", "--temp 0 --top-k 1 --top-p 1.0")
        sampling_args = shlex.split(sampling) if isinstance(sampling, str) else list(sampling)
        func_ctx = int(item.get("func_ctx", 4096))
        aime_ctx = int(item.get("aime_ctx", 16384))
        models.append(base.ModelConfig(
            dim=name,
            model_path=model_path,
            sampling_args=sampling_args,
            func_ctx=func_ctx,
            aime_ctx=aime_ctx,
        ))
    return models


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base = load_base_module()

    repo_root = Path(args.repo_root).resolve()
    build_bin = Path(args.build_bin).resolve() if args.build_bin else (repo_root / "build" / "bin" / "Release").resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.types:
        wanted = {name.lower() for name in args.types}
        available = {cfg.name.lower() for cfg in base.TYPES}
        unknown = sorted(wanted - available)
        if unknown:
            raise ValueError(f"unknown cache types requested: {', '.join(unknown)}")
        base.TYPES = [cfg for cfg in base.TYPES if cfg.name.lower() in wanted]

    if args.function_n_predict is not None:
        override_n_predict = args.function_n_predict

        def function_n_predict_override(model, cli_mode):
            return override_n_predict

        base.function_n_predict = function_n_predict_override

    base.REPO_ROOT = repo_root
    base.BUILD_BIN = build_bin
    base.MODELS = load_models(base, Path(args.models_json).resolve())
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