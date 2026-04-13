#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests


REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_BIN = REPO_ROOT / "build" / "bin" / "Release"
# This base file is intended to be wrapped by 00_run_pqtq_suite.py, which injects
# repo-local model and dataset paths at runtime.
WIKI_TEST_RAW = Path("REQUIRED_WIKI_TEST_RAW")

AIME_PROMPT = (
    "Solve the following AIME problem step by step. "
    "Keep reasoning coherent, and end with a final line exactly in the form "
    "'Final answer: <integer>'.\n\n"
    "Patrick started walking at a constant speed along a straight road from his school to the park. "
    "One hour after Patrick left, Tanya started running at a constant speed of 2 miles per hour faster than Patrick walked, "
    "following the same straight road from the school to the park. One hour after Tanya left, Jose started bicycling at a constant speed "
    "of 7 miles per hour faster than Tanya ran, following the same straight road from the school to the park. "
    "All three people arrived at the park at the same time. The distance from the school to the park is m/n miles, "
    "where m and n are relatively prime positive integers. Find m + n."
)


@dataclass(frozen=True)
class ModelConfig:
    dim: str
    model_path: str
    sampling_args: list[str]
    func_ctx: int
    aime_ctx: int


@dataclass(frozen=True)
class CacheTypeConfig:
    name: str
    ctk: str
    ctv: str


@dataclass(frozen=True)
class NeedleLengthConfig:
    name: str
    ctx_size: int
    en_length_buffer: int
    zh_length_buffer: int


MODELS: list[ModelConfig] = [
    ModelConfig(
        dim="D64",
        model_path="REQUIRED_MODEL_D64.gguf",
        sampling_args=["--temp", "1.0", "--top-p", "1.0", "--top-k", "0", "--min-p", "0.0"],
        func_ctx=4096,
        aime_ctx=65536,
    ),
    ModelConfig(
        dim="D128",
        model_path="REQUIRED_MODEL_D128.gguf",
        sampling_args=["--temp", "0.6", "--top-p", "0.95", "--top-k", "20", "--min-p", "0.0"],
        func_ctx=4096,
        aime_ctx=65536,
    ),
    ModelConfig(
        dim="D256",
        model_path="REQUIRED_MODEL_D256.gguf",
        sampling_args=["--temp", "0.6", "--top-p", "0.95", "--top-k", "20", "--min-p", "0.0"],
        func_ctx=4096,
        aime_ctx=65536,
    ),
]

TYPES: list[CacheTypeConfig] = [
    CacheTypeConfig("f16", "f16", "f16"),
    CacheTypeConfig("q8_0", "q8_0", "q8_0"),
    CacheTypeConfig("q4_0", "q4_0", "q4_0"),
    CacheTypeConfig("pq2", "pq2", "pq2"),
    CacheTypeConfig("tq2", "tq2", "pq2"),
    CacheTypeConfig("pq3", "pq3", "pq3"),
    CacheTypeConfig("tq3", "tq3", "pq3"),
    CacheTypeConfig("pq4", "pq4", "pq4"),
    CacheTypeConfig("tq4", "tq4", "pq4"),
]

NEEDLE_DEPTHS = [0, 20, 40, 60, 80, 100]
NEEDLE_MAX_TOKENS = 256
NEEDLE_RETRY_MAX_TOKENS = 512
NEEDLE_LENGTHS: list[NeedleLengthConfig] = [
    NeedleLengthConfig("32K", 32000, 3000, 200),
    NeedleLengthConfig("128K", 128000, 600, 200),
]
SERVER_ALIAS = "local-model"
GPT_OSS_STOP_THINK_SUFFIX = "<|end|><|start|>assistant<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>"
NEEDLE_SYSTEM_PROMPT = (
    "Answer with exactly one sentence in the user-requested fill-in-the-blank format. "
    "Do not include reasoning, explanations, quotes, bullet points, or any extra sentences."
)
GPT_OSS_CHANNEL_RE = re.compile(
    r"<\|channel\|>(analysis|final)<\|message\|>(.*?)(?=(?:<\|end\|><\|start\|>assistant<\|channel\|>)|$)",
    re.DOTALL,
)

FUNCTION_QUESTIONS = [
    {
        "id": "arith_2p2",
        "prompt": "What is 2 + 2? Reply with just the answer.",
        "expect": re.compile(r"\b4\b", re.IGNORECASE),
    },
    {
        "id": "capital_france",
        "prompt": "Which city is the capital of France? Reply with just the city name.",
        "expect": re.compile(r"\bparis\b", re.IGNORECASE),
    },
    {
        "id": "larger_9_12",
        "prompt": "Which is larger, 9 or 12? Reply with just the larger number.",
        "expect": re.compile(r"\b12\b", re.IGNORECASE),
    },
]


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def json_dump(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_json_array(text: str) -> list[dict]:
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end < 0 or end < start:
        raise ValueError("JSON array not found")
    return json.loads(text[start:end + 1])


def extract_generation_text(log_text: str) -> str:
    prompt_match = re.search(r"(^|\n)(> .*)", log_text)
    if prompt_match:
        text = log_text[prompt_match.start(2):]
        nl = text.find("\n")
        if nl >= 0:
            text = text[nl + 1:]
    else:
        text = log_text

    perf_markers = [
        "\n[ Prompt:",
        "\nllama_perf_context_print:",
        "\nExiting...",
    ]
    end = len(text)
    for marker in perf_markers:
        idx = text.find(marker)
        if idx >= 0:
            end = min(end, idx)
    text = text[:end]
    filtered = []
    for line in text.splitlines():
        if line.startswith("llama_memory_breakdown_print:"):
            continue
        filtered.append(line)
    return "\n".join(filtered).strip()


def parse_sampling_args(args: list[str]) -> dict[str, float | int]:
    parsed: dict[str, float | int] = {}
    i = 0
    while i + 1 < len(args):
        key = args[i]
        value = args[i + 1]
        if key == "--temp":
            parsed["temperature"] = float(value)
        elif key == "--top-p":
            parsed["top_p"] = float(value)
        elif key == "--top-k":
            parsed["top_k"] = int(value)
        elif key == "--min-p":
            parsed["min_p"] = float(value)
        i += 2
    return parsed


def needle_prompt_for_model(model: ModelConfig, prompt: str) -> str:
    model_path = model.model_path.lower()
    if "gpt-oss" in model_path:
        return prompt + GPT_OSS_STOP_THINK_SUFFIX
    if "qwen3.5" in model_path:
        return prompt
    if "qwen3" in model_path:
        return prompt.rstrip() + "\n/no_think"
    return prompt


def needle_system_prompt_for_model(model: ModelConfig) -> str | None:
    model_path = model.model_path.lower()
    if "qwen3" in model_path:
        return NEEDLE_SYSTEM_PROMPT
    return None


def normalize_needle_prediction(text: str) -> str:
    text = re.sub(r"^\s*<think>\s*</think>\s*", "", text, flags=re.DOTALL)
    return text.strip()


def model_path_lower(model: ModelConfig) -> str:
    return model.model_path.lower()


def is_gpt_oss_model(model: ModelConfig) -> bool:
    return "gpt-oss" in model_path_lower(model)


def is_qwen35_model(model: ModelConfig) -> bool:
    return "qwen3.5" in model_path_lower(model)


def is_qwen3_model(model: ModelConfig) -> bool:
    model_path = model_path_lower(model)
    return "qwen3" in model_path and "qwen3.5" not in model_path


def strip_think_blocks(text: str) -> str:
    text = re.sub(r"<think>\s*</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def extract_gpt_oss_channels(text: str) -> list[dict[str, str]]:
    channels: list[dict[str, str]] = []
    for match in GPT_OSS_CHANNEL_RE.finditer(text):
        channels.append({
            "channel": match.group(1),
            "content": match.group(2).strip(),
        })
    return channels


def extract_gpt_oss_final_message(text: str) -> str | None:
    channels = extract_gpt_oss_channels(text)
    finals = [entry["content"] for entry in channels if entry["channel"] == "final" and entry["content"].strip()]
    if finals:
        return finals[-1].strip()
    marker = "<|channel|>final<|message|>"
    idx = text.rfind(marker)
    if idx >= 0:
        return text[idx + len(marker):].strip()
    return None


def normalize_raw_cli_output(model: ModelConfig, text: str, phase: str) -> str:
    cleaned = text.strip()
    if is_gpt_oss_model(model):
        channels = extract_gpt_oss_channels(cleaned)
        if channels:
            if phase == "function":
                final_message = extract_gpt_oss_final_message(cleaned)
                if final_message:
                    cleaned = final_message
            else:
                parts = [entry["content"] for entry in channels if entry["content"].strip()]
                if parts:
                    cleaned = "\n\n".join(parts)
    cleaned = strip_think_blocks(cleaned)
    cleaned = re.sub(r"<\|[^>]+\|>", " ", cleaned)
    cleaned = cleaned.replace("/no_think", " ")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def prompt_for_cli_phase(model: ModelConfig, phase: str, prompt: str, cli_mode: str) -> str:
    if cli_mode != "raw":
        return prompt
    if phase == "function" and is_qwen3_model(model):
        return prompt.rstrip() + "\n/no_think"
    return prompt


def function_n_predict(model: ModelConfig, cli_mode: str) -> int:
    if cli_mode == "raw" and is_qwen3_model(model):
        # Raw Qwen3 often emits a long <think> block before the final answer.
        # Keep the smoke test in the answer region instead of truncating inside <think>.
        return 512
    return 96


def classify_reasoning_text(text: str, final_correct: bool) -> str:
    lower = text.lower()
    if not text.strip():
        return "empty"
    if "<|channel|>" in text or "<|message|>" in text:
        return "chaotic"
    if "2018-200-200-200" in text or "der res is a" in lower:
        return "chaotic"
    if re.search(r"(.)\1{24,}", text):
        return "chaotic"
    if final_correct:
        return "coherent"
    if len(re.findall(r"[A-Za-z]{3,}", text)) >= 30:
        return "structured_wrong"
    return "chaotic"


def run_command(cmd: list[str], log_path: Path, timeout_s: int, env: dict[str, str] | None = None) -> tuple[int, str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=merged_env,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        encoding="utf-8",
        errors="replace",
    )
    text = proc.stdout
    if proc.stderr:
        if text and not text.endswith("\n"):
            text += "\n"
        text += proc.stderr
    log_path.write_text(text, encoding="utf-8", errors="replace")
    return proc.returncode, text


def wait_for_server(port: int, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{port}/v1/models"
    last_error = "unknown"
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return
            last_error = f"status {response.status_code}: {response.text[:400]}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(1)
    raise RuntimeError(f"server did not become ready: {last_error}")


def terminate_process(proc: subprocess.Popen[str] | subprocess.Popen[bytes]) -> int | None:
    if proc.poll() is not None:
        return proc.returncode
    proc.terminate()
    try:
        return proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        return proc.wait(timeout=30)


def start_server(model: ModelConfig, cache_type: CacheTypeConfig, ctx_size: int, port: int, log_path: Path) -> tuple[subprocess.Popen[str], object]:
    log_handle = log_path.open("w", encoding="utf-8", errors="replace")
    cmd = [
        str(BUILD_BIN / "llama-server.exe"),
        "-m", model.model_path,
        "-ngl", "999",
        "-c", str(ctx_size),
        "-fa", "on",
        "-ctk", cache_type.ctk,
        "-ctv", cache_type.ctv,
        "-np", "1",
        "--threads-http", "4",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--alias", SERVER_ALIAS,
        "--reasoning", "off",
        "--reasoning-budget", "0",
        "--chat-template-kwargs", "{\"enable_thinking\":false}",
        "--no-webui",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc, log_handle


def generate_needle_samples(length_cfg: NeedleLengthConfig) -> list[dict]:
    from opencompass.datasets.needlebench.origin import NeedleBenchOriginDataset

    samples: list[dict] = []
    dataset_specs = [
        ("English", "PaulGrahamEssays.jsonl", length_cfg.en_length_buffer),
    ]
    for language, file_name, length_buffer in dataset_specs:
        for depth in NEEDLE_DEPTHS:
            dataset = NeedleBenchOriginDataset.load(
                path="opencompass/needlebench",
                length=length_cfg.ctx_size,
                depth=depth,
                tokenizer_model="gpt-4",
                file_list=[file_name],
                num_repeats_per_file=1,
                length_buffer=length_buffer,
                guide=True,
                language=language,
                needle_file_name="needles.jsonl",
            )
            if len(dataset) != 1:
                raise RuntimeError(f"unexpected dataset size for {length_cfg.name} {language} depth={depth}: {len(dataset)}")
            row = dataset[0]
            samples.append({
                "language": language,
                "depth": depth,
                "prompt": row["prompt"],
                "answer": row["answer"],
            })
    return samples


def request_server_completion(port: int, prompt: str, sampling: dict[str, float | int], max_tokens: int, system_prompt: str | None = None) -> str:
    token_budgets = [max_tokens]
    if max_tokens < NEEDLE_RETRY_MAX_TOKENS:
        token_budgets.append(NEEDLE_RETRY_MAX_TOKENS)

    last_text = ""
    for token_budget in token_budgets:
        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        payload = {
            "model": SERVER_ALIAS,
            "messages": messages,
            "max_tokens": token_budget,
            "temperature": sampling["temperature"],
            "top_p": sampling["top_p"],
            "seed": 123,
            "stream": False,
            "cache_prompt": False,
            "reasoning_format": "none",
            "thinking_budget_tokens": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        payload["top_k"] = sampling["top_k"]
        payload["min_p"] = sampling["min_p"]

        response = requests.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json=payload,
            timeout=(30, 3600),
        )
        response.raise_for_status()
        body = response.json()
        choice = body["choices"][0]
        message = choice["message"]
        content = message.get("content") or ""
        reasoning_content = message.get("reasoning_content") or ""
        last_text = normalize_needle_prediction(content if content else reasoning_content)
        if choice.get("finish_reason") != "length":
            return last_text

    return last_text


def token_count(model_path: str, text: str, outdir: Path, tag: str) -> int | None:
    text_path = outdir / f"{tag}.txt"
    token_log = outdir / f"{tag}_tokenize.log"
    text_path.write_text(text, encoding="utf-8", errors="replace")
    cmd = [
        str(BUILD_BIN / "llama-tokenize.exe"),
        "--model", model_path,
        "--file", str(text_path),
        "--show-count",
        "--log-disable",
    ]
    code, output = run_command(cmd, token_log, timeout_s=1800)
    if code != 0:
        return None
    match = re.search(r"(\d+)\s*$", output.strip())
    if match:
        return int(match.group(1))
    return None


def common_cli_args(
    model: ModelConfig,
    cache_type: CacheTypeConfig,
    ctx: int,
    n_predict: int,
    prompt: str,
    *,
    phase: str,
    cli_mode: str,
) -> list[str]:
    cmd = [
        str(BUILD_BIN / "llama-cli.exe"),
        "-m", model.model_path,
        "-ngl", "999",
        "-c", str(ctx),
        "-n", str(n_predict),
        "-p", prompt_for_cli_phase(model, phase, prompt, cli_mode),
        "-fa", "on",
        "--seed", "123",
        "-ctk", cache_type.ctk,
        "-ctv", cache_type.ctv,
        "-st",
        "--simple-io",
        *model.sampling_args,
    ]
    if cli_mode == "raw":
        cmd.append("--no-jinja")
    return cmd


def run_function_phase(outdir: Path, cli_mode: str) -> list[dict]:
    results: list[dict] = []
    for model in MODELS:
        for cache_type in TYPES:
            for question in FUNCTION_QUESTIONS:
                tag = f"func_{model.dim.lower()}_{cache_type.name}_{question['id']}"
                log_path = outdir / f"{tag}.log"
                cmd = common_cli_args(
                    model,
                    cache_type,
                    model.func_ctx,
                    function_n_predict(model, cli_mode),
                    question["prompt"],
                    phase="function",
                    cli_mode=cli_mode,
                )
                code, text = run_command(cmd, log_path, timeout_s=1800)
                generated = extract_generation_text(text)
                normalized = normalize_raw_cli_output(model, generated, "function") if cli_mode == "raw" else generated
                answer_ok = bool(question["expect"].search(normalized))
                raw_tags = "<|channel|>" in generated or "<|message|>" in generated
                chaotic = classify_reasoning_text(normalized, answer_ok) == "chaotic"
                results.append({
                    "phase": "function",
                    "dim": model.dim,
                    "type": cache_type.name,
                    "question_id": question["id"],
                    "exit_code": code,
                    "cli_mode": cli_mode,
                    "answer_ok": answer_ok,
                    "raw_tags": raw_tags,
                    "chaotic": chaotic,
                    "pass": code == 0 and answer_ok and not chaotic,
                    "log": str(log_path),
                    "output_excerpt": generated[:600],
                    "normalized_output_excerpt": normalized[:600],
                })
                print(f"[function] {model.dim} {cache_type.name} {question['id']} exit={code} pass={code == 0 and answer_ok and not chaotic}", flush=True)
    return results


def run_bench_phase(outdir: Path) -> list[dict]:
    results: list[dict] = []
    for model in MODELS:
        for cache_type in TYPES:
            tag = f"bench_{model.dim.lower()}_{cache_type.name}"
            log_path = outdir / f"{tag}.json"
            cmd = [
                str(BUILD_BIN / "llama-bench.exe"),
                "-m", model.model_path,
                "-ngl", "999",
                "-fa", "1",
                "-ctk", cache_type.ctk,
                "-ctv", cache_type.ctv,
                "-p", "512",
                "-n", "128",
                "-pg", "32768,256",
                "-r", "3",
                "-o", "json",
            ]
            code, text = run_command(cmd, log_path, timeout_s=5400)
            bench_rows = []
            if code == 0:
                bench_rows = extract_json_array(text)
            record = {
                "phase": "bench",
                "dim": model.dim,
                "type": cache_type.name,
                "exit_code": code,
                "log": str(log_path),
                "pp512_tps": None,
                "tg128_tps": None,
                "pg32768_256_tps": None,
            }
            for row in bench_rows:
                if row["n_prompt"] == 512 and row["n_gen"] == 0:
                    record["pp512_tps"] = row["avg_ts"]
                elif row["n_prompt"] == 0 and row["n_gen"] == 128:
                    record["tg128_tps"] = row["avg_ts"]
                elif row["n_prompt"] == 32768 and row["n_gen"] == 256:
                    record["pg32768_256_tps"] = row["avg_ts"]
            results.append(record)
            print(f"[bench] {model.dim} {cache_type.name} exit={code}", flush=True)
    return results


def run_ppl_phase(outdir: Path) -> list[dict]:
    results: list[dict] = []
    for model in MODELS:
        for cache_type in TYPES:
            tag = f"ppl_{model.dim.lower()}_{cache_type.name}"
            log_path = outdir / f"{tag}.log"
            cmd = [
                str(BUILD_BIN / "llama-perplexity.exe"),
                "-m", model.model_path,
                "-f", str(WIKI_TEST_RAW),
                "-ngl", "999",
                "-fa", "on",
                "-ctk", cache_type.ctk,
                "-ctv", cache_type.ctv,
                "-c", "512",
            ]
            code, text = run_command(cmd, log_path, timeout_s=14400)
            ppl = None
            match = re.search(r"Final estimate: PPL = ([0-9.]+)", text)
            if match:
                ppl = float(match.group(1))
            results.append({
                "phase": "ppl",
                "dim": model.dim,
                "type": cache_type.name,
                "exit_code": code,
                "ppl": ppl,
                "log": str(log_path),
            })
            print(f"[ppl] {model.dim} {cache_type.name} exit={code} ppl={ppl}", flush=True)
    return results


def run_aime_phase(outdir: Path, cli_mode: str) -> list[dict]:
    results: list[dict] = []
    for model in MODELS:
        for cache_type in TYPES:
            tag = f"aime_{model.dim.lower()}_{cache_type.name}"
            log_path = outdir / f"{tag}.log"
            cmd = common_cli_args(
                model,
                cache_type,
                model.aime_ctx,
                60000,
                AIME_PROMPT,
                phase="aime",
                cli_mode=cli_mode,
            )
            code, text = run_command(cmd, log_path, timeout_s=10800)
            generated = extract_generation_text(text)
            normalized = normalize_raw_cli_output(model, generated, "aime") if cli_mode == "raw" else generated
            answer_match = re.findall(r"Final answer:\s*([0-9]+)", normalized, re.IGNORECASE)
            final_answer = answer_match[-1] if answer_match else None
            final_correct = final_answer == "277"
            coherence = classify_reasoning_text(normalized, final_correct)
            token_len = token_count(model.model_path, normalized, outdir, tag)
            results.append({
                "phase": "aime",
                "dim": model.dim,
                "type": cache_type.name,
                "exit_code": code,
                "cli_mode": cli_mode,
                "final_answer": final_answer,
                "pass": code == 0 and final_correct,
                "coherence": coherence,
                "token_count": token_len,
                "log": str(log_path),
                "output_excerpt": generated[:2000],
                "normalized_output_excerpt": normalized[:2000],
            })
            print(
                f"[aime] {model.dim} {cache_type.name} exit={code} answer={final_answer} pass={code == 0 and final_correct} coherence={coherence} tokens={token_len}",
                flush=True,
            )
    return results


def run_needle_phase(outdir: Path) -> list[dict]:
    from opencompass.datasets.needlebench.origin import NeedleBenchOriginEvaluator

    results: list[dict] = []
    case_index = 0
    max_ctx_size = max(length_cfg.ctx_size for length_cfg in NEEDLE_LENGTHS)
    if max_ctx_size == 128000:
        max_ctx_size = 131072
    needle_samples = {length_cfg.name: generate_needle_samples(length_cfg) for length_cfg in NEEDLE_LENGTHS}
    for model in MODELS:
        sampling = parse_sampling_args(model.sampling_args)
        for cache_type in TYPES:
            case_index += 1
            port = 18080 + case_index
            server_log_path = outdir / f"needle_server_{model.dim.lower()}_{cache_type.name}.log"
            proc = None
            log_handle = None
            try:
                proc, log_handle = start_server(model, cache_type, max_ctx_size, port, server_log_path)
                wait_for_server(port, timeout_s=1800)
                for length_cfg in NEEDLE_LENGTHS:
                    tag = f"needle_{length_cfg.name.lower()}_{model.dim.lower()}_{cache_type.name}"
                    detail_path = outdir / f"{tag}.json"
                    samples = needle_samples[length_cfg.name]
                    predictions: list[str] = []
                    per_sample: list[dict] = []
                    try:
                        for sample in samples:
                            prediction = request_server_completion(
                                port,
                                needle_prompt_for_model(model, sample["prompt"]),
                                sampling,
                                max_tokens=NEEDLE_MAX_TOKENS,
                                system_prompt=needle_system_prompt_for_model(model),
                            )
                            predictions.append(prediction)
                            per_sample.append({
                                "language": sample["language"],
                                "depth": sample["depth"],
                                "answer": sample["answer"],
                                "prediction": prediction,
                            })
                        evaluator = NeedleBenchOriginEvaluator()
                        with contextlib.redirect_stdout(io.StringIO()):
                            overall = evaluator.score(predictions, [sample["answer"] for sample in samples])
                        en_pairs = [(predictions[i], samples[i]["answer"]) for i in range(len(samples)) if samples[i]["language"] == "English"]
                        zh_pairs = [(predictions[i], samples[i]["answer"]) for i in range(len(samples)) if samples[i]["language"] == "Chinese"]
                        with contextlib.redirect_stdout(io.StringIO()):
                            en_score = evaluator.score([pred for pred, _ in en_pairs], [gold for _, gold in en_pairs])["score"]
                            zh_score = evaluator.score([pred for pred, _ in zh_pairs], [gold for _, gold in zh_pairs])["score"] if zh_pairs else None
                        result = {
                            "phase": "needle",
                            "dim": model.dim,
                            "type": cache_type.name,
                            "needle_size": length_cfg.name,
                            "exit_code": 0,
                            "needle_score": round(overall["score"], 4),
                            "needle_score_en": round(en_score, 4) if en_score is not None else None,
                            "needle_score_zh": round(zh_score, 4) if zh_score is not None else None,
                            "n_samples": len(samples),
                            "log": str(detail_path),
                            "server_log": str(server_log_path),
                        }
                        detail_path.write_text(json.dumps({
                            "result": result,
                            "details": overall["details"],
                            "samples": per_sample,
                        }, ensure_ascii=False, indent=2), encoding="utf-8")
                        results.append(result)
                        print(f"[needle] {length_cfg.name} {model.dim} {cache_type.name} score={result['needle_score']:.4f}", flush=True)
                    except Exception as exc:
                        exit_code = proc.poll() if proc is not None else 1
                        if exit_code is None:
                            exit_code = 1
                        error = str(exc)
                        if proc is not None and proc.poll() is not None:
                            exit_code = proc.returncode
                        detail_path.write_text(json.dumps({
                            "phase": "needle",
                            "dim": model.dim,
                            "type": cache_type.name,
                            "needle_size": length_cfg.name,
                            "error": error,
                            "exit_code": exit_code,
                            "samples": per_sample,
                        }, ensure_ascii=False, indent=2), encoding="utf-8")
                        results.append({
                            "phase": "needle",
                            "dim": model.dim,
                            "type": cache_type.name,
                            "needle_size": length_cfg.name,
                            "exit_code": exit_code,
                            "needle_score": None,
                            "needle_score_en": None,
                            "needle_score_zh": None,
                            "n_samples": len(samples),
                            "log": str(detail_path),
                            "server_log": str(server_log_path),
                            "error": error,
                        })
                        print(f"[needle] {length_cfg.name} {model.dim} {cache_type.name} failed: {error}", flush=True)
                        if proc is not None and proc.poll() is not None:
                            break
            finally:
                if proc is not None:
                    terminate_process(proc)
                if log_handle is not None:
                    log_handle.close()
    return results


def compute_relatives(records: Iterable[dict], key: str, group_keys: tuple[str, ...] = ("dim",)) -> None:
    baselines: dict[tuple[object, ...], float] = {}
    for record in records:
        if record.get("type") == "q8_0" and record.get(key) is not None:
            group = tuple(record[group_key] for group_key in group_keys)
            baselines[group] = record[key]
    for record in records:
        group = tuple(record[group_key] for group_key in group_keys)
        base = baselines.get(group)
        value = record.get(key)
        rel_key = f"{key}_vs_q8_pct"
        record[rel_key] = None
        if base and value is not None:
            record[rel_key] = round(100.0 * value / base, 2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases", nargs="+", choices=["function", "bench", "ppl", "aime", "needle"], required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--cli-mode", choices=["templated", "raw"], default="raw")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list[dict]] = {}
    if "function" in args.phases:
        all_results["function"] = run_function_phase(outdir, args.cli_mode)
        json_dump(outdir / "function_results.json", all_results["function"])
    if "bench" in args.phases:
        all_results["bench"] = run_bench_phase(outdir)
        for metric in ("pp512_tps", "tg128_tps", "pg32768_256_tps"):
            compute_relatives(all_results["bench"], metric)
        json_dump(outdir / "bench_results.json", all_results["bench"])
    if "ppl" in args.phases:
        all_results["ppl"] = run_ppl_phase(outdir)
        compute_relatives(all_results["ppl"], "ppl")
        json_dump(outdir / "ppl_results.json", all_results["ppl"])
    if "aime" in args.phases:
        all_results["aime"] = run_aime_phase(outdir, args.cli_mode)
        compute_relatives(all_results["aime"], "token_count")
        json_dump(outdir / "aime_results.json", all_results["aime"])
    if "needle" in args.phases:
        all_results["needle"] = run_needle_phase(outdir)
        compute_relatives(all_results["needle"], "needle_score", ("dim", "needle_size"))
        json_dump(outdir / "needle_results.json", all_results["needle"])

    summary_path = outdir / "summary.json"
    json_dump(summary_path, all_results)
    print(f"SUMMARY={summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
