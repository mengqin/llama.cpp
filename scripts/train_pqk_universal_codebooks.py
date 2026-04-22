#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


DIMENSION = 16
DEFAULT_SAMPLE_COUNT = 600000
DEFAULT_SEED = 1234
POSITIVE_LEVEL_COUNT = {
    2: 2,
    3: 4,
    4: 8,
}


def sample_spherical_coordinates(sample_count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((sample_count, DIMENSION), dtype=np.float64)
    u = math.sqrt(DIMENSION) * z / np.linalg.norm(z, axis=1, keepdims=True)
    return u.reshape(-1)


def lloyd_symmetric_positive(samples_abs: np.ndarray, n_levels: int, max_iter: int = 200, tol: float = 1e-10) -> np.ndarray:
    quantiles = np.linspace(0.5 / n_levels, 1.0 - 0.5 / n_levels, n_levels)
    levels = np.quantile(samples_abs, quantiles)
    levels = np.maximum.accumulate(levels)

    for _ in range(max_iter):
        thresholds = np.empty(n_levels + 1, dtype=np.float64)
        thresholds[0] = 0.0
        thresholds[-1] = np.inf
        thresholds[1:-1] = 0.5 * (levels[:-1] + levels[1:])

        updated = levels.copy()
        for index in range(n_levels):
            lo = thresholds[index]
            hi = thresholds[index + 1]
            bucket = samples_abs[(samples_abs >= lo) & (samples_abs < hi)]
            if bucket.size:
                updated[index] = bucket.mean()

        updated = np.maximum.accumulate(updated)
        if np.max(np.abs(updated - levels)) < tol:
            levels = updated
            break
        levels = updated

    return levels


def format_float_list(values: list[float]) -> str:
    return ", ".join(f"{value:.9f}" for value in values)


def emit_c_switch(bits: int, full_levels: list[float]) -> str:
    mask = hex((1 << bits) - 1)
    lines = [
        f"static GGML_PQK_HOST_DEVICE inline float ggml_pqk_centroid_{bits}bit(uint8_t q) {{",
        f"    switch (q & {mask}u) {{",
    ]
    for index, value in enumerate(full_levels):
        prefix = "default" if index == len(full_levels) - 1 else f"case {index}"
        lines.append(f"        {prefix}: return {value:.9f}f;")
    lines.extend([
        "    }",
        "}",
    ])
    return "\n".join(lines)


def build_payload(sample_count: int, seed: int) -> dict:
    coords = sample_spherical_coordinates(sample_count, seed)
    coords_abs = np.abs(coords)
    variance = float(coords.var())
    excess_kurtosis = float((coords**4).mean() / (variance * variance) - 3.0)

    payload = {
        "distribution": {
            "kind": "16d_sphere",
            "formula": "u = sqrt(16) * z / ||z||_2, z ~ N(0, I_16)",
            "dimension": DIMENSION,
        },
        "sample_count": sample_count,
        "seed": seed,
        "coordinate_stats": {
            "mean_abs": float(coords_abs.mean()),
            "variance": variance,
            "excess_kurtosis": excess_kurtosis,
        },
        "codebooks": {},
    }

    for bits, n_positive in POSITIVE_LEVEL_COUNT.items():
        positive = lloyd_symmetric_positive(coords_abs, n_positive)
        full = np.concatenate([-positive[::-1], positive])
        payload["codebooks"][f"PQ{bits}_K"] = {
            "bits": bits,
            "positive": [float(value) for value in positive],
            "full": [float(value) for value in full],
            "max_centroid": float(positive[-1]),
        }

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate universal single-family PQK codebooks from a 16D spherical distribution.")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLE_COUNT, help="number of 16D spherical samples")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="PRNG seed")
    parser.add_argument("--output-json", type=Path, help="optional JSON output path")
    parser.add_argument("--emit-c", action="store_true", help="print C centroid switches and max-centroid macros")
    args = parser.parse_args()

    payload = build_payload(args.samples, args.seed)

    if args.output_json is not None:
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if args.emit_c:
        for name, entry in payload["codebooks"].items():
            bits = int(entry["bits"])
            print(f"#define GGML_{name.replace('_K', 'K')}_MAX_CENTROID {entry['max_centroid']:.9f}f")
            print(emit_c_switch(bits, entry["full"]))
            print()
        return

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()