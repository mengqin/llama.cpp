#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


RUNNER_PATH = Path(__file__).with_name("00_run_pqtq_suite.py")


def main() -> int:
    spec = importlib.util.spec_from_file_location("pqtq_suite_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load runner: {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.main(["--phases", "aime", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
