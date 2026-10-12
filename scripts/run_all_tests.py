#!/usr/bin/env python3
"""Single entry point for Ron's repository-wide regression and runtime checks.

Run with: python scripts/run_all_tests.py
The script stops on the first failing command so training cannot continue past a
failed regression gate.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(label: str, *command: str) -> None:
    print(f"\n{'=' * 16} {label} {'=' * 16}", flush=True)
    print("$ " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode:
        raise SystemExit(result.returncode)


def main() -> None:
    required = [
        "README.md",
        "docs/architecture.md",
        "index.html",
        "app.js",
        "assets/style.css",
        "ron",
        "tests",
    ]
    print("Validating repository structure", flush=True)
    missing = [item for item in required if not (ROOT / item).exists()]
    if missing:
        raise SystemExit("Missing required repository paths: " + ", ".join(missing))

    run("Python unit, integration, training, evaluation and quality tests",
        sys.executable, "-m", "pytest")

    if not shutil.which("node"):
        raise SystemExit("Node.js is required for Ron's JavaScript/runtime tests.")

    for path in ("app.js", "ron-core.js", "ron-agent-core.js", "ron-browser-reasoner.js"):
        run(f"JavaScript syntax: {path}", "node", "--check", path)

    run("Ron core tests", "node", "--test", "ron-core.test.js",
        "tests/test_core_browser_runtime.mjs")
    run("Browser and reasoner runtime tests", "node", "--test",
        "tests/test_frontend_runtime.mjs", "tests/test_reasoner_runtime.mjs")

    print("\nALL RON TEST GROUPS PASSED", flush=True)


if __name__ == "__main__":
    main()
