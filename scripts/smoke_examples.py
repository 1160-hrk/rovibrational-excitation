#!/usr/bin/env python
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> int:
    print("$", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> int:
    env = os.environ.copy()
    env["EXAMPLES_QUICK"] = "1"

    # Use launcher to run a few representative examples in quick mode
    launcher = [sys.executable, str(ROOT / "examples" / "launcher.py")]

    cases = [
        ["--run", "example_typed_twolevel", "--quick"],
        ["--run", "example_typed_spectral_modulation", "--quick"],
        ["--run", "example_external_scalar_field", "--quick"],
    ]

    failed = 0
    for args in cases:
        code = subprocess.call(launcher + args, env=env)
        if code != 0:
            failed += 1

    if failed:
        print(f"Smoke tests finished with {failed} failure(s)")
        return 1
    print("Smoke tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
