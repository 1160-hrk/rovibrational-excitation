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
        ["--run", "examples/example_rovibrational_excitation.py", "--quick"],
        ["--run", "examples/example_twolevel_2d_map.py", "--quick"],
        ["--run", "examples/example_nondimensional_propagation.py", "--quick"],
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


