#!/usr/bin/env python
"""Execute every supported example and the public parameter template."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def run(command: list[str], *, env: dict[str, str]) -> int:
    print("$", " ".join(command), flush=True)
    return subprocess.call(command, cwd=ROOT, env=env)


def main() -> int:
    env = os.environ.copy()
    env["EXAMPLES_QUICK"] = "1"
    existing_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{SRC}{os.pathsep}{existing_path}" if existing_path else str(SRC)
    )

    launcher = [sys.executable, str(ROOT / "examples" / "launcher.py")]
    commands = [
        launcher + ["--run", "example_typed_twolevel", "--quick"],
        launcher + ["--run", "example_typed_spectral_modulation", "--quick"],
        launcher + ["--run", "example_external_scalar_field", "--quick"],
        [
            sys.executable,
            "-m",
            "rovibrational_excitation.cli.simulate",
            "examples/params_template.py",
            "--no-save",
        ],
    ]

    failed = sum(run(command, env=env) != 0 for command in commands)
    if failed:
        print(f"Smoke tests finished with {failed} failure(s)")
        return 1
    print("Smoke tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
