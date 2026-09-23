"""Ownership and dependency contracts for visualization helpers."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "rovibrational_excitation"
VISUALIZATION = PACKAGE / "visualization"
STRICT_RESULT_READER = (
    "src/rovibrational_excitation/visualization/result_data.py",
    "rovibrational_excitation.io.result_schema",
)
LEGACY_PLOTS = PACKAGE / "plots"


def test_plotting_helpers_are_owned_by_visualization():
    import rovibrational_excitation as rve
    from rovibrational_excitation.visualization.plot_all import plot_all
    from rovibrational_excitation.visualization.plot_electric_field import (
        plot_electric_field,
    )
    from rovibrational_excitation.visualization.plot_electric_field_vector import (
        plot_electric_vector,
    )
    from rovibrational_excitation.visualization.plot_population import plot_population
    from rovibrational_excitation.visualization.spectrogram import spectrogram_fast

    assert (VISUALIZATION / "__init__.py").is_file()
    assert (VISUALIZATION / "result_data.py").is_file()
    assert (VISUALIZATION / "plot_all.py").is_file()
    assert (VISUALIZATION / "plot_electric_field.py").is_file()
    assert (VISUALIZATION / "plot_electric_field_vector.py").is_file()
    assert (VISUALIZATION / "plot_population.py").is_file()
    assert (VISUALIZATION / "spectrogram.py").is_file()
    assert not LEGACY_PLOTS.exists()
    assert rve.visualization.__name__ == "rovibrational_excitation.visualization"
    assert plot_all.__module__ == "rovibrational_excitation.visualization.plot_all"
    assert plot_electric_field.__module__ == (
        "rovibrational_excitation.visualization.plot_electric_field"
    )
    assert plot_electric_vector.__module__ == (
        "rovibrational_excitation.visualization.plot_electric_field_vector"
    )
    assert plot_population.__module__ == (
        "rovibrational_excitation.visualization.plot_population"
    )
    assert spectrogram_fast.__module__ == (
        "rovibrational_excitation.visualization.spectrogram"
    )


def test_visualization_does_not_depend_on_application_workflows():
    forbidden = {
        "cli",
        "io",
        "models",
        "optimization",
        "simulation",
        "spectroscopy",
    }
    violations: list[str] = []

    for path in sorted(VISUALIZATION.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
                level = 0
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                modules = [node.module]
                level = node.level
            else:
                continue
            for module in modules:
                parts = module.split(".")
                if module == "rovibrational_excitation":
                    violations.append(f"{path.relative_to(ROOT)} imports package root")
                elif parts[0] == "rovibrational_excitation" and len(parts) > 1:
                    if parts[1] in forbidden:
                        dependency = (str(path.relative_to(ROOT)), module)
                        if dependency == STRICT_RESULT_READER:
                            continue
                        violations.append(f"{path.relative_to(ROOT)} imports {module}")
                elif level > 1 and parts[0] in forbidden:
                    violations.append(f"{path.relative_to(ROOT)} imports {module}")

    assert violations == []


def test_root_import_keeps_optional_matplotlib_lazy():
    environment = {**os.environ, "PYTHONPATH": str(PACKAGE.parent)}
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import rovibrational_excitation as rve; "
                "assert rve.visualization.__name__ == "
                "'rovibrational_excitation.visualization'; "
                "assert 'matplotlib' not in sys.modules; "
                "assert 'matplotlib.pyplot' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
