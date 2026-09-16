"""Ownership of vibration kernels shared by LinMol and VibLadder."""

from pathlib import Path

from rovibrational_excitation.models.linear_molecule import dipole_builder
from rovibrational_excitation.models.vib_ladder import dipole as vib_ladder_dipole
from rovibrational_excitation.models.vibration import (
    omega01_domega_to_N,
    tdm_vib_harm,
    tdm_vib_morse,
    validate_morse_v_max,
)

PACKAGE = Path(__file__).resolve().parents[2] / "src/rovibrational_excitation"


def test_vibration_kernels_have_one_shared_model_owner() -> None:
    assert tdm_vib_harm.__module__ == (
        "rovibrational_excitation.models.vibration.harmonic"
    )
    for function in (omega01_domega_to_N, tdm_vib_morse, validate_morse_v_max):
        assert function.__module__ == "rovibrational_excitation.models.vibration.morse"
    shared_functions = {
        "tdm_vib_harm": tdm_vib_harm,
        "tdm_vib_morse": tdm_vib_morse,
        "omega01_domega_to_N": omega01_domega_to_N,
        "validate_morse_v_max": validate_morse_v_max,
    }
    for function_name, shared in shared_functions.items():
        assert getattr(dipole_builder, function_name) is shared
        assert getattr(vib_ladder_dipole, function_name) is shared
    for old_file in ("__init__.py", "harmonic.py", "morse.py"):
        assert not (PACKAGE / "dipole/vib" / old_file).exists()
