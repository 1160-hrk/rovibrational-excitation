"""Input validation for optimization workflows."""

import pytest

from rovibrational_excitation.optimization.krotov import run_krotov_optimization


def test_krotov_requires_explicit_positive_initial_pulse_duration():
    with pytest.raises(
        ValueError, match="duration_initial must be finite and positive"
    ):
        run_krotov_optimization(
            basis=None,
            hamiltonian=None,
            dipole=None,
            states={},
            time_cfg={},
            params={},
        )
