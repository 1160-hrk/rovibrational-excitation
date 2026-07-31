"""Independent VibLadder references for the Phase 0 physics baseline.

Energies are angular frequencies in rad/fs. The fixed tolerance of 2e-15 is
near float64 roundoff for the O(1) analytic values used below.
"""

from __future__ import annotations

import numpy as np
import pytest

from rovibrational_excitation.core.basis import VibLadderBasis

pytestmark = pytest.mark.physics

OMEGA01_RAD_PER_FS = 1.2
ANHARMONIC_SHIFT_RAD_PER_FS = 0.08
V_MAX = 4
ENERGY_ATOL = 2.0e-15


def _expected_energies(
    levels: np.ndarray,
    omega01: float,
    anharmonic_shift: float,
) -> np.ndarray:
    vterm = levels + 0.5
    return (omega01 + anharmonic_shift) * vterm - (anharmonic_shift / 2) * vterm**2


def test_harmonic_energies_and_adjacent_spacings():
    """For delta=0, E_v=omega01(v+1/2) and every spacing is omega01."""
    basis = VibLadderBasis(
        V_max=V_MAX,
        omega=OMEGA01_RAD_PER_FS,
        delta_omega=0.0,
        input_units="rad/fs",
        output_units="rad/fs",
    )
    energies = np.diag(basis.generate_H0().matrix)
    levels = np.arange(V_MAX + 1)

    np.testing.assert_allclose(
        energies,
        OMEGA01_RAD_PER_FS * (levels + 0.5),
        rtol=0.0,
        atol=ENERGY_ATOL,
    )
    np.testing.assert_allclose(
        np.diff(energies),
        OMEGA01_RAD_PER_FS,
        rtol=0.0,
        atol=ENERGY_ATOL,
    )


def test_anharmonic_energies_and_spacings_follow_omega01_convention():
    """E_v=(omega01+delta)x-delta*x^2/2 gives dE_v=omega01-v*delta."""
    basis = VibLadderBasis(
        V_max=V_MAX,
        omega=OMEGA01_RAD_PER_FS,
        delta_omega=ANHARMONIC_SHIFT_RAD_PER_FS,
        input_units="rad/fs",
        output_units="rad/fs",
    )
    energies = np.diag(basis.generate_H0().matrix)
    levels = np.arange(V_MAX + 1)

    np.testing.assert_allclose(
        energies,
        _expected_energies(
            levels,
            OMEGA01_RAD_PER_FS,
            ANHARMONIC_SHIFT_RAD_PER_FS,
        ),
        rtol=0.0,
        atol=ENERGY_ATOL,
    )
    np.testing.assert_allclose(
        np.diff(energies),
        OMEGA01_RAD_PER_FS - levels[:-1] * ANHARMONIC_SHIFT_RAD_PER_FS,
        rtol=0.0,
        atol=ENERGY_ATOL,
    )


def test_stored_and_override_hamiltonian_paths_use_the_same_formula():
    """Stored and temporary parameters must not reinterpret omega01 or delta."""
    configured = VibLadderBasis(
        V_max=V_MAX,
        omega=OMEGA01_RAD_PER_FS,
        delta_omega=ANHARMONIC_SHIFT_RAD_PER_FS,
        input_units="rad/fs",
        output_units="rad/fs",
    )
    override_host = VibLadderBasis(
        V_max=V_MAX,
        omega=0.5,
        delta_omega=0.0,
        input_units="rad/fs",
        output_units="rad/fs",
    )

    stored = configured.generate_H0().matrix
    overridden = override_host.generate_H0_with_params(
        omega=OMEGA01_RAD_PER_FS,
        delta_omega=ANHARMONIC_SHIFT_RAD_PER_FS,
        input_units="rad/fs",
        units="rad/fs",
    ).matrix

    np.testing.assert_allclose(overridden, stored, rtol=0.0, atol=ENERGY_ATOL)
