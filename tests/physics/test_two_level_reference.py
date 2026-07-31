"""Independent TwoLevel references for the Phase 0 physics baseline."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from rovibrational_excitation.core.basis import TwoLevelBasis
from rovibrational_excitation.core.electric_field import ElectricField
from rovibrational_excitation.core.propagation import (
    LiouvillePropagator,
    SchrodingerPropagator,
)
from rovibrational_excitation.dipole import TwoLevelDipoleMatrix

pytestmark = pytest.mark.physics

GAP_RAD_PER_FS = 0.37
DIPOLE_C_M = 2.0e-29
FIELD_DT_FS = 1.0e-3
FINAL_TIME_FS = 0.2
FIELD_V_PER_M = 5.0e8


def _flat_envelope(
    time_fs: np.ndarray, center_fs: float, duration_fs: float
) -> np.ndarray:
    """Return an exactly constant envelope; center and duration are irrelevant."""
    del center_fs, duration_fs
    return np.ones_like(time_fs)


def _two_level_system(*, backend: str = "numpy"):
    basis = TwoLevelBasis(
        energy_gap=GAP_RAD_PER_FS,
        input_units="rad/fs",
        output_units="rad/fs",
    )
    hamiltonian = basis.generate_H0()
    dipole = TwoLevelDipoleMatrix(
        basis,
        mu0=DIPOLE_C_M,
        backend=backend,
        units="C*m",
        units_input="C*m",
    )
    return hamiltonian, dipole


def _constant_field(
    amplitude: float,
    *,
    polarization: np.ndarray | None = None,
) -> ElectricField:
    if polarization is None:
        polarization = np.array([1.0, 0.0])
    intervals = round(FINAL_TIME_FS / FIELD_DT_FS)
    time_fs = np.linspace(0.0, FINAL_TIME_FS, intervals + 1)
    field = ElectricField(time_fs)
    field.add_dispersed_Efield(
        _flat_envelope,
        duration=1.0,
        t_center=0.0,
        carrier_freq=0.0,
        amplitude=amplitude,
        polarization=polarization,
        const_polarisation=True,
    )
    return field


def _pure_propagation(
    field: ElectricField,
    initial_state: np.ndarray,
    *,
    nondimensional: bool = False,
    sparse: bool = False,
    backend: str = "numpy",
    dipole_backend: str = "numpy",
    return_traj: bool = True,
):
    hamiltonian, dipole = _two_level_system(backend=dipole_backend)
    return SchrodingerPropagator(
        backend=backend,
        validate_units=False,
        sparse=sparse,
    ).propagate(
        hamiltonian,
        field,
        dipole,
        initial_state,
        coupling_mode="scalar",
        coupling_axis="x",
        return_traj=return_traj,
        return_time_psi=True,
        nondimensional=nondimensional,
        sparse=sparse,
    )


def test_free_superposition_matches_analytic_phase():
    """For E=0, psi_n(t)=exp(-i E_n t) psi_n(0), with energies in rad/fs."""
    initial = np.array([np.sqrt(0.3), np.sqrt(0.7) * np.exp(0.23j)])
    time_fs, trajectory = _pure_propagation(_constant_field(0.0), initial)

    elapsed_fs = time_fs - time_fs[0]
    energies_rad_per_fs = np.array([0.0, GAP_RAD_PER_FS])
    expected = initial[None, :] * np.exp(
        -1j * elapsed_fs[:, None] * energies_rad_per_fs[None, :]
    )

    np.testing.assert_allclose(time_fs[[0, -1]], [0.0, FINAL_TIME_FS])
    np.testing.assert_allclose(trajectory, expected, rtol=0.0, atol=2.0e-14)


def test_density_evolution_is_outer_product_of_pure_evolution():
    """Unitary Liouville evolution must equal |psi(t)><psi(t)|."""
    initial = np.array([np.sqrt(0.4), np.sqrt(0.6) * np.exp(-0.31j)])
    field = _constant_field(FIELD_V_PER_M)
    hamiltonian, dipole = _two_level_system()

    time_psi, psi = _pure_propagation(field, initial)
    time_rho, rho = LiouvillePropagator(validate_units=False).propagate(
        hamiltonian,
        field,
        dipole,
        np.outer(initial, initial.conj()),
        coupling_mode="scalar",
        coupling_axis="x",
        return_traj=True,
        return_time_rho=True,
    )
    expected_rho = np.einsum("ti,tj->tij", psi, psi.conj())

    np.testing.assert_array_equal(time_rho, time_psi)
    np.testing.assert_allclose(rho, expected_rho, rtol=0.0, atol=3.0e-13)


def test_constant_drive_rk4_matches_matrix_exponential_and_minus_mu_e_sign():
    """A constant field has exact U(T)=exp[-i(H0-mu E)T]."""
    initial = np.array([1.0, 0.0], dtype=np.complex128)
    field = _constant_field(FIELD_V_PER_M)
    hamiltonian, dipole = _two_level_system()

    time_fs, trajectory = _pure_propagation(field, initial)
    h0 = hamiltonian.get_matrix("rad/fs")
    mu_x = dipole.get_mu_in_units("x", "rad/fs/(V/m)")
    effective_hamiltonian = h0 - mu_x * FIELD_V_PER_M
    expected_final = expm(-1j * effective_hamiltonian * FINAL_TIME_FS) @ initial

    assert time_fs[-1] == pytest.approx(FINAL_TIME_FS)
    np.testing.assert_allclose(trajectory[-1], expected_final, rtol=0.0, atol=2.0e-13)


@pytest.mark.parametrize(
    "polarization",
    [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([1.0, 1.0]) / np.sqrt(2.0),
        np.array([1.0, 1.0j]) / np.sqrt(2.0),
    ],
)
def test_scalar_workflow_is_independent_of_polarization(polarization):
    """TwoLevel scalar coupling ignores the laboratory polarization direction."""
    initial = np.array([1.0, 0.0], dtype=np.complex128)
    reference_time, reference = _pure_propagation(
        _constant_field(FIELD_V_PER_M), initial
    )
    time_fs, trajectory = _pure_propagation(
        _constant_field(FIELD_V_PER_M, polarization=polarization), initial
    )

    np.testing.assert_array_equal(time_fs, reference_time)
    np.testing.assert_allclose(trajectory, reference, rtol=0.0, atol=2.0e-14)


def test_dimensional_and_nondimensional_population_and_time_agree():
    """Scaling may change amplitudes and phase units, not physical time/population."""
    initial = np.array([np.sqrt(0.65), np.sqrt(0.35) * np.exp(0.17j)])
    field = _constant_field(FIELD_V_PER_M)

    time_dim, psi_dim = _pure_propagation(field, initial, nondimensional=False)
    time_nd, psi_nd = _pure_propagation(field, initial, nondimensional=True)

    np.testing.assert_allclose(time_nd, time_dim, rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(
        np.abs(psi_nd) ** 2,
        np.abs(psi_dim) ** 2,
        rtol=0.0,
        atol=2.0e-12,
    )


def test_dense_and_sparse_numpy_paths_agree():
    """Dense and CSR application paths must implement the same Hamiltonian."""
    initial = np.array([np.sqrt(0.2), np.sqrt(0.8) * np.exp(-0.4j)])
    field = _constant_field(FIELD_V_PER_M)

    time_dense, psi_dense = _pure_propagation(field, initial, sparse=False)
    time_sparse, psi_sparse = _pure_propagation(field, initial, sparse=True)

    np.testing.assert_array_equal(time_sparse, time_dense)
    np.testing.assert_allclose(psi_sparse, psi_dense, rtol=0.0, atol=2.0e-14)


def _has_cuda_device() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.mark.gpu
@pytest.mark.skipif(not _has_cuda_device(), reason="real CuPy/CUDA device unavailable")
def test_numpy_and_cupy_final_state_agree():
    """The field, dipole, and propagator backend must stay aligned on GPU."""
    initial = np.array([np.sqrt(0.2), np.sqrt(0.8) * np.exp(-0.4j)])
    field = _constant_field(FIELD_V_PER_M)

    _, numpy_final = _pure_propagation(field, initial, return_traj=False)
    _, cupy_final = _pure_propagation(
        field,
        initial,
        backend="cupy",
        dipole_backend="cupy",
        return_traj=False,
    )

    assert cupy_final.shape == numpy_final.shape == (2,)
    np.testing.assert_allclose(cupy_final, numpy_final, rtol=2.0e-12, atol=2.0e-13)
