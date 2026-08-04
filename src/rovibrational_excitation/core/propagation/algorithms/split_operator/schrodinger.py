"""_splitop_schrodinger.py
=================================
Split‑Operator time propagator that mirrors the API of ``rk4_schrodinger.py``
but allows two execution back‑ends:

* **CuPy**  – for GPU acceleration (if ``cupy`` is available and the user passes
  ``backend='cupy'``).
* **NumPy + Numba** – CPU execution with an inner loop compiled by ``@njit`` when
  CuPy is not selected (or not installed).

Cartesian propagation uses the real field components directly.  For an
M-resolved linear molecule, rotations in the xy plane are applied through
diagonal M-dependent phases, so the interaction eigensystem is built once.
The optional helicity-projected model constructs a one-way transition
operator and adds its adjoint explicitly.

The returned trajectory has exactly the same shape as the one produced by
``rk4_schrodinger_traj`` so the two integrators can be swapped freely in user
code.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.sparse

from ..validation import validate_wavefunction_problem

# ---------------------------------------------------------------------------
# Optional back‑ends ----------------------------------------------------------
# ---------------------------------------------------------------------------

try:
    import cupy as cp  # type: ignore
except ImportError:  # CuPy が無い環境でも読み込めるように動作
    cp = None  # noqa: N816

try:
    from numba import njit  # type: ignore

    _HAS_NUMBA = True
except ImportError:  # NumPy fallback（遅くなるが動く）

    def njit(**_kwargs):  # type: ignore
        """Dummy decorator when numba is absent."""

        def _decorator(func):
            return func

        return _decorator

    _HAS_NUMBA = False

__all__ = ["build_helicity_projected_interaction", "splitop_schrodinger"]

_HERMITICITY_ROUNDOFF_FACTOR = 128.0

# ---------------------------------------------------------------------------
# Helper (CPU, Numba) --------------------------------------------------------
# ---------------------------------------------------------------------------


@njit(cache=True)
def _propagate_numpy(
    U: np.ndarray,  # (dim, dim)  unitary eigenvector matrix
    U_H: np.ndarray,  # U.conj().T  – Hermitian adjoint
    eigvals: np.ndarray,  # (dim,)   eigenvalues of A (real)
    psi0: np.ndarray,  # (dim,)
    exp_half: np.ndarray,  # (dim,)   element‑wise ½‑step phase from H0
    e_mid: np.ndarray,  # (steps,)   midpoint values of E(t)
    phase_coeff: complex,  # −i·2·dt/hbar   (scalar complex)
    return_traj: bool,
    stride: int,
    renorm: bool,
) -> np.ndarray:
    """Numba‑accelerated inner loop (CPU, NumPy)."""

    dim = psi0.shape[0]
    steps = e_mid.size
    n_samples = steps // stride + 1 if return_traj else 1
    traj = np.empty((n_samples, dim), dtype=np.complex128)

    psi = psi0.copy()
    traj[0] = psi
    s_idx = 1

    for k in range(steps):
        # H0 – 前半
        psi *= exp_half

        # Interaction part   exp[ phase_coeff * E * eigvals ]
        phase = np.exp(phase_coeff * e_mid[k] * eigvals)
        psi = U @ (phase * (U_H @ psi))

        # H0 – 後半
        psi *= exp_half
        if renorm:
            norm = np.sqrt((psi.conj() @ psi).real)
            if norm > 1e-12:
                psi *= 1.0 / norm

        if return_traj and (k + 1) % stride == 0:
            traj[s_idx] = psi
            s_idx += 1

    if return_traj:
        return traj
    else:
        return psi.reshape((1, dim))


@njit(cache=True, fastmath=True)
def _propagate_rotating_xy_numpy(
    U: np.ndarray,
    U_H: np.ndarray,
    eigvals: np.ndarray,
    magnetic_quantum_numbers: np.ndarray,
    psi0: np.ndarray,
    exp_half: np.ndarray,
    ex_mid: np.ndarray,
    ey_mid: np.ndarray,
    dt: float,
    return_traj: bool,
    stride: int,
    renorm: bool,
) -> np.ndarray:
    """Propagate a Cartesian xy interaction using diagonal M rotations."""
    dim = psi0.shape[0]
    steps = ex_mid.size
    n_samples = steps // stride + 1 if return_traj else 1
    traj = np.empty((n_samples, dim), dtype=np.complex128)

    psi = psi0.copy()
    traj[0] = psi
    s_idx = 1

    for k in range(steps):
        psi *= exp_half

        amplitude = np.hypot(ex_mid[k], ey_mid[k])
        if amplitude != 0.0:
            angle = np.arctan2(ey_mid[k], ex_mid[k])
            for index in range(dim):
                psi[index] *= np.exp(-1j * magnetic_quantum_numbers[index] * angle)
            phase = np.exp(1j * dt * amplitude * eigvals)
            psi = U @ (phase * (U_H @ psi))
            for index in range(dim):
                psi[index] *= np.exp(1j * magnetic_quantum_numbers[index] * angle)

        psi *= exp_half
        if renorm:
            norm = np.sqrt((psi.conj() @ psi).real)
            if norm > 0.0:
                psi *= 1.0 / norm

        if return_traj and (k + 1) % stride == 0:
            traj[s_idx] = psi
            s_idx += 1

    if return_traj:
        return traj
    return psi.reshape((1, dim))


def _as_numpy_dense(matrix) -> np.ndarray:
    """Return a dense NumPy complex matrix from a supported backend."""
    if scipy.sparse.issparse(matrix):
        matrix = matrix.toarray()
    elif cp is not None and isinstance(matrix, cp.ndarray):
        matrix = cp.asnumpy(matrix)
    return np.asarray(matrix, dtype=np.complex128)


def _matrix_roundoff_tolerance(matrix: np.ndarray) -> float:
    scale = float(np.linalg.norm(matrix, ord=np.inf))
    return (
        _HERMITICITY_ROUNDOFF_FACTOR
        * max(1, matrix.shape[0])
        * np.finfo(np.float64).eps
        * max(scale, np.finfo(np.float64).tiny)
    )


def _validate_hermitian(name: str, matrix: np.ndarray) -> None:
    tolerance = _matrix_roundoff_tolerance(matrix)
    residual = float(np.linalg.norm(matrix - matrix.conj().T, ord=np.inf))
    if residual > tolerance:
        raise ValueError(
            f"{name} must be Hermitian; residual {residual:.3e} exceeds "
            f"the roundoff tolerance {tolerance:.3e}"
        )


def _validate_unit_polarization(polarization: np.ndarray) -> np.ndarray:
    pol = np.asarray(polarization, dtype=np.complex128)
    if pol.shape != (2,) or not np.all(np.isfinite(pol)):
        raise ValueError("polarization must be a finite two-element Jones vector")
    norm = float(np.linalg.norm(pol))
    tolerance = _HERMITICITY_ROUNDOFF_FACTOR * np.finfo(np.float64).eps
    if not np.isclose(norm, 1.0, rtol=tolerance, atol=tolerance):
        raise ValueError("polarization must be normalized before propagation")
    return pol


def build_helicity_projected_interaction(
    mu_x: np.ndarray,
    mu_y: np.ndarray,
    polarization: np.ndarray,
) -> np.ndarray:
    """Build T + T-adjoint from the upper-triangular one-way transition T.

    The canonical LinMol basis is ordered by increasing vibrational manifold.
    With the current Cartesian tensor convention, (1, +i)/sqrt(2) selects
    resonant Delta M = +1 absorption and the opposite sign selects
    Delta M = -1.
    """
    mux = _as_numpy_dense(mu_x)
    muy = _as_numpy_dense(mu_y)
    _validate_hermitian("mu_x", mux)
    _validate_hermitian("mu_y", muy)
    pol = _validate_unit_polarization(polarization)

    combined = -pol[0] * mux - pol[1] * muy
    tolerance = _matrix_roundoff_tolerance(combined)
    if np.any(np.abs(np.diag(combined)) > tolerance):
        raise ValueError(
            "helicity_projected requires a transition dipole with zero diagonal"
        )
    one_way = np.triu(combined, k=1)
    return one_way + one_way.conj().T


def _factor_fixed_cartesian_field(
    field_x: np.ndarray, field_y: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return a real unit direction and signed scalar field when fixed."""
    vectors = np.column_stack((field_x, field_y))
    magnitudes = np.linalg.norm(vectors, axis=1)
    peak_index = int(np.argmax(magnitudes))
    peak = float(magnitudes[peak_index])
    if peak == 0.0:
        return np.array([1.0, 0.0]), np.zeros_like(field_x)

    direction = vectors[peak_index] / peak
    scalar = vectors @ direction
    residual = vectors - scalar[:, None] * direction
    tolerance = _HERMITICITY_ROUNDOFF_FACTOR * np.finfo(np.float64).eps * peak
    if float(np.max(np.abs(residual))) > tolerance:
        return None
    return direction, scalar


def _validate_xy_rotation_covariance(
    mu_x: np.ndarray, mu_y: np.ndarray, magnetic_quantum_numbers: np.ndarray
) -> None:
    """Require D(pi/2) mu_x D-adjoint = mu_y for the supplied M labels."""
    rotation = np.exp(0.5j * np.pi * magnetic_quantum_numbers)
    rotated = rotation[:, None] * mu_x * rotation.conj()[None, :]
    tolerance = max(
        _matrix_roundoff_tolerance(mu_x),
        _matrix_roundoff_tolerance(mu_y),
    )
    residual = float(np.linalg.norm(rotated - mu_y, ord=np.inf))
    if residual > tolerance:
        raise ValueError(
            "cartesian split propagation with changing field direction requires "
            "xy vector dipoles satisfying M-rotation covariance"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _as_numpy_field(field) -> np.ndarray:
    if cp is not None and isinstance(field, cp.ndarray):
        field = cp.asnumpy(field)
    return np.asarray(field, dtype=np.float64)


def splitop_schrodinger(
    H0: np.ndarray,
    mu_x: np.ndarray,
    mu_y: np.ndarray,
    field_x: np.ndarray,
    field_y: np.ndarray,
    psi: np.ndarray,
    dt: float,
    return_traj: bool = True,
    sample_stride: int = 1,
    *,
    interaction_mode: Literal["cartesian", "helicity_projected"] = "cartesian",
    magnetic_quantum_numbers: np.ndarray | None = None,
    polarization: np.ndarray | None = None,
    scalar_field: np.ndarray | None = None,
    backend: Literal["numpy", "cupy"] = "numpy",
    sparse: bool = False,
    renorm: bool = False,
) -> np.ndarray:
    """Propagate with a static or M-rotated spectral split operator.

    Cartesian mode uses the real field components and therefore represents the
    same Hamiltonian as RK4.  A field with changing direction requires an
    M-resolved xy vector operator.  Helicity-projected mode is an explicit
    one-way-transition approximation driven by a real scalar waveform.

    Sparse input matrices are accepted, but their spectral eigenvectors are
    dense.  The sparse flag is retained only at the high-level API boundary.
    """
    if interaction_mode not in {"cartesian", "helicity_projected"}:
        raise ValueError("interaction_mode must be 'cartesian' or 'helicity_projected'")
    del sparse

    ex = _as_numpy_field(field_x)
    ey = _as_numpy_field(field_y)
    if interaction_mode == "helicity_projected":
        if polarization is None or scalar_field is None:
            raise ValueError(
                "helicity_projected requires polarization and scalar_field"
            )
        active_field = _as_numpy_field(scalar_field)
        fields_to_validate = (active_field,)
    else:
        active_field = ex
        fields_to_validate = (ex, ey)

    validate_wavefunction_problem(
        H0,
        (mu_x, mu_y),
        fields_to_validate,
        psi,
        dt=dt,
        stride=sample_stride,
        backend=backend,
        require_diagonal_h0=True,
        require_odd_field=True,
    )

    h0_dense = _as_numpy_dense(H0)
    diag_h0_complex = np.diag(h0_dense) if h0_dense.ndim == 2 else h0_dense
    h0_tolerance = _matrix_roundoff_tolerance(np.diag(diag_h0_complex))
    if float(np.max(np.abs(np.imag(diag_h0_complex)))) > h0_tolerance:
        raise ValueError("split-operator requires real diagonal H0 eigenvalues")
    diag_h0 = np.asarray(np.real(diag_h0_complex), dtype=np.float64)

    mux = _as_numpy_dense(mu_x)
    muy = _as_numpy_dense(mu_y)
    _validate_hermitian("mu_x", mux)
    _validate_hermitian("mu_y", muy)
    psi_numpy = np.asarray(psi, dtype=np.complex128).reshape(-1)

    steps = (active_field.size - 1) // 2
    exp_half = np.exp(-1j * diag_h0 * dt / 2.0)

    rotating = False
    if interaction_mode == "helicity_projected":
        assert polarization is not None
        interaction = build_helicity_projected_interaction(mux, muy, polarization)
        scalar_mid = active_field[1 : 2 * steps + 1 : 2]
    else:
        fixed = _factor_fixed_cartesian_field(ex, ey)
        if fixed is not None:
            direction, scalar = fixed
            interaction = -direction[0] * mux - direction[1] * muy
            _validate_hermitian("fixed Cartesian interaction", interaction)
            scalar_mid = scalar[1 : 2 * steps + 1 : 2]
        else:
            if magnetic_quantum_numbers is None:
                raise ValueError(
                    "changing Cartesian field direction requires "
                    "magnetic_quantum_numbers"
                )
            m_values = np.asarray(magnetic_quantum_numbers, dtype=np.float64)
            if m_values.shape != (mux.shape[0],) or not np.all(np.isfinite(m_values)):
                raise ValueError(
                    "magnetic_quantum_numbers must be a finite vector "
                    "matching the Hilbert-space dimension"
                )
            _validate_xy_rotation_covariance(mux, muy, m_values)
            ex_mid = ex[1 : 2 * steps + 1 : 2]
            ey_mid = ey[1 : 2 * steps + 1 : 2]
            rotating = True

    if backend == "cupy":
        if cp is None:
            raise RuntimeError(
                "backend='cupy' was requested but CuPy is not installed."
            )
        if rotating:
            return _splitop_rotating_xy_cupy(
                diag_h0,
                mux,
                m_values,
                ex_mid,
                ey_mid,
                psi_numpy,
                dt,
                return_traj,
                sample_stride,
                renorm,
            )
        return _splitop_static_cupy(
            diag_h0,
            interaction,
            scalar_mid,
            psi_numpy,
            dt,
            return_traj,
            sample_stride,
            renorm,
        )

    if rotating:
        eigvals, eigenvectors = np.linalg.eigh(mux)
        return _propagate_rotating_xy_numpy(
            np.ascontiguousarray(eigenvectors),
            np.ascontiguousarray(eigenvectors.conj().T),
            np.ascontiguousarray(eigvals),
            np.ascontiguousarray(m_values),
            psi_numpy,
            exp_half,
            ex_mid,
            ey_mid,
            dt,
            return_traj,
            sample_stride,
            renorm,
        )

    eigvals, eigenvectors = np.linalg.eigh(interaction)
    return _propagate_numpy(
        np.ascontiguousarray(eigenvectors),
        np.ascontiguousarray(eigenvectors.conj().T),
        np.ascontiguousarray(eigvals),
        psi_numpy,
        exp_half,
        scalar_mid,
        -1j * dt,
        return_traj,
        sample_stride,
        renorm,
    )


# ---------------------------------------------------------------------------
# CuPy back-end
# ---------------------------------------------------------------------------


def _splitop_static_cupy(
    diag_h0: np.ndarray,
    interaction: np.ndarray,
    scalar_mid: np.ndarray,
    psi: np.ndarray,
    dt: float,
    return_traj: bool,
    sample_stride: int,
    renorm: bool,
) -> np.ndarray:
    """GPU implementation for a static interaction operator."""
    assert cp is not None
    h0_cp = cp.asarray(diag_h0)
    interaction_cp = cp.asarray(interaction)
    field_cp = cp.asarray(scalar_mid)
    psi_cp = cp.asarray(psi)
    exp_half = cp.exp(-1j * h0_cp * dt / 2.0)
    eigvals, eigenvectors = cp.linalg.eigh(interaction_cp)
    eigenvectors_h = eigenvectors.conj().T

    steps = field_cp.size
    n_samples = steps // sample_stride + 1 if return_traj else 1
    trajectory = cp.empty((n_samples, psi_cp.size), dtype=cp.complex128)
    trajectory[0] = psi_cp
    sample_index = 1

    for step in range(steps):
        psi_cp *= exp_half
        phase = cp.exp(-1j * dt * field_cp[step] * eigvals)
        psi_cp = eigenvectors @ (phase * (eigenvectors_h @ psi_cp))
        psi_cp *= exp_half
        if renorm:
            norm = cp.sqrt((psi_cp.conj() @ psi_cp).real)
            if float(norm.item()) > 0.0:
                psi_cp *= 1.0 / norm
        if return_traj and (step + 1) % sample_stride == 0:
            trajectory[sample_index] = psi_cp
            sample_index += 1

    if return_traj:
        return cp.asnumpy(trajectory)
    return cp.asnumpy(psi_cp).reshape(1, -1)


def _splitop_rotating_xy_cupy(
    diag_h0: np.ndarray,
    mu_x: np.ndarray,
    magnetic_quantum_numbers: np.ndarray,
    ex_mid: np.ndarray,
    ey_mid: np.ndarray,
    psi: np.ndarray,
    dt: float,
    return_traj: bool,
    sample_stride: int,
    renorm: bool,
) -> np.ndarray:
    """GPU implementation for an M-rotated Cartesian xy interaction."""
    assert cp is not None
    h0_cp = cp.asarray(diag_h0)
    mu_x_cp = cp.asarray(mu_x)
    m_cp = cp.asarray(magnetic_quantum_numbers)
    ex_cp = cp.asarray(ex_mid)
    ey_cp = cp.asarray(ey_mid)
    psi_cp = cp.asarray(psi)
    exp_half = cp.exp(-1j * h0_cp * dt / 2.0)
    eigvals, eigenvectors = cp.linalg.eigh(mu_x_cp)
    eigenvectors_h = eigenvectors.conj().T

    steps = ex_cp.size
    n_samples = steps // sample_stride + 1 if return_traj else 1
    trajectory = cp.empty((n_samples, psi_cp.size), dtype=cp.complex128)
    trajectory[0] = psi_cp
    sample_index = 1

    for step in range(steps):
        psi_cp *= exp_half
        amplitude = cp.hypot(ex_cp[step], ey_cp[step])
        if float(amplitude.item()) != 0.0:
            angle = cp.arctan2(ey_cp[step], ex_cp[step])
            rotation = cp.exp(1j * m_cp * angle)
            psi_cp *= rotation.conj()
            phase = cp.exp(1j * dt * amplitude * eigvals)
            psi_cp = eigenvectors @ (phase * (eigenvectors_h @ psi_cp))
            psi_cp *= rotation
        psi_cp *= exp_half
        if renorm:
            norm = cp.sqrt((psi_cp.conj() @ psi_cp).real)
            if float(norm.item()) > 0.0:
                psi_cp *= 1.0 / norm
        if return_traj and (step + 1) % sample_stride == 0:
            trajectory[sample_index] = psi_cp
            sample_index += 1

    if return_traj:
        return cp.asnumpy(trajectory)
    return cp.asnumpy(psi_cp).reshape(1, -1)
