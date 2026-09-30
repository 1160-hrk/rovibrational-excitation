"""Device-native CuPy implementation of Schrödinger split propagation."""

from __future__ import annotations

from typing import Any, Literal

import scipy.sparse

try:
    import cupy as cp
except ImportError:  # pragma: no cover - optional GPU dependency
    cp = None


def _require_cupy() -> Any:
    if cp is None:
        raise RuntimeError("backend='cupy' was requested but CuPy is not installed")
    return cp


def _item_float(value: Any) -> float:
    return float(value.item())


def _item_bool(value: Any) -> bool:
    return bool(value.item())


def _as_device_dense(xp: Any, matrix: Any) -> Any:
    if scipy.sparse.issparse(matrix):
        matrix = matrix.toarray()
    return xp.asarray(matrix, dtype=xp.complex128)


def _matrix_roundoff_tolerance(xp: Any, matrix: Any, roundoff_factor: float) -> float:
    scale = _item_float(xp.linalg.norm(matrix, ord=xp.inf))
    return float(
        roundoff_factor
        * max(1, matrix.shape[0])
        * xp.finfo(xp.float64).eps
        * max(scale, xp.finfo(xp.float64).tiny)
    )


def _validate_hermitian(
    xp: Any, name: str, matrix: Any, roundoff_factor: float
) -> None:
    tolerance = _matrix_roundoff_tolerance(xp, matrix, roundoff_factor)
    residual = _item_float(xp.linalg.norm(matrix - matrix.conj().T, ord=xp.inf))
    if residual > tolerance:
        raise ValueError(
            f"{name} must be Hermitian; residual {residual:.3e} exceeds "
            f"the roundoff tolerance {tolerance:.3e}"
        )


def _validate_unit_polarization(
    xp: Any, polarization: Any, roundoff_factor: float
) -> Any:
    pol = xp.asarray(polarization, dtype=xp.complex128)
    if pol.shape != (2,) or not _item_bool(xp.all(xp.isfinite(pol))):
        raise ValueError("polarization must be a finite two-element Jones vector")
    norm = _item_float(xp.linalg.norm(pol))
    tolerance = roundoff_factor * xp.finfo(xp.float64).eps
    if not _item_bool(xp.isclose(norm, 1.0, rtol=tolerance, atol=tolerance)):
        raise ValueError("polarization must be normalized before propagation")
    return pol


def _build_helicity_projected_interaction(
    xp: Any,
    mu_x: Any,
    mu_y: Any,
    polarization: Any,
    roundoff_factor: float,
) -> Any:
    pol = _validate_unit_polarization(xp, polarization, roundoff_factor)
    combined = -pol[0] * mu_x - pol[1] * mu_y
    tolerance = _matrix_roundoff_tolerance(xp, combined, roundoff_factor)
    if _item_bool(xp.any(xp.abs(xp.diag(combined)) > tolerance)):
        raise ValueError(
            "helicity_projected requires a transition dipole with zero diagonal"
        )
    one_way = xp.triu(combined, k=1)
    return one_way + one_way.conj().T


def _factor_fixed_cartesian_field(
    xp: Any, field_x: Any, field_y: Any, roundoff_factor: float
) -> tuple[Any, Any] | None:
    vectors = xp.column_stack((field_x, field_y))
    magnitudes = xp.linalg.norm(vectors, axis=1)
    peak_index = int(xp.argmax(magnitudes).item())
    peak = _item_float(magnitudes[peak_index])
    if peak == 0.0:
        return xp.asarray([1.0, 0.0]), xp.zeros_like(field_x)

    direction = vectors[peak_index] / peak
    scalar = vectors @ direction
    residual = vectors - scalar[:, None] * direction
    tolerance = roundoff_factor * xp.finfo(xp.float64).eps * peak
    if _item_float(xp.max(xp.abs(residual))) > tolerance:
        return None
    return direction, scalar


def _validate_xy_rotation_covariance(
    xp: Any,
    mu_x: Any,
    mu_y: Any,
    magnetic_quantum_numbers: Any,
    roundoff_factor: float,
) -> None:
    rotation = xp.exp(0.5j * xp.pi * magnetic_quantum_numbers)
    rotated = rotation[:, None] * mu_x * rotation.conj()[None, :]
    tolerance = max(
        _matrix_roundoff_tolerance(xp, mu_x, roundoff_factor),
        _matrix_roundoff_tolerance(xp, mu_y, roundoff_factor),
    )
    residual = _item_float(xp.linalg.norm(rotated - mu_y, ord=xp.inf))
    if residual > tolerance:
        raise ValueError(
            "cartesian split propagation with changing field direction requires "
            "xy vector dipoles satisfying M-rotation covariance"
        )


def _propagate_static(
    xp: Any,
    diag_h0: Any,
    interaction: Any,
    scalar_mid: Any,
    psi: Any,
    dt: float,
    return_traj: bool,
    sample_stride: int,
    renorm: bool,
) -> Any:
    exp_half = xp.exp(-1j * diag_h0 * dt / 2.0)
    eigvals, eigenvectors = xp.linalg.eigh(interaction)
    eigenvectors_h = eigenvectors.conj().T

    steps = scalar_mid.size
    n_samples = steps // sample_stride + 1 if return_traj else 1
    trajectory = xp.empty((n_samples, psi.size), dtype=xp.complex128)
    trajectory[0] = psi
    sample_index = 1

    for step in range(steps):
        psi *= exp_half
        phase = xp.exp(-1j * dt * scalar_mid[step] * eigvals)
        psi = eigenvectors @ (phase * (eigenvectors_h @ psi))
        psi *= exp_half
        if renorm:
            norm = xp.sqrt((psi.conj() @ psi).real)
            if _item_float(norm) > 0.0:
                psi *= 1.0 / norm
        if return_traj and (step + 1) % sample_stride == 0:
            trajectory[sample_index] = psi
            sample_index += 1

    if return_traj:
        return trajectory
    return psi.reshape(1, -1)


def _propagate_rotating_xy(
    xp: Any,
    diag_h0: Any,
    mu_x: Any,
    magnetic_quantum_numbers: Any,
    ex_mid: Any,
    ey_mid: Any,
    psi: Any,
    dt: float,
    return_traj: bool,
    sample_stride: int,
    renorm: bool,
) -> Any:
    exp_half = xp.exp(-1j * diag_h0 * dt / 2.0)
    eigvals, eigenvectors = xp.linalg.eigh(mu_x)
    eigenvectors_h = eigenvectors.conj().T

    steps = ex_mid.size
    n_samples = steps // sample_stride + 1 if return_traj else 1
    trajectory = xp.empty((n_samples, psi.size), dtype=xp.complex128)
    trajectory[0] = psi
    sample_index = 1

    for step in range(steps):
        psi *= exp_half
        amplitude = xp.hypot(ex_mid[step], ey_mid[step])
        if _item_float(amplitude) != 0.0:
            angle = xp.arctan2(ey_mid[step], ex_mid[step])
            rotation = xp.exp(1j * magnetic_quantum_numbers * angle)
            psi *= rotation.conj()
            phase = xp.exp(1j * dt * amplitude * eigvals)
            psi = eigenvectors @ (phase * (eigenvectors_h @ psi))
            psi *= rotation
        psi *= exp_half
        if renorm:
            norm = xp.sqrt((psi.conj() @ psi).real)
            if _item_float(norm) > 0.0:
                psi *= 1.0 / norm
        if return_traj and (step + 1) % sample_stride == 0:
            trajectory[sample_index] = psi
            sample_index += 1

    if return_traj:
        return trajectory
    return psi.reshape(1, -1)


def splitop_schrodinger_cupy(
    H0: Any,
    mu_x: Any,
    mu_y: Any,
    field_x: Any,
    field_y: Any,
    psi: Any,
    dt: float,
    *,
    return_traj: bool,
    sample_stride: int,
    interaction_mode: Literal["cartesian", "helicity_projected"],
    magnetic_quantum_numbers: Any | None,
    polarization: Any | None,
    scalar_field: Any | None,
    renorm: bool,
    roundoff_factor: float,
) -> Any:
    """Prepare and run split propagation without copying arrays to the host."""
    xp = _require_cupy()
    h0 = _as_device_dense(xp, H0)
    diag_h0_complex = xp.diag(h0) if h0.ndim == 2 else h0
    h0_tolerance = _matrix_roundoff_tolerance(
        xp, xp.diag(diag_h0_complex), roundoff_factor
    )
    if _item_float(xp.max(xp.abs(xp.imag(diag_h0_complex)))) > h0_tolerance:
        raise ValueError("split-operator requires real diagonal H0 eigenvalues")
    diag_h0 = xp.asarray(xp.real(diag_h0_complex), dtype=xp.float64)

    mux = _as_device_dense(xp, mu_x)
    muy = _as_device_dense(xp, mu_y)
    _validate_hermitian(xp, "mu_x", mux, roundoff_factor)
    _validate_hermitian(xp, "mu_y", muy, roundoff_factor)
    state = xp.asarray(psi, dtype=xp.complex128).reshape(-1).copy()
    ex = xp.asarray(field_x, dtype=xp.float64)
    ey = xp.asarray(field_y, dtype=xp.float64)

    if interaction_mode == "helicity_projected":
        assert polarization is not None
        assert scalar_field is not None
        active_field = xp.asarray(scalar_field, dtype=xp.float64)
        steps = (active_field.size - 1) // 2
        interaction = _build_helicity_projected_interaction(
            xp, mux, muy, polarization, roundoff_factor
        )
        scalar_mid = active_field[1 : 2 * steps + 1 : 2]
        return _propagate_static(
            xp,
            diag_h0,
            interaction,
            scalar_mid,
            state,
            dt,
            return_traj,
            sample_stride,
            renorm,
        )

    steps = (ex.size - 1) // 2
    fixed = _factor_fixed_cartesian_field(xp, ex, ey, roundoff_factor)
    if fixed is not None:
        direction, scalar = fixed
        interaction = -direction[0] * mux - direction[1] * muy
        _validate_hermitian(
            xp, "fixed Cartesian interaction", interaction, roundoff_factor
        )
        scalar_mid = scalar[1 : 2 * steps + 1 : 2]
        return _propagate_static(
            xp,
            diag_h0,
            interaction,
            scalar_mid,
            state,
            dt,
            return_traj,
            sample_stride,
            renorm,
        )

    if magnetic_quantum_numbers is None:
        raise ValueError(
            "changing Cartesian field direction requires magnetic_quantum_numbers"
        )
    m_values = xp.asarray(magnetic_quantum_numbers, dtype=xp.float64)
    if m_values.shape != (mux.shape[0],) or not _item_bool(
        xp.all(xp.isfinite(m_values))
    ):
        raise ValueError(
            "magnetic_quantum_numbers must be a finite vector "
            "matching the Hilbert-space dimension"
        )
    _validate_xy_rotation_covariance(xp, mux, muy, m_values, roundoff_factor)
    ex_mid = ex[1 : 2 * steps + 1 : 2]
    ey_mid = ey[1 : 2 * steps + 1 : 2]
    return _propagate_rotating_xy(
        xp,
        diag_h0,
        mux,
        m_values,
        ex_mid,
        ey_mid,
        state,
        dt,
        return_traj,
        sample_stride,
        renorm,
    )
