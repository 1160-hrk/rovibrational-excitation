"""Validation shared by low-level wavefunction propagators."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

try:
    import cupy as cp
except ImportError:  # pragma: no cover - optional GPU dependency
    cp = None


def _shape(operator: Any) -> tuple[int, ...]:
    shape = getattr(operator, "shape", None)
    if shape is None:
        shape = np.asarray(operator).shape
    return tuple(shape)


def _all_finite(value: Any) -> bool:
    if sp.issparse(value):
        return bool(np.all(np.isfinite(value.data)))
    if cp is not None and isinstance(value, cp.ndarray):
        return bool(cp.all(cp.isfinite(value)).item())
    return bool(np.all(np.isfinite(np.asarray(value))))


def validate_wavefunction_problem(
    h0: Any,
    dipoles: tuple[Any, ...],
    fields: tuple[Any, ...],
    psi0: Any,
    *,
    dt: float,
    stride: int,
    backend: str,
    require_odd_field: bool = False,
    require_diagonal_h0: bool = False,
) -> int:
    """Validate a propagation problem and return its Hilbert-space dimension."""
    if backend not in {"numpy", "cupy"}:
        raise ValueError("backend must be 'numpy' or 'cupy'")
    if not np.isfinite(dt) or dt == 0:
        raise ValueError("dt must be a non-zero finite number")
    if not isinstance(stride, (int, np.integer)) or stride < 1:
        raise ValueError("stride must be a positive integer")

    h0_shape = _shape(h0)
    if len(h0_shape) == 1:
        dim = h0_shape[0]
    elif len(h0_shape) == 2 and h0_shape[0] == h0_shape[1]:
        dim = h0_shape[0]
    else:
        raise ValueError("H0 must be a one-dimensional diagonal or square matrix")
    if dim < 1:
        raise ValueError("H0 must contain at least one state")

    if require_diagonal_h0 and len(h0_shape) == 2:
        h0_array = h0.toarray() if sp.issparse(h0) else h0
        xp = cp if cp is not None and isinstance(h0_array, cp.ndarray) else np
        h0_array = xp.asarray(h0_array)
        if not bool(xp.allclose(h0_array, xp.diag(xp.diag(h0_array)))):
            raise ValueError("split-operator requires a diagonal H0")

    for index, dipole in enumerate(dipoles):
        if _shape(dipole) != (dim, dim):
            raise ValueError(
                f"dipole[{index}] must have shape {(dim, dim)}, got {_shape(dipole)}"
            )

    psi_shape = _shape(psi0)
    valid_state_shape = psi_shape == (dim,) or psi_shape == (dim, 1)
    if not valid_state_shape:
        raise ValueError(
            f"psi0 must have shape {(dim,)} or {(dim, 1)}, got {psi_shape}"
        )

    if not fields:
        raise ValueError("at least one electric-field component is required")
    field_arrays = [
        field if hasattr(field, "shape") else np.asarray(field) for field in fields
    ]
    field_length = field_arrays[0].size
    if field_arrays[0].ndim != 1 or field_length < 3:
        raise ValueError(
            "electric fields must be one-dimensional with at least 3 points"
        )
    for index, field in enumerate(field_arrays):
        if field.ndim != 1 or field.size != field_length:
            raise ValueError(
                f"field[{index}] must be one-dimensional with length {field_length}"
            )
    if require_odd_field and field_length % 2 == 0:
        raise ValueError("electric field must contain 2*n_steps + 1 points")

    values = (h0, *dipoles, psi0, *field_arrays)
    if not all(_all_finite(value) for value in values):
        raise ValueError("propagation inputs must contain only finite values")

    return dim


_DENSITY_MATRIX_EPSILON_FACTOR = 100.0


def _density_matrix_tolerance(density: np.ndarray) -> float:
    """Return a scale-aware tolerance for Hermitian eigenvalue calculations."""
    dimension = max(1, density.shape[0])
    spectral_scale = float(np.linalg.norm(density, ord=2))
    machine_epsilon = np.finfo(density.real.dtype).eps
    # LAPACK errors scale as O(n * eps * ||A||). The factor leaves margin for
    # roundoff accumulated while constructing a density matrix.
    return _DENSITY_MATRIX_EPSILON_FACTOR * dimension * machine_epsilon * spectral_scale


def validate_density_matrix_properties(rho: Any) -> None:
    """Reject non-Hermitian, non-positive, or numerically invalid density matrices."""
    density_source = rho.toarray() if sp.issparse(rho) else rho
    density = np.asarray(density_source, dtype=np.complex128)
    if density.ndim != 2 or density.shape[0] != density.shape[1]:
        raise ValueError("density matrix must be square")
    if not np.all(np.isfinite(density)):
        raise ValueError("density matrix must contain only finite values")

    tolerance = _density_matrix_tolerance(density)
    trace = np.trace(density)
    if abs(trace.imag) > tolerance:
        raise ValueError("density-matrix trace must be real within numerical tolerance")
    if trace.real <= 0.0:
        raise ValueError("density-matrix trace must be positive")

    hermiticity_error = float(np.linalg.norm(density - density.conj().T, ord=2))
    if hermiticity_error > tolerance:
        raise ValueError("density matrix must be Hermitian within numerical tolerance")

    hermitian_density = 0.5 * (density + density.conj().T)
    minimum_eigenvalue = float(np.linalg.eigvalsh(hermitian_density)[0])
    if minimum_eigenvalue < -tolerance:
        raise ValueError(
            "density matrix must be positive semidefinite; "
            f"minimum eigenvalue {minimum_eigenvalue:.3e} is below "
            f"the tolerance {-tolerance:.3e}"
        )


def validate_density_matrix_problem(
    h0: Any,
    dipoles: tuple[Any, ...],
    fields: tuple[Any, ...],
    rho0: Any,
    *,
    dt: float,
    stride: int,
    backend: str,
    require_odd_field: bool = False,
) -> int:
    """Validate a density-matrix propagation problem and return its dimension."""
    h0_shape = _shape(h0)
    if not h0_shape:
        raise ValueError("H0 must be a one-dimensional diagonal or square matrix")
    dimension_hint = h0_shape[0]
    dim = validate_wavefunction_problem(
        h0,
        dipoles,
        fields,
        np.zeros(dimension_hint, dtype=np.complex128),
        dt=dt,
        stride=stride,
        backend=backend,
        require_odd_field=require_odd_field,
    )
    if _shape(rho0) != (dim, dim):
        raise ValueError(f"rho0 must have shape {(dim, dim)}, got {_shape(rho0)}")
    validate_density_matrix_properties(rho0)
    return dim
