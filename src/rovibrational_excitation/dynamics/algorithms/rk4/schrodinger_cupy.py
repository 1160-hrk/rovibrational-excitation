"""Device-native CuPy implementation of dense Schrödinger RK4."""

from __future__ import annotations

from typing import Any

try:
    import cupy as cp
except ImportError:  # pragma: no cover - optional GPU dependency
    cp = None


def _require_cupy() -> Any:
    if cp is None:
        raise RuntimeError("backend='cupy' but CuPy is not installed")
    return cp


def _derivative(
    H0: Any,
    mu_x: Any,
    mu_y: Any,
    field_x: Any,
    field_y: Any,
    state: Any,
) -> Any:
    """Return ``-1j * (H0 - mu_x*Ex - mu_y*Ey) @ state`` on device."""
    return -1j * ((H0 - field_x * mu_x - field_y * mu_y) @ state)


def rk4_schrodinger_cupy(
    H0: Any,
    mu_x: Any,
    mu_y: Any,
    field_x: Any,
    field_y: Any,
    psi0: Any,
    dt: float,
    *,
    return_traj: bool,
    stride: int,
    renorm: bool,
) -> Any:
    """Propagate with the same RK4 graph as the dense NumPy implementation.

    Operators, fields, stage vectors, and returned states remain CuPy arrays.
    With ``renorm=True`` only the scalar validity check synchronizes the host;
    no state or trajectory array crosses the backend boundary.
    """
    xp = _require_cupy()
    h0_device = xp.asarray(H0, dtype=xp.complex128)
    mu_x_device = xp.asarray(mu_x, dtype=xp.complex128)
    mu_y_device = xp.asarray(mu_y, dtype=xp.complex128)
    field_x_device = xp.asarray(field_x, dtype=xp.float64)
    field_y_device = xp.asarray(field_y, dtype=xp.float64)
    psi = xp.asarray(psi0, dtype=xp.complex128).reshape(-1).copy()

    steps = (field_x_device.size - 1) // 2
    output_rows = steps // stride + 1 if return_traj else 1
    output = xp.empty((output_rows, psi.size), dtype=xp.complex128)
    output_index = 0
    if return_traj:
        output[0] = psi
        output_index = 1

    for step_index in range(steps):
        field_index = 2 * step_index
        k1 = _derivative(
            h0_device,
            mu_x_device,
            mu_y_device,
            field_x_device[field_index],
            field_y_device[field_index],
            psi,
        )
        k2 = _derivative(
            h0_device,
            mu_x_device,
            mu_y_device,
            field_x_device[field_index + 1],
            field_y_device[field_index + 1],
            psi + 0.5 * dt * k1,
        )
        k3 = _derivative(
            h0_device,
            mu_x_device,
            mu_y_device,
            field_x_device[field_index + 1],
            field_y_device[field_index + 1],
            psi + 0.5 * dt * k2,
        )
        k4 = _derivative(
            h0_device,
            mu_x_device,
            mu_y_device,
            field_x_device[field_index + 2],
            field_y_device[field_index + 2],
            psi + dt * k3,
        )
        psi += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        if renorm:
            norm_squared = xp.vdot(psi, psi).real
            norm_value = float(norm_squared.item())
            if norm_value <= 0.0 or not bool(xp.isfinite(norm_squared).item()):
                raise ValueError("cannot renormalize a zero or non-finite wavefunction")
            psi *= 1.0 / xp.sqrt(norm_squared)

        if return_traj and (step_index + 1) % stride == 0:
            output[output_index] = psi
            output_index += 1

    if not return_traj:
        output[0] = psi
    return output
