"""Strict structural validation at the canonical propagation-unit boundary."""

from __future__ import annotations

from typing import Any

import numpy as np


def _shape_of(value: Any, *, name: str) -> tuple[int, ...]:
    """Return an object's shape without converting backend-native storage."""
    shape = getattr(value, "shape", None)
    if shape is None:
        raise TypeError(f"{name} must expose a shape")
    try:
        return tuple(int(length) for length in shape)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} has an invalid shape") from exc


def _square_dimension(value: Any, *, name: str) -> int:
    shape = _shape_of(value, name=name)
    if len(shape) != 2 or shape[0] != shape[1] or shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty square matrix; got shape {shape}")
    return shape[0]


def _required_method(owner: Any, name: str):
    method = getattr(owner, name, None)
    if not callable(method):
        raise TypeError(f"{type(owner).__name__} must provide callable {name}()")
    return method


class UnitValidator:
    """Validate formal access to the canonical units used by propagation.

    This validator intentionally does not classify numerical values as typical
    or atypical. Physical-scale adequacy belongs to an explicit convergence
    analysis, not to a warning or fallback at the propagation boundary.
    """

    def validate_propagation_units(
        self,
        hamiltonian: Any,
        dipole_matrix: Any,
        efield: Any,
        expected_H0_units: str = "J",
        expected_dipole_units: str = "C*m",
    ) -> None:
        """Require canonical unit accessors and mutually consistent shapes.

        No exception is downgraded to a warning, and no raw attribute is used
        when a canonical SI accessor is absent.
        """
        if expected_H0_units != "J":
            raise ValueError("expected_H0_units must be 'J'")
        if expected_dipole_units != "C*m":
            raise ValueError("expected_dipole_units must be 'C*m'")

        h0 = _required_method(hamiltonian, "get_matrix")("J")
        dimension = _square_dimension(h0, name="Hamiltonian in J")

        for axis in ("x", "y"):
            component = _required_method(dipole_matrix, f"get_mu_{axis}_SI")()
            shape = _shape_of(component, name=f"mu_{axis} in C*m")
            if shape != (dimension, dimension):
                raise ValueError(
                    f"mu_{axis} in C*m must have shape "
                    f"{(dimension, dimension)}; got {shape}"
                )

        time_fs = _required_method(efield, "get_time_SI")()
        time_shape = _shape_of(time_fs, name="time grid in fs")
        if len(time_shape) != 1 or time_shape[0] < 2:
            raise ValueError(
                "time grid in fs must be one-dimensional with at least 2 points"
            )

        field_v_per_m = _required_method(efield, "get_Efield_SI")()
        field_shape = _shape_of(field_v_per_m, name="electric field in V/m")
        if len(field_shape) not in (1, 2) or field_shape[0] != time_shape[0]:
            raise ValueError(
                "electric field in V/m must have one sample per time-grid point; "
                f"got field shape {field_shape} and time shape {time_shape}"
            )

        try:
            field_dt_fs = float(efield.dt)
        except (AttributeError, TypeError, ValueError) as exc:
            raise TypeError("electric field must expose scalar dt in fs") from exc
        if not np.isfinite(field_dt_fs) or field_dt_fs <= 0.0:
            raise ValueError("electric-field dt in fs must be finite and positive")


validator = UnitValidator()

__all__ = ["UnitValidator", "validator"]
