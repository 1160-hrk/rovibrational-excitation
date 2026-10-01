"""Explicit, report-only convergence assessment for simulation cases."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from rovibrational_excitation.core.time import TimeGrid
from rovibrational_excitation.fields import SampledField, ScalarField

from .runner import _execute_one
from .validation import SimulationConfigurationError, validate_simulation_case

ObservableArray: TypeAlias = NDArray[np.float64] | NDArray[np.complex128]
FieldKind: TypeAlias = Literal["generated", "scalar", "cartesian"]

_GENERATED_GRID_KEYS = frozenset({"t_start", "t_end", "dt"})


class ConvergenceConfigurationError(ValueError):
    """Raised before propagation when a convergence request is invalid."""


@dataclass(frozen=True, slots=True, eq=False)
class ConvergenceReport:
    """Immutable comparison of one observable on coarse and fine grids."""

    observable_name: str
    field_kind: FieldKind
    coarse_field_dt_fs: float
    fine_field_dt_fs: float
    refinement_ratio: float
    tolerance: float
    max_absolute_difference: float
    converged: bool
    coarse_observable: ObservableArray
    fine_observable: ObservableArray


def _validate_request(
    *,
    observable_name: str,
    observable: Callable[[NDArray[np.float64]], Any],
    tolerance: float,
) -> tuple[str, float]:
    if not isinstance(observable_name, str) or not observable_name.strip():
        raise ConvergenceConfigurationError("observable_name must be a nonempty string")
    if not callable(observable):
        raise ConvergenceConfigurationError("observable must be callable")
    if isinstance(tolerance, (bool, np.bool_)):
        raise ConvergenceConfigurationError(
            "tolerance must be a finite nonnegative number"
        )
    try:
        validated_tolerance = float(tolerance)
    except (TypeError, ValueError) as exc:
        raise ConvergenceConfigurationError(
            "tolerance must be a finite nonnegative number"
        ) from exc
    if not np.isfinite(validated_tolerance) or validated_tolerance < 0.0:
        raise ConvergenceConfigurationError(
            "tolerance must be a finite nonnegative number"
        )
    return observable_name.strip(), validated_tolerance


def _values_equal(left: Any, right: Any) -> bool:
    """Compare validated parameter values without ambiguous array truth values."""
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if left.keys() != right.keys():
            return False
        return all(_values_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _values_equal(left_value, right_value)
            for left_value, right_value in zip(left, right)
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return bool(np.array_equal(np.asarray(left), np.asarray(right)))
        except (TypeError, ValueError):
            return False
    try:
        result = left == right
    except (TypeError, ValueError):
        return False
    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    try:
        return bool(np.all(result))
    except (TypeError, ValueError):
        return False


def _mismatched_parameter_names(
    coarse: Mapping[str, Any],
    fine: Mapping[str, Any],
    *,
    excluded: frozenset[str],
) -> list[str]:
    names = (set(coarse) | set(fine)) - excluded
    missing = object()
    return sorted(
        name
        for name in names
        if not _values_equal(coarse.get(name, missing), fine.get(name, missing))
    )


def _generated_grid(params: Mapping[str, Any]) -> TimeGrid:
    return TimeGrid.from_bounds(params["t_start"], params["t_end"], params["dt"])


def _validate_grid_pair(coarse: TimeGrid, fine: TimeGrid) -> None:
    if coarse.t_start_fs != fine.t_start_fs or coarse.t_end_fs != fine.t_end_fs:
        raise ConvergenceConfigurationError(
            "coarse and fine grids must have identical endpoints"
        )
    if fine.field_dt_fs >= coarse.field_dt_fs:
        raise ConvergenceConfigurationError(
            "fine field-grid step must be strictly smaller than coarse field-grid step"
        )


def _validate_case(
    label: str,
    params: Mapping[str, Any],
    *,
    field: SampledField | None,
) -> dict[str, Any]:
    if not isinstance(params, Mapping):
        raise ConvergenceConfigurationError(f"{label}_params must be a mapping")
    copied = dict(params)
    if copied.get("save", False) is True:
        raise ConvergenceConfigurationError(
            f"{label}_params must use save=False; convergence assessment never "
            "writes simulation results"
        )
    copied["save"] = False
    try:
        validate_simulation_case(copied, field=field)
    except SimulationConfigurationError as exc:
        raise ConvergenceConfigurationError(
            f"invalid {label} simulation case: {exc}"
        ) from exc
    return copied


def _validate_observable_output(value: Any, *, label: str) -> ObservableArray:
    raw = np.asarray(value)
    if raw.size == 0:
        raise ConvergenceConfigurationError(
            f"{label} observable output must be nonempty"
        )
    if np.issubdtype(raw.dtype, np.bool_) or not np.issubdtype(raw.dtype, np.number):
        raise ConvergenceConfigurationError(
            f"{label} observable output must be numeric and not boolean"
        )
    dtype = np.complex128 if np.iscomplexobj(raw) else np.float64
    result = np.array(raw, dtype=dtype, copy=True)
    if not np.all(np.isfinite(result)):
        raise ConvergenceConfigurationError(
            f"{label} observable output must contain only finite values"
        )
    result.setflags(write=False)
    return cast(ObservableArray, result)


def _evaluate_observable(
    observable: Callable[[NDArray[np.float64]], Any],
    population: NDArray[np.float64],
    *,
    label: str,
) -> ObservableArray:
    protected_population = np.array(population, dtype=np.float64, copy=True)
    protected_population.setflags(write=False)
    return _validate_observable_output(
        observable(protected_population),
        label=label,
    )


def assess_simulation_convergence(
    coarse_params: Mapping[str, Any],
    fine_params: Mapping[str, Any],
    *,
    observable_name: str,
    observable: Callable[[NDArray[np.float64]], Any],
    tolerance: float,
    coarse_field: SampledField | None = None,
    fine_field: SampledField | None = None,
) -> ConvergenceReport:
    """Run two explicit grids and compare a caller-selected observable.

    This operation only reports convergence. It never changes either grid,
    selects a time step, resamples a field, or writes simulation results.
    The two cases must be identical except for ``dt`` on the generated-field
    route, or for the supplied :class:`TimeGrid` on the external-field route.
    Both grids must span the same exact endpoints and the fine field-grid step
    must be strictly smaller. ``tolerance`` is applied to the maximum absolute
    elementwise difference of the two observable outputs.
    """
    validated_name, validated_tolerance = _validate_request(
        observable_name=observable_name,
        observable=observable,
        tolerance=tolerance,
    )

    if (coarse_field is None) != (fine_field is None):
        raise ConvergenceConfigurationError(
            "both sampled fields (coarse_field and fine_field) must be supplied, "
            "or both omitted"
        )
    if coarse_field is not None and type(coarse_field) is not type(fine_field):
        raise ConvergenceConfigurationError(
            "coarse_field and fine_field must have the same sampled-field type"
        )

    coarse = _validate_case("coarse", coarse_params, field=coarse_field)
    fine = _validate_case("fine", fine_params, field=fine_field)

    if coarse_field is None:
        coarse_grid = _generated_grid(coarse)
        fine_grid = _generated_grid(fine)
        field_kind: FieldKind = "generated"
        excluded = _GENERATED_GRID_KEYS
    else:
        coarse_grid = coarse_field.time_grid
        assert fine_field is not None
        fine_grid = fine_field.time_grid
        field_kind = "scalar" if isinstance(coarse_field, ScalarField) else "cartesian"
        excluded = frozenset()

    _validate_grid_pair(coarse_grid, fine_grid)
    mismatched = _mismatched_parameter_names(coarse, fine, excluded=excluded)
    if mismatched:
        raise ConvergenceConfigurationError(
            "coarse and fine calculation parameters must match; mismatched: "
            + ", ".join(mismatched)
        )

    coarse_population = _execute_one(coarse, field=coarse_field)
    fine_population = _execute_one(fine, field=fine_field)
    coarse_observable = _evaluate_observable(
        observable, coarse_population, label="coarse"
    )
    fine_observable = _evaluate_observable(observable, fine_population, label="fine")
    if coarse_observable.shape != fine_observable.shape:
        raise ConvergenceConfigurationError(
            "coarse and fine observable outputs must have the same shape"
        )

    common_dtype = (
        np.complex128
        if np.iscomplexobj(coarse_observable) or np.iscomplexobj(fine_observable)
        else np.float64
    )
    coarse_result = cast(
        ObservableArray,
        np.array(coarse_observable, dtype=common_dtype, copy=True),
    )
    fine_result = cast(
        ObservableArray,
        np.array(fine_observable, dtype=common_dtype, copy=True),
    )
    coarse_result.setflags(write=False)
    fine_result.setflags(write=False)
    maximum_difference = float(np.max(np.abs(coarse_result - fine_result)))

    return ConvergenceReport(
        observable_name=validated_name,
        field_kind=field_kind,
        coarse_field_dt_fs=coarse_grid.field_dt_fs,
        fine_field_dt_fs=fine_grid.field_dt_fs,
        refinement_ratio=coarse_grid.field_dt_fs / fine_grid.field_dt_fs,
        tolerance=validated_tolerance,
        max_absolute_difference=maximum_difference,
        converged=maximum_difference <= validated_tolerance,
        coarse_observable=coarse_result,
        fine_observable=fine_result,
    )


__all__ = [
    "ConvergenceConfigurationError",
    "ConvergenceReport",
    "assess_simulation_convergence",
]
