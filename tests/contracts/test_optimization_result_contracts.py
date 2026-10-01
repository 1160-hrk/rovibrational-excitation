"""Typed optimization-result contracts shared by every algorithm."""

from __future__ import annotations

import numpy as np
import pytest

from rovibrational_excitation.fields import ElectricField
from rovibrational_excitation.optimization import ControlLayout, OptimizationResult


def _field() -> ElectricField:
    field = ElectricField(tlist=np.array([0.0, 0.1, 0.2]), time_units="fs")
    field.add_arbitrary_Efield(np.zeros((3, 2)), field_units="V/m")
    return field


def _values() -> dict[str, object]:
    return {
        "trajectory_times_fs": np.array([0.0, 0.2]),
        "trajectory": np.array([[1.0, 0.0], [0.9, 0.1j]], dtype=np.complex128),
        "control_times_fs": np.array([0.0, 0.1, 0.2]),
        "controls_v_per_m": np.zeros((3, 2)),
        "target_index": 1,
        "metrics": {"fidelity": 0.01},
        "control_layout": ControlLayout.RK4_FIELD_SAMPLES,
        "electric_field": _field(),
    }


def test_result_preserves_exact_algorithm_owned_arrays_without_repair() -> None:
    values = _values()
    result = OptimizationResult(**values)  # type: ignore[arg-type]

    assert result.trajectory_times_fs is values["trajectory_times_fs"]
    assert result.trajectory is values["trajectory"]
    assert result.control_times_fs is values["control_times_fs"]
    assert result.controls_v_per_m is values["controls_v_per_m"]
    assert result.metrics is values["metrics"]
    assert result.electric_field is values["electric_field"]


def test_interval_controls_are_explicit_and_have_no_electric_field() -> None:
    values = _values()
    values.update(
        control_times_fs=np.array([0.1, 0.3]),
        controls_v_per_m=np.zeros((2, 2)),
        control_layout=ControlLayout.PIECEWISE_CONSTANT_INTERVALS,
        electric_field=None,
    )

    result = OptimizationResult(**values)  # type: ignore[arg-type]

    assert result.control_layout is ControlLayout.PIECEWISE_CONSTANT_INTERVALS
    assert result.electric_field is None


def test_result_represents_an_optional_optimizer_target_as_none() -> None:
    values = _values()
    values["target_index"] = None

    result = OptimizationResult(**values)  # type: ignore[arg-type]

    assert result.target_index is None


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"control_layout": "rk4_field_samples"}, "ControlLayout"),
        ({"trajectory_times_fs": np.array([])}, "trajectory_times_fs"),
        ({"trajectory": np.zeros((3, 2))}, "trajectory.*aligned"),
        ({"control_times_fs": np.array([0.0, np.nan])}, "control_times_fs"),
        ({"controls_v_per_m": np.zeros((2, 2))}, "controls_v_per_m"),
        ({"controls_v_per_m": np.zeros((3, 2), dtype=complex)}, "controls_v_per_m"),
        ({"target_index": True}, "target_index"),
        ({"target_index": 2}, "target_index"),
        ({"metrics": None}, "metrics"),
        ({"electric_field": None}, "require an ElectricField"),
    ],
)
def test_result_rejects_inconsistent_common_data(
    updates: dict[str, object],
    match: str,
) -> None:
    values = _values()
    values.update(updates)

    with pytest.raises((TypeError, ValueError), match=match):
        OptimizationResult(**values)  # type: ignore[arg-type]


def test_interval_layout_rejects_a_misleading_sampled_electric_field() -> None:
    values = _values()
    values["control_layout"] = ControlLayout.PIECEWISE_CONSTANT_INTERVALS

    with pytest.raises(ValueError, match="do not have an ElectricField"):
        OptimizationResult(**values)  # type: ignore[arg-type]
