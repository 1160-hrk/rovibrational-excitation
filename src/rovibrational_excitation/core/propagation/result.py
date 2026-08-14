"""Endpoint-complete, backend-explicit propagation results."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .options import PropagationOptions
from .problem import PropagationProblem

RESULT_SCHEMA_VERSION = 1
_HASH_SCOPE = "declared_model_metadata_and_propagation_contract"

StateKind = Literal["wavefunction", "density_matrix"]
BackendName = Literal["numpy", "cupy"]


@lru_cache(maxsize=1)
def _package_version() -> str:
    try:
        return version("rovibrational-excitation")
    except PackageNotFoundError:
        return "0.0.0+dev"


def _json_value(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"{path} must contain only finite JSON values")
        return number
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            converted[key] = _json_value(item, path=f"{path}.{key}")
        return converted
    if isinstance(value, (list, tuple)):
        return [
            _json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{path} must contain JSON-compatible values; got {type(value).__name__}"
    )


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _state_shape(state: Any) -> tuple[int, ...]:
    shape = getattr(state, "shape", None)
    if not isinstance(shape, tuple) or not all(
        isinstance(item, (int, np.integer)) for item in shape
    ):
        raise TypeError("state must expose an integer shape")
    return tuple(int(item) for item in shape)


def _validate_state_shape(
    *,
    times_size: int,
    state_shape: tuple[int, ...],
    state_kind: StateKind,
    trajectory: bool,
) -> None:
    if trajectory:
        expected_ndim = 2 if state_kind == "wavefunction" else 3
        if len(state_shape) != expected_ndim or state_shape[0] != times_size:
            raise ValueError(
                "trajectory state must have a leading dimension equal to times_fs"
            )
        if state_kind == "density_matrix" and state_shape[1] != state_shape[2]:
            raise ValueError("density-matrix trajectory states must be square")
        return

    if times_size != 1:
        raise ValueError("a final-state result must contain exactly one time")
    if state_kind == "wavefunction":
        if len(state_shape) != 1:
            raise ValueError("final wavefunction state must be one-dimensional")
        return
    if len(state_shape) != 2 or state_shape[0] != state_shape[1]:
        raise ValueError("final density-matrix state must be square")


@dataclass(frozen=True, slots=True, eq=False)
class PropagationResult:
    """One unconditional high-level propagation return value."""

    times_fs: NDArray[np.float64]
    state: Any
    state_kind: StateKind
    trajectory: bool
    backend: BackendName
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        times = np.array(self.times_fs, dtype=np.float64, copy=True)
        if times.ndim != 1 or times.size < 1:
            raise ValueError("times_fs must be a nonempty one-dimensional array")
        if not np.all(np.isfinite(times)):
            raise ValueError("times_fs must contain only finite values")
        if times.size > 1:
            intervals = np.diff(times)
            if not (np.all(intervals > 0.0) or np.all(intervals < 0.0)):
                raise ValueError("times_fs must be strictly monotonic")
        if self.state_kind not in {"wavefunction", "density_matrix"}:
            raise ValueError("state_kind must be wavefunction or density_matrix")
        if not isinstance(self.trajectory, bool):
            raise TypeError("trajectory must be a bool")
        if self.backend not in {"numpy", "cupy"}:
            raise ValueError("backend must be numpy or cupy")
        if self.backend == "numpy" and not isinstance(self.state, np.ndarray):
            raise TypeError("backend='numpy' requires a NumPy state array")
        if self.backend == "cupy" and not hasattr(
            self.state, "__cuda_array_interface__"
        ):
            raise TypeError("backend='cupy' requires a device state array")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")

        _validate_state_shape(
            times_size=int(times.size),
            state_shape=_state_shape(self.state),
            state_kind=self.state_kind,
            trajectory=self.trajectory,
        )
        metadata = _json_value(self.metadata, path="metadata")
        times.setflags(write=False)
        object.__setattr__(self, "times_fs", times)
        object.__setattr__(self, "metadata", _freeze_json(metadata))

    def to_numpy(self) -> PropagationResult:
        """Create an explicit host-owned result without changing provenance."""
        getter = getattr(self.state, "get", None)
        host_state = getter() if callable(getter) else self.state
        return PropagationResult(
            times_fs=self.times_fs,
            state=np.array(host_state, copy=True),
            state_kind=self.state_kind,
            trajectory=self.trajectory,
            backend="numpy",
            metadata=self.metadata,
        )


def _scale_metadata(scales: Any | None) -> dict[str, Any] | None:
    if scales is None:
        return None
    return {
        "energy_J": scales.E0,
        "dipole_Cm": scales.mu0,
        "field_V_per_m": scales.Efield0,
        "time_s": scales.t0,
        "lambda_coupling": scales.lambda_coupling,
        "energy_offset_J": scales.energy_offset,
        "free_energy_span_J": scales.free_energy_span,
        "interaction_energy_J": scales.interaction_energy,
        "physical_coupling_ratio": scales.physical_coupling_ratio,
        "energy_source": scales.reference_energy.source,
        "energy_method": scales.reference_energy.method,
        "dipole_source": scales.dipole_scale.source,
        "field_source": scales.field_scale.source,
    }


def _result_metadata(
    problem: PropagationProblem,
    options: PropagationOptions,
    *,
    scales: Any | None,
) -> dict[str, Any]:
    coupling: dict[str, Any] = (
        {"mode": "scalar", "axis": problem.coupling.axes[0]}
        if problem.coupling_mode == "scalar"
        else {"mode": "cartesian", "axes": list(problem.coupling.axes)}
    )
    configuration = {
        "model": {
            "name": problem.model.name,
            "dimension": problem.model.dimension,
            "metadata": dict(problem.model.metadata),
        },
        "coupling": coupling,
        "algorithm": options.algorithm.value,
        "execution_backend": options.execution.backend.value,
        "matrix_storage": options.execution.storage.value,
        "scaling": options.scaling.value,
        "renormalization": options.renormalization.value,
        "return_trajectory": options.return_trajectory,
        "sample_stride": options.sample_stride,
        "time_grid": {
            "start_fs": problem.time_grid.t_start_fs,
            "end_fs": problem.time_grid.t_end_fs,
            "field_dt_fs": problem.time_grid.field_dt_fs,
            "propagation_dt_fs": problem.time_grid.propagation_dt_fs,
            "propagation_steps": problem.time_grid.propagation_steps,
        },
        "nondimensionalization_scales": _scale_metadata(scales),
    }
    canonical = _json_value(configuration, path="configuration")
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "package_version": _package_version(),
        **configuration,
        "configuration_hash": hashlib.sha256(encoded).hexdigest(),
        "configuration_hash_scope": _HASH_SCOPE,
    }


def _native_state(state: Any, backend: BackendName) -> Any:
    if backend == "numpy":
        return np.asarray(state)
    if hasattr(state, "__cuda_array_interface__"):
        return state
    cupy = import_module("cupy")
    return cupy.asarray(state)


def finalize_propagation_result(
    *,
    problem: PropagationProblem,
    options: PropagationOptions,
    times_fs: Any,
    state: Any,
    state_kind: StateKind,
    scales: Any | None,
    backward: bool = False,
) -> PropagationResult:
    """Apply output-only stride and endpoint completion to a full result."""
    times = np.asarray(times_fs, dtype=np.float64)
    state_output = state
    if options.return_trajectory:
        expected = problem.time_grid.propagation_steps + 1
        shape = _state_shape(state)
        if times.shape != (expected,) or not shape or shape[0] != expected:
            raise RuntimeError(
                "private propagator did not return the complete internal trajectory"
            )
        if options.sample_stride == 1:
            times = np.array(times, copy=True)
        else:
            indices = np.arange(0, expected, options.sample_stride, dtype=np.int64)
            if indices[-1] != expected - 1:
                indices = np.append(indices, expected - 1)
            state_output = state[indices]
            times = np.array(times[indices], copy=True)
        times[0] = (
            problem.time_grid.t_end_fs if backward else problem.time_grid.t_start_fs
        )
        times[-1] = (
            problem.time_grid.t_start_fs if backward else problem.time_grid.t_end_fs
        )
    else:
        times = np.array(
            [problem.time_grid.t_start_fs if backward else problem.time_grid.t_end_fs],
            dtype=np.float64,
        )

    backend: BackendName = options.backend_name
    state_output = _native_state(state_output, backend)
    return PropagationResult(
        times_fs=times,
        state=state_output,
        state_kind=state_kind,
        trajectory=options.return_trajectory,
        backend=backend,
        metadata=_result_metadata(problem, options, scales=scales),
    )


__all__ = [
    "BackendName",
    "PropagationResult",
    "RESULT_SCHEMA_VERSION",
    "StateKind",
]
