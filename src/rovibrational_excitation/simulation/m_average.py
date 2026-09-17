"""Fixed-linear-polarization, M-averaged linear-molecule propagation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from rovibrational_excitation.core.execution import ExecutionPolicy
from rovibrational_excitation.core.states import PureState
from rovibrational_excitation.core.time import TimeGrid
from rovibrational_excitation.dynamics.options import PropagationOptions
from rovibrational_excitation.dynamics.problem import (
    Axis,
    CouplingSpec,
    PropagationProblem,
    SystemModel,
)
from rovibrational_excitation.dynamics.schrodinger import SchrodingerPropagator
from rovibrational_excitation.models.linear_molecule import (
    FixedMLinMolBasis,
    LinMolDipoleMatrix,
    LinMolParameters,
)
from rovibrational_excitation.models.validation import model_parameters_from_mapping

_LINEAR_POLARIZATION_TOL = 128.0 * np.finfo(np.float64).eps


def canonicalize_fixed_linear_polarization(polarization: Any) -> np.ndarray:
    """Return a real unit Jones vector, rejecting non-linear polarization."""
    vector = np.asarray(polarization, dtype=np.complex128)
    if vector.shape != (2,):
        raise ValueError("polarization must be a 2-element vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError("polarization must contain only finite values")
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("polarization must be non-zero")
    vector = vector / norm

    pivot = int(np.argmax(np.abs(vector)))
    vector = vector * np.exp(-1j * np.angle(vector[pivot]))
    if np.linalg.norm(vector.imag) > _LINEAR_POLARIZATION_TOL:
        raise ValueError(
            "representation=m_incoherent_average requires fixed linear "
            "polarization; circular and elliptical polarization require "
            "representation=m_resolved"
        )
    real_vector = vector.real
    return real_vector / np.linalg.norm(real_vector)


@dataclass(frozen=True)
class MBlockProblem:
    """One representative |M| block and its incoherent ensemble weight."""

    abs_m: int
    multiplicity: int
    weight: float
    basis: FixedMLinMolBasis
    hamiltonian: Any
    dipole: LinMolDipoleMatrix
    initial_state: np.ndarray
    reduced_indices: np.ndarray


@dataclass(frozen=True)
class MAveragePropagationResult:
    """Reduced populations plus auditable representative block trajectories."""

    time_fs: np.ndarray
    population: np.ndarray
    blocks: tuple[MBlockProblem, ...]
    block_wavefunctions: tuple[np.ndarray, ...]


def _reduced_initial_states(
    v_max: int,
    j_max: int,
    initial_states: Any,
) -> tuple[list[tuple[int, int]], int]:
    j_count = j_max + 1
    dimension = (v_max + 1) * j_count
    raw_indices = list(initial_states)
    if not raw_indices:
        raise ValueError("initial_states must contain at least one state index")

    indices: list[int] = []
    for raw in raw_indices:
        if isinstance(raw, bool) or not isinstance(raw, (int, np.integer)):
            raise ValueError("initial_states must contain integer state indices")
        index = int(raw)
        if index < 0 or index >= dimension:
            raise ValueError(
                f"initial state index {index} is outside reduced dimension {dimension}"
            )
        if index not in indices:
            indices.append(index)

    states = [divmod(index, j_count) for index in indices]
    initial_j = states[0][1]
    if any(j != initial_j for _, j in states[1:]):
        raise ValueError(
            "representation=m_incoherent_average cannot assign an isotropic M "
            "average to a coherent "
            "superposition spanning different J values; use one J value or "
            "an explicit incoherent ensemble"
        )
    return states, initial_j


def validate_m_average_initial_states(params: dict[str, Any]) -> None:
    """Validate reduced initial-state semantics without building any matrices."""
    _reduced_initial_states(
        params["V_max"],
        params["J_max"],
        params["initial_states"],
    )


def build_m_average_blocks(
    params: dict[str, Any], *, execution_policy: ExecutionPolicy
) -> tuple[MBlockProblem, ...]:
    """Build the non-negative |M| representatives for an isotropic M mixture."""
    model_params = model_parameters_from_mapping(params)
    if not isinstance(model_params, LinMolParameters):
        raise TypeError("M-average builder requires LinMolParameters")
    return build_m_average_blocks_from_parameters(
        model_params,
        params["initial_states"],
        execution_policy=execution_policy,
    )


def build_m_average_blocks_from_parameters(
    model_params: LinMolParameters,
    raw_initial_states: Any,
    *,
    execution_policy: ExecutionPolicy,
) -> tuple[MBlockProblem, ...]:
    """Build M blocks from the frozen model schema and immutable state indices."""
    initial_states, initial_j = _reduced_initial_states(
        model_params.v_max,
        model_params.j_max,
        raw_initial_states,
    )
    degeneracy = 2 * initial_j + 1
    amplitude = 1.0 / np.sqrt(len(initial_states))
    dense = execution_policy.dense
    blocks: list[MBlockProblem] = []

    for abs_m in range(initial_j + 1):
        basis = FixedMLinMolBasis(
            model_params.v_max,
            model_params.j_max,
            M=abs_m,
            omega=model_params.vibrational_frequency.angular_rad_per_fs,
            delta_omega=model_params.anharmonic_shift.angular_rad_per_fs,
            B=model_params.rotational_constant.angular_rad_per_fs,
            alpha=model_params.vibration_rotation_coupling.angular_rad_per_fs,
            output_units="J",
            input_units="rad/fs",
        )
        initial = np.zeros(basis.size(), dtype=np.complex128)
        for v, j in initial_states:
            initial[basis.get_index((v, j, abs_m))] = amplitude

        multiplicity = 1 if abs_m == 0 else 2
        dipole = LinMolDipoleMatrix(
            basis,
            mu0=model_params.dipole_c_m,
            potential_type=model_params.potential_type,
            backend=execution_policy.backend.value,
            dense=dense,
        )
        reduced_indices = (
            basis.V_array * (model_params.j_max + 1) + basis.J_array
        ).astype(np.int64)
        blocks.append(
            MBlockProblem(
                abs_m=abs_m,
                multiplicity=multiplicity,
                weight=multiplicity / degeneracy,
                basis=basis,
                hamiltonian=basis.generate_H0(),
                dipole=dipole,
                initial_state=initial,
                reduced_indices=reduced_indices,
            )
        )
    return tuple(blocks)


def propagate_m_average(
    model_params: LinMolParameters,
    initial_states: Any,
    electric_field: Any,
    *,
    time_grid: TimeGrid,
    options: PropagationOptions,
    validate_units: bool,
    verbose: bool,
) -> MAveragePropagationResult:
    """Propagate fixed-M blocks and incoherently sum reduced populations."""
    blocks = build_m_average_blocks_from_parameters(
        model_params,
        initial_states,
        execution_policy=options.execution,
    )
    sparse = options.sparse
    algorithm_name = options.algorithm_name
    split_interaction: Literal["cartesian"] = "cartesian"
    propagator = SchrodingerPropagator(
        backend=options.backend_name,
        algorithm=algorithm_name,
        split_interaction=split_interaction,
        validate_units=validate_units,
        renorm=options.renorm,
        sparse=sparse,
    )
    reduced_dimension = (model_params.v_max + 1) * (model_params.j_max + 1)
    time_reference: np.ndarray | None = None
    population: np.ndarray | None = None
    trajectories: list[np.ndarray] = []

    for block in blocks:
        problem = PropagationProblem(
            model=SystemModel(
                name="linmol_m_average_block",
                basis=block.basis,
                hamiltonian=block.hamiltonian,
                dipole=block.dipole,
                coupling=CouplingSpec.scalar(Axis.Z),
                metadata={"abs_m": block.abs_m},
            ),
            field=electric_field,
            time_grid=time_grid,
            initial_state=PureState(block.initial_state),
        )
        propagation_result = propagator.propagate(
            problem,
            options=options,
            verbose=verbose,
            split_interaction=(
                split_interaction if algorithm_name == "split_operator" else None
            ),
        )
        host_result = propagation_result.to_numpy()
        time_fs = host_result.times_fs
        wavefunction = host_result.state
        if wavefunction.ndim == 1:
            wavefunction = wavefunction.reshape(1, -1)

        if time_reference is None:
            time_reference = time_fs
            population = np.zeros(
                (wavefunction.shape[0], reduced_dimension), dtype=np.float64
            )
        elif not np.array_equal(time_fs, time_reference):
            raise RuntimeError("M blocks produced different physical output time grids")
        assert population is not None
        if wavefunction.shape[0] != population.shape[0]:
            raise RuntimeError("M blocks produced different trajectory lengths")

        block_population = np.abs(wavefunction) ** 2
        for block_index, reduced_index in enumerate(block.reduced_indices):
            population[:, reduced_index] += (
                block.weight * block_population[:, block_index]
            )
        trajectories.append(wavefunction)

    assert time_reference is not None and population is not None
    return MAveragePropagationResult(
        time_fs=time_reference,
        population=population,
        blocks=blocks,
        block_wavefunctions=tuple(trajectories),
    )
