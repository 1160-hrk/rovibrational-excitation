"""Prepare and propagate one validated normal-simulation case."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from ..dynamics.result import PropagationResult
from ..fields import SampledField
from ..models.factory import build_model_from_parameters
from ..models.linear_molecule import LinMolParameters
from ..models.validation import LinMolRepresentation
from .case import SimulationCase
from .field_preparation import _generated_sampled_field
from .m_average import MAveragePropagationResult, propagate_m_average


@dataclass(frozen=True, slots=True)
class WavefunctionCaseResult:
    """Host result values consumed by simulation persistence and callers."""

    propagation: PropagationResult
    population: np.ndarray
    regime_info: dict[str, Any] | None


def prepare_simulation_case(
    params: Mapping[str, Any],
    *,
    field: SampledField | None,
) -> SimulationCase:
    """Validate, sample, and freeze one generated or externally supplied case."""
    from .validation import _resolve_simulation_case

    validated = _resolve_simulation_case(params, field=field)
    options = validated.options
    use_m_average = (
        params["basis_type"].lower() == "linmol"
        and params["representation"] == LinMolRepresentation.M_INCOHERENT_AVERAGE.value
    )
    expects_cartesian = params["basis_type"].lower() == "symtop" or (
        params["basis_type"].lower() == "linmol"
        and params["representation"] == LinMolRepresentation.M_RESOLVED.value
    )
    if field is None:
        generated_parameters = validated.generated_field
        if generated_parameters is None:
            raise RuntimeError("validated generated field parameters are unavailable")
        sampled_field = _generated_sampled_field(
            params,
            generated_parameters=generated_parameters,
            use_m_average=use_m_average,
            expects_cartesian=expects_cartesian,
        )
    else:
        sampled_field = field
    return SimulationCase.from_validated_mapping(
        params,
        field=sampled_field,
        options=options,
    )


def propagate_simulation_case(
    simulation_case: SimulationCase,
) -> MAveragePropagationResult | WavefunctionCaseResult:
    """Propagate one immutable case through its explicitly selected route."""
    from rovibrational_excitation.core.states import PureState
    from rovibrational_excitation.dynamics.problem import PropagationProblem
    from rovibrational_excitation.dynamics.scaling.reporting import analyze_regime
    from rovibrational_excitation.dynamics.schrodinger import SchrodingerPropagator

    options = simulation_case.options
    execution_policy = options.execution
    time_grid = simulation_case.time_grid
    sampled_field = simulation_case.field

    if simulation_case.uses_m_average:
        model_parameters = simulation_case.model_parameters
        if not isinstance(model_parameters, LinMolParameters):
            raise RuntimeError("M-average case does not contain LinMolParameters")
        return propagate_m_average(
            model_parameters,
            simulation_case.initial_states,
            sampled_field,
            time_grid=time_grid,
            options=options,
            validate_units=simulation_case.validate_units,
            verbose=simulation_case.verbose,
        )

    model = build_model_from_parameters(
        simulation_case.model_parameters,
        initial_states=simulation_case.initial_states,
        representation=simulation_case.representation,
        axes=simulation_case.axes,
        execution_policy=execution_policy,
    )
    problem = PropagationProblem(
        model=model.to_system_model(),
        field=cast(Any, sampled_field),
        time_grid=time_grid,
        initial_state=PureState(model.state.data.ravel()),
    )
    hamiltonian = problem.model.hamiltonian
    dipole = problem.model.dipole

    algorithm_name = options.algorithm_name
    split_interaction = simulation_case.split_interaction
    propagator = SchrodingerPropagator(
        backend=options.backend_name,
        algorithm=algorithm_name,
        split_interaction=split_interaction,
        validate_units=simulation_case.validate_units,
        renorm=options.renorm,
        sparse=options.sparse,
    )
    propagation_result = propagator.propagate(
        problem,
        options=options,
        verbose=simulation_case.verbose,
        split_interaction=(
            split_interaction if algorithm_name == "split_operator" else None
        ),
    )

    regime_info = None
    if options.nondimensional:
        from rovibrational_excitation.dynamics.scaling.converter import (
            nondimensionalize_from_objects,
        )

        coupling_axes = problem.coupling.axes
        *_, scales = nondimensionalize_from_objects(
            hamiltonian,
            dipole,
            cast(Any, sampled_field),
            coupling_axes=coupling_axes,
            scalar_coupling=problem.coupling_mode == "scalar",
            verbose=False,
        )
        regime_info = analyze_regime(scales)

    host_result = propagation_result.to_numpy()
    state = cast(np.ndarray, host_result.state)
    population = np.abs(state) ** 2
    if isinstance(population, np.ndarray):
        if population.ndim == 0:
            population = np.array([[float(population)]], dtype=float)
        elif population.ndim == 1:
            population = population.reshape(1, -1)

    return WavefunctionCaseResult(
        propagation=host_result,
        population=population,
        regime_info=regime_info,
    )


__all__ = [
    "WavefunctionCaseResult",
    "prepare_simulation_case",
    "propagate_simulation_case",
]
