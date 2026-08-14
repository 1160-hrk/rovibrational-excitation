"""Public split-operator interaction-mode contracts."""

import numpy as np
import pytest

from rovibrational_excitation.core.propagation import SchrodingerPropagator
from rovibrational_excitation.core.propagation.capabilities import (
    PropagationAlgorithm,
)
from rovibrational_excitation.core.states import PureState
from tests.propagation_options import propagation_options
from tests.propagation_problem import propagation_problem


def _split_options():
    return propagation_options(
        algorithm=PropagationAlgorithm.SPLIT_OPERATOR,
        return_trajectory=False,
    )


def _state():
    return PureState(np.array([1.0, 0.0], dtype=np.complex128))


def test_split_public_boundary_requires_interaction_mode_before_work():
    solver = SchrodingerPropagator(
        algorithm="split_operator",
        validate_units=False,
    )

    with pytest.raises(ValueError, match="split_interaction is required"):
        solver.propagate(
            propagation_problem(_state()),
            options=_split_options(),
        )


def test_split_public_boundary_rejects_constructor_mode_conflict_before_work():
    solver = SchrodingerPropagator(
        algorithm="split_operator",
        split_interaction="helicity_projected",
        validate_units=False,
    )

    with pytest.raises(ValueError, match="conflicts with the propagator"):
        solver.propagate(
            propagation_problem(_state()),
            options=_split_options(),
            split_interaction="cartesian",
        )


def test_rk4_public_boundary_rejects_split_interaction():
    solver = SchrodingerPropagator(validate_units=False)

    with pytest.raises(ValueError, match="only to split-operator"):
        solver.propagate(
            propagation_problem(_state()),
            options=propagation_options(return_trajectory=False),
            split_interaction="cartesian",
        )
