"""Construction shared by simulation model builders."""

from __future__ import annotations

from typing import Any

from rovibrational_excitation.core.basis import StateVector


def build_initial_state(basis: Any, state_indices: Any) -> StateVector:
    """Build a normalized, equal-amplitude coherent superposition."""
    indices = list(state_indices)
    if not indices:
        raise ValueError("initial_states must contain at least one state index")

    state = StateVector(basis)
    for index in indices:
        state.set_amplitude(basis.get_state(index), 1.0)
    state.normalize()
    return state
