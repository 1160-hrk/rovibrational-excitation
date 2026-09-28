"""Exact arithmetic contracts for typed optimization objectives."""

from __future__ import annotations

import numpy as np

from rovibrational_excitation.optimization import (
    DiscreteL2TargetObjective,
    IndexedTargetPopulation,
    TargetPopulationEvaluator,
    VectorTargetPopulation,
)


def _evaluate(
    evaluator: TargetPopulationEvaluator,
    state: np.ndarray,
) -> tuple[float, float]:
    evaluation = evaluator.evaluate(state)
    return evaluation.fidelity, evaluation.infidelity


def test_indexed_target_preserves_runner_arithmetic_exactly() -> None:
    state = np.array([0.2 + 0.3j, -0.4 + 0.5j], dtype=np.complex128)
    expected = float(np.abs(state[1]) ** 2)

    fidelity, infidelity = _evaluate(IndexedTargetPopulation(1), state)

    assert fidelity == expected
    assert infidelity == 1.0 - expected


def test_vector_target_preserves_vdot_and_target_identity_exactly() -> None:
    target = np.array([1.0j, 2.0 - 0.5j], dtype=np.complex128)
    state = np.array([0.2 + 0.3j, -0.4 + 0.5j], dtype=np.complex128)
    evaluator = VectorTargetPopulation(target)
    expected_overlap = complex(np.vdot(target, state))
    expected_fidelity = float(np.abs(expected_overlap) ** 2)

    overlap, evaluation = evaluator.evaluate_with_overlap(state)

    assert evaluator.target_state is target
    assert overlap == expected_overlap
    assert evaluation.fidelity == expected_fidelity
    assert evaluation.infidelity == 1.0 - expected_fidelity


def test_discrete_l2_objective_preserves_grape_expression_exactly() -> None:
    state = np.array([0.6 + 0.1j, 0.2 - 0.3j], dtype=np.complex128)
    controls = np.array([[2.0, -3.0], [5.0, 7.0]], dtype=np.float64)
    lambda_a = 0.125
    target = IndexedTargetPopulation(1).evaluate(state)
    expected = float(
        1.0 - target.fidelity + 0.5 * lambda_a * float(np.sum(controls * controls))
    )

    actual = DiscreteL2TargetObjective(lambda_a).evaluate(target, controls)

    assert actual == expected


def test_indexed_and_vector_paths_remain_explicit_for_one_hot_target() -> None:
    state = np.array([0.6 + 0.1j, 0.2 - 0.3j], dtype=np.complex128)
    target = np.array([0.0, 1.0], dtype=np.complex128)

    indexed = IndexedTargetPopulation(1).evaluate(state)
    vector = VectorTargetPopulation(target).evaluate(state)

    assert indexed == vector
