"""Input and forwarding contracts for density-matrix propagators."""

from types import SimpleNamespace

import numpy as np
import pytest

from rovibrational_excitation.core.propagation import (
    LiouvillePropagator,
    MixedStatePropagator,
)
from rovibrational_excitation.core.propagation.algorithms.rk4.lvne import (
    rk4_lvne,
    rk4_lvne_traj,
)
from rovibrational_excitation.core.propagation.algorithms.rk4.schrodinger import (
    rk4_schrodinger,
)
from rovibrational_excitation.core.propagation.algorithms.validation import (
    validate_density_matrix_properties,
)


def _low_level_problem(field_size=5):
    h0 = np.diag([0.0, 1.0]).astype(np.complex128)
    mu = np.zeros((2, 2), dtype=np.complex128)
    fields = np.zeros(field_size)
    rho0 = np.eye(2, dtype=np.complex128) / 2.0
    steps = (field_size - 1) // 2
    return h0, mu, fields, rho0, steps


def test_lvne_rejects_malformed_density_matrix():
    h0, mu, fields, _, steps = _low_level_problem()

    with pytest.raises(ValueError, match="rho0 must have shape"):
        rk4_lvne(h0, mu, mu, fields, fields, np.ones(2), 0.1, steps)


def test_lvne_rejects_even_field_grid():
    h0, mu, fields, rho0, steps = _low_level_problem(field_size=4)

    with pytest.raises(ValueError, match=r"2\*n_steps \+ 1"):
        rk4_lvne(h0, mu, mu, fields, fields, rho0, 0.1, steps)


def test_lvne_rejects_steps_inconsistent_with_field_grid():
    h0, mu, fields, rho0, steps = _low_level_problem()

    with pytest.raises(ValueError, match="steps must be"):
        rk4_lvne_traj(h0, mu, mu, fields, fields, rho0, 0.1, steps + 1)


def test_liouville_advertises_only_implemented_backend():
    solver = LiouvillePropagator(validate_units=False)

    assert solver.get_supported_backends() == ["numpy"]
    with pytest.raises(ValueError, match="only backend='numpy'"):
        LiouvillePropagator(backend="cupy", validate_units=False)


def test_liouville_rejects_ignored_timestep_override():
    solver = LiouvillePropagator(validate_units=False)

    with pytest.raises(ValueError, match="dt override is unsupported"):
        solver.propagate(None, None, None, np.eye(2), dt=0.1)


def test_liouville_returns_physical_time_and_forwards_preparation_options(
    monkeypatch,
):
    import rovibrational_excitation.core.propagation.liouville as liouville_module

    captured = {}
    h0, mu, fields, rho0, _ = _low_level_problem()

    def fake_prepare(*args, **kwargs):
        captured.update(kwargs)
        return h0, mu, mu, fields, fields, None, None, 0.5, 1.0

    monkeypatch.setattr(liouville_module, "prepare_propagation_args", fake_prepare)
    efield = SimpleNamespace(tlist=np.linspace(-1.0, 0.0, fields.size))

    time, rho = LiouvillePropagator(validate_units=False).propagate(
        object(),
        efield,
        object(),
        rho0,
        return_traj=True,
        return_time_rho=True,
        sample_stride=1,
        target_accuracy="high",
        coupling_mode="scalar",
        coupling_axis="z",
    )

    np.testing.assert_allclose(time, [-1.0, -0.5, 0.0])
    assert rho.shape == (3, 2, 2)
    assert captured["target_accuracy"] == "high"
    assert captured["coupling_mode"] == "scalar"
    assert captured["coupling_axis"] == "z"


def test_liouville_final_state_time_is_field_endpoint(monkeypatch):
    import rovibrational_excitation.core.propagation.liouville as liouville_module

    h0, mu, fields, rho0, _ = _low_level_problem()
    monkeypatch.setattr(
        liouville_module,
        "prepare_propagation_args",
        lambda *args, **kwargs: (
            h0,
            mu,
            mu,
            fields,
            fields,
            None,
            None,
            0.5,
            1.0,
        ),
    )
    efield = SimpleNamespace(tlist=np.linspace(-1.0, 0.0, fields.size))

    time, rho = LiouvillePropagator(validate_units=False).propagate(
        object(),
        efield,
        object(),
        rho0,
        return_traj=False,
        return_time_rho=True,
    )

    np.testing.assert_array_equal(time, [0.0])
    assert rho.shape == (2, 2)


def test_mixed_state_forwards_solver_configuration_and_returns_final_time():
    solver = MixedStatePropagator(
        algorithm="split_operator",
        sparse=True,
        validate_units=False,
    )
    assert solver._schrodinger_prop.algorithm == "split_operator"
    assert solver._schrodinger_prop.sparse is True

    calls = []

    def fake_propagate(*args, **kwargs):
        calls.append(kwargs)
        return np.array([3.0]), np.asarray(args[3])

    solver._schrodinger_prop.propagate = fake_propagate
    states = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]

    time, rho = solver.propagate(
        object(),
        object(),
        object(),
        states,
        return_traj=False,
        return_time_rho=True,
        sample_stride=3,
        nondimensional=True,
        auto_timestep=True,
        target_accuracy="fast",
        coupling_mode="scalar",
        coupling_axis="x",
    )

    np.testing.assert_array_equal(time, [3.0])
    np.testing.assert_allclose(rho, np.eye(2) / 2.0)
    assert len(calls) == 2
    for call in calls:
        assert call["return_time_psi"] is True
        assert call["return_traj"] is False
        assert call["sample_stride"] == 3
        assert call["nondimensional"] is True
        assert call["auto_timestep"] is True
        assert call["target_accuracy"] == "fast"
        assert call["coupling_mode"] == "scalar"
        assert call["coupling_axis"] == "x"
        assert call["algorithm"] == "split_operator"
        assert call["sparse"] is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"algorithm": "split_operator"}, "only algorithm='rk4'"),
        ({"sparse": True}, "does not support sparse"),
    ],
)
def test_liouville_rejects_unsupported_solver_options(kwargs, message):
    solver = LiouvillePropagator(validate_units=False)

    with pytest.raises(ValueError, match=message):
        solver.propagate(None, None, None, np.eye(2), **kwargs)


@pytest.mark.parametrize(
    ("constructor_kwargs", "message"),
    [
        ({"algorithm": "split_operator"}, "only algorithm='rk4'"),
        ({"sparse": True}, "does not support sparse"),
    ],
)
def test_mixed_state_rejects_unsupported_options_for_explicit_density(
    constructor_kwargs,
    message,
):
    solver = MixedStatePropagator(validate_units=False, **constructor_kwargs)

    with pytest.raises(ValueError, match=message):
        solver.propagate(object(), object(), object(), np.eye(2))


def test_liouville_matches_schrodinger_for_a_pure_state():
    h0 = np.diag([0.0, 0.4]).astype(np.complex128)
    mu_x = np.array([[0.2, 1.0], [1.0, -0.1]], dtype=np.complex128)
    mu_y = np.zeros((2, 2), dtype=np.complex128)
    field_x = np.array([0.7, 0.9, 1.1])
    field_y = np.zeros(3)
    psi0 = np.array([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2.0)
    rho0 = np.outer(psi0, psi0.conj())
    dt = 1.0e-3

    psi = rk4_schrodinger(
        h0,
        mu_x,
        mu_y,
        field_x,
        field_y,
        psi0,
        dt,
        return_traj=True,
    )[-1]
    rho = rk4_lvne_traj(
        h0,
        mu_x,
        mu_y,
        field_x,
        field_y,
        rho0,
        dt,
        steps=1,
    )[-1]

    np.testing.assert_allclose(rho, np.outer(psi, psi.conj()), atol=1.0e-13)


@pytest.mark.parametrize(
    ("rho", "message"),
    [
        (
            np.array([[0.5, 0.2], [0.0, 0.5]], dtype=np.complex128),
            "Hermitian",
        ),
        (
            np.diag([1.001, -0.001]).astype(np.complex128),
            "positive semidefinite",
        ),
        (
            np.array([[np.nan, 0.0], [0.0, 1.0]], dtype=np.complex128),
            "finite",
        ),
    ],
)
def test_density_matrix_validation_rejects_nonphysical_input(rho, message):
    with pytest.raises(ValueError, match=message):
        validate_density_matrix_properties(rho)


def test_density_matrix_validation_allows_roundoff_scale_negative_eigenvalue():
    roundoff_eigenvalue = -1.0e-15
    rho = np.diag([1.0 - roundoff_eigenvalue, roundoff_eigenvalue])

    validate_density_matrix_properties(rho)
