"""Propagation of incoherent statistical mixtures of pure states."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np

from ..units.validators import validator
from .base import PropagatorBase
from .schrodinger import SchrodingerPropagator
from .utils import get_backend


def _normalized_ensemble(
    initial_states: Iterable[np.ndarray],
) -> tuple[list[np.ndarray], np.ndarray]:
    """Normalize state vectors and their norm-squared statistical weights."""
    raw_states = list(initial_states)
    if not raw_states:
        raise ValueError("initial_state ensemble must not be empty")

    states: list[np.ndarray] = []
    raw_weights: list[float] = []
    dimension: int | None = None
    for index, state in enumerate(raw_states):
        state_array = np.asarray(state, dtype=np.complex128).reshape(-1)
        if dimension is None:
            dimension = state_array.size
        elif state_array.size != dimension:
            raise ValueError(
                "all initial states must have the same dimension; "
                f"state 0 has {dimension}, state {index} has {state_array.size}"
            )

        weight = float(np.vdot(state_array, state_array).real)
        if not np.isfinite(weight):
            raise ValueError("initial-state weights must be finite")
        if weight == 0.0:
            continue
        states.append(state_array / np.sqrt(weight))
        raw_weights.append(weight)

    if not raw_weights:
        raise ValueError("at least one initial state must have non-zero norm")

    weights = np.asarray(raw_weights, dtype=float)
    weights /= weights.sum()
    return states, weights


def _normalized_density_matrix(initial_state: np.ndarray) -> np.ndarray:
    """Normalize an explicitly supplied density matrix to unit trace."""
    density = np.asarray(initial_state, dtype=np.complex128)
    trace = np.trace(density)
    if not np.isfinite(trace) or not np.isclose(trace.imag, 0.0):
        raise ValueError("density-matrix trace must be finite and real")
    if trace.real <= 0.0:
        raise ValueError("density-matrix trace must be positive")
    return density / trace.real


class MixedStatePropagator(PropagatorBase):
    """Propagate a normalized statistical mixture of pure states."""

    def __init__(
        self,
        algorithm: Literal["rk4", "split_operator"] = "rk4",
        backend: Literal["numpy", "cupy"] = "numpy",
        sparse: bool = False,
        validate_units: bool = True,
        renorm: bool = False,
    ):
        super().__init__(validate_units)
        self.algorithm = algorithm
        self.backend = backend
        self.sparse = sparse
        self._schrodinger_prop = SchrodingerPropagator(
            backend=backend,
            algorithm=algorithm,
            validate_units=validate_units,
            renorm=renorm,
            sparse=sparse,
        )

    def get_algorithm_name(self) -> str:
        """Return the selected mixed-state algorithm name."""
        return f"MixedState-{self.algorithm}"

    def get_supported_backends(self) -> list:
        """Return computational backends supported by the pure-state solver."""
        return self._schrodinger_prop.get_supported_backends()

    def propagate(
        self,
        hamiltonian,
        efield,
        dipole_matrix,
        initial_state: np.ndarray | Iterable[np.ndarray],
        **kwargs,
    ) -> np.ndarray | tuple:
        """Propagate a density matrix or an ensemble of pure states.

        For an ensemble, each input vector contributes a raw statistical weight
        ``w_i = ||psi_i||^2``. The weights are normalized to sum to one and each
        vector is normalized before propagation. Unit-norm vectors therefore
        form an equal mixture; arbitrary weights can be encoded as
        ``sqrt(w_i) * psi_i``.
        """
        return_traj = kwargs.get("return_traj", True)
        return_time_rho = kwargs.get("return_time_rho", False)
        verbose = kwargs.get("verbose", False)

        if self.validate_units:
            warnings = validator.validate_propagation_units(
                hamiltonian, dipole_matrix, efield
            )
            if warnings:
                self._last_validation_warnings = warnings
                if verbose:
                    self.print_validation_warnings()

        if (
            isinstance(initial_state, np.ndarray)
            and initial_state.ndim == 2
            and initial_state.shape[0] == initial_state.shape[1]
        ):
            from .liouville import LiouvillePropagator

            if self.algorithm != "rk4":
                raise ValueError("density matrices support only algorithm='rk4'")
            if self.sparse:
                raise ValueError(
                    "density-matrix propagation does not support sparse matrices"
                )

            liouville_prop = LiouvillePropagator(
                backend=self.backend,
                validate_units=False,
            )
            density = _normalized_density_matrix(initial_state)
            return liouville_prop.propagate(
                hamiltonian, efield, dipole_matrix, density, **kwargs
            )

        states, weights = _normalized_ensemble(initial_state)
        xp = get_backend(self.backend)
        rho_out = None
        time_psi = None

        propagation_kwargs = dict(kwargs)
        propagation_kwargs.pop("return_time_rho", None)
        propagation_kwargs["return_time_psi"] = return_time_rho
        propagation_kwargs["algorithm"] = self.algorithm
        propagation_kwargs.setdefault("sparse", self.sparse)
        propagation_kwargs["verbose"] = False

        for psi0, weight in zip(states, weights):
            result = self._schrodinger_prop.propagate(
                hamiltonian,
                efield,
                dipole_matrix,
                psi0,
                **propagation_kwargs,
            )

            if isinstance(result, tuple):
                component_time, psi_t = result
                if time_psi is None:
                    time_psi = component_time
                elif not np.allclose(time_psi, component_time):
                    raise RuntimeError(
                        "ensemble components returned inconsistent times"
                    )
            else:
                psi_t = result

            psi_backend = xp.asarray(psi_t)
            if return_traj:
                component_density = xp.einsum(
                    "ti, tj -> tij", psi_backend, psi_backend.conj()
                )
            else:
                component_density = xp.outer(psi_backend, psi_backend.conj())

            if rho_out is None:
                rho_out = xp.zeros_like(component_density)
            rho_out += float(weight) * component_density

        if return_time_rho and time_psi is not None:
            return time_psi, rho_out
        return rho_out
