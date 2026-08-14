"""Propagation of incoherent statistical mixtures of pure states."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ..states import DensityState, IncoherentEnsemble
from ..units.validators import validator
from .base import PropagatorBase
from .options import PropagationOptions
from .problem import PropagationProblem
from .result import PropagationResult, finalize_propagation_result
from .schrodinger import SchrodingerPropagator
from .utils import get_backend


class MixedStatePropagator(PropagatorBase[DensityState | IncoherentEnsemble]):
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

    def get_supported_backends(self) -> list[str]:
        """Return computational backends supported by the pure-state solver."""
        return self._schrodinger_prop.get_supported_backends()

    def propagate(
        self,
        problem: PropagationProblem,
        *,
        options: PropagationOptions,
        verbose: bool = False,
        split_interaction: Literal["cartesian", "helicity_projected"] | None = None,
    ) -> PropagationResult:
        """Propagate one complete typed statistical-state problem."""
        if not isinstance(problem, PropagationProblem):
            raise TypeError("problem must be a PropagationProblem")
        initial_state = problem.initial_state
        if not isinstance(initial_state, (DensityState, IncoherentEnsemble)):
            raise TypeError(
                "problem initial_state must be an IncoherentEnsemble or DensityState"
            )
        self._schrodinger_prop._validate_public_options(options)
        self._schrodinger_prop._validate_public_split_interaction(
            options, split_interaction
        )
        hamiltonian = problem.model.hamiltonian
        efield = problem.field
        dipole_matrix = problem.model.dipole

        if self.validate_units:
            warnings = validator.validate_propagation_units(
                hamiltonian, dipole_matrix, efield
            )
            if warnings:
                self._last_validation_warnings = warnings
                if verbose:
                    self.print_validation_warnings()

        if isinstance(initial_state, DensityState):
            from .liouville import LiouvillePropagator

            liouville_prop = LiouvillePropagator(
                backend=self.backend,
                validate_units=False,
            )
            liouville_prop._validate_public_options(options)
            times_fs, state, scales = liouville_prop._propagate_array(
                hamiltonian,
                efield,
                dipole_matrix,
                initial_state.matrix,
                return_traj=options.return_trajectory,
                return_time_rho=True,
                sample_stride=1,
                _return_context=True,
                nondimensional=options.nondimensional,
                coupling_mode=problem.coupling_mode,
                **problem.coupling_kwargs,
                verbose=False,
                algorithm=options.algorithm_name,
                sparse=options.sparse,
            )
            return finalize_propagation_result(
                problem=problem,
                options=options,
                times_fs=times_fs,
                state=state,
                state_kind="density_matrix",
                scales=scales,
            )

        xp = get_backend(self.backend)
        rho_out = None
        time_psi = None
        scales_reference = None
        propagation_kwargs = {
            "return_traj": options.return_trajectory,
            "return_time_psi": True,
            "sample_stride": 1,
            "_return_context": True,
            "nondimensional": options.nondimensional,
            "coupling_mode": problem.coupling_mode,
            **problem.coupling_kwargs,
            "verbose": False,
            "algorithm": options.algorithm_name,
            "sparse": options.sparse,
            "renorm": options.renorm,
            **(
                {"split_interaction": split_interaction}
                if split_interaction is not None
                else {}
            ),
        }

        for state, weight in zip(initial_state.states, initial_state.weights):
            result = self._schrodinger_prop._propagate_array(
                hamiltonian,
                efield,
                dipole_matrix,
                state.amplitudes,
                **propagation_kwargs,
            )

            component_time, psi_t, component_scales = result
            if time_psi is None:
                time_psi = component_time
                scales_reference = component_scales
            elif not np.allclose(time_psi, component_time):
                raise RuntimeError("ensemble components returned inconsistent times")

            psi_backend = xp.asarray(psi_t)
            if options.return_trajectory:
                component_density = xp.einsum(
                    "ti, tj -> tij", psi_backend, psi_backend.conj()
                )
            else:
                component_density = xp.outer(psi_backend, psi_backend.conj())

            if rho_out is None:
                rho_out = xp.zeros_like(component_density)
            rho_out += float(weight) * component_density

        if time_psi is None or rho_out is None:
            raise RuntimeError("ensemble propagation produced no result")
        return finalize_propagation_result(
            problem=problem,
            options=options,
            times_fs=time_psi,
            state=rho_out,
            state_kind="density_matrix",
            scales=scales_reference,
        )
