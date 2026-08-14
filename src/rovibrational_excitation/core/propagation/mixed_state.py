"""Propagation of incoherent statistical mixtures of pure states."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from ..states import DensityState, IncoherentEnsemble
from ..units.validators import validator
from .base import PropagatorBase
from .options import PropagationOptions
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
        hamiltonian: Any,
        efield: Any,
        dipole_matrix: Any,
        initial_state: DensityState | IncoherentEnsemble,
        *,
        options: PropagationOptions,
        coupling_mode: Literal["cartesian", "scalar"],
        axes: str | None = None,
        coupling_axis: Literal["x", "y", "z"] | None = None,
        return_times: bool = False,
        verbose: bool = False,
        split_interaction: Literal["cartesian", "helicity_projected"] | None = None,
    ) -> Any:
        """Propagate a typed statistical state through the explicit boundary."""
        if not isinstance(initial_state, (DensityState, IncoherentEnsemble)):
            raise TypeError(
                "initial_state must be an IncoherentEnsemble or DensityState"
            )
        self._schrodinger_prop._validate_public_options(options)
        self._schrodinger_prop._validate_public_split_interaction(
            options, split_interaction
        )
        coupling_kwargs = self._schrodinger_prop._validate_public_coupling(
            coupling_mode=coupling_mode,
            axes=axes,
            coupling_axis=coupling_axis,
        )

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
            return liouville_prop._propagate_array(
                hamiltonian,
                efield,
                dipole_matrix,
                initial_state.matrix,
                return_traj=options.return_trajectory,
                return_time_rho=return_times,
                sample_stride=options.sample_stride,
                nondimensional=options.nondimensional,
                coupling_mode=coupling_mode,
                **coupling_kwargs,
                verbose=False,
                algorithm=options.algorithm_name,
                sparse=options.sparse,
            )

        xp = get_backend(self.backend)
        rho_out = None
        time_psi = None
        propagation_kwargs = {
            "return_traj": options.return_trajectory,
            "return_time_psi": return_times,
            "sample_stride": options.sample_stride,
            "nondimensional": options.nondimensional,
            "coupling_mode": coupling_mode,
            **coupling_kwargs,
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
            if options.return_trajectory:
                component_density = xp.einsum(
                    "ti, tj -> tij", psi_backend, psi_backend.conj()
                )
            else:
                component_density = xp.outer(psi_backend, psi_backend.conj())

            if rho_out is None:
                rho_out = xp.zeros_like(component_density)
            rho_out += float(weight) * component_density

        if return_times and time_psi is not None:
            return time_psi, rho_out
        return rho_out
