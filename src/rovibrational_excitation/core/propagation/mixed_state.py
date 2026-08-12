"""Propagation of incoherent statistical mixtures of pure states."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ..states import DensityState, IncoherentEnsemble
from ..units.validators import validator
from .base import PropagatorBase
from .schrodinger import SchrodingerPropagator
from .utils import get_backend


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
        initial_state: DensityState | IncoherentEnsemble,
        **kwargs,
    ) -> np.ndarray | tuple:
        """Propagate an explicitly typed density state or incoherent ensemble.

        ``IncoherentEnsemble`` owns normalized components and statistical
        weights. ``DensityState`` owns a validated trace-one density matrix.
        This boundary never infers a state kind from array shape and never
        repairs or renormalizes typed input.
        """
        removed_timestep_options = {
            key for key in ("auto_timestep", "target_accuracy") if key in kwargs
        }
        if removed_timestep_options:
            names = ", ".join(sorted(removed_timestep_options))
            raise ValueError(
                f"{names} were removed; define the ElectricField grid explicitly"
            )
        allowed_options = {
            "axes",
            "return_traj",
            "return_time_rho",
            "sample_stride",
            "nondimensional",
            "coupling_mode",
            "coupling_axis",
            "verbose",
            "algorithm",
            "sparse",
            "split_interaction",
            "propagator_func",
            "renorm",
            "dt",
        }
        unknown_options = sorted(set(kwargs) - allowed_options)
        if unknown_options:
            raise ValueError(
                "unsupported propagation options: " + ", ".join(unknown_options)
            )
        return_traj = kwargs.get("return_traj", True)
        if "algorithm" in kwargs and kwargs["algorithm"] != self.algorithm:
            raise ValueError(
                "algorithm propagation override conflicts with the "
                "MixedStatePropagator constructor"
            )

        if not isinstance(initial_state, (DensityState, IncoherentEnsemble)):
            raise TypeError(
                "initial_state must be an IncoherentEnsemble or DensityState"
            )

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

        if isinstance(initial_state, DensityState):
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
            return liouville_prop.propagate(
                hamiltonian,
                efield,
                dipole_matrix,
                initial_state.matrix,
                **kwargs,
            )

        xp = get_backend(self.backend)
        rho_out = None
        time_psi = None

        propagation_kwargs = dict(kwargs)
        propagation_kwargs.pop("return_time_rho", None)
        propagation_kwargs["return_time_psi"] = return_time_rho
        propagation_kwargs["algorithm"] = self.algorithm
        propagation_kwargs.setdefault("sparse", self.sparse)
        propagation_kwargs["verbose"] = False

        for state, weight in zip(initial_state.states, initial_state.weights):
            result = self._schrodinger_prop.propagate(
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
