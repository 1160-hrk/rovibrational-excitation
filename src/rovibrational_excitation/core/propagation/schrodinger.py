"""
Schrödinger equation propagator implementation.

This module provides the SchrodingerPropagator class for time-dependent
Schrödinger equation propagation.
"""

from collections.abc import Callable, Sized
from typing import Any, Literal, Union, cast

import numpy as np

from ..units.validators import validator
from .base import PropagatorBase
from .utils import (
    HAS_CUPY,
    ensure_sparse_matrix,
    prepare_propagation_args,
)


class SchrodingerPropagator(PropagatorBase):
    """
    Time-dependent Schrödinger equation propagator.

    This class implements various algorithms for solving the time-dependent
    Schrödinger equation with external fields.
    """

    def __init__(
        self,
        backend: Literal["numpy", "cupy"] = "numpy",
        algorithm: Literal["rk4", "split_operator"] = "rk4",
        validate_units: bool = True,
        split_interaction: Literal["cartesian", "helicity_projected"] = "cartesian",
        renorm: bool = False,
        sparse: bool = False,
        custom_propagator: Callable | None = None,
    ):
        """
        Initialize Schrödinger propagator.

        Parameters
        ----------
        backend : {"numpy", "cupy"}
            Computational backend
        validate_units : bool
            Whether to validate physical units
        renorm : bool
            Renormalize wavefunction during propagation
        custom_propagator : callable, optional
            Custom propagation function to inject from outside.
            Should have signature: func(H0, mu_x, mu_y, Ex, Ey, initial_state, dt, return_traj, sample_stride)
        """
        super().__init__(validate_units)
        if algorithm not in {"rk4", "split_operator"}:
            raise ValueError("algorithm must be 'rk4' or 'split_operator'")
        if split_interaction not in {"cartesian", "helicity_projected"}:
            raise ValueError(
                "split_interaction must be 'cartesian' or 'helicity_projected'"
            )

        self.backend = backend
        self.renorm = renorm
        self.algorithm = algorithm
        self.custom_propagator = custom_propagator
        self.split_interaction = split_interaction
        self.sparse = sparse

        # Validate backend availability
        if backend == "cupy" and not HAS_CUPY:
            raise RuntimeError("CuPy backend requested but CuPy not installed")

        if backend == "cupy" and sparse:
            raise ValueError("sparse=True is not supported by the CuPy propagator")

    def set_custom_propagator(self, propagator_func: Callable) -> None:
        """
        Set custom propagation function from outside.

        Parameters
        ----------
        propagator_func : callable
            Custom propagation function.
            Should have signature: func(H0, mu_x, mu_y, Ex, Ey, initial_state, dt, return_traj, sample_stride)
        """
        self.custom_propagator = propagator_func

    def get_algorithm_name(self) -> str:
        """Get the name of the propagation algorithm."""
        if self.custom_propagator is not None:
            return f"Schrödinger-Custom-{getattr(self.custom_propagator, '__name__', 'Unknown')}"
        return f"Schrödinger-{self.algorithm}"

    def get_supported_backends(self) -> list:
        """Get list of supported computational backends."""
        backends = ["numpy"]
        if HAS_CUPY:
            backends.append("cupy")
        return backends

    def propagate(
        self,
        hamiltonian,
        efield,
        dipole_matrix,
        initial_state: np.ndarray,
        **kwargs,
    ) -> Union[np.ndarray, tuple]:
        """
        Propagate wavefunction using time-dependent Schrödinger equation.

        Parameters
        ----------
        hamiltonian : Hamiltonian
            Hamiltonian object with internal unit management
        efield : ElectricField
            Electric field object
        dipole_matrix : DipoleMatrixBase
            Dipole moment matrices with internal unit management
        initial_state : np.ndarray
            Initial wavefunction
        **kwargs
            Additional propagation parameters:
            - axes : str, default "xy"
                Polarization axes mapping ("xy", "zx", etc.)
            - return_traj : bool, default True
                Return full trajectory vs final state only
            - return_time_psi : bool, default False
                Return time array along with trajectory
            - sample_stride : int, default 1
                Sampling stride for trajectory
            - nondimensional : bool, default False
                Use nondimensional propagation
            - auto_timestep : bool, default False
                Automatically select optimal timestep
            - target_accuracy : str, default "standard"
                Target accuracy for auto timestep
            - verbose : bool, default False
                Print detailed information
            - algorithm : {"rk4", "split_operator"}, default "rk4"
                Propagation algorithm to use
            - sparse : bool, default False
                Use sparse matrix operations
            - propagator_func : callable, optional
                Override propagation function for this call only

        Returns
        -------
        np.ndarray or tuple
            Propagated wavefunction(s), optionally with time array
        """
        # Extract parameters with defaults
        axes = kwargs.get("axes", "xy")
        return_traj = kwargs.get("return_traj", True)
        return_time_psi = kwargs.get("return_time_psi", False)
        sample_stride = kwargs.get("sample_stride", 1)
        nondimensional = kwargs.get("nondimensional", False)
        auto_timestep = kwargs.get("auto_timestep", False)
        target_accuracy = kwargs.get("target_accuracy", "standard")
        coupling_mode = kwargs.get("coupling_mode", "cartesian")
        coupling_axis = kwargs.get("coupling_axis")
        verbose = kwargs.get("verbose", False)
        algorithm = kwargs.get("algorithm", self.algorithm)
        sparse = kwargs.get("sparse", self.sparse)
        split_interaction = kwargs.get("split_interaction", self.split_interaction)
        propagator_func = kwargs.get("propagator_func", None)
        renorm = kwargs.get("renorm", self.renorm)

        # Unit validation
        if algorithm not in {"rk4", "split_operator"}:
            raise ValueError("algorithm must be 'rk4' or 'split_operator'")
        if split_interaction not in {"cartesian", "helicity_projected"}:
            raise ValueError(
                "split_interaction must be 'cartesian' or 'helicity_projected'"
            )
        if self.backend == "cupy" and sparse:
            raise ValueError("sparse=True is not supported by the CuPy propagator")

        if self.validate_units:
            warnings = validator.validate_propagation_units(
                hamiltonian, dipole_matrix, efield
            )
            if warnings:
                self._last_validation_warnings = warnings
                if verbose:
                    self.print_validation_warnings()

        # Prepare arguments
        H0, mu_x, mu_y, Ex, Ey, pol, E_scalar, dt_calc, t0_calc = (
            prepare_propagation_args(
                hamiltonian,
                efield,
                dipole_matrix,
                axes=axes,
                nondimensional=nondimensional,
                auto_timestep=auto_timestep,
                target_accuracy=target_accuracy,
                coupling_mode=coupling_mode,
                coupling_axis=coupling_axis,
            )
        )

        # Handle sparse matrices if requested
        if sparse:
            H0 = ensure_sparse_matrix(H0)
            mu_x = ensure_sparse_matrix(mu_x)
            mu_y = ensure_sparse_matrix(mu_y)

        # Select and run algorithm
        # Priority: kwargs propagator_func > instance custom_propagator > built-in algorithms
        active_propagator = propagator_func or self.custom_propagator

        if active_propagator is not None:
            result = active_propagator(
                H0,
                mu_x,
                mu_y,
                Ex,
                Ey,
                initial_state,
                dt_calc,
                return_traj,
                sample_stride,
            )
        else:
            if algorithm == "rk4":
                result = self._propagate_rk4(
                    H0,
                    mu_x,
                    mu_y,
                    Ex,
                    Ey,
                    initial_state,
                    dt_calc,
                    return_traj,
                    sample_stride,
                    sparse,
                    renorm,
                )
            elif algorithm == "split_operator":
                basis = getattr(dipole_matrix, "basis", None)
                magnetic_quantum_numbers = getattr(basis, "M_array", None)
                result = self._propagate_split_operator(
                    H0,
                    mu_x,
                    mu_y,
                    Ex,
                    Ey,
                    initial_state,
                    dt_calc,
                    return_traj,
                    sample_stride,
                    sparse,
                    renorm,
                    pol,
                    E_scalar,
                    magnetic_quantum_numbers,
                    split_interaction,
                )
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

        # Handle return values
        if return_traj:
            psi = result
        else:
            psi = result[-1] if hasattr(result, "__len__") else result

        if return_time_psi:
            if return_traj:
                step_fs = dt_calc * sample_stride * t0_calc
                if nondimensional:
                    step_fs *= 1e15
                t = (
                    efield.tlist[0]
                    + np.arange(0, len(cast(Sized, psi)), dtype=np.float64) * step_fs
                )
            else:
                t = np.array([efield.tlist[-1]], dtype=np.float64)
            return t, psi

        return psi

    def _propagate_rk4(
        self,
        H0: Union[np.ndarray, Any],
        mu_x: Union[np.ndarray, Any],
        mu_y: Union[np.ndarray, Any],
        Ex: np.ndarray,
        Ey: np.ndarray,
        initial_state: np.ndarray,
        dt: float,
        return_traj: bool,
        stride: int,
        sparse: bool,
        renorm: bool,
    ) -> np.ndarray:
        """Run RK4 propagation algorithm."""
        from .algorithms.rk4.schrodinger import rk4_schrodinger

        backend_typed = cast(Literal["numpy", "cupy"], self.backend)

        return rk4_schrodinger(
            H0,
            mu_x,
            mu_y,
            Ex,
            Ey,
            initial_state,
            dt,
            return_traj=return_traj,
            stride=stride,
            renorm=renorm,
            sparse=sparse,
            backend=backend_typed,
        )

    def _propagate_split_operator(
        self,
        H0: Union[np.ndarray, Any],
        mu_x: Union[np.ndarray, Any],
        mu_y: Union[np.ndarray, Any],
        field_x: np.ndarray,
        field_y: np.ndarray,
        initial_state: np.ndarray,
        dt: float,
        return_traj: bool,
        stride: int,
        sparse: bool,
        renorm: bool,
        pol: np.ndarray,
        E_scalar: np.ndarray,
        magnetic_quantum_numbers: np.ndarray | None,
        split_interaction: Literal["cartesian", "helicity_projected"],
    ) -> np.ndarray:
        """Run split-operator propagation algorithm."""
        from .algorithms.split_operator.schrodinger import splitop_schrodinger

        backend_typed = cast(Literal["numpy", "cupy"], self.backend)

        return splitop_schrodinger(
            H0,
            mu_x,
            mu_y,
            field_x,
            field_y,
            initial_state,
            dt,
            return_traj=return_traj,
            sample_stride=stride,
            interaction_mode=split_interaction,
            magnetic_quantum_numbers=magnetic_quantum_numbers,
            polarization=pol,
            scalar_field=E_scalar,
            backend=backend_typed,
            sparse=sparse,
            renorm=renorm,
        )
