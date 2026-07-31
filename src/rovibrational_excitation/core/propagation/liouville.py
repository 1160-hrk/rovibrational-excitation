"""
Liouville-von Neumann equation propagator implementation.

This module provides the LiouvillePropagator class for density matrix
propagation using the Liouville-von Neumann equation.
"""

from typing import Literal

import numpy as np

from ..units.validators import validator
from .algorithms.rk4.lvne import rk4_lvne, rk4_lvne_traj
from .base import PropagatorBase
from .utils import prepare_propagation_args


class LiouvillePropagator(PropagatorBase):
    """
    Liouville-von Neumann equation propagator for density matrices.

    This class implements the propagation of density matrices using
    the Liouville-von Neumann equation.
    """

    def __init__(
        self,
        backend: Literal["numpy", "cupy"] = "numpy",
        validate_units: bool = True,
    ):
        """
        Initialize Liouville propagator.

        Parameters
        ----------
        backend : {"numpy"}
            Computational backend. Density-matrix propagation currently uses
            the NumPy/Numba implementation only.
        validate_units : bool
            Whether to validate physical units
        """
        super().__init__(validate_units)
        self.backend = backend

        if backend != "numpy":
            raise ValueError(
                "LiouvillePropagator currently supports only backend='numpy'"
            )

    def get_algorithm_name(self) -> str:
        """Get the name of the propagation algorithm."""
        return "Liouville-von-Neumann"

    def get_supported_backends(self) -> list:
        """Get list of supported computational backends."""
        return ["numpy"]

    def propagate(
        self,
        hamiltonian,
        efield,
        dipole_matrix,
        initial_state: np.ndarray,
        **kwargs,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Propagate density matrix using Liouville-von Neumann equation.

        Parameters
        ----------
        hamiltonian : Hamiltonian
            Hamiltonian object with internal unit management
        efield : ElectricField
            Electric field object
        dipole_matrix : DipoleMatrixBase
            Dipole moment matrices with internal unit management
        initial_state : np.ndarray
            Initial density matrix
        **kwargs
            Additional parameters:
            - axes : str, default "xy"
                Polarization axes mapping
            - return_traj : bool, default True
                Return full trajectory vs final state only
            - sample_stride : int, default 1
                Sampling stride for trajectory
            - return_time_rho : bool, default False
                Return the physical time array together with the density matrix
            - verbose : bool, default False
                Print detailed information
            - dt : None
                A time-step override is not supported. The electric-field grid
                is the single source of truth for the integration time step.

        Returns
        -------
        np.ndarray or tuple[np.ndarray, np.ndarray]
            Propagated density matrix or trajectory, optionally paired with time.
        """
        # Extract kwargs
        axes = kwargs.get("axes", "xy")
        return_traj = kwargs.get("return_traj", True)
        return_time_rho = kwargs.get("return_time_rho", False)
        sample_stride = kwargs.get("sample_stride", 1)
        verbose = kwargs.get("verbose", False)
        nondimensional = kwargs.get("nondimensional", False)
        auto_timestep = kwargs.get("auto_timestep", False)
        target_accuracy = kwargs.get("target_accuracy", "standard")
        coupling_mode = kwargs.get("coupling_mode", "cartesian")
        coupling_axis = kwargs.get("coupling_axis")
        if kwargs.get("dt") is not None:
            raise ValueError(
                "dt override is unsupported; construct ElectricField with the desired grid"
            )
        if kwargs.get("algorithm", "rk4") != "rk4":
            raise ValueError("LiouvillePropagator supports only algorithm='rk4'")
        if kwargs.get("sparse", False):
            raise ValueError(
                "LiouvillePropagator does not support sparse matrix propagation"
            )

        rho0 = initial_state

        if self.validate_units:
            warnings = validator.validate_propagation_units(
                hamiltonian, dipole_matrix, efield
            )
            if warnings:
                self._last_validation_warnings = warnings
                if verbose:
                    self.print_validation_warnings()

        # Prepare arguments using the same utility as SchrodingerPropagator
        H0, mu_x, mu_y, Ex, Ey, _, _, dt_calc, t0_calc = prepare_propagation_args(
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

        # Calculate number of steps
        steps = (len(Ex) - 1) // 2

        # Prepare arguments for RK4
        rk4_args = (H0, mu_x, mu_y, Ex, Ey, np.asarray(rho0), dt_calc, steps)

        # Call the appropriate low-level propagator
        if return_traj:
            rho = rk4_lvne_traj(*rk4_args, sample_stride)
        else:
            rho = rk4_lvne(*rk4_args)

        if return_time_rho:
            if return_traj:
                step_fs = dt_calc * sample_stride * t0_calc
                if nondimensional:
                    step_fs *= 1e15
                time = efield.tlist[0] + np.arange(rho.shape[0]) * step_fs
            else:
                time = np.array([efield.tlist[-1]], dtype=np.float64)
            return time, rho
        return rho
