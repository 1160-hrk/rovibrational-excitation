#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spectroscopy Module
===================

This module provides tools for calculating spectroscopic properties
from quantum mechanical simulations of rovibrational excitation.

Classes
-------
AbsorbanceCalculator
    Main class for calculating absorbance spectra from density matrices
ExperimentalConditions
    Dataclass for experimental conditions (temperature, pressure, etc.)

Functions
---------
create_calculator_from_params
    Helper function to create AbsorbanceCalculator from parameters

Examples
--------
Basic usage:

>>> from rovibrational_excitation.spectroscopy import AbsorbanceCalculator, ExperimentalConditions
>>> from rovibrational_excitation.core.basis import LinMolBasis
>>> 
>>> # Create basis and other components
>>> basis = LinMolBasis(V_max=5, J_max=10, use_M=True)
>>> H0 = basis.generate_H0()
>>> dipole_matrix = LinMolDipoleMatrix(basis=basis, mu0=1e-30)
>>> 
>>> # Set up experimental conditions
>>> conditions = ExperimentalConditions(
...     temperature=300,  # K
...     pressure=1e5,     # Pa
...     optical_length=1e-3,  # m
...     T2=500            # ps
... )
>>> 
>>> # Create calculator
>>> calculator = AbsorbanceCalculator(
...     basis=basis,
...     hamiltonian=H0,
...     dipole_matrix=dipole_matrix,
...     conditions=conditions,
...     axes='xy'
... )
>>> 
>>> # Calculate spectrum
>>> wavenumber = np.arange(2000, 2500, 0.1)
>>> absorbance = calculator.calculate(rho, wavenumber)

Advanced usage with 3D dipole components:

>>> # Using all three dipole components
>>> calculator = AbsorbanceCalculator(
...     basis=basis,
...     hamiltonian=H0,
...     dipole_matrix=dipole_matrix,
...     conditions=conditions,
...     axes='xyz',
...     pol_int=np.array([1, 0, 0]),    # x-polarized interaction
...     pol_det=np.array([0, 1, 0])     # y-polarized detection
... )

Memory-efficient calculation for large systems:

>>> # Automatic optimization for large basis sets
>>> absorbance = calculator.calculate(
...     rho, wavenumber, 
...     method='optimized',  # Automatically selects best method
...     chunk_size=1000      # Controls memory usage
... )
"""

from .absorbance_calculator import (
    AbsorbanceCalculator,
    ExperimentalConditions,
    create_calculator_from_params
)


# Define what gets imported with "from spectroscopy import *"
__all__ = [
    'AbsorbanceCalculator',
    'ExperimentalConditions', 
    'create_calculator_from_params'
]


# Version information
__version__ = '1.0.0'
__author__ = 'Rovibrational Excitation Team'
__email__ = 'contact@example.com'

# Module-level documentation
if __doc__ is None:
    __doc__ = ""

__doc__ += f"""

Available Components
--------------------
AbsorbanceCalculator : {AbsorbanceCalculator.__doc__.split('.')[0] if AbsorbanceCalculator.__doc__ else 'Main calculator class'}
ExperimentalConditions : {ExperimentalConditions.__doc__.split('.')[0] if ExperimentalConditions.__doc__ else 'Experimental parameters dataclass'}

Optional Components
-------------------
"""


__doc__ += f"""

Module Version: {__version__}
"""
