"""
Parameter conversion utilities for rovibrational excitation calculations.

This module provides object-oriented parameter conversion that replaces
the functional approach in units.py with a more integrated design.
"""

from __future__ import annotations

from typing import Any, Dict, Union
import numpy as np

from .basis.hamiltonian import Hamiltonian
from .electric_field import ElectricField


class ParameterConverter:
    """
    パラメータ変換ユーティリティクラス
    
    units.pyの機能をオブジェクト指向アプローチで代替します。
    各物理量のオブジェクトクラスと統合して、より安全で
    一貫性のある単位変換を提供します。
    """
    
    # Physical constants
    _C = 2.99792458e10  # speed of light [cm/s]
    _H = 6.62607015e-34  # Planck constant [J·s]
    _E = 1.602176634e-19  # elementary charge [C]
    _DEBYE = 3.33564e-30  # Debye unit [C·m]
    _A0 = 5.29177210903e-11  # Bohr radius [m]
    _MU0 = 1.25663706212e-6  # vacuum permeability [H/m]
    
    # Unit conversion factors
    _FREQUENCY_CONVERSIONS = {
        "rad/fs": 1.0,
        "THz": 2 * np.pi * 1e-3,
        "GHz": 2 * np.pi * 1e-6,
        "cm^-1": 2 * np.pi * _C * 1e-15,
        "cm-1": 2 * np.pi * _C * 1e-15,
        "wavenumber": 2 * np.pi * _C * 1e-15,
        "PHz": 2 * np.pi,
        "Hz": 2 * np.pi * 1e-15,
        "rad/s": 1e-15,
    }
    
    _DIPOLE_CONVERSIONS = {
        "C*m": 1.0,
        "C·m": 1.0,
        "Cm": 1.0,
        "D": _DEBYE,
        "Debye": _DEBYE,
        "ea0": _E * _A0,
        "e*a0": _E * _A0,
        "atomic": _E * _A0,
    }
    
    _FIELD_CONVERSIONS = {
        "V/m": 1.0,
        "V/nm": 1e9,
        "MV/cm": 1e8,
        "kV/cm": 1e5,
    }
    
    _INTENSITY_CONVERSIONS = {
        "W/cm^2": lambda I: np.sqrt(2 * I * ParameterConverter._MU0 * ParameterConverter._C * 1e2 * 1e4),
        "W/cm2": lambda I: np.sqrt(2 * I * ParameterConverter._MU0 * ParameterConverter._C * 1e2 * 1e4),
        "TW/cm^2": lambda I: np.sqrt(2 * I * 1e12 * ParameterConverter._MU0 * ParameterConverter._C * 1e2 * 1e4),
        "TW/cm2": lambda I: np.sqrt(2 * I * 1e12 * ParameterConverter._MU0 * ParameterConverter._C * 1e2 * 1e4),
        "GW/cm^2": lambda I: np.sqrt(2 * I * 1e9 * ParameterConverter._MU0 * ParameterConverter._C * 1e2 * 1e4),
        "GW/cm2": lambda I: np.sqrt(2 * I * 1e9 * ParameterConverter._MU0 * ParameterConverter._C * 1e2 * 1e4),
        "MW/cm^2": lambda I: np.sqrt(2 * I * 1e6 * ParameterConverter._MU0 * ParameterConverter._C * 1e2 * 1e4),
        "MW/cm2": lambda I: np.sqrt(2 * I * 1e6 * ParameterConverter._MU0 * ParameterConverter._C * 1e2 * 1e4),
    }
    
    _ENERGY_CONVERSIONS = {
        "J": 1.0,
        "eV": _E,
        "meV": _E * 1e-3,
        "μJ": 1e-6,
        "uJ": 1e-6,
        "mJ": 1e-3,
        "nJ": 1e-9,
        "pJ": 1e-12,
        "cm^-1": _H * _C,
        "cm-1": _H * _C,
        "wavenumber": _H * _C,
    }
    
    _TIME_CONVERSIONS = {
        "fs": 1.0,
        "ps": 1e3,
        "ns": 1e6,
        "s": 1e15,
    }
    
    @classmethod
    def convert_frequency(cls, value: Union[float, np.ndarray], from_unit: str) -> Union[float, np.ndarray]:
        """Convert frequency to rad/fs"""
        if from_unit not in cls._FREQUENCY_CONVERSIONS:
            raise ValueError(f"Unknown frequency unit: {from_unit}. "
                            f"Supported: {list(cls._FREQUENCY_CONVERSIONS.keys())}")
        return value * cls._FREQUENCY_CONVERSIONS[from_unit]
    
    @classmethod
    def convert_dipole_moment(cls, value: Union[float, np.ndarray], from_unit: str) -> Union[float, np.ndarray]:
        """Convert dipole moment to C·m"""
        if from_unit not in cls._DIPOLE_CONVERSIONS:
            raise ValueError(f"Unknown dipole unit: {from_unit}. "
                            f"Supported: {list(cls._DIPOLE_CONVERSIONS.keys())}")
        return value * cls._DIPOLE_CONVERSIONS[from_unit]
    
    @classmethod
    def convert_electric_field(cls, value: Union[float, np.ndarray], from_unit: str) -> Union[float, np.ndarray]:
        """Convert electric field to V/m"""
        if from_unit in cls._FIELD_CONVERSIONS:
            return value * cls._FIELD_CONVERSIONS[from_unit]
        elif from_unit in cls._INTENSITY_CONVERSIONS:
            converter = cls._INTENSITY_CONVERSIONS[from_unit]
            return converter(value)
        else:
            raise ValueError(f"Unknown electric field unit: {from_unit}. "
                            f"Supported field units: {list(cls._FIELD_CONVERSIONS.keys())}. "
                            f"Supported intensity units: {list(cls._INTENSITY_CONVERSIONS.keys())}")
    
    @classmethod
    def convert_energy(cls, value: Union[float, np.ndarray], from_unit: str) -> Union[float, np.ndarray]:
        """Convert energy to J"""
        if from_unit not in cls._ENERGY_CONVERSIONS:
            raise ValueError(f"Unknown energy unit: {from_unit}. "
                            f"Supported: {list(cls._ENERGY_CONVERSIONS.keys())}")
        return value * cls._ENERGY_CONVERSIONS[from_unit]
    
    @classmethod
    def convert_time(cls, value: Union[float, np.ndarray], from_unit: str) -> Union[float, np.ndarray]:
        """Convert time to fs"""
        if from_unit not in cls._TIME_CONVERSIONS:
            raise ValueError(f"Unknown time unit: {from_unit}. "
                            f"Supported: {list(cls._TIME_CONVERSIONS.keys())}")
        return value * cls._TIME_CONVERSIONS[from_unit]
    
    @classmethod
    def auto_convert_parameters(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Automatically convert parameters with unit specifications to standard units.
        
        This method replaces the units.py auto_convert_parameters function.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Parameter dictionary potentially containing unit specifications
            
        Returns
        -------
        Dict[str, Any]
            Parameter dictionary with values converted to standard units
        """
        converted_params = params.copy()
        
        # Parameter groups
        frequency_params = [
            "omega_rad_phz", "delta_omega_rad_phz", "B_rad_phz", "alpha_rad_phz",
            "carrier_freq", "vibrational_frequency_rad_per_fs", 
            "rotational_constant_rad_per_fs", "vibration_rotation_coupling_rad_per_fs",
            "anharmonicity_correction_rad_per_fs"
        ]
        
        dipole_params = ["mu0_Cm", "transition_dipole_moment"]
        field_params = ["amplitude"]
        energy_params = ["energy_gap"]
        time_params = ["duration", "t_center", "t_start", "t_end", "dt",
                      "coherence_relaxation_time_ps"]
        
        # Convert frequency parameters
        for param in frequency_params:
            unit_key = f"{param}_units"
            if param in converted_params and unit_key in converted_params:
                original_value = converted_params[param]
                unit = converted_params[unit_key]
                try:
                    converted_params[param] = cls.convert_frequency(original_value, unit)
                    print(f"✓ Converted {param}: {original_value} {unit} → {converted_params[param]:.6g} rad/fs")
                except ValueError as e:
                    print(f"⚠ Warning: {e}")
        
        # Convert dipole moment parameters
        for param in dipole_params:
            unit_key = f"{param}_units"
            if param in converted_params and unit_key in converted_params:
                original_value = converted_params[param]
                unit = converted_params[unit_key]
                try:
                    converted_params[param] = cls.convert_dipole_moment(original_value, unit)
                    print(f"✓ Converted {param}: {original_value} {unit} → {converted_params[param]:.6g} C·m")
                except ValueError as e:
                    print(f"⚠ Warning: {e}")
        
        # Convert electric field parameters
        for param in field_params:
            unit_key = f"{param}_units"
            if param in converted_params and unit_key in converted_params:
                original_value = converted_params[param]
                unit = converted_params[unit_key]
                try:
                    converted_params[param] = cls.convert_electric_field(original_value, unit)
                    print(f"✓ Converted {param}: {original_value} {unit} → {converted_params[param]:.6g} V/m")
                except ValueError as e:
                    print(f"⚠ Warning: {e}")
        
        # Convert energy parameters
        for param in energy_params:
            unit_key = f"{param}_units"
            if param in converted_params and unit_key in converted_params:
                original_value = converted_params[param]
                unit = converted_params[unit_key]
                try:
                    converted_params[param] = cls.convert_energy(original_value, unit)
                    print(f"✓ Converted {param}: {original_value} {unit} → {converted_params[param]:.6g} J")
                except ValueError as e:
                    print(f"⚠ Warning: {e}")
        
        # Convert time parameters
        for param in time_params:
            unit_key = f"{param}_units"
            if param in converted_params and unit_key in converted_params:
                original_value = converted_params[param]
                unit = converted_params[unit_key]
                try:
                    if "ps" in param:
                        if unit != "ps":
                            converted_value = cls.convert_time(original_value, unit) / 1000
                            converted_params[param] = converted_value
                            print(f"✓ Converted {param}: {original_value} {unit} → {converted_value:.6g} ps")
                    else:
                        converted_params[param] = cls.convert_time(original_value, unit)
                        print(f"✓ Converted {param}: {original_value} {unit} → {converted_params[param]:.6g} fs")
                except ValueError as e:
                    print(f"⚠ Warning: {e}")
        
        return converted_params
    
    @classmethod
    def create_hamiltonian_from_params(cls, params: Dict[str, Any], matrix: np.ndarray) -> Hamiltonian:
        """
        Create Hamiltonian object from parameters and matrix.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Parameter dictionary containing unit information
        matrix : np.ndarray
            Hamiltonian matrix
            
        Returns
        -------
        Hamiltonian
            Hamiltonian object with proper unit management
        """
        # Extract unit information
        input_units = params.get("hamiltonian_units", "J")
        target_units = params.get("target_units", "J")
        
        # Create basis info for debugging
        basis_info = {
            "source": "parameter_converter",
            "input_units": input_units,
            "target_units": target_units,
        }
        
        return Hamiltonian.from_input_units(matrix, input_units, target_units, basis_info)
    
    @classmethod
    def create_efield_from_params(cls, params: Dict[str, Any], tlist: np.ndarray) -> ElectricField:
        """
        Create ElectricField object from parameters.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Parameter dictionary containing field and time unit information
        tlist : np.ndarray
            Time array
            
        Returns
        -------
        ElectricField
            ElectricField object with proper unit management
        """
        time_units = params.get("time_units", "fs")
        field_units = params.get("field_units", "V/m")
        
        return ElectricField(tlist, time_units=time_units, field_units=field_units) 