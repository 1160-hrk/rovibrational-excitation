"""
Parameter conversion utilities for rovibrational excitation calculations.

This module provides object-oriented parameter conversion that replaces
the functional approach in units.py with a more integrated design.

DEPRECATED: This module is now a compatibility layer for the new
units.parameter_processor module. Please use the new module directly
for new code.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Union
import numpy as np

from .basis.hamiltonian import Hamiltonian
from .electric_field import ElectricField


class ParameterConverter:
    """
    パラメータ変換ユーティリティクラス
    
    DEPRECATED: This class is now a compatibility layer that delegates
    to the new units.parameter_processor.ParameterProcessor class.
    Please use the new class directly for better functionality.
    """
    
    # 後方互換性のために物理定数を保持
    _C = 2.99792458e10  # speed of light [cm/s] 
    _H = 6.62607015e-34  # Planck constant [J·s]
    _E = 1.602176634e-19  # elementary charge [C]
    _DEBYE = 3.33564e-30  # Debye unit [C·m]
    _A0 = 5.29177210903e-11  # Bohr radius [m]
    _MU0 = 1.25663706212e-6  # vacuum permeability [H/m]
    
    @classmethod
    def _get_processor(cls):
        """新しいParameterProcessorのインスタンスを取得"""
        from .units.parameter_processor import parameter_processor
        return parameter_processor
    
    @classmethod
    def _issue_deprecation_warning(cls, method_name: str):
        """非推奨警告を発行"""
        warnings.warn(
            f"ParameterConverter.{method_name}() is deprecated. "
            "Please use rovibrational_excitation.core.units.parameter_processor instead.",
            DeprecationWarning,
            stacklevel=3
        )
    
    @classmethod
    def convert_frequency(cls, value: Union[float, np.ndarray], from_unit: str) -> Union[float, np.ndarray]:
        """Convert frequency to rad/fs"""
        cls._issue_deprecation_warning("convert_frequency")
        processor = cls._get_processor()
        return processor.converter.convert_frequency(value, from_unit, "rad/fs")
    
    @classmethod
    def convert_dipole_moment(cls, value: Union[float, np.ndarray], from_unit: str) -> Union[float, np.ndarray]:
        """Convert dipole moment to C·m"""
        cls._issue_deprecation_warning("convert_dipole_moment")
        processor = cls._get_processor()
        return processor.converter.convert_dipole_moment(value, from_unit, "C*m")
    
    @classmethod
    def convert_electric_field(cls, value: Union[float, np.ndarray], from_unit: str) -> Union[float, np.ndarray]:
        """Convert electric field to V/m"""
        cls._issue_deprecation_warning("convert_electric_field")
        processor = cls._get_processor()
        return processor.converter.convert_electric_field(value, from_unit, "V/m")
    
    @classmethod
    def convert_energy(cls, value: Union[float, np.ndarray], from_unit: str) -> Union[float, np.ndarray]:
        """Convert energy to J"""
        cls._issue_deprecation_warning("convert_energy")
        processor = cls._get_processor()
        return processor.converter.convert_energy(value, from_unit, "J")
    
    @classmethod
    def convert_time(cls, value: Union[float, np.ndarray], from_unit: str) -> Union[float, np.ndarray]:
        """Convert time to fs"""
        cls._issue_deprecation_warning("convert_time")
        processor = cls._get_processor()
        return processor.converter.convert_time(value, from_unit, "fs")
    
    @classmethod
    def auto_convert_parameters(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Automatically convert parameters with unit specifications to standard units.
        
        DEPRECATED: This method now delegates to the new ParameterProcessor.
        """
        cls._issue_deprecation_warning("auto_convert_parameters")
        processor = cls._get_processor()
        return processor.auto_convert_parameters(params, validate=False)
    
    @classmethod
    def create_hamiltonian_from_params(cls, params: Dict[str, Any], matrix: np.ndarray) -> Hamiltonian:
        """
        Create Hamiltonian object from parameters and matrix.
        
        DEPRECATED: This method now delegates to the new ParameterProcessor.
        """
        cls._issue_deprecation_warning("create_hamiltonian_from_params")
        processor = cls._get_processor()
        return processor.create_hamiltonian_from_params(params, matrix)
    
    @classmethod
    def create_efield_from_params(cls, params: Dict[str, Any], tlist: np.ndarray) -> ElectricField:
        """
        Create ElectricField object from parameters.
        
        DEPRECATED: This method now delegates to the new ParameterProcessor.
        """
        cls._issue_deprecation_warning("create_efield_from_params")
        processor = cls._get_processor()
        return processor.create_efield_from_params(params, tlist)


# 後方互換性のために変換テーブルも保持（使用は推奨されません）
_FREQUENCY_CONVERSIONS = {
    "rad/fs": 1.0,
    "THz": 2 * np.pi * 1e-3,
    "GHz": 2 * np.pi * 1e-6,
    "cm^-1": 2 * np.pi * ParameterConverter._C * 1e-15,
    "cm-1": 2 * np.pi * ParameterConverter._C * 1e-15,
    "wavenumber": 2 * np.pi * ParameterConverter._C * 1e-15,
    "PHz": 2 * np.pi,
    "Hz": 2 * np.pi * 1e-15,
    "rad/s": 1e-15,
}

_DIPOLE_CONVERSIONS = {
    "C*m": 1.0,
    "C·m": 1.0,
    "Cm": 1.0,
    "D": ParameterConverter._DEBYE,
    "Debye": ParameterConverter._DEBYE,
    "ea0": ParameterConverter._E * ParameterConverter._A0,
    "e*a0": ParameterConverter._E * ParameterConverter._A0,
    "atomic": ParameterConverter._E * ParameterConverter._A0,
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
    "eV": ParameterConverter._E,
    "meV": ParameterConverter._E * 1e-3,
    "μJ": 1e-6,
    "uJ": 1e-6,
    "mJ": 1e-3,
    "nJ": 1e-9,
    "pJ": 1e-12,
    "cm^-1": ParameterConverter._H * ParameterConverter._C,
    "cm-1": ParameterConverter._H * ParameterConverter._C,
    "wavenumber": ParameterConverter._H * ParameterConverter._C,
}

_TIME_CONVERSIONS = {
    "fs": 1.0,
    "ps": 1e3,
    "ns": 1e6,
    "s": 1e15,
} 