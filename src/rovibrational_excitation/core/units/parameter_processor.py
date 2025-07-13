"""
Parameter processing utilities for rovibrational excitation calculations.

This module provides high-level parameter processing that combines
unit conversion with automatic parameter detection and processing.
"""

from typing import Any, Dict, Union
import numpy as np

from .converters import converter
from .validators import validator


class ParameterProcessor:
    """
    高レベルパラメータ処理クラス
    
    parameter_converter.pyの機能をunitsモジュールに統合し、
    より拡張性があり保守しやすい設計で提供します。
    """
    
    def __init__(self):
        """Initialize parameter processor."""
        self.converter = converter
        self.validator = validator
        
        # Parameter groups for automatic processing
        self.frequency_params = [
            "omega_rad_phz", "delta_omega_rad_phz", "B_rad_phz", "alpha_rad_phz",
            "carrier_freq", "vibrational_frequency_rad_per_fs", 
            "rotational_constant_rad_per_fs", "vibration_rotation_coupling_rad_per_fs",
            "anharmonicity_correction_rad_per_fs"
        ]
        
        self.dipole_params = ["mu0_Cm", "transition_dipole_moment"]
        self.field_params = ["amplitude"]
        self.energy_params = ["energy_gap"]
        self.time_params = [
            "duration", "t_center", "t_start", "t_end", "dt",
            "coherence_relaxation_time_ps"
        ]
    
    def auto_convert_parameters(self, params: Dict[str, Any], 
                               validate: bool = False) -> Dict[str, Any]:
        """
        Automatically convert parameters with unit specifications to standard units.
        
        This method replaces parameter_converter.py's auto_convert_parameters function
        with enhanced functionality.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Parameter dictionary potentially containing unit specifications
        validate : bool, optional
            Whether to validate converted values, default False
            
        Returns
        -------
        Dict[str, Any]
            Parameter dictionary with values converted to standard units
        """
        converted_params = params.copy()
        warnings = []
        
        # Process frequency parameters
        for param in self.frequency_params:
            unit_key = f"{param}_units"
            if param in converted_params and unit_key in converted_params:
                original_value = converted_params[param]
                unit = converted_params[unit_key]
                try:
                    converted_value = self.converter.convert_frequency(
                        original_value, unit, "rad/fs"
                    )
                    converted_params[param] = converted_value
                    print(f"✓ Converted {param}: {original_value} {unit} → {converted_value:.6g} rad/fs")
                    
                    if validate:
                        valid, val_warnings = self.validator.validate_frequency(
                            converted_value, "rad/fs", "molecular"
                        )
                        warnings.extend(val_warnings)
                        
                except ValueError as e:
                    print(f"⚠ Warning: {e}")
                    warnings.append(str(e))
        
        # Process dipole moment parameters
        for param in self.dipole_params:
            unit_key = f"{param}_units"
            if param in converted_params and unit_key in converted_params:
                original_value = converted_params[param]
                unit = converted_params[unit_key]
                try:
                    converted_value = self.converter.convert_dipole_moment(
                        original_value, unit, "C*m"
                    )
                    converted_params[param] = converted_value
                    print(f"✓ Converted {param}: {original_value} {unit} → {converted_value:.6g} C·m")
                    
                    if validate:
                        valid, val_warnings = self.validator.validate_dipole_moment(
                            converted_value, "C*m", "molecular"
                        )
                        warnings.extend(val_warnings)
                        
                except ValueError as e:
                    print(f"⚠ Warning: {e}")
                    warnings.append(str(e))
        
        # Process electric field parameters
        for param in self.field_params:
            unit_key = f"{param}_units"
            if param in converted_params and unit_key in converted_params:
                original_value = converted_params[param]
                unit = converted_params[unit_key]
                try:
                    converted_value = self.converter.convert_electric_field(
                        original_value, unit, "V/m"
                    )
                    converted_params[param] = converted_value
                    print(f"✓ Converted {param}: {original_value} {unit} → {converted_value:.6g} V/m")
                    
                    if validate:
                        valid, val_warnings = self.validator.validate_electric_field(
                            converted_value, "V/m", "moderate"
                        )
                        warnings.extend(val_warnings)
                        
                except ValueError as e:
                    print(f"⚠ Warning: {e}")
                    warnings.append(str(e))
        
        # Process energy parameters
        for param in self.energy_params:
            unit_key = f"{param}_units"
            if param in converted_params and unit_key in converted_params:
                original_value = converted_params[param]
                unit = converted_params[unit_key]
                try:
                    converted_value = self.converter.convert_energy(
                        original_value, unit, "J"
                    )
                    converted_params[param] = converted_value
                    print(f"✓ Converted {param}: {original_value} {unit} → {converted_value:.6g} J")
                    
                    if validate:
                        valid, val_warnings = self.validator.validate_energy(
                            converted_value, "J", "molecular"
                        )
                        warnings.extend(val_warnings)
                        
                except ValueError as e:
                    print(f"⚠ Warning: {e}")
                    warnings.append(str(e))
        
        # Process time parameters
        for param in self.time_params:
            unit_key = f"{param}_units"
            if param in converted_params and unit_key in converted_params:
                original_value = converted_params[param]
                unit = converted_params[unit_key]
                try:
                    if "ps" in param:
                        # Special handling for ps parameters
                        if unit != "ps":
                            converted_value = self.converter.convert_time(
                                original_value, unit, "ps"
                            )
                            converted_params[param] = converted_value
                            print(f"✓ Converted {param}: {original_value} {unit} → {converted_value:.6g} ps")
                    else:
                        converted_value = self.converter.convert_time(
                            original_value, unit, "fs"
                        )
                        converted_params[param] = converted_value
                        print(f"✓ Converted {param}: {original_value} {unit} → {converted_value:.6g} fs")
                    
                    if validate:
                        target_unit = "ps" if "ps" in param else "fs"
                        valid, val_warnings = self.validator.validate_time(
                            converted_params[param], target_unit, "fast"
                        )
                        warnings.extend(val_warnings)
                        
                except ValueError as e:
                    print(f"⚠ Warning: {e}")
                    warnings.append(str(e))
        
        # Store warnings for access
        if warnings:
            converted_params["_conversion_warnings"] = warnings
        
        return converted_params
    
    def create_hamiltonian_from_params(self, params: Dict[str, Any], matrix: np.ndarray):
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
        # Lazy import to avoid circular dependency
        from ..basis.hamiltonian import Hamiltonian
        
        # Extract unit information
        input_units = params.get("hamiltonian_units", "J")
        target_units = params.get("target_units", "J")
        
        # Create basis info for debugging
        basis_info = {
            "source": "parameter_processor",
            "input_units": input_units,
            "target_units": target_units,
        }
        
        return Hamiltonian.from_input_units(matrix, input_units, target_units, basis_info)
    
    def create_efield_from_params(self, params: Dict[str, Any], tlist: np.ndarray):
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
        # Lazy import to avoid circular dependency
        from ..electric_field import ElectricField
        
        time_units = params.get("time_units", "fs")
        field_units = params.get("field_units", "V/m")
        
        return ElectricField(tlist, time_units=time_units, field_units=field_units)
    
    def add_parameter_group(self, group_name: str, param_list: list, 
                           quantity_type: str):
        """
        Add a custom parameter group for automatic processing.
        
        Parameters
        ----------
        group_name : str
            Name of the parameter group
        param_list : list
            List of parameter names
        quantity_type : str
            Type of physical quantity ("frequency", "energy", etc.)
        """
        if quantity_type not in ["frequency", "dipole", "field", "energy", "time"]:
            raise ValueError(f"Unknown quantity type: {quantity_type}")
        
        attr_name = f"{quantity_type}_params"
        if hasattr(self, attr_name):
            current_list = getattr(self, attr_name)
            current_list.extend(param_list)
        else:
            setattr(self, attr_name, param_list)
        
        print(f"✓ Added parameter group '{group_name}' with {len(param_list)} parameters")


# Create singleton instance
parameter_processor = ParameterProcessor() 