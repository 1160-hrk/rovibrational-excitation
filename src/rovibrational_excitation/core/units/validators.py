"""
Unit validation utilities for rovibrational excitation calculations.

This module provides validation functions to ensure physical quantities
are within reasonable ranges for molecular physics calculations.
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np

from .constants import CONSTANTS
from .converters import converter


class UnitValidator:
    """
    Validator for physical quantities and their units.
    
    This class provides methods to validate that physical quantities
    are within reasonable ranges for molecular physics calculations.
    """
    
    def __init__(self):
        """Initialize validator with reasonable ranges."""
        self._setup_ranges()
    
    def _setup_ranges(self):
        """Set up reasonable ranges for different quantities."""
        # Energy ranges in J
        self._energy_ranges_J = {
            "molecular": (1e-25, 1e-15),  # Typical molecular energies
            "electronic": (1e-19, 1e-17),  # Electronic transitions
            "vibrational": (1e-21, 1e-19),  # Vibrational energies
            "rotational": (1e-23, 1e-21),  # Rotational energies
        }
        
        # Frequency ranges in rad/fs
        self._frequency_ranges_rad_fs = {
            "molecular": (1e-6, 1e3),  # Typical molecular frequencies
            "electronic": (1e2, 1e4),  # Electronic transitions
            "vibrational": (1e-2, 1e1),  # Vibrational frequencies
            "rotational": (1e-5, 1e-2),  # Rotational frequencies
        }
        
        # Dipole moment ranges in C·m
        self._dipole_ranges_Cm = {
            "molecular": (1e-35, 1e-25),  # Typical molecular dipoles
            "small": (1e-32, 1e-29),  # Small dipoles
            "large": (1e-29, 1e-26),  # Large dipoles
        }
        
        # Electric field ranges in V/m
        self._field_ranges_Vm = {
            "weak": (1e3, 1e6),  # Weak fields
            "moderate": (1e6, 1e9),  # Moderate fields
            "strong": (1e9, 1e12),  # Strong fields
            "extreme": (1e12, 1e15),  # Extreme fields
        }
        
        # Time ranges in fs
        self._time_ranges_fs = {
            "ultrafast": (0.001, 10),  # Ultrafast processes
            "fast": (10, 1000),  # Fast processes
            "slow": (1000, 1e6),  # Slow processes
        }
    
    def validate_energy(self, value: Union[float, np.ndarray], 
                       unit: str, context: str = "molecular") -> Tuple[bool, List[str]]:
        """
        Validate energy value.
        
        Parameters
        ----------
        value : float or np.ndarray
            Energy value(s) to validate
        unit : str
            Unit of the energy
        context : str
            Context for validation ("molecular", "electronic", etc.)
            
        Returns
        -------
        valid : bool
            Whether the value is valid
        warnings : List[str]
            List of warning messages
        """
        warnings = []
        
        # Convert to J for validation
        try:
            value_J = converter.convert_energy(value, unit, "J")
        except ValueError as e:
            return False, [str(e)]
        
        # Get appropriate range
        if context not in self._energy_ranges_J:
            warnings.append(f"Unknown context '{context}', using 'molecular'")
            context = "molecular"
        
        min_E, max_E = self._energy_ranges_J[context]
        
        # Check range
        if np.any(np.abs(value_J) < min_E):
            warnings.append(
                f"Energy {np.min(np.abs(value_J)):.2e} J is below typical "
                f"{context} range ({min_E:.2e} - {max_E:.2e} J)"
            )
        
        if np.any(np.abs(value_J) > max_E):
            warnings.append(
                f"Energy {np.max(np.abs(value_J)):.2e} J is above typical "
                f"{context} range ({min_E:.2e} - {max_E:.2e} J)"
            )
        
        return len(warnings) == 0, warnings
    
    def validate_frequency(self, value: Union[float, np.ndarray], 
                          unit: str, context: str = "molecular") -> Tuple[bool, List[str]]:
        """Validate frequency value."""
        warnings = []
        
        # Convert to rad/fs for validation
        try:
            value_rad_fs = converter.convert_frequency(value, unit, "rad/fs")
        except ValueError as e:
            return False, [str(e)]
        
        # Get appropriate range
        if context not in self._frequency_ranges_rad_fs:
            warnings.append(f"Unknown context '{context}', using 'molecular'")
            context = "molecular"
        
        min_f, max_f = self._frequency_ranges_rad_fs[context]
        
        # Check range
        if np.any(np.abs(value_rad_fs) < min_f):
            warnings.append(
                f"Frequency {np.min(np.abs(value_rad_fs)):.2e} rad/fs is below typical "
                f"{context} range ({min_f:.2e} - {max_f:.2e} rad/fs)"
            )
        
        if np.any(np.abs(value_rad_fs) > max_f):
            warnings.append(
                f"Frequency {np.max(np.abs(value_rad_fs)):.2e} rad/fs is above typical "
                f"{context} range ({min_f:.2e} - {max_f:.2e} rad/fs)"
            )
        
        return len(warnings) == 0, warnings
    
    def validate_dipole_moment(self, value: Union[float, np.ndarray], 
                              unit: str, context: str = "molecular") -> Tuple[bool, List[str]]:
        """Validate dipole moment value."""
        warnings = []
        
        # Convert to C·m for validation
        try:
            value_Cm = converter.convert_dipole_moment(value, unit, "C*m")
        except ValueError as e:
            return False, [str(e)]
        
        # Get appropriate range
        if context not in self._dipole_ranges_Cm:
            warnings.append(f"Unknown context '{context}', using 'molecular'")
            context = "molecular"
        
        min_d, max_d = self._dipole_ranges_Cm[context]
        
        # Check range
        if np.any(np.abs(value_Cm) < min_d):
            warnings.append(
                f"Dipole moment {np.min(np.abs(value_Cm)):.2e} C·m is below typical "
                f"{context} range ({min_d:.2e} - {max_d:.2e} C·m)"
            )
        
        if np.any(np.abs(value_Cm) > max_d):
            warnings.append(
                f"Dipole moment {np.max(np.abs(value_Cm)):.2e} C·m is above typical "
                f"{context} range ({min_d:.2e} - {max_d:.2e} C·m)"
            )
        
        return len(warnings) == 0, warnings
    
    def validate_electric_field(self, value: Union[float, np.ndarray], 
                               unit: str, context: str = "moderate") -> Tuple[bool, List[str]]:
        """Validate electric field value."""
        warnings = []
        
        # Convert to V/m for validation
        try:
            value_Vm = converter.convert_electric_field(value, unit, "V/m")
        except ValueError as e:
            return False, [str(e)]
        
        # Get appropriate range
        if context not in self._field_ranges_Vm:
            warnings.append(f"Unknown context '{context}', using 'moderate'")
            context = "moderate"
        
        min_E, max_E = self._field_ranges_Vm[context]
        
        # Check range
        if np.any(np.abs(value_Vm) < min_E):
            warnings.append(
                f"Electric field {np.min(np.abs(value_Vm)):.2e} V/m is below typical "
                f"{context} range ({min_E:.2e} - {max_E:.2e} V/m)"
            )
        
        if np.any(np.abs(value_Vm) > max_E):
            warnings.append(
                f"Electric field {np.max(np.abs(value_Vm)):.2e} V/m is above typical "
                f"{context} range ({min_E:.2e} - {max_E:.2e} V/m)"
            )
        
        return len(warnings) == 0, warnings
    
    def validate_time(self, value: Union[float, np.ndarray], 
                     unit: str, context: str = "fast") -> Tuple[bool, List[str]]:
        """Validate time value."""
        warnings = []
        
        # Convert to fs for validation
        try:
            value_fs = converter.convert_time(value, unit, "fs")
        except ValueError as e:
            return False, [str(e)]
        
        # Get appropriate range
        if context not in self._time_ranges_fs:
            warnings.append(f"Unknown context '{context}', using 'fast'")
            context = "fast"
        
        min_t, max_t = self._time_ranges_fs[context]
        
        # Check range
        if np.any(value_fs < min_t):
            warnings.append(
                f"Time {np.min(value_fs):.2e} fs is below typical "
                f"{context} range ({min_t:.2e} - {max_t:.2e} fs)"
            )
        
        if np.any(value_fs > max_t):
            warnings.append(
                f"Time {np.max(value_fs):.2e} fs is above typical "
                f"{context} range ({min_t:.2e} - {max_t:.2e} fs)"
            )
        
        return len(warnings) == 0, warnings
    
    def validate_propagation_units(self, hamiltonian, dipole_matrix, 
                                  efield, expected_H0_units: str = "J",
                                  expected_dipole_units: str = "C*m") -> List[str]:
        """
        Validate units before propagation calculation.
        
        This is a refactored version of the validate_propagation_units
        function from propagator.py.
        """
        warnings = []
        H0 = hamiltonian.get_matrix("J")
        try:
            # Energy scale analysis
            if H0.ndim == 2:
                eigenvals = np.diag(H0)
            else:
                eigenvals = H0
            
            energy_range = np.ptp(eigenvals)
            
            # Validate energy scale
            valid, energy_warnings = self.validate_energy(
                energy_range, expected_H0_units, "molecular"
            )
            warnings.extend(energy_warnings)
            
            # Dipole moment analysis
            mu_x = self._get_dipole_component(dipole_matrix, 'x')
            mu_y = self._get_dipole_component(dipole_matrix, 'y')
            max_dipole = max(np.max(np.abs(mu_x)), np.max(np.abs(mu_y)))
            
            # Validate dipole scale
            valid, dipole_warnings = self.validate_dipole_moment(
                max_dipole, expected_dipole_units, "molecular"
            )
            warnings.extend(dipole_warnings)
            
            # Time scale analysis
            if hasattr(efield, 'dt') and energy_range > 0:
                if expected_H0_units == "J":
                    char_time_fs = CONSTANTS.HBAR / energy_range * 1e15
                elif expected_H0_units == "rad/fs":
                    char_time_fs = 1 / energy_range
                else:
                    char_time_fs = 1000  # rough estimate
                
                if efield.dt > char_time_fs / 5:
                    warnings.append(
                        f"時間ステップ {efield.dt:.3f} fs が特性時間 "
                        f"{char_time_fs:.3f} fs に対して大きすぎます "
                        f"(推奨: < {char_time_fs/5:.3f} fs)"
                    )
            
            # Electric field magnitude check
            if hasattr(efield, 'Efield'):
                max_field = np.max(np.abs(efield.Efield))
                valid, field_warnings = self.validate_electric_field(
                    max_field, "V/m", "moderate"
                )
                warnings.extend(field_warnings)
                
                # Interaction strength analysis
                if max_dipole > 0 and energy_range > 0:
                    if expected_H0_units == "J" and expected_dipole_units == "C*m":
                        interaction_strength = max_field * max_dipole / energy_range
                        if interaction_strength > 0.1:
                            warnings.append(
                                f"強電場域です (相互作用強度 = {interaction_strength:.3f}). "
                                "小さな時間ステップを検討してください"
                            )
            
        except Exception as e:
            warnings.append(f"単位検証中にエラーが発生しました: {e}")
        
        return warnings
    
    def _get_dipole_component(self, dipole_matrix, axis: str) -> np.ndarray:
        """Extract dipole matrix component."""
        # Try to get SI units first (preferred method)
        si_method = f"get_mu_{axis}_SI"
        if hasattr(dipole_matrix, si_method):
            return getattr(dipole_matrix, si_method)()
        
        # Fallback to direct attribute access
        attr = f"mu_{axis}"
        if hasattr(dipole_matrix, attr):
            return getattr(dipole_matrix, attr)
        
        raise AttributeError(
            f"{type(dipole_matrix).__name__} has no attribute '{attr}' or '{si_method}'"
        )


# Create a singleton instance
validator = UnitValidator() 