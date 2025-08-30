"""
Unit conversion utilities for rovibrational excitation calculations.

This module provides a clean, extensible system for unit conversions.
"""

from typing import Dict, Union, Callable
import numpy as np

from .constants import CONSTANTS


class UnitConverter:
    """
    Centralized unit conversion system.
    
    This class provides methods for converting between different unit systems
    used in molecular physics calculations.
    """
    
    def __init__(self):
        """Initialize converter with conversion tables."""
        self._setup_conversions()
    
    def _setup_conversions(self):
        """Set up conversion factors and functions."""
        # Frequency conversions (target: rad/fs)
        self._frequency_to_rad_fs = {
            "rad/fs": 1.0,
            "THz": 2 * np.pi * 1e-3,
            "GHz": 2 * np.pi * 1e-6,
            "MHz": 2 * np.pi * 1e-9,
            "kHz": 2 * np.pi * 1e-12,
            "Hz": 2 * np.pi * 1e-15,
            "cm^-1": 2 * np.pi * CONSTANTS.C * 1e-13,  # c in cm/s, result in rad/fs
            "cm-1": 2 * np.pi * CONSTANTS.C * 1e-13,
            "wavenumber": 2 * np.pi * CONSTANTS.C * 1e-13,
            "PHz": 2 * np.pi,
            "rad/s": 1e-15,
            "rad/ps": 1e-3,
        }
        
        # Energy conversions (target: J)
        self._energy_to_J = {
            "J": 1.0,
            "eV": CONSTANTS.EV_TO_J,
            "meV": CONSTANTS.EV_TO_J * 1e-3,
            "keV": CONSTANTS.EV_TO_J * 1e3,
            "Ry": 13.605693122994 * CONSTANTS.EV_TO_J,  # Rydberg
            "Ha": 27.211386245988 * CONSTANTS.EV_TO_J,  # Hartree
            "rad/fs": CONSTANTS.HBAR * 1e15,
            "rad/ps": CONSTANTS.HBAR * 1e12,
            "PHz": CONSTANTS.H * 1e15,
            "THz": CONSTANTS.H * 1e12,
            "cm^-1": CONSTANTS.H * CONSTANTS.C * 100,  # h*c with c in m/s
            "cm-1": CONSTANTS.H * CONSTANTS.C * 100,
            "wavenumber": CONSTANTS.H * CONSTANTS.C * 100,
            "kJ/mol": 1e3 / 6.02214076e23,  # per molecule
            "kcal/mol": 4184 / 6.02214076e23,
        }
        
        # Dipole moment conversions (target: C·m)
        self._dipole_to_Cm = {
            "C*m": 1.0,
            "C·m": 1.0,
            "Cm": 1.0,
            "D": CONSTANTS.DEBYE_TO_CM,
            "Debye": CONSTANTS.DEBYE_TO_CM,
            "ea0": CONSTANTS.E * CONSTANTS.A0,
            "e*a0": CONSTANTS.E * CONSTANTS.A0,
            "atomic": CONSTANTS.E * CONSTANTS.A0,
            "rad/fs/(V/m)": 1.0 * CONSTANTS.HBAR * 1e15,
            "rad*PHz/(V/m)": 1.0 * CONSTANTS.HBAR * 1e15,
        }
        
        # Electric field conversions (target: V/m)
        self._field_to_Vm = {
            "V/m": 1.0,
            "V/nm": 1e9,
            "V/Å": 1e10,
            "V/A": 1e10,
            "kV/m": 1e3,
            "kV/cm": 1e5,
            "MV/m": 1e6,
            "MV/cm": 1e8,
            "GV/m": 1e9,
            "TV/m": 1e12,
            "atomic": 5.14220674763e11,  # E_h/(e*a_0)
        }
        
        # Time conversions (target: fs)
        self._time_to_fs = {
            "fs": 1.0,
            "ps": 1e3,
            "ns": 1e6,
            "μs": 1e9,
            "us": 1e9,
            "ms": 1e12,
            "s": 1e15,
            "atomic": 2.4188843265857e-2,  # ℏ/E_h in fs
        }
        
        # Dispersion (GDD: time^2, TOD: time^3)
        # Use (time_conversion)^power factors derived from _time_to_fs
        self._gdd_to_fs2 = {unit + "^2": factor ** 2 for unit, factor in self._time_to_fs.items() if unit != "atomic"}
        self._tod_to_fs3 = {unit + "^3": factor ** 3 for unit, factor in self._time_to_fs.items() if unit != "atomic"}
        
        # Intensity to field conversions (functions)
        self._intensity_to_field: Dict[str, Callable] = {
            "W/cm^2": lambda I: np.sqrt(2 * I * 1e4 * CONSTANTS.MU0 * CONSTANTS.C),
            "W/cm2": lambda I: np.sqrt(2 * I * 1e4 * CONSTANTS.MU0 * CONSTANTS.C),
            "W/m^2": lambda I: np.sqrt(2 * I * CONSTANTS.MU0 * CONSTANTS.C),
            "W/m2": lambda I: np.sqrt(2 * I * CONSTANTS.MU0 * CONSTANTS.C),
            "TW/cm^2": lambda I: np.sqrt(2 * I * 1e16 * CONSTANTS.MU0 * CONSTANTS.C),
            "TW/cm2": lambda I: np.sqrt(2 * I * 1e16 * CONSTANTS.MU0 * CONSTANTS.C),
            "GW/cm^2": lambda I: np.sqrt(2 * I * 1e13 * CONSTANTS.MU0 * CONSTANTS.C),
            "GW/cm2": lambda I: np.sqrt(2 * I * 1e13 * CONSTANTS.MU0 * CONSTANTS.C),
            "MW/cm^2": lambda I: np.sqrt(2 * I * 1e10 * CONSTANTS.MU0 * CONSTANTS.C),
            "MW/cm2": lambda I: np.sqrt(2 * I * 1e10 * CONSTANTS.MU0 * CONSTANTS.C),
        }
    
    def convert_frequency(self, value: Union[float, np.ndarray], 
                         from_unit: str, to_unit: str = "rad/fs") -> Union[float, np.ndarray]:
        """Convert frequency between units."""
        if from_unit not in self._frequency_to_rad_fs:
            raise ValueError(f"Unknown frequency unit: {from_unit}")
        
        # Convert to rad/fs first
        value_rad_fs = value * self._frequency_to_rad_fs[from_unit]
        
        # Convert to target unit if not rad/fs
        if to_unit == "rad/fs":
            return value_rad_fs
        elif to_unit in self._frequency_to_rad_fs:
            return value_rad_fs / self._frequency_to_rad_fs[to_unit]
        else:
            raise ValueError(f"Unknown target frequency unit: {to_unit}")
    
    def convert_energy(self, value: Union[float, np.ndarray], 
                      from_unit: str, to_unit: str = "J") -> Union[float, np.ndarray]:
        """Convert energy between units."""
        if from_unit not in self._energy_to_J:
            raise ValueError(f"Unknown energy unit: {from_unit}")
        
        # Convert to J first
        value_J = value * self._energy_to_J[from_unit]
        
        # Convert to target unit if not J
        if to_unit == "J":
            return value_J
        elif to_unit in self._energy_to_J:
            return value_J / self._energy_to_J[to_unit]
        else:
            raise ValueError(f"Unknown target energy unit: {to_unit}")
    
    def convert_hamiltonian(self, value: Union[float, np.ndarray], 
                            from_unit: str, to_unit: str = "J") -> Union[float, np.ndarray]:
        """Convert Hamiltonian between units."""
        if from_unit in self._energy_to_J and to_unit in self._energy_to_J:
                return self.convert_energy(value, from_unit, to_unit)
        elif from_unit in self._frequency_to_rad_fs and to_unit in self._frequency_to_rad_fs:
                return self.convert_frequency(value, from_unit, to_unit)
        elif from_unit in self._energy_to_J and to_unit in self._frequency_to_rad_fs:
            return self.energy_to_frequency(value, from_unit, to_unit)
        elif from_unit in self._frequency_to_rad_fs and to_unit in self._energy_to_J:
            return self.frequency_to_energy(value, from_unit, to_unit)
        else:
            raise ValueError(f"Unknown target Hamiltonian unit: {to_unit}")
    
    def convert_dipole_moment(self, value: Union[float, np.ndarray], 
                             from_unit: str, to_unit: str = "C*m") -> Union[float, np.ndarray]:
        """Convert dipole moment between units."""
        if from_unit not in self._dipole_to_Cm:
            raise ValueError(f"Unknown dipole unit: {from_unit}")
        
        # Convert to C·m first
        value_Cm = value * self._dipole_to_Cm[from_unit]
        
        # Convert to target unit if not C·m
        if to_unit in ["C*m", "C·m", "Cm"]:
            return value_Cm
        elif to_unit in self._dipole_to_Cm:
            return value_Cm / self._dipole_to_Cm[to_unit]
        else:
            raise ValueError(f"Unknown target dipole unit: {to_unit}")
    
    def convert_electric_field(self, value: Union[float, np.ndarray], 
                              from_unit: str, to_unit: str = "V/m") -> Union[float, np.ndarray]:
        """Convert electric field between units."""
        # Check if it's an intensity unit
        if from_unit in self._intensity_to_field:
            # Convert intensity to field in V/m
            value_Vm = self._intensity_to_field[from_unit](value)
        elif from_unit in self._field_to_Vm:
            # Direct field conversion
            value_Vm = value * self._field_to_Vm[from_unit]
        else:
            raise ValueError(f"Unknown electric field/intensity unit: {from_unit}")
        
        # Convert to target unit if not V/m
        if to_unit == "V/m":
            return value_Vm
        elif to_unit in self._field_to_Vm:
            return value_Vm / self._field_to_Vm[to_unit]
        else:
            raise ValueError(f"Unknown target field unit: {to_unit}")
    
    def convert_time(self, value: Union[float, np.ndarray], 
                    from_unit: str, to_unit: str = "fs") -> Union[float, np.ndarray]:
        """Convert time between units."""
        if from_unit not in self._time_to_fs:
            raise ValueError(f"Unknown time unit: {from_unit}")
        
        # Convert to fs first
        value_fs = value * self._time_to_fs[from_unit]
        
        # Convert to target unit if not fs
        if to_unit == "fs":
            return value_fs
        elif to_unit in self._time_to_fs:
            return value_fs / self._time_to_fs[to_unit]
        else:
            raise ValueError(f"Unknown target time unit: {to_unit}")
    
    def frequency_to_energy(self, freq: Union[float, np.ndarray], 
                           freq_unit: str = "rad/fs", 
                           energy_unit: str = "J") -> Union[float, np.ndarray]:
        """Convert frequency to energy using E = ℏω."""
        # Convert frequency to rad/s
        freq_rad_s = self.convert_frequency(freq, freq_unit, "rad/s")
        
        # Calculate energy in J
        energy_J = CONSTANTS.HBAR * freq_rad_s
        
        # Convert to target unit
        if energy_unit == "J":
            return energy_J
        else:
            return self.convert_energy(energy_J, "J", energy_unit)
    
    def energy_to_frequency(self, energy: Union[float, np.ndarray], 
                           energy_unit: str = "J", 
                           freq_unit: str = "rad/fs") -> Union[float, np.ndarray]:
        """Convert energy to frequency using ω = E/ℏ."""
        # Convert energy to J
        energy_J = self.convert_energy(energy, energy_unit, "J")
        
        # Calculate frequency in rad/s
        freq_rad_s = energy_J / CONSTANTS.HBAR
        
        # Convert to target unit
        return self.convert_frequency(freq_rad_s, "rad/s", freq_unit)
    
    def get_supported_units(self, quantity: str) -> list:
        """Get list of supported units for a physical quantity."""
        mapping = {
            "frequency": list(self._frequency_to_rad_fs.keys()),
            "energy": list(self._energy_to_J.keys()),
            "dipole": list(self._dipole_to_Cm.keys()),
            "field": list(self._field_to_Vm.keys()) + list(self._intensity_to_field.keys()),
            "time": list(self._time_to_fs.keys()),
            "gdd": list(self._gdd_to_fs2.keys()),
            "tod": list(self._tod_to_fs3.keys()),
        }
        
        if quantity not in mapping:
            raise ValueError(f"Unknown quantity: {quantity}. "
                           f"Supported: {list(mapping.keys())}")
        
        return mapping[quantity]

    # ------------------------------------------------------------------
    # Dispersion converters --------------------------------------------
    # ------------------------------------------------------------------
    def convert_gdd(self, value: Union[float, np.ndarray], from_unit: str, to_unit: str = "fs^2") -> Union[float, np.ndarray]:
        """Convert group delay dispersion (time^2) to requested units."""
        if from_unit not in self._gdd_to_fs2:
            raise ValueError(f"Unknown GDD unit: {from_unit}")
        if to_unit not in self._gdd_to_fs2:
            raise ValueError(f"Unknown target GDD unit: {to_unit}")
        value_fs2 = value * self._gdd_to_fs2[from_unit]
        return value_fs2 / self._gdd_to_fs2[to_unit]

    def convert_tod(self, value: Union[float, np.ndarray], from_unit: str, to_unit: str = "fs^3") -> Union[float, np.ndarray]:
        """Convert third-order dispersion (time^3) to requested units."""
        if from_unit not in self._tod_to_fs3:
            raise ValueError(f"Unknown TOD unit: {from_unit}")
        if to_unit not in self._tod_to_fs3:
            raise ValueError(f"Unknown target TOD unit: {to_unit}")
        value_fs3 = value * self._tod_to_fs3[from_unit]
        return value_fs3 / self._tod_to_fs3[to_unit]


# Create a singleton instance for convenience
converter = UnitConverter() 