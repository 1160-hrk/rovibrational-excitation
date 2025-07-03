"""
Nondimensionalization scale factors for quantum dynamics.

This module provides the NondimensionalizationScales class that manages
scale factors for converting between dimensional and dimensionless systems.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

from ..units.constants import CONSTANTS


@dataclass
class NondimensionalizationScales:
    """
    Scale factors for nondimensionalization.
    
    This class manages the characteristic scales used to convert
    physical quantities to dimensionless form.
    
    Attributes
    ----------
    E0 : float
        Energy scale [J]
    mu0 : float
        Dipole moment scale [C·m]
    Efield0 : float
        Electric field scale [V/m]
    t0 : float
        Time scale [s]
    lambda_coupling : float
        Dimensionless coupling strength
    """
    
    E0: float  # Energy scale [J]
    mu0: float  # Dipole moment scale [C·m]
    Efield0: float  # Electric field scale [V/m]
    t0: float  # Time scale [s]
    lambda_coupling: float  # Dimensionless coupling
    
    def __post_init__(self):
        """Validate scales after initialization."""
        if self.E0 <= 0:
            raise ValueError("Energy scale E0 must be positive")
        if self.mu0 <= 0:
            raise ValueError("Dipole scale mu0 must be positive")
        if self.Efield0 <= 0:
            raise ValueError("Field scale Efield0 must be positive")
        if self.t0 <= 0:
            raise ValueError("Time scale t0 must be positive")
    
    @classmethod
    def from_physical_system(cls, H0: np.ndarray, mu_values: np.ndarray,
                           field_amplitude: float) -> "NondimensionalizationScales":
        """
        Create scales from physical system parameters.
        
        Parameters
        ----------
        H0 : np.ndarray
            Hamiltonian matrix in J
        mu_values : np.ndarray
            Dipole matrix elements in C·m
        field_amplitude : float
            Electric field amplitude in V/m
            
        Returns
        -------
        NondimensionalizationScales
            Calculated scale factors
        """
        # Energy scale from Hamiltonian
        if H0.ndim == 2:
            eigvals = np.diag(H0)
        else:
            eigvals = H0.copy()
        
        energy_diffs = []
        for i in range(len(eigvals)):
            for j in range(i + 1, len(eigvals)):
                diff = abs(eigvals[i] - eigvals[j])
                if diff > 1e-20:  # Threshold for numerical zero
                    energy_diffs.append(diff)
        
        if energy_diffs:
            E0 = max(energy_diffs)
        else:
            E0 = max(abs(eigvals)) if len(eigvals) > 0 else CONSTANTS.HBAR / 1e-15
        
        # Time scale from energy
        t0 = CONSTANTS.HBAR / E0
        
        # Dipole scale from matrix elements
        mu_offdiag = mu_values.copy()
        if mu_values.ndim == 2:
            np.fill_diagonal(mu_offdiag, 0)
        
        mu0 = np.max(np.abs(mu_offdiag))
        if mu0 == 0:
            mu0 = CONSTANTS.DEBYE_TO_CM  # 1 Debye default
        
        # Field scale
        Efield0 = field_amplitude if field_amplitude > 0 else 1e8  # 1 MV/cm default
        
        # Coupling strength
        lambda_coupling = (Efield0 * mu0) / E0
        
        return cls(E0=E0, mu0=mu0, Efield0=Efield0, t0=t0,
                  lambda_coupling=lambda_coupling)
    
    def get_time_scale_fs(self) -> float:
        """Get time scale in femtoseconds."""
        return self.t0 * 1e15
    
    def get_energy_scale_eV(self) -> float:
        """Get energy scale in eV."""
        return self.E0 / CONSTANTS.EV_TO_J
    
    def get_field_scale_MV_cm(self) -> float:
        """Get field scale in MV/cm."""
        return self.Efield0 / 1e8
    
    def get_dipole_scale_D(self) -> float:
        """Get dipole scale in Debye."""
        return self.mu0 / CONSTANTS.DEBYE_TO_CM
    
    def get_regime(self) -> str:
        """
        Determine the physical regime based on lambda.
        
        Returns
        -------
        str
            "weak", "intermediate", or "strong" coupling regime
        """
        if self.lambda_coupling < 0.1:
            return "weak"
        elif self.lambda_coupling < 1.0:
            return "intermediate"
        else:
            return "strong"
    
    def get_recommended_timestep(self, method: str = "adaptive",
                               safety_factor: float = 0.1) -> Dict[str, float]:
        """
        Get recommended timestep for numerical integration.
        
        Parameters
        ----------
        method : str
            Timestep calculation method
        safety_factor : float
            Safety factor for stability
            
        Returns
        -------
        dict
            Timestep in both dimensionless and fs units
        """
        λ = self.lambda_coupling
        
        if method == "adaptive":
            if λ < 0.01:
                dt_base = 1.0
            elif λ < 0.1:
                dt_base = 1.0 - 9.0 * (λ - 0.01) / 0.09
            elif λ < 1.0:
                dt_base = 0.2 / λ
            else:
                dt_base = 0.2 / (λ ** 1.2)
        
        elif method == "rabi":
            rabi_period = 2 * np.pi / max(λ, 0.01)
            dt_base = rabi_period / 10
        
        elif method == "stability":
            dt_base = 0.5 / max(λ, 0.1)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        dt_dimensionless = dt_base * safety_factor
        dt_fs = dt_dimensionless * self.get_time_scale_fs()
        
        return {
            "dimensionless": dt_dimensionless,
            "fs": dt_fs,
            "rabi_periods_per_step": dt_dimensionless * λ / (2 * np.pi) if λ > 0 else float('inf')
        }
    
    def summary(self) -> str:
        """Get a summary of the scale factors."""
        lines = [
            "Nondimensionalization Scales:",
            f"  Energy: {self.get_energy_scale_eV():.3f} eV ({self.E0:.3e} J)",
            f"  Dipole: {self.get_dipole_scale_D():.3f} D ({self.mu0:.3e} C·m)",
            f"  Field: {self.get_field_scale_MV_cm():.3f} MV/cm ({self.Efield0:.3e} V/m)",
            f"  Time: {self.get_time_scale_fs():.3f} fs ({self.t0:.3e} s)",
            f"  λ: {self.lambda_coupling:.3f} ({self.get_regime()} coupling)",
        ]
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return (
            f"NondimensionalizationScales(\n"
            f"  E0={self.E0:.3e} J,\n"
            f"  mu0={self.mu0:.3e} C·m,\n"
            f"  Efield0={self.Efield0:.3e} V/m,\n"
            f"  t0={self.t0:.3e} s,\n"
            f"  λ={self.lambda_coupling:.3f}\n"
            f")"
        ) 