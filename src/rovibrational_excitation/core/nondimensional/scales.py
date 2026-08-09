"""
Nondimensionalization scale factors for quantum dynamics.

This module provides the NondimensionalizationScales class that manages
scale factors for converting between dimensional and dimensionless systems.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..units.constants import CONSTANTS


@dataclass(frozen=True)
class ScaleValue:
    """A scale value together with the rule that produced it."""

    value: float | None
    source: Literal["derived", "explicit", "inactive"]
    method: str


class NondimensionalizationScales:
    """
    無次元化のスケールファクターを管理するクラス
    
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

    def __init__(
        self,
        E0: float,
        mu0: float | None,
        Efield0: float | None,
        t0: float,
        lambda_coupling: float,
        *,
        energy_offset: float = 0.0,
        free_energy_span: float | None = None,
        interaction_energy: float | None = None,
        physical_coupling_ratio: float | None = None,
        energy_source: Literal["derived", "explicit"] = "derived",
        energy_method: str = "legacy",
        dipole_source: Literal["derived", "inactive"] | None = None,
        field_source: Literal["derived", "inactive"] | None = None,
    ):
        required_values = {
            "E0": E0,
            "t0": t0,
            "lambda_coupling": lambda_coupling,
            "energy_offset": energy_offset,
        }
        if any(not np.isfinite(value) for value in required_values.values()):
            raise ValueError("all active scale metadata must be finite")
        optional_values = (
            mu0,
            Efield0,
            free_energy_span,
            interaction_energy,
            physical_coupling_ratio,
        )
        if any(value is not None and not np.isfinite(value) for value in optional_values):
            raise ValueError("optional scale metadata must be finite when active")
        # 値の検証
        if E0 <= 0:
            raise ValueError("Energy scale E0 must be positive")
        if mu0 is not None and mu0 <= 0:
            raise ValueError("Dipole scale mu0 must be positive when active")
        if lambda_coupling < 0:
            raise ValueError("lambda_coupling must be non-negative")
        if Efield0 is not None and Efield0 <= 0:
            raise ValueError("Field scale Efield0 must be positive when active")
        if t0 <= 0:
            raise ValueError("Time scale t0 must be positive")

        self.E0 = E0
        self.mu0 = mu0
        self.Efield0 = Efield0
        self.t0 = t0
        self.lambda_coupling = lambda_coupling
        self.energy_offset = energy_offset
        self.free_energy_span = E0 if free_energy_span is None else free_energy_span
        self.interaction_energy = (
            E0 * lambda_coupling
            if interaction_energy is None
            else interaction_energy
        )
        self.physical_coupling_ratio = physical_coupling_ratio
        self.reference_energy = ScaleValue(E0, energy_source, energy_method)
        self.dipole_scale = ScaleValue(
            mu0,
            dipole_source or ("inactive" if mu0 is None else "derived"),
            "operator_2_norm" if mu0 is not None else "zero_operator",
        )
        self.field_scale = ScaleValue(
            Efield0,
            field_source or ("inactive" if Efield0 is None else "derived"),
            "peak_vector_magnitude" if Efield0 is not None else "ZeroField",
        )

    def __repr__(self) -> str:
        return (
            f"NondimensionalizationScales(\n"
            f"  E0={self.E0:.3e} J,\n"
            f"  mu0={self.mu0 if self.mu0 is None else f'{self.mu0:.3e}'} C·m,\n"
            f"  Efield0={self.Efield0 if self.Efield0 is None else f'{self.Efield0:.3e}'} V/m,\n"
            f"  t0={self.t0:.3e} s,\n"
            f"  λ={self.lambda_coupling:.3f}\n"
            f")"
        )

    # -----------------------------
    # 単位変換ユーティリティ
    # -----------------------------
    def get_time_scale_fs(self) -> float:
        """時間スケールをフェムト秒で取得"""
        return self.t0 * 1e15

    def get_energy_scale_eV(self) -> float:
        """エネルギースケールをeVで取得"""
        return self.E0 / CONSTANTS.EV_TO_J

    def get_field_scale_MV_cm(self) -> float:
        """電場スケールをMV/cmで取得"""
        if self.Efield0 is None:
            raise ValueError("field scale is inactive for ZeroField")
        return self.Efield0 / 1e8

    def get_dipole_scale_D(self) -> float:
        """双極子スケールをDebyeで取得"""
        if self.mu0 is None:
            raise ValueError("dipole scale is inactive for a zero coupling operator")
        return self.mu0 / CONSTANTS.DEBYE_TO_CM

    def get_regime(self) -> str:
        """
        λに基づく物理レジーム判定
        
        Returns
        -------
        str
            "weak", "intermediate", or "strong" coupling regime
        """
        if self.interaction_energy == 0:
            return "field_free"
        if self.physical_coupling_ratio is None:
            return "gapless_driven"
        return "unclassified"

    # Legacy heuristic APIs intentionally remain as explicit errors for callers
    # that have not yet migrated to an explicit grid and convergence study.
    def get_recommended_timestep_dimensionless(self, *args, **kwargs) -> float:
        del args, kwargs
        raise RuntimeError(
            "heuristic timestep selection was removed; provide the "
            "ElectricField grid explicitly and validate convergence"
        )

    get_recommended_timestep_fs = get_recommended_timestep_dimensionless
    get_recommended_timestep = get_recommended_timestep_dimensionless

    def analyze_timestep_requirements(self):
        raise RuntimeError(
            "heuristic timestep analysis was removed; use a convergence study"
        )

    @classmethod
    def from_physical_system(cls, *args, **kwargs):
        del cls, args, kwargs
        raise RuntimeError(
            "from_physical_system() was removed because it invented zero "
            "dipole and zero-field scales; use determine_SI_based_scales()"
        )

    # -----------------------------
    # サマリー表示
    # -----------------------------
    def summary(self) -> str:
        """Return scale values, inactive states, and provenance."""
        dipole = (
            "inactive"
            if self.mu0 is None
            else f"{self.get_dipole_scale_D():.3f} D ({self.mu0:.3e} C·m)"
        )
        field = (
            "inactive"
            if self.Efield0 is None
            else (
                f"{self.get_field_scale_MV_cm():.3f} MV/cm "
                f"({self.Efield0:.3e} V/m)"
            )
        )
        return "\n".join(
            [
                "Nondimensionalization Scales:",
                f"  Energy: {self.get_energy_scale_eV():.3f} eV ({self.E0:.3e} J)",
                f"  Energy source: {self.reference_energy.source}",
                f"  Energy method: {self.reference_energy.method}",
                f"  Dipole: {dipole}",
                f"  Field: {field}",
                f"  Time: {self.get_time_scale_fs():.3f} fs ({self.t0:.3e} s)",
                f"  Numerical coupling coefficient: {self.lambda_coupling:.3f}",
                f"  Regime: {self.get_regime()}",
            ]
        )
