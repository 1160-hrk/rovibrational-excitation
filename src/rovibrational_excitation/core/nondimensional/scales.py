"""
Nondimensionalization scale factors for quantum dynamics.

This module provides the NondimensionalizationScales class that manages
scale factors for converting between dimensional and dimensionless systems.
"""

from typing import Dict, Any
import numpy as np

from ..units.constants import CONSTANTS


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
        mu0: float,
        Efield0: float,
        t0: float,
        lambda_coupling: float,
    ):
        # 値の検証
        if E0 <= 0:
            raise ValueError("Energy scale E0 must be positive")
        if mu0 <= 0:
            raise ValueError("Dipole scale mu0 must be positive")
        if Efield0 <= 0:
            raise ValueError("Field scale Efield0 must be positive")
        if t0 <= 0:
            raise ValueError("Time scale t0 must be positive")
        
        self.E0 = E0
        self.mu0 = mu0
        self.Efield0 = Efield0
        self.t0 = t0
        self.lambda_coupling = lambda_coupling

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
        return self.Efield0 / 1e8

    def get_dipole_scale_D(self) -> float:
        """双極子スケールをDebyeで取得"""
        return self.mu0 / CONSTANTS.DEBYE_TO_CM

    def get_regime(self) -> str:
        """
        λに基づく物理レジーム判定
        
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

    # -----------------------------
    # 推奨時間ステップ計算
    # -----------------------------
    def get_recommended_timestep_dimensionless(
        self,
        safety_factor: float = 0.02,
        min_dt: float = 1e-4,
        max_dt: float = 1.0,
        method: str = "adaptive",
        numerical_method: str = "split_operator",
    ) -> float:
        """λ に基づく推奨時間ステップ（無次元）を計算"""

        λ = self.lambda_coupling

        method_corrections = {
            "split_operator": 1.0,
            "rk4": 0.4,
            "magnus": 0.6,
            "crank_nicolson": 0.8,
            "implicit": 1.2,
        }

        correction_factor = method_corrections.get(numerical_method, 1.0)

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

        dt_recommended = dt_base * safety_factor * correction_factor
        dt_recommended = max(min_dt, min(max_dt, dt_recommended))
        return dt_recommended

    def get_recommended_timestep_fs(
        self,
        safety_factor: float = 0.5,
        min_dt_fs: float | None = None,
        max_dt_fs: float | None = None,
        method: str = "adaptive",
        numerical_method: str = "split_operator",
    ) -> float:
        """推奨時間ステップ（fs）を返す"""

        t0_fs = self.t0 * 1e15

        if min_dt_fs is None:
            min_dt_fs = t0_fs * 1e-3
        if max_dt_fs is None:
            max_dt_fs = t0_fs * 10

        min_dim = min_dt_fs / t0_fs
        max_dim = max_dt_fs / t0_fs

        dt_dim = self.get_recommended_timestep_dimensionless(
            safety_factor=safety_factor,
            min_dt=min_dim,
            max_dt=max_dim,
            method=method,
            numerical_method=numerical_method,
        )

        return dt_dim * t0_fs

    def get_recommended_timestep(
        self, 
        method: str = "adaptive",
        safety_factor: float = 0.1
    ) -> Dict[str, float]:
        """
        推奨時間ステップを取得（後方互換性用）
        
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
        
        dt_dimensionless = self.get_recommended_timestep_dimensionless(
            safety_factor=safety_factor, method=method
        )
        dt_fs = dt_dimensionless * self.get_time_scale_fs()
        
        return {
            "dimensionless": dt_dimensionless,
            "fs": dt_fs,
            "rabi_periods_per_step": dt_dimensionless * λ / (2 * np.pi) if λ > 0 else float('inf')
        }

    # -----------------------------
    # 解析ユーティリティ
    # -----------------------------
    def analyze_timestep_requirements(self) -> Dict[str, Any]:
        """時間ステップ要件を多角的に分析"""
        λ = self.lambda_coupling
        t0_fs = self.t0 * 1e15

        methods = ["adaptive", "rabi", "stability"]
        rec: Dict[str, Any] = {}
        for m in methods:
            dt_dim = self.get_recommended_timestep_dimensionless(method=m)
            rec[m] = {
                "dt_dimensionless": dt_dim,
                "dt_fs": dt_dim * t0_fs,
                "steps_per_rabi_period": 2 * np.pi / (λ * dt_dim) if λ > 0 else np.inf,
            }

        regime, advice = (
            ("weak_coupling", "大きな時間ステップで計算効率を重視可能")
            if λ < 0.1
            else ("intermediate_coupling", "適度な時間ステップでバランスを取る")
            if λ < 1.0
            else ("strong_coupling", "Rabi振動解像のため小さな時間ステップが必要")
        )

        return {
            "lambda_coupling": λ,
            "regime": regime,
            "advice": advice,
            "time_scale_fs": t0_fs,
            "rabi_period_dimensionless": 2 * np.pi / λ if λ > 0 else np.inf,
            "rabi_period_fs": (2 * np.pi / λ) * t0_fs if λ > 0 else np.inf,
            "recommendations": rec,
            "default_choice": rec["adaptive"],
        }

    # -----------------------------
    # ファクトリーメソッド
    # -----------------------------
    @classmethod
    def from_physical_system(
        cls, 
        H0: np.ndarray, 
        mu_values: np.ndarray,
        field_amplitude: float
    ) -> "NondimensionalizationScales":
        """
        物理システムのパラメータからスケールを作成
        
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

    # -----------------------------
    # サマリー表示
    # -----------------------------
    def summary(self) -> str:
        """スケールファクターのサマリーを取得"""
        lines = [
            "Nondimensionalization Scales:",
            f"  Energy: {self.get_energy_scale_eV():.3f} eV ({self.E0:.3e} J)",
            f"  Dipole: {self.get_dipole_scale_D():.3f} D ({self.mu0:.3e} C·m)",
            f"  Field: {self.get_field_scale_MV_cm():.3f} MV/cm ({self.Efield0:.3e} V/m)",
            f"  Time: {self.get_time_scale_fs():.3f} fs ({self.t0:.3e} s)",
            f"  λ: {self.lambda_coupling:.3f} ({self.get_regime()} coupling)",
        ]
        return "\n".join(lines) 