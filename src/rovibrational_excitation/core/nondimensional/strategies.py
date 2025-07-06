"""
strategies.py
=============
λ (lambda) のスケーリングに関する戦略を提供するモジュール。

無次元化後の正しい形: i d/dτ |ψ⟩ = (H₀' - λ μ' E'(τ)) |ψ⟩

Strategy 1: 実効電場アプローチ (推奨)
Strategy 2: 実効双極子アプローチ  
Strategy 3: 明示的λ処理アプローチ
Strategy 4: スケール統合アプローチ
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Tuple, Dict, TYPE_CHECKING
import numpy as np

from .scales import NondimensionalizationScales

if TYPE_CHECKING:  # pragma: no cover
    from rovibrational_excitation.core.electric_field import ElectricField


class LambdaScalingStrategy(Enum):
    """列挙型: λ スケーリング戦略の種類"""

    EFFECTIVE_FIELD = "effective_field"
    EFFECTIVE_DIPOLE = "effective_dipole"
    EXPLICIT = "explicit_lambda"  # Strategy 3
    UNIFIED_SCALE = "unified_scale"  # Strategy 4


def create_effective_field_scaling(
    scales: NondimensionalizationScales,
    Efield_prime: np.ndarray
) -> Tuple[np.ndarray, str]:
    """
    Strategy 1: 実効電場アプローチ (推奨)
    
    E_effective = λ * E' として電場にλを事前積算
    これにより propagator では μ' * E_effective を計算するだけ
    
    Parameters
    ----------
    scales : NondimensionalizationScales
        スケールファクター
    Efield_prime : np.ndarray
        無次元電場
        
    Returns
    -------
    tuple
        (E_effective, strategy_description)
    """
    λ = scales.lambda_coupling
    E_effective = λ * Efield_prime
    
    strategy_description = f"""
Strategy 1: Effective Field Scaling
- 実効電場: E_eff = λ * E' = {λ:.3f} * E'
- Propagator使用法: H_interaction = μ' * E_eff
- 利点: 電場の「実効強度」として物理的に直感的
- 利点: propagatorの変更が最小限
- 注意: E_effective は無次元だがλ倍されているので注意
    """
    
    return E_effective, strategy_description


def create_effective_dipole_scaling(
    scales: NondimensionalizationScales,
    mu_x_prime: np.ndarray,
    mu_y_prime: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Strategy 2: 実効双極子アプローチ
    
    μ_effective = λ * μ' として双極子にλを事前積算
    
    Parameters
    ----------
    scales : NondimensionalizationScales
        スケールファクター
    mu_x_prime, mu_y_prime : np.ndarray
        無次元双極子行列
        
    Returns
    -------
    tuple
        (mu_x_effective, mu_y_effective, strategy_description)
    """
    λ = scales.lambda_coupling
    mu_x_effective = λ * mu_x_prime
    mu_y_effective = λ * mu_y_prime
    
    strategy_description = f"""
Strategy 2: Effective Dipole Scaling  
- 実効双極子: μ_eff = λ * μ' = {λ:.3f} * μ'
- Propagator使用法: H_interaction = μ_eff * E'
- 利点: 双極子の「実効強度」として理解可能
- 欠点: x,y両成分に同じλが適用される
    """
    
    return mu_x_effective, mu_y_effective, strategy_description


class NondimensionalizedSystem:
    """
    Strategy 3: 明示的λ処理アプローチ
    
    λを明示的に保持し、propagatorで適切に処理
    """
    
    def __init__(
        self,
        H0_prime: np.ndarray,
        mu_x_prime: np.ndarray, 
        mu_y_prime: np.ndarray,
        Efield_prime: np.ndarray,
        scales: NondimensionalizationScales
    ):
        self.H0_prime = H0_prime
        self.mu_x_prime = mu_x_prime
        self.mu_y_prime = mu_y_prime
        self.Efield_prime = Efield_prime
        self.scales = scales
        self.lambda_coupling = scales.lambda_coupling
        
    def get_interaction_hamiltonian(self, time_index: int) -> np.ndarray:
        """
        正しい相互作用ハミルトニアンを計算: λ μ' E'(τ)
        
        Parameters
        ----------
        time_index : int
            時間インデックス
            
        Returns
        -------
        np.ndarray
            相互作用ハミルトニアン
        """
        Ex = self.Efield_prime[time_index, 0]
        Ey = self.Efield_prime[time_index, 1]
        
        # λ μ' E'(τ) = λ * (μ_x' * Ex + μ_y' * Ey)
        H_int = self.lambda_coupling * (
            self.mu_x_prime * Ex + self.mu_y_prime * Ey
        )
        
        return H_int
    
    def get_total_hamiltonian(self, time_index: int) -> np.ndarray:
        """
        全ハミルトニアンを計算: H₀' - λ μ' E'(τ)
        """
        H_int = self.get_interaction_hamiltonian(time_index)
        return self.H0_prime - H_int


def create_unified_scaling_approach(
    H0: np.ndarray,
    mu_x: np.ndarray,
    mu_y: np.ndarray, 
    efield: "ElectricField"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, NondimensionalizationScales]:
    """
    Strategy 4: スケール統合アプローチ
    
    λを双極子か電場のスケールに統合して、自然に正しい式になるようにする
    
    Parameters
    ----------
    H0, mu_x, mu_y : np.ndarray
        物理量（SI単位）
    efield : ElectricField
        電場オブジェクト
        
    Returns
    -------
    tuple
        統合スケーリング済みの無次元量
    """
    from .analysis import calculate_nondimensionalization_scales_strict
    
    # 厳密なスケール計算
    scales_original = calculate_nondimensionalization_scales_strict(
        H0, mu_x, mu_y, efield, verbose=False
    )
    
    # Strategy 4a: 双極子スケールにλを統合
    mu0_effective = scales_original.mu0 / scales_original.lambda_coupling
    
    # Strategy 4b: 電場スケールにλを統合  
    Efield0_effective = scales_original.Efield0 / scales_original.lambda_coupling
    
    # 統合スケールでの無次元化
    H0_prime = H0 / scales_original.E0
    
    # 方法A: 双極子統合（λが自動的に含まれる）
    mu_x_integrated = mu_x / mu0_effective  # = λ * μ/μ₀
    mu_y_integrated = mu_y / mu0_effective
    
    # 方法B: 電場統合（λが自動的に含まれる）
    Efield_integrated = efield.get_Efield() / Efield0_effective  # = λ * E/E₀
    
    # 新しいスケール情報
    scales_integrated = NondimensionalizationScales(
        E0=scales_original.E0,
        mu0=mu0_effective,  # または scales_original.mu0
        Efield0=Efield0_effective,  # または scales_original.Efield0  
        t0=scales_original.t0,
        lambda_coupling=1.0  # 既に統合済みなので1
    )
    
    print(f"""
Strategy 4: Unified Scaling Approach
- Original λ: {scales_original.lambda_coupling:.3f}
- Integrated into scales, so effective λ = 1.0
- Propagator can use: H_int = μ_integrated * E' (or μ' * E_integrated)
- 利点: λの明示的な処理が不要
- 利点: 数学的に自然
    """)
    
    return H0_prime, mu_x_integrated, mu_y_integrated, Efield_integrated, scales_integrated


def recommend_lambda_strategy(
    scales: NondimensionalizationScales,
    propagator_type: str = "split_operator"
) -> Dict[str, Any]:
    """
    λ処理戦略の推奨
    
    Parameters
    ----------
    scales : NondimensionalizationScales
        スケールファクター
    propagator_type : str, optional
        使用するpropagatorの種類
        
    Returns
    -------
    dict
        推奨戦略と実装ガイド
    """
    λ = scales.lambda_coupling
    
    # λの大きさに基づく推奨
    if λ < 0.1:
        # 弱結合: λの影響は小さいが、正確性のため必要
        primary_recommendation = "Strategy 1: Effective Field"
        risk_level = "Low"
        reason = "弱結合だが長時間計算で累積誤差の可能性"
        
    elif λ < 1.0:
        # 中間結合: λの正確な処理が重要
        primary_recommendation = "Strategy 1: Effective Field"  
        risk_level = "Medium"
        reason = "中間結合域、λの正確な処理が精度に影響"
        
    else:
        # 強結合: λの処理が極めて重要
        primary_recommendation = "Strategy 4: Unified Scaling"
        risk_level = "High"
        reason = "強結合域、λ抜けは大きな物理誤差を生む"
    
    # Propagator種別による推奨
    propagator_specific = {
        "split_operator": {
            "preferred": ["Strategy 1", "Strategy 4"],
            "reason": "高速性とユニタリ性を両立"
        },
        "rk4": {
            "preferred": ["Strategy 3", "Strategy 1"], 
            "reason": "明示的処理が高精度計算に適合"
        },
        "magnus": {
            "preferred": ["Strategy 4", "Strategy 3"],
            "reason": "数学的な厳密性を重視"
        }
    }
    
    return {
        "lambda_coupling": λ,
        "primary_recommendation": primary_recommendation,
        "risk_level": risk_level,
        "physical_reason": reason,
        "propagator_specific": propagator_specific.get(propagator_type, {}),
        "implementation_priority": "CRITICAL" if λ > 1.0 else "HIGH",
        "strategies_ranked": [
            "Strategy 1: Effective Field (推奨・汎用性)",
            "Strategy 4: Unified Scaling (推奨・厳密性)", 
            "Strategy 3: Explicit Lambda (完全制御)",
            "Strategy 2: Effective Dipole (特殊用途)"
        ]
    }


class EffectiveFieldStrategy:
    """Strategy 1: Effective Field Scaling"""

    @staticmethod
    def apply(
        scales: NondimensionalizationScales,
        Efield_prime: np.ndarray,
        *_,
        **__,
    ) -> Tuple[np.ndarray, str]:
        """λ を電場に吸収した実効電場を返す"""
        return create_effective_field_scaling(scales, Efield_prime)


class EffectiveDipoleStrategy:
    """Strategy 2: Effective Dipole Scaling"""

    @staticmethod
    def apply(
        scales: NondimensionalizationScales,
        mu_x_prime: np.ndarray,
        mu_y_prime: np.ndarray,
        *_,
        **__,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """λ を双極子に吸収した実効双極子を返す"""
        return create_effective_dipole_scaling(
            scales, mu_x_prime, mu_y_prime
        )


# Strategy 3 のエイリアス
ExplicitLambdaSystem = NondimensionalizedSystem 