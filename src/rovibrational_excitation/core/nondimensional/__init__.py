"""
Nondimensionalization system for quantum dynamics calculations.

This package provides a modular system for nondimensionalizing
quantum mechanical equations to improve numerical stability.

Modules:
--------
- scales: 無次元化スケールファクター管理
- converter: 物理量の無次元化変換
- analysis: 無次元化関連の分析機能
- strategies: λスケーリング戦略
- utils: 共通ユーティリティ関数

Main Classes:
-------------
- NondimensionalizationScales: スケールファクター管理
- NondimensionalConverter: 高レベル変換インターフェース
- NondimensionalAnalyzer: 分析機能
- LambdaScalingStrategy: λスケーリング戦略の列挙型

Main Functions:
---------------
- nondimensionalize_system: 基本的な無次元化
- nondimensionalize_with_SI_base_units: SI基本単位での無次元化
- nondimensionalize_from_objects: HamiltonianとDipoleMatrixBaseクラスから自動SI変換
- auto_nondimensionalize: 完全自動無次元化（最適時間ステップ自動選択）
- analyze_regime: 物理レジーム分析
- verify_nondimensional_equation: 無次元方程式の検証
"""

# コアクラス
from .scales import NondimensionalizationScales

# 変換機能
from .converter import (
    NondimensionalConverter,
    nondimensionalize_system,
    nondimensionalize_with_SI_base_units,
    nondimensionalize_from_objects,
    auto_nondimensionalize,
    create_dimensionless_time_array,
)

# 分析機能
from .analysis import (
    NondimensionalAnalyzer,
    analyze_regime,
    verify_nondimensional_equation,
    optimize_timestep_for_coupling,
    calculate_nondimensionalization_scales_strict,
)

# 戦略機能
from .strategies import (
    LambdaScalingStrategy,
    EffectiveFieldStrategy,
    EffectiveDipoleStrategy,
    NondimensionalizedSystem,
    ExplicitLambdaSystem,
    create_effective_field_scaling,
    create_effective_dipole_scaling,
    create_unified_scaling_approach,
    recommend_lambda_strategy,
)

# ユーティリティ関数
from .utils import (
    convert_default_units_to_SI_base,
    dimensionalize_wavefunction,
    get_physical_time,
    create_SI_demo_parameters,
)

__all__ = [
    # コアクラス
    "NondimensionalizationScales",
    
    # 変換機能
    "NondimensionalConverter",
    "nondimensionalize_system",
    "nondimensionalize_with_SI_base_units",
    "nondimensionalize_from_objects",
    "auto_nondimensionalize",
    "create_dimensionless_time_array",
    
    # 分析機能
    "NondimensionalAnalyzer",
    "analyze_regime",
    "verify_nondimensional_equation",
    "optimize_timestep_for_coupling",
    "calculate_nondimensionalization_scales_strict",
    
    # 戦略機能
    "LambdaScalingStrategy",
    "EffectiveFieldStrategy",
    "EffectiveDipoleStrategy",
    "NondimensionalizedSystem",
    "ExplicitLambdaSystem",
    "create_effective_field_scaling",
    "create_effective_dipole_scaling",
    "create_unified_scaling_approach",
    "recommend_lambda_strategy",
    
    # ユーティリティ関数
    "convert_default_units_to_SI_base",
    "dimensionalize_wavefunction",
    "get_physical_time",
    "create_SI_demo_parameters",
] 