"""
impl.py
=======
後方互換性のためのスタブモジュール。

このモジュールは廃止予定です。
すべての機能は以下のモジュールに移動されました：
- converter.py: 基本的な無次元化機能
- analysis.py: 分析機能
- strategies.py: λスケーリング戦略
- utils.py: ユーティリティ関数
- scales.py: スケールファクター管理

新しいコードでは適切なモジュールから直接インポートしてください。
"""

from __future__ import annotations

# 後方互換性のための re-export
from .converter import (
    nondimensionalize_system,
    nondimensionalize_with_SI_base_units,
    determine_SI_based_scales,
    create_dimensionless_time_array,
)

from .analysis import (
    analyze_regime,
    verify_nondimensional_equation,
    optimize_timestep_for_coupling,
    calculate_nondimensionalization_scales_strict,
)

from .strategies import (
    create_effective_field_scaling,
    create_effective_dipole_scaling,
    create_unified_scaling_approach,
    recommend_lambda_strategy,
    NondimensionalizedSystem,
)

from .utils import (
    convert_default_units_to_SI_base,
    dimensionalize_wavefunction,
    get_physical_time,
    create_SI_demo_parameters,
)

from .scales import NondimensionalizationScales

# 廃止予定警告
import warnings
warnings.warn(
    "The 'impl' module is deprecated. Please import functions directly from "
    "'converter', 'analysis', 'strategies', or 'utils' modules instead.",
    DeprecationWarning,
    stacklevel=2
) 