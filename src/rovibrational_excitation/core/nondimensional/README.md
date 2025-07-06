# 無次元化モジュール使用ガイド

## 概要

`rovibrational_excitation.core.nondimensional` パッケージは、量子力学的方程式の無次元化を体系的に行うためのモジュラーシステムです。数値計算の安定性と効率性を向上させることを目的としています。

### 目標式

シュレディンガー方程式を以下の無次元形式に変換します：

```
i ∂ψ/∂τ = (H₀' - λ μ' E'(τ)) ψ
```

ここで：
- `τ`: 無次元時間
- `H₀'`: 無次元ハミルトニアン  
- `μ'`: 無次元双極子行列
- `E'(τ)`: 無次元電場
- `λ`: 無次元結合強度パラメータ

## モジュール構造

```
nondimensional/
├── __init__.py         # 統合API（推奨エントリーポイント）
├── scales.py          # スケールファクター管理
├── utils.py           # 基本ユーティリティ関数
├── converter.py       # 無次元化変換機能
├── analysis.py        # 分析機能  
├── strategies.py      # λスケーリング戦略
└── impl.py           # 後方互換性スタブ（廃止予定）
```

### 各モジュールの役割

- **`scales.py`**: `NondimensionalizationScales` クラス（スケールファクター管理）
- **`utils.py`**: 基本定数、単位変換、共通計算ユーティリティ
- **`converter.py`**: 物理量の無次元化変換の実装
- **`analysis.py`**: 物理レジーム分析、方程式検証、時間ステップ最適化
- **`strategies.py`**: λスケーリング戦略の実装
- **`__init__.py`**: 統合されたAPI（すべての機能を公開）

## 基本的な使用方法

### 1. 基本的な無次元化

```python
import numpy as np
from rovibrational_excitation.core.nondimensional import (
    nondimensionalize_system,
    NondimensionalizationScales
)

# 物理量の準備
H0 = np.diag([0, 1e-21, 2e-21])  # ハミルトニアン [J]
mu_x = np.array([[0, 1e-30, 0], [1e-30, 0, 1e-30], [0, 1e-30, 0]])  # 双極子 [C·m]
mu_y = np.zeros_like(mu_x)
# efield は ElectricField オブジェクト

# 無次元化実行
H0_prime, mu_x_prime, mu_y_prime, Efield_prime, tlist_prime, dt_prime, scales = \
    nondimensionalize_system(H0, mu_x, mu_y, efield)

print(f"結合強度 λ = {scales.lambda_coupling:.3f}")
print(f"物理レジーム: {scales.get_regime()}")
```

### 2. SI基本単位での無次元化（推奨）

```python
from rovibrational_excitation.core.nondimensional import (
    nondimensionalize_with_SI_base_units
)

# デフォルト単位から自動的にSI基本単位に変換して無次元化
result = nondimensionalize_with_SI_base_units(
    H0, mu_x, mu_y, efield,
    auto_timestep=True,  # 推奨時間ステップを自動選択
    timestep_method="adaptive",
    timestep_safety_factor=0.1
)

H0_prime, mu_x_prime, mu_y_prime, Efield_prime, tlist_prime, dt_prime, scales = result
```

### 3. オブジェクトベースの無次元化（新機能）

```python
from rovibrational_excitation.core.nondimensional import (
    nondimensionalize_from_objects,
    auto_nondimensionalize
)

# HamiltonianとDipoleMatrixBaseクラスから自動的にSI単位系に変換
# hamiltonian: Hamiltonianオブジェクト（内部単位管理）
# dipole_matrix: DipoleMatrixBaseオブジェクト（内部単位管理）
result = nondimensionalize_from_objects(
    hamiltonian, dipole_matrix, efield,
    auto_timestep=True,
    verbose=True
)

# 戻り値には3つの双極子成分が含まれる
H0_prime, mu_x_prime, mu_y_prime, mu_z_prime, Efield_prime, tlist_prime, dt_prime, scales = result

# 完全自動モード（最も簡単）
result = auto_nondimensionalize(
    hamiltonian, dipole_matrix, efield,
    target_accuracy="standard",  # "fast", "standard", "high"
    verbose=True
)

# 戻り値の展開
H0_prime, mu_x_prime, mu_y_prime, mu_z_prime, Efield_prime, tlist_prime, dt_prime, scales = result
```

### 4. スケールファクターの分析

```python
from rovibrational_excitation.core.nondimensional import NondimensionalAnalyzer

# 物理レジーム分析
regime_info = NondimensionalAnalyzer.analyze_regime(scales)
print(f"レジーム: {regime_info['description']}")
print(f"エネルギースケール: {regime_info['energy_scale_eV']:.3f} eV")

# 無次元方程式の検証
verification = NondimensionalAnalyzer.verify_equation(
    H0_prime, mu_x_prime, mu_y_prime, Efield_prime, scales, verbose=True
)
print(f"方程式が妥当: {verification['overall_valid']}")
```

## 高度な使用方法

### 1. 時間ステップの最適化

```python
from rovibrational_excitation.core.nondimensional import (
    NondimensionalAnalyzer,
    create_dimensionless_time_array
)

# 結合強度に最適化された時間ステップの提案
optimization = NondimensionalAnalyzer.optimize_timestep_for_coupling(
    scales,
    target_accuracy="high",  # "fast", "standard", "high", "ultrahigh"
    verbose=True
)

print(f"推奨時間ステップ: {optimization['recommended_dt_fs']:.3f} fs")
print(f"計算コスト（相対）: {optimization['computational_cost_estimate']:.1f}x")

# 最適化された時間配列の作成
tlist_opt, dt_opt = create_dimensionless_time_array(
    scales,
    duration_fs=100.0,
    auto_timestep=True,
    target_accuracy="high"
)
```

### 2. λスケーリング戦略

#### Strategy 1: 実効電場アプローチ（推奨）

```python
from rovibrational_excitation.core.nondimensional import (
    EffectiveFieldStrategy,
    create_effective_field_scaling
)

# λを電場に事前積算
E_effective, description = create_effective_field_scaling(scales, Efield_prime)
print(description)

# Propagatorで使用: H_interaction = μ' * E_effective
```

#### Strategy 2: 実効双極子アプローチ

```python
from rovibrational_excitation.core.nondimensional import (
    EffectiveDipoleStrategy,
    create_effective_dipole_scaling
)

# λを双極子に事前積算
mu_x_eff, mu_y_eff, description = create_effective_dipole_scaling(
    scales, mu_x_prime, mu_y_prime
)
print(description)

# Propagatorで使用: H_interaction = μ_eff * E'
```

#### Strategy 3: 明示的λ処理

```python
from rovibrational_excitation.core.nondimensional import NondimensionalizedSystem

# λを明示的に管理するシステム
ndsystem = NondimensionalizedSystem(
    H0_prime, mu_x_prime, mu_y_prime, Efield_prime, scales
)

# 時刻tでの相互作用ハミルトニアン
t_index = 100
H_int = ndsystem.get_interaction_hamiltonian(t_index)
H_total = ndsystem.get_total_hamiltonian(t_index)
```

#### Strategy 4: スケール統合アプローチ

```python
from rovibrational_excitation.core.nondimensional import create_unified_scaling_approach

# λをスケールに統合（高度な使用法）
H0_unified, mu_x_unified, mu_y_unified, E_unified, scales_unified = \
    create_unified_scaling_approach(H0, mu_x, mu_y, efield)

# この場合、effective λ = 1.0
print(f"統合後のλ: {scales_unified.lambda_coupling}")
```

### 3. 戦略の推奨取得

```python
from rovibrational_excitation.core.nondimensional import recommend_lambda_strategy

# 使用するpropagatorに応じた推奨戦略
recommendation = recommend_lambda_strategy(
    scales,
    propagator_type="split_operator"  # "rk4", "magnus"など
)

print(f"主要推奨: {recommendation['primary_recommendation']}")
print(f"リスクレベル: {recommendation['risk_level']}")
print(f"理由: {recommendation['physical_reason']}")
```

## 実践例

### 例1: CO分子の回転振動励起

```python
import numpy as np
from rovibrational_excitation.core.nondimensional import *

# CO分子のパラメータ（デフォルト単位）
H0_cm_inv = np.array([0, 2143.3, 4286.6])  # cm⁻¹
H0_J = H0_cm_inv * 1.986e-23  # J に変換

mu_D = np.array([[0, 0.3, 0], [0.3, 0, 0.3], [0, 0.3, 0]])  # Debye
mu_Cm = mu_D * 3.33564e-30  # C·m に変換

# 電場は別途定義済みと仮定: efield

# SI基本単位での無次元化
result = nondimensionalize_with_SI_base_units(
    H0_J, mu_Cm, np.zeros_like(mu_Cm), efield,
    auto_timestep=True,
    timestep_method="adaptive"
)

H0_prime, mu_x_prime, mu_y_prime, E_prime, t_prime, dt_prime, scales = result

# 分析
print(f"📊 物理パラメータ:")
print(f"   λ = {scales.lambda_coupling:.3f}")
print(f"   レジーム = {scales.get_regime()}")
print(f"   推奨dt = {dt_prime * scales.get_time_scale_fs():.3f} fs")

# 戦略選択
strategy_rec = recommend_lambda_strategy(scales, "split_operator")
print(f"🎯 推奨戦略: {strategy_rec['primary_recommendation']}")

# 実効電場を作成（Strategy 1）
E_effective, _ = create_effective_field_scaling(scales, E_prime)
```

### 例1b: CO分子の回転振動励起（オブジェクトベース）

```python
from rovibrational_excitation.core.basis.hamiltonian import Hamiltonian
from rovibrational_excitation.dipole.base import DipoleMatrixBase
from rovibrational_excitation.core.nondimensional import auto_nondimensionalize

# HamiltonianとDipoleMatrixBaseオブジェクトを作成
hamiltonian = Hamiltonian.from_input_units(
    np.array([0, 2143.3, 4286.6]),  # cm⁻¹
    "cm^-1",
    target_units="J"
)

# dipole_matrix = DipoleMatrixBase子クラスのインスタンス
# （例: LinMolDipoleMatrix, VibLadderDipoleMatrix等）
# 内部で mu_x, mu_y, mu_z の3成分を管理

# 完全自動無次元化（最も簡単）
result = auto_nondimensionalize(
    hamiltonian, dipole_matrix, efield,
    target_accuracy="standard",
    verbose=True
)

# 3つの双極子成分を含む戻り値
H0_prime, mu_x_prime, mu_y_prime, mu_z_prime, E_prime, t_prime, dt_prime, scales = result

print(f"🎯 自動選択結果:")
print(f"   λ = {scales.lambda_coupling:.3f}")
print(f"   推奨dt = {dt_prime * scales.get_time_scale_fs():.3f} fs")
print(f"   双極子成分:")
print(f"     x: {np.max(np.abs(mu_x_prime)):.3f}")
print(f"     y: {np.max(np.abs(mu_y_prime)):.3f}")
print(f"     z: {np.max(np.abs(mu_z_prime)):.3f}")
```

### 例2: 高精度計算のための設定

```python
# 高精度計算用の設定
optimization = NondimensionalAnalyzer.optimize_timestep_for_coupling(
    scales,
    target_accuracy="ultrahigh",
    verbose=True
)

# 厳密なスケール計算
strict_scales = NondimensionalAnalyzer.calculate_strict_scales(
    H0_J, mu_Cm, np.zeros_like(mu_Cm), efield,
    verbose=True
)

# 方程式の厳密な検証
verification = NondimensionalAnalyzer.verify_equation(
    H0_prime, mu_x_prime, mu_y_prime, E_prime, strict_scales,
    verbose=True
)

if not verification["overall_valid"]:
    print("⚠️ 警告: 無次元化が O(1) になっていません")
```

## API リファレンス

### メインクラス

#### `NondimensionalizationScales`

スケールファクター管理クラス

```python
scales = NondimensionalizationScales(E0, mu0, Efield0, t0, lambda_coupling)

# プロパティ
scales.E0               # エネルギースケール [J]
scales.mu0              # 双極子スケール [C·m]
scales.Efield0          # 電場スケール [V/m]
scales.t0               # 時間スケール [s]
scales.lambda_coupling  # 結合強度 λ

# メソッド
scales.get_time_scale_fs()        # 時間スケール [fs]
scales.get_energy_scale_eV()      # エネルギースケール [eV]
scales.get_regime()               # 物理レジーム
scales.get_recommended_timestep() # 推奨時間ステップ
scales.summary()                  # サマリー表示
```

### メイン関数

#### 変換関数

```python
# 基本無次元化（従来型）
nondimensionalize_system(H0, mu_x, mu_y, efield, **kwargs)
# 戻り値: (H0_prime, mu_x_prime, mu_y_prime, Efield_prime, tlist_prime, dt_prime, scales)

# SI基本単位無次元化（推奨）
nondimensionalize_with_SI_base_units(H0, mu_x, mu_y, efield, **kwargs)
# 戻り値: (H0_prime, mu_x_prime, mu_y_prime, Efield_prime, tlist_prime, dt_prime, scales)

# オブジェクトベース無次元化（新機能）
nondimensionalize_from_objects(hamiltonian, dipole_matrix, efield, **kwargs)
# 戻り値: (H0_prime, mu_x_prime, mu_y_prime, mu_z_prime, Efield_prime, tlist_prime, dt_prime, scales)

# 完全自動無次元化（新機能）
auto_nondimensionalize(hamiltonian, dipole_matrix, efield, **kwargs)
# 戻り値: (H0_prime, mu_x_prime, mu_y_prime, mu_z_prime, Efield_prime, tlist_prime, dt_prime, scales)

# 時間配列作成
create_dimensionless_time_array(scales, duration_fs, **kwargs)
# 戻り値: (tlist_dimensionless, dt_dimensionless)
```

#### 分析関数

```python
# 物理レジーム分析
analyze_regime(scales)

# 方程式検証
verify_nondimensional_equation(H0_prime, mu_x_prime, mu_y_prime, E_prime, scales)

# 時間ステップ最適化
optimize_timestep_for_coupling(scales, target_accuracy="standard")

# 厳密スケール計算
calculate_nondimensionalization_scales_strict(H0, mu_x, mu_y, efield)
```

#### 戦略関数

```python
# λスケーリング戦略
create_effective_field_scaling(scales, Efield_prime)
create_effective_dipole_scaling(scales, mu_x_prime, mu_y_prime)
create_unified_scaling_approach(H0, mu_x, mu_y, efield)
recommend_lambda_strategy(scales, propagator_type)
```

## 移行ガイド

### 既存コードからの移行

**旧方式（非推奨）:**
```python
# 廃止予定
from rovibrational_excitation.core.nondimensional.impl import nondimensionalize_system
```

**新方式（推奨）:**
```python
# 推奨
from rovibrational_excitation.core.nondimensional import nondimensionalize_system
```

### 段階的移行手順

1. **警告の確認**: 既存コードを実行して DeprecationWarning を確認
2. **インポートの更新**: `impl` からのインポートを `__init__` からに変更
3. **機能の最適化**: 新しい分析機能やSI基本単位機能を導入
4. **戦略の選択**: λスケーリング戦略を明示的に選択

## トラブルシューティング

### よくある問題

#### 1. λが異常に大きい/小さい

```python
if scales.lambda_coupling > 10:
    print("⚠️ 強結合域: 非常に小さな時間ステップが必要")
    recommendation = recommend_lambda_strategy(scales)
    print(f"推奨: {recommendation['primary_recommendation']}")

elif scales.lambda_coupling < 0.001:
    print("⚠️ 極弱結合域: スケールの見直しを検討")
```

#### 2. 無次元量が O(1) にならない

```python
verification = NondimensionalAnalyzer.verify_equation(
    H0_prime, mu_x_prime, mu_y_prime, E_prime, scales, verbose=True
)

if not verification["overall_valid"]:
    # 厳密スケール計算を試す
    strict_scales = NondimensionalAnalyzer.calculate_strict_scales(
        H0, mu_x, mu_y, efield, verbose=True
    )
```

#### 3. 時間ステップが小さすぎる

```python
# より高速な設定を試す
fast_optimization = NondimensionalAnalyzer.optimize_timestep_for_coupling(
    scales, target_accuracy="fast", verbose=True
)
```

### デバッグ支援

```python
# 詳細な診断情報
print(scales.summary())

# 時間ステップ要件分析
analysis = scales.analyze_timestep_requirements()
print(f"アドバイス: {analysis['advice']}")

# 物理レジーム確認
regime = analyze_regime(scales)
print(f"物理解釈: {regime['description']}")
```

## 注意点と制限事項

1. **単位の一貫性**: 入力の物理量は一貫した単位系で与えること
2. **数値精度**: 極端に小さい/大きい値では数値誤差に注意
3. **電場の時間依存性**: 電場オブジェクトが適切に設定されていること
4. **メモリ使用量**: 大きなシステムでは時間配列のサイズに注意

## サポート

問題や質問がある場合は、以下を確認してください：

1. このドキュメントのトラブルシューティング
2. 各関数のdocstring
3. デバッグ支援機能の使用
4. コードの例外メッセージとスタックトレース 