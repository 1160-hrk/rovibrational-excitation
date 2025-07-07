# Propagation Module

量子状態の時間発展を計算するためのモジュールです。シュレーディンガー方程式、リウビル方程式、および混合状態の伝播をサポートしています。

## 主な機能

- 複数の時間発展方程式のサポート
  - シュレーディンガー方程式（純粋状態）
  - リウビル-フォンノイマン方程式（密度行列）
  - 混合状態の時間発展
- 複数の数値計算アルゴリズム
  - RK4法（4次のルンゲ・クッタ法）
  - Split-operator法
- 高度な最適化機能
  - スパース行列対応（非ゼロ成分の比率が5%以下または行列の次元が50を超える時に有効）
  - NumPy/CuPyバックエンドの切り替え（GPU計算対応）
  - 適応的時間ステップ制御（RK4法のみ）
- 物理単位の自動検証機能

## 基本的な使い方

### シュレーディンガー方程式の時間発展

```python
from rovibrational_excitation.core.propagation import SchrodingerPropagator

# プロパゲータの初期化
propagator = SchrodingerPropagator(
    algorithm="rk4",         # "rk4" または "split_operator"
    backend="numpy",         # "numpy" または "cupy"
    sparse=False,           # スパース行列を使用するかどうか
    validate_units=True,    # 物理単位の検証を行うかどうか
)

# 時間発展の計算
final_state = propagator.propagate(
    hamiltonian=H0,          # ハミルトニアンオブジェクト
    efield=efield,           # 電場オブジェクト
    dipole_matrix=dipole,    # 双極子モーメント行列オブジェクト
    psi0=initial_state,      # 初期状態
    axes="xy",               # 偏光軸の指定
    return_traj=True,        # 軌跡を返すかどうか
    sample_stride=1,         # サンプリング間隔
)
```

### リウビル方程式の時間発展

```python
from rovibrational_excitation.core.propagation import LiouvillePropagator

# プロパゲータの初期化
propagator = LiouvillePropagator(
    backend="numpy",         # "numpy" または "cupy"
    validate_units=True,     # 物理単位の検証を行うかどうか
)

# 時間発展の計算
final_state = propagator.propagate(
    hamiltonian=H0,          # ハミルトニアンオブジェクト
    efield=efield,           # 電場オブジェクト
    dipole_matrix=dipole,    # 双極子モーメント行列オブジェクト
    initial_state=rho0,      # 初期密度行列
    axes="xy",               # 偏光軸の指定
    return_traj=True,        # 軌跡を返すかどうか
)
```

### 混合状態の時間発展

```python
from rovibrational_excitation.core.propagation import MixedStatePropagator

# プロパゲータの初期化
propagator = MixedStatePropagator(
    algorithm="rk4",         # "rk4" または "split_operator"
    backend="numpy",         # "numpy" または "cupy"
    sparse=False,           # スパース行列を使用するかどうか
    validate_units=True,    # 物理単位の検証を行うかどうか
)

# 時間発展の計算
final_states = propagator.propagate(
    hamiltonian=H0,          # ハミルトニアンオブジェクト
    efield=efield,           # 電場オブジェクト
    dipole_matrix=dipole,    # 双極子モーメント行列オブジェクト
    initial_state=psi0_list, # 初期状態のリスト
    return_traj=True,        # 軌跡を返すかどうか
)
```

## 専用プロパゲータ

### Split-operator法専用プロパゲータ

偏光が固定されている場合に最適化された実装を提供します。

```python
from rovibrational_excitation.core.propagation import SplitOperatorPropagator

propagator = SplitOperatorPropagator(
    backend="numpy",
    validate_units=True,
)

# 偏光の設定（オプション）
propagator.set_polarization(pol_x=1.0, pol_y=0.0)

# 時間発展の計算
final_state = propagator.propagate(
    hamiltonian=H0,
    efield=efield,
    dipole_matrix=dipole,
    psi0=initial_state,
)
```

### RK4法専用プロパゲータ

適応的時間ステップ制御機能を提供します。

```python
from rovibrational_excitation.core.propagation import RK4Propagator

propagator = RK4Propagator(
    backend="numpy",
    sparse=False,
    validate_units=True,
    adaptive=True,           # 適応的時間ステップ制御を有効化
)

# 時間発展の計算（誤差制御付き）
final_state = propagator.propagate_with_error_control(
    hamiltonian=H0,
    efield=efield,
    dipole_matrix=dipole,
    psi0=initial_state,
    tolerance=1e-6,          # 許容誤差
)
```

## アルゴリズムの選択指針

1. スパース行列の使用
   - 非ゼロ成分の比率が5%以下の場合
   - 行列の次元が50を超える場合

2. Split-operator法の使用
   - 偏光が固定されている場合
   - 大規模な系で高速な計算が必要な場合

3. RK4法の使用
   - 偏光が時間変化する場合
   - 高精度な計算が必要な場合（適応的時間ステップ制御）

4. バックエンドの選択
   - GPU利用可能な場合：CuPyバックエンド
   - それ以外：NumPyバックエンド

## モジュール構成

```
propagation/
├── algorithms/          # 数値計算アルゴリズムの実装
│   ├── rk4/            # RK4法関連
│   │   ├── lvne.py     # リウビル方程式用RK4
│   │   └── schrodinger.py  # シュレーディンガー方程式用RK4
│   └── split_operator/  # Split-operator法関連
│       └── schrodinger.py  # シュレーディンガー方程式用Split-operator
├── base.py             # 基底クラス（PropagatorBase）
├── liouville.py        # リウビル方程式プロパゲータ
├── mixed_state.py      # 混合状態プロパゲータ
├── rk4.py             # RK4専用プロパゲータ
├── schrodinger.py      # シュレーディンガー方程式プロパゲータ
├── split_operator.py   # Split-operator専用プロパゲータ
└── utils.py           # ユーティリティ関数（単位変換など）
```

## 注意事項

1. 物理単位について
   - ハミルトニアン：ジュール (J)
   - 双極子モーメント：クーロン・メートル (C·m)
   - 電場：ボルト/メートル (V/m)
   - 時間：フェムト秒 (fs)

2. バックエンド切り替え時の注意
   - CuPyバックエンド使用時は入力データも自動的にGPUに転送
   - スパース行列はCuPyバックエンド使用時に自動的に密行列に変換

3. エラー処理
   - 物理単位の不整合：`ValueError`
   - バックエンドの問題：`RuntimeError`
   - 入力データの形式エラー：`TypeError`

4. パフォーマンス最適化
   - スパース行列使用時は事前にパターンを解析して高速化
   - Split-operator法は偏光が固定の場合に最適化
   - RK4法は適応的時間ステップ制御で精度と速度のバランスを調整
