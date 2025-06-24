# 新しい量子システムの使用方法

このドキュメントでは、新たに追加された二準位系と振動ラダー系の使用方法を説明します。

## 追加されたシステム

### 1. 二準位系 (Two-Level System)
- **基底クラス**: `TwoLevelBasis`
- **双極子行列クラス**: `TwoLevelDipoleMatrix`
- **用途**: 量子光学、原子物理学、基本的な量子遷移

### 2. 振動ラダー系 (Vibrational Ladder System)
- **基底クラス**: `VibLadderBasis`
- **双極子行列クラス**: `VibLadderDipoleMatrix`
- **用途**: 分子振動励起（回転なし）、調和振動子・Morse振動子

## ディレクトリ構造

```
src/rovibrational_excitation/
├── core/
│   └── basis/                    # 新しい基底モジュール
│       ├── __init__.py
│       ├── base.py               # 抽象基底クラス
│       ├── linmol.py            # 線形分子基底（改良版）
│       ├── twolevel.py          # 二準位系基底
│       └── viblad.py            # 振動ラダー基底
└── dipole/
    ├── twolevel/                # 二準位系双極子行列
    │   ├── __init__.py
    │   └── builder.py
    └── viblad/                  # 振動ラダー双極子行列
        ├── __init__.py
        └── builder.py
```

## 使用例

### 二準位系

```python
from rovibrational_excitation.core.basis import TwoLevelBasis
from rovibrational_excitation.dipole.twolevel import TwoLevelDipoleMatrix

# 基底の作成
basis = TwoLevelBasis()
print(f"基底サイズ: {basis.size()}")  # → 2

# ハミルトニアンの生成
H0 = basis.generate_H0(energy_gap=2.0)  # |1⟩ - |0⟩ = 2.0 a.u.

# 双極子行列の生成
dipole = TwoLevelDipoleMatrix(basis, mu0=1.0)
print(f"σ_x 行列:\n{dipole.mu_x}")
print(f"σ_y 行列:\n{dipole.mu_y}")
print(f"σ_z 行列:\n{dipole.mu_z}")
```

### 振動ラダー系

```python
from rovibrational_excitation.core.basis import VibLadderBasis
from rovibrational_excitation.dipole.viblad import VibLadderDipoleMatrix

# 基底の作成
basis = VibLadderBasis(V_max=5, omega_rad_phz=1.0, delta_omega_rad_phz=0.01)
print(f"基底サイズ: {basis.size()}")  # → 6

# ハミルトニアンの生成
H0 = basis.generate_H0()

# 双極子行列の生成（調和振動子）
dipole_harm = VibLadderDipoleMatrix(basis, mu0=1e-30, potential_type="harmonic")

# 双極子行列の生成（Morse振動子）
dipole_morse = VibLadderDipoleMatrix(basis, mu0=1e-30, potential_type="morse")
```

## 時間発展シミュレーション

### 二準位系のラビ振動

```python
from rovibrational_excitation.core.states import StateVector
from rovibrational_excitation.core.electric_field import ElectricField, gaussian
from rovibrational_excitation.core.propagator import schrodinger_propagation

# 初期状態: |0⟩
state = StateVector(basis)
state.set_state(0, 1.0)
psi0 = state.data

# 電場設定
time_array = np.arange(0, 100, 0.1)
Efield = ElectricField(tlist=time_array)
Efield.add_dispersed_Efield(
    envelope_func=gaussian,
    duration=50.0,
    t_center=50.0,
    carrier_freq=energy_gap/(2*np.pi),  # 共鳴
    amplitude=0.1,
    polarization=np.array([1, 0]),  # Ex方向
    const_polarisation=False,
)

# 時間発展計算
time_psi, psi_t = schrodinger_propagation(
    H0=H0,
    Efield=Efield,
    dipole_matrix=dipole,
    psi0=psi0,
    axes="xy",  # Exとμ_x、Eyとμ_yをカップリング
    return_traj=True,
    return_time_psi=True,
    sample_stride=5
)
```

### 振動励起シミュレーション

```python
# 初期状態: |v=0⟩
state = StateVector(basis)
state.set_state((0,), 1.0)
psi0 = state.data

# 電場設定（振動遷移）
Efield.add_dispersed_Efield(
    envelope_func=gaussian,
    duration=100.0,
    t_center=250.0,
    carrier_freq=omega_rad_phz/(2*np.pi),  # 振動周波数に共鳴
    amplitude=1e10,
    polarization=np.array([0, 1]),  # Ey方向
    const_polarisation=False,
)

# 時間発展計算
time_psi, psi_t = schrodinger_propagation(
    H0=H0,
    Efield=Efield,
    dipole_matrix=dipole_harm,
    psi0=psi0,
    axes="zy",  # Eyとμ_zをカップリング
    return_traj=True,
    return_time_psi=True,
    sample_stride=5
)
```

## 電場と双極子行列の軸対応

| 電場成分 | 双極子行列 | axes設定例 |
|----------|------------|------------|
| Ex | μ_x | "xy" (Ex↔μ_x, Ey↔μ_y) |
| Ey | μ_y | "xy" (Ex↔μ_x, Ey↔μ_y) |
| Ey | μ_z | "zy" (Ex↔μ_z, Ey↔μ_y) |

## サンプルコード

- **チェック用**: `scripts/check_new_basis.py`, `scripts/check_new_simulations.py`
- **実例**: `examples/example_twolevel_excitation.py`, `examples/example_vibrational_excitation.py`

## 実行方法

```bash
# 基本的な動作確認
python scripts/check_new_basis.py
python scripts/check_new_simulations.py

# 詳細なシミュレーション例
python examples/example_twolevel_excitation.py
python examples/example_vibrational_excitation.py
```

## 後方互換性

既存の `LinMolBasis` は引き続き利用可能ですが、新しいパッケージからの import を推奨します：

```python
# 推奨（新）
from rovibrational_excitation.core.basis import LinMolBasis

# 非推奨（旧）- deprecation warning が表示されます
from rovibrational_excitation.core.basis import LinMolBasis
```

## 注意事項

1. **電場の polarization**: ElectricField では2要素ベクトル `[Ex, Ey]` を使用
2. **axes パラメータ**: propagator では2文字の組み合わせ（例："xy", "zy"）
3. **双極子行列**: 全てのクラスで `mu_x`, `mu_y`, `mu_z` プロパティが利用可能

## 開発ノート

- 新しい量子システムを追加する場合は `BasisBase` を継承
- 双極子行列クラスは `mu_x`, `mu_y`, `mu_z` プロパティを実装する必要がある
- propagator との互換性を保つため、適切な型ヒントを追加することを推奨 