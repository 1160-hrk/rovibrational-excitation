# 量子基底モジュール (basis)

このモジュールは、分子の量子状態を表現するための基底クラス群を提供します。

## 概要

basisモジュールは以下の基底クラスを提供します：

- `BasisBase`: 全ての基底クラスの抽象基底クラス
- `LinMolBasis`: 線形分子の振動・回転基底
- `SymTopBasis`: 対称コマ分子の振動・回転基底
- `TwoLevelBasis`: 二準位系の基底
- `VibLadderBasis`: 振動準位のみの基底（回転なし）
- `StateVector`: 純粋状態を表現するクラス
- `DensityMatrix`: 混合状態を表現するクラス

## 基本的な使い方

各基底クラスは以下の共通インターフェースを持ちます：

```python
def size(self) -> int:
    """基底の次元数を返す"""
    
def get_index(self, state) -> int:
    """量子数から基底のインデックスを取得"""
    
def get_state(self, index: int):
    """インデックスから量子数を取得"""
    
def generate_H0(self) -> Hamiltonian:
    """自由ハミルトニアンを生成"""
```

## 使用例

### 1. 二準位系の例

```python
from rovibrational_excitation.core.basis import TwoLevelBasis, StateVector

# 2.35 eVのエネルギーギャップを持つ二準位系
basis = TwoLevelBasis(energy_gap=2.35, input_units="eV")

# 基底状態のベクトルを作成
state = StateVector(basis)
state.set_state(0)  # |0⟩を設定

# ハミルトニアンを生成（J単位）
H0 = basis.generate_H0()
print(f"エネルギーギャップ: {H0.eigenvalues[1]} J")
```

### 2. 線形分子（CO2など）の例

```python
from rovibrational_excitation.core.basis import LinMolBasis

# CO2分子のパラメータ（cm^-1単位）
basis = LinMolBasis(
    V_max=2,          # 最大振動量子数
    J_max=10,         # 最大回転量子数
    omega=2350,       # ν3モードの振動周波数
    B=0.39,           # 回転定数
    alpha=0.0042,     # 振動回転相互作用定数
    input_units="cm^-1"
)

# 基底の大きさを確認
print(f"基底の次元: {basis.size()}")

# 特定の状態のインデックスを取得
# V=0, J=1, M=0の状態
idx = basis.get_index([0, 1, 0])

# ハミルトニアンを生成（rad/fs単位）
H0 = basis.generate_H0(output_units="rad/fs")
```

### 3. 振動準位のみの系

```python
from rovibrational_excitation.core.basis import VibLadderBasis, DensityMatrix

# 非調和振動子
basis = VibLadderBasis(
    V_max=5,              # 最大振動量子数
    omega=500,            # 振動周波数
    delta_omega=5,        # 非調和性パラメータ
    input_units="cm^-1"
)

# 混合状態を作成
rho = DensityMatrix(basis)
# 熱平衡分布を仮定
populations = [0.5, 0.3, 0.1, 0.06, 0.03, 0.01]
rho.set_diagonal(populations)

# ハミルトニアンのエネルギー準位を確認
H0 = basis.generate_H0()
print("振動エネルギー準位:")
for v, E in enumerate(H0.eigenvalues):
    print(f"|v={v}⟩: {E:.2e} J")
```

### 4. 対称コマ分子の例

```python
from rovibrational_excitation.core.basis import SymTopBasis

# メチルフルオライド（CH3F）のような対称コマ分子
basis = SymTopBasis(
    V_max=1,          # 最大振動量子数
    J_max=5,          # 最大回転量子数
    omega=1000,       # 代表的な振動モード
    B=1.0,           # 回転定数B
    C=0.8,           # 回転定数C
    input_units="cm^-1"
)

# ハミルトニアンを生成
H0 = basis.generate_H0()
print(f"基底の次元: {basis.size()}")
print(f"最大エネルギー差: {H0.max_energy_difference():.2e} J")
```

## 注意事項

1. 単位系について
   - 入力パラメータは様々な単位（cm^-1, THz, eV等）で指定可能
   - ハミルトニアンの出力は"J"または"rad/fs"のみ
   - 内部計算は全てrad/fsで行われる

2. 基底の制限
   - 各基底クラスは特定の分子系に特化
   - 量子数の範囲は初期化時に固定
   - メモリ使用量は量子数の範囲に応じて増加

3. パフォーマンス
   - ハミルトニアンは対角行列として生成
   - 状態とインデックスの変換は辞書を使用して高速化
   - 大きな量子数での使用時はメモリ使用量に注意

## 参考文献

1. Cohen-Tannoudji, C., et al. "Quantum Mechanics"
2. Herzberg, G. "Molecular Spectra and Molecular Structure" 