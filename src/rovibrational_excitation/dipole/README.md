# 双極子行列モジュール

このモジュールは、量子分子系の電気双極子モーメントを計算するためのPythonライブラリです。
異なる分子系に対応し、NumPyとCuPyの両方のバックエンドをサポートしています。

## 特徴

- 複数の分子系（対称コマ分子、直線分子、振動ラダーシステム、二準位系）に対応
- NumPy/CuPyバックエンド対応
- 密/疎行列形式サポート
- 自動単位変換機能
- 効率的なキャッシュシステム
- 物理的選択則の自動適用
- 基底クラスに応じた双極子行列クラスの自動選択

## 使用例

### 0. ファクトリー関数を使用した自動選択

```python
from rovibrational_excitation.core.basis import LinMolBasis, TwoLevelBasis
from rovibrational_excitation.dipole import create_dipole_matrix

# 線形分子の例
basis_linmol = LinMolBasis(V_max=2, J_max=10, omega=2350, input_units="cm^-1")
dipole_linmol = create_dipole_matrix(
    basis_linmol,
    mu0=1.0,
    potential_type="morse",  # 振動を含む系のみ指定可能
    backend="numpy",
    dense=True,
    units_input="D"  # 入力値の単位（デバイ）
)

# 二準位系の例
basis_twolevel = TwoLevelBasis(energy_gap=2.35, input_units="eV")
dipole_twolevel = create_dipole_matrix(
    basis_twolevel,
    mu0=0.5,
    units_input="D"  # potential_typeは不要（二準位系）
)

# 双極子行列の取得
mu_x = dipole_linmol.mu_x  # x方向
mu_y = dipole_linmol.mu_y  # y方向
mu_z = dipole_linmol.mu_z  # z方向

# 全方向の双極子行列をスタック
mu_xyz = dipole_linmol.stacked("xyz")  # 3次元配列として取得

# 単位変換
mu_x_SI = dipole_linmol.get_mu_x_SI()  # SI単位（C·m）
mu_x_debye = dipole_linmol.get_mu_in_units("x", "D")  # デバイ単位
```

### 1. 対称コマ分子系

```python
from rovibrational_excitation.core.basis import SymTopBasis
from rovibrational_excitation.dipole import SymTopDipoleMatrix

# 基底の作成（V_max=2, J_max=2, K_max=2の場合）
basis = SymTopBasis(
    V_max=2, 
    J_max=2,
    omega=1.0,  # 振動周波数
    delta_omega=0.0,  # 非調和性
    B = 0.01,
    C = 0.005,
)

# 双極子行列の生成
dipole = SymTopDipoleMatrix(
    basis,
    mu0=1.0,              # 双極子モーメントの大きさ
    potential_type="harmonic",  # または "morse"
    backend="numpy",      # 現在はnumpyのみサポート
    dense=True,           # または False で疎行列形式
    units="C*m",         # 内部保存単位（"C*m", "D", "ea0"）
    units_input="D"      # 入力値の単位
)

# 双極子行列要素の取得
mu_x = dipole.mu_x  # x成分
mu_y = dipole.mu_y  # y成分
mu_z = dipole.mu_z  # z成分

# 単位変換と行列の取得
mu_x_SI = dipole.get_mu_x_SI()  # SI単位（C·m）
mu_xyz = dipole.stacked("xyz")  # 全成分を3次元配列として取得
```

### 2. 直線分子系

```python
from rovibrational_excitation.core.basis import LinMolBasis
from rovibrational_excitation.dipole import LinMolDipoleMatrix

# 基底の作成（V_max=2, J_max=1の場合）
basis = LinMolBasis(V_max=2, J_max=1)

# 双極子行列の生成（調和ポテンシャル、CPU計算）
dipole = LinMolDipoleMatrix(
    basis,
    mu0=1.0,              # 双極子モーメントの大きさ
    potential_type="harmonic",
    backend="numpy",      # または "cupy" でGPU計算
    dense=True,           # または False で疎行列形式
    units="C*m",           # 内部保存単位（"C*m", "D", "ea0"）
    units_input="D"      # 入力値の単位
)

# 双極子行列要素の取得
mu_x = dipole.mu_x       # x成分
mu_y = dipole.mu_y       # y成分
mu_z = dipole.mu_z       # z成分

# 単位変換
mu_x_SI = dipole.get_mu_x_SI()                # SI単位（C·m）
mu_x_debye = dipole.get_mu_in_units("x", "D") # デバイ単位
mu_x_atomic = dipole.get_mu_in_units("x", "ea0") # 原子単位

# 全成分の一括取得
mu_xyz = dipole.stacked("xyz")  # 3次元配列として取得
```

### 2. 二準位系

```python
from rovibrational_excitation.core.basis import TwoLevelBasis
from rovibrational_excitation.dipole import TwoLevelDipoleMatrix

# 基底の作成
basis = TwoLevelBasis()

# 双極子行列の生成
dipole = TwoLevelDipoleMatrix(
    basis,
    mu0=1.0,
    backend="numpy",
    units="C*m"
)

# Pauli行列に基づく双極子行列
mu_x = dipole.mu_x  # σx型（|0⟩⟨1| + |1⟩⟨0|）
mu_y = dipole.mu_y  # σy型（i(|1⟩⟨0| - |0⟩⟨1|)）
mu_z = dipole.mu_z  # ゼロ行列
```

### 3. 振動ラダー系

```python
from rovibrational_excitation.core.basis import VibLadderBasis
from rovibrational_excitation.dipole import VibLadderDipoleMatrix

# 基底の作成（3準位系）
basis = VibLadderBasis(
    V_max=2,
    omega_rad_pfs=1.0,    # 振動周波数
    delta_omega_rad_pfs=0.0  # 非調和性
)

# 双極子行列の生成
dipole = VibLadderDipoleMatrix(
    basis,
    mu0=1.0,
    potential_type="harmonic",  # または "morse"
    backend="numpy"
)

# z方向のみ非ゼロの遷移
mu_z = dipole.mu_z  # ΔV = ±1 の遷移のみ許容
```

### 4. 大規模系での最適化

```python
# 疎行列形式の使用
dipole_sparse = LinMolDipoleMatrix(
    basis,
    mu0=1.0,
    dense=False  # CSR形式の疎行列を使用
)

# GPU計算の活用
dipole_gpu = LinMolDipoleMatrix(
    basis,
    mu0=1.0,
    backend="cupy",  # GPU上で計算
    dense=True
)
```

## 物理的選択則

各分子系で自動的に適用される選択則：

### 対称コマ分子
- 振動遷移: ΔV = ±1
- 回転遷移: 
  - ΔJ = ±1
  - ΔK = 0
  - ΔM = 0, ±1
- 行列要素は回転部分（tdm_jmk_{x,y,z}）と振動部分の積

### 線形分子
- 振動遷移: ΔV = ±1
- 回転遷移: ΔJ = ±1
- 磁気遷移: ΔM = 0, ±1

### 二準位系
- x方向: |0⟩⟨1| + |1⟩⟨0|
- y方向: i(|1⟩⟨0| - |0⟩⟨1|)
- z方向: 0

### 振動はしご
- z方向: ΔV = ±1
- x,y方向: 遷移禁止

## 注意事項

1. メモリ管理
   - 大規模系では疎行列形式を推奨
   - GPU使用時はメモリ容量に注意
   - 対称コマ分子では現在GPUバックエンドは未実装

2. 単位系
   - 内部計算はSI単位系で実施
   - 入出力時の単位変換に注意

3. キャッシュ
   - 行列要素は初回計算時にキャッシュ
   - メモリ使用量と計算速度のバランスを考慮

## 参考文献

1. Cohen-Tannoudji, C., et al. "Quantum Mechanics"
2. Zare, R. N. "Angular Momentum"
3. Herzberg, G. "Molecular Spectra and Molecular Structure"
4. Townes, C. H. & Schawlow, A. L. "Microwave Spectroscopy" 