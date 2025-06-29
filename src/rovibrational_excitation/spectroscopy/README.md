# Linear Response Theory for Rovibrational Spectroscopy

## 概要

このモジュールは、振動回転（rovibrational）分光における線形応答理論の実装を提供します。密度行列形式を使用して、分子の量子状態から各種スペクトル（吸収、発光、PFID等）を計算できます。

## 理論的背景

### 線形応答理論

分子と電磁場の相互作用において、分子の応答が外部場に対して線形である範囲での理論です。密度行列 ρ の時間発展は以下のリウヴィル方程式で記述されます：

```
∂ρ/∂t = -i/ℏ [H, ρ] - Γρ
```

ここで：
- H: 分子のハミルトニアン
- Γ: 緩和演算子
- [A, B]: 交換子 AB - BA

### 分光スペクトルの計算

#### 吸収スペクトル
分子が電磁場を吸収する過程。遷移双極子モーメント μ と密度行列の交換子から計算されます：

```
α(ω) ∝ Im[Tr(μ† · χ(ω) · μ)]
```

#### 発光スペクトル  
励起状態からの自発放出。吸収とは逆の過程として計算されます。

#### PFID (Polarization-resolved Free Induction Decay)
偏光分解自由誘導減衰。分子の分極の時間発展を周波数領域で解析します。

### 振動回転エネルギー準位

線形分子のエネルギーは以下で表されます：

```
E(v,J) = ωe(v + 1/2) - ωexe(v + 1/2)² + [Be - αe(v + 1/2)]J(J+1)
```

- v: 振動量子数
- J: 回転量子数  
- ωe: 振動周波数
- ωexe: 非調和性定数
- Be: 回転定数
- αe: 振動-回転相互作用定数

## アーキテクチャ

### クラス設計

```
LinearResponseCalculator
├── SpectroscopyParameters (実験条件)
├── MolecularParameters (分子パラメータ)  
├── LinMolBasis (基底状態管理)
└── LinMolDipoleMatrix (遷移双極子行列)
```

### データフロー

1. **初期化**: 量子数と分子パラメータから基底系を構築
2. **行列計算**: エネルギー行列と遷移双極子行列を生成
3. **応答計算**: 密度行列から線形応答を計算
4. **スペクトル変換**: 物理的観測量（吸光度等）に変換

## 主要な実装特徴

### 効率的な計算
- 非ゼロ遷移のみを計算（疎行列活用）
- 振動コヒーレンス射影による計算の最適化
- エネルギー差分行列による高速化

### 広がり効果
- **ドップラー広がり**: 分子の熱運動による周波数分布
- **均質広がり**: コヒーレンス緩和時間 T₂ による
- **装置関数**: FTIR等の分光器による broadening

### 偏光効果
- レーザー偏光と放射偏光を独立に設定可能
- x, y, z 軸の遷移双極子成分を考慮

## 使用例

### Modern API (推奨)

```python
from rovibrational_excitation.spectroscopy import (
    LinearResponseCalculator, SpectroscopyParameters, MolecularParameters
)

# 分子パラメータ設定
molecular_params = MolecularParameters(
    vibrational_frequency_rad_per_fs=1000.0,  # CO₂ 対称伸縮振動
    rotational_constant_rad_per_fs=1.2,       # 回転定数
    transition_dipole_moment=0.3              # 遷移双極子 [ea₀]
)

# 分光条件設定  
spectroscopy_params = SpectroscopyParameters(
    temperature_K=300,                         # 室温
    pressure_Pa=3e4,                          # 300 hPa
    coherence_relaxation_time_ps=500,         # T₂ 緩和時間
    wavenumbers_cm=np.arange(2000, 2400, 0.1) # 波数範囲
)

# 計算器初期化
calculator = LinearResponseCalculator()
calculator.initialize(
    num_vibrational_levels=3,    # v = 0, 1, 2
    num_rotational_levels=10,    # J = 0-9  
    use_magnetic_quantum_numbers=True,  # M 量子数使用
    spectroscopy_params=spectroscopy_params,
    molecular_params=molecular_params
)

# 熱平衡密度行列（ボルツマン分布）
rho_thermal = create_thermal_density_matrix(calculator.basis, T=300)

# 吸収スペクトル計算
absorption = calculate_absorption_spectrum(rho_thermal, calculator)
```

### Legacy API (後方互換性)

```python
# 既存コードとの互換性
prepare_variables(Nv=3, Nj=10, T2=500, temp=300)
spectrum = absorbance_spectrum_for_loop(rho_thermal)
```

## 物理的妥当性の確認

### 選択則
- Δv = ±1 (基本振動遷移)
- ΔJ = ±1 (双極子許容遷移)
- ΔM = 0, ±1 (偏光依存)

### 強度の正規化
- アインシュタインのA係数との整合性
- ベール・ランバート則への準拠
- 単位系の一貫性（SI単位）

### 温度依存性
- ボルツマン分布による状態占有数
- ドップラー広がりの √T 依存性

## 拡張可能性

### 新しい分光手法
- 非線形分光（SFG, CARS等）
- 時間分解分光
- 多次元分光

### 分子系の拡張  
- 多原子分子（C₂H₄, NH₃等）
- 固体中の分子
- 溶液中の分子

### 計算手法の改良
- GPU並列化（CuPy対応済み）
- 疎行列最適化
- 近似手法の導入 