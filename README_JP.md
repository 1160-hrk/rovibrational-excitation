# rovibrational-excitation

<!-- Core Information -->
[![PyPI version](https://img.shields.io/pypi/v/rovibrational-excitation.svg)](https://pypi.org/project/rovibrational-excitation/)
[![Python](https://img.shields.io/pypi/pyversions/rovibrational-excitation.svg)](https://pypi.org/project/rovibrational-excitation/)
[![Downloads](https://img.shields.io/pypi/dm/rovibrational-excitation.svg)](https://pypi.org/project/rovibrational-excitation/)
[![License](https://img.shields.io/github/license/1160-hrk/rovibrational-excitation.svg)](https://github.com/1160-hrk/rovibrational-excitation/blob/main/LICENSE)

<!-- Quality Assurance -->
[![Tests](https://github.com/1160-hrk/rovibrational-excitation/actions/workflows/tests.yml/badge.svg)](https://github.com/1160-hrk/rovibrational-excitation/actions/workflows/tests.yml)
[![Coverage](https://codecov.io/gh/1160-hrk/rovibrational-excitation/branch/main/graph/badge.svg)](https://codecov.io/gh/1160-hrk/rovibrational-excitation)
[![Code Quality](https://github.com/1160-hrk/rovibrational-excitation/actions/workflows/ci.yml/badge.svg)](https://github.com/1160-hrk/rovibrational-excitation/actions/workflows/ci.yml)

**線形分子の振動回転励起シミュレーションのためのPythonパッケージ**

フェムト秒〜ピコ秒レーザーパルスによって駆動される線形分子（回転×振動）の時間依存量子動力学計算を行うためのツールです。

<div align="center">

| CPU / GPU (CuPy) | Numba-JIT RK4 propagator | Lazy, cached dipole matrices |
|------------------|--------------------------|------------------------------|

</div>

---

## 🚀 主要機能

### 🔧 高性能な時間発展計算エンジン
- **Runge-Kutta 4次法 (RK-4)** による高精度時間発展
  - シュレーディンガー方程式
  - リウヴィル・フォン・ノイマン方程式
  - `complex128`精度、キャッシュ効率最適化
- **スプリット演算子法** CPU/GPUバックエンド対応

### ⚡ 高速双極子行列構築
- **遅延評価・キャッシュ機能**による高速計算
- 剛体回転子 + 調和振動子/モース振動子
- Numba (CPU) / CuPy (GPU) バックエンド対応

### 🌊 柔軟な電場制御
- ガウシアンエンベロープ
- チャープ機能
- 正弦波・分割変調オプション
- ベクトル電場対応

### 📊 バッチ処理・解析機能
- ポンプ・プローブ実験シミュレーション
- パラメータスイープ
- 自動ディレクトリ作成
- プログレスバー表示
- 圧縮出力 (`.npz`)

### 🔬 対応分子
- **線形分子専用設計**
- 回転量子数 J, M を考慮
- 将来的に非線形分子への拡張予定

### 🏗️ Pure Python実装
- 100 % pure-Python、**コンパイル済み拡張なし** (Numbaが実行時にコンパイル)

---

## テスト・カバレッジ

パッケージには包括的なテストスイートが含まれており、全モジュールで **63%のコードカバレッジ** を達成しています。

- 🟢 **Basis classes**: 100% カバレッジ (LinMol, TwoLevel, VibLadder)
- 🟡 **Core physics**: 55% 全体カバレッジ
  - States: 98% カバレッジ
  - Propagator: 83% カバレッジ
  - Hamiltonian: 67% カバレッジ
- 🟡 **Electric field**: 53% カバレッジ
- 🟡 **Dipole matrices**: 52-96% カバレッジ (サブシステムにより異なる)
- 🔴 **Low-level propagators**: 25-38% カバレッジ (開発継続中)
- 🟡 **Simulation runner**: 62% カバレッジ

詳細なカバレッジレポートとテスト手順については [`tests/README.md`](tests/README.md) を参照してください。

```bash
# テスト実行
cd tests/ && python -m pytest -v

# カバレッジレポート生成
coverage run -m pytest && coverage report
```

---

## 📦 インストール

### 安定版（PyPI）
```bash
pip install rovibrational-excitation
```

### 開発版（GitHub）
```bash
pip install git+https://github.com/1160-hrk/rovibrational-excitation.git
```

### GPU計算用オプション
```bash
# CUDA版CuPy（GPU加速用）
pip install cupy-cuda12x  # CUDA版に合わせて選択
```

---

## 📋 必要要件

### Python環境
- **Python**: 3.10以上
- **NumPy**: 配列操作・数値計算
- **SciPy**: 科学計算ライブラリ
- **Numba**: JITコンパイル（CPU高速化）

### オプション
- **CuPy**: GPU計算（CUDA必須）
- **Matplotlib**: グラフ作成
- **tqdm**: プログレスバー

---

## 📚 ドキュメント

詳細な使用方法とパラメータリファレンス:

| ドキュメント | 説明 | 対象 |
|----------|-------------|----------|
| **[docs/PARAMETER_REFERENCE.md](docs/PARAMETER_REFERENCE.md)** | **完全なパラメータリファレンス** | 全ユーザー |
| [docs/SWEEP_SPECIFICATION.md](docs/SWEEP_SPECIFICATION.md) | パラメータスイープ仕様 | 中級者 |
| [docs/README.md](docs/README.md) | ドキュメント索引・クイックガイド | 全ユーザー |
| [examples/params_template.py](examples/params_template.py) | パラメータファイルテンプレート | 初心者 |

### 🚀 はじめに

1. **パラメータリファレンスを読む**: [docs/PARAMETER_REFERENCE.md](docs/PARAMETER_REFERENCE.md)
2. **テンプレートをコピー**: `cp examples/params_template.py my_params.py`
3. **システムに応じてパラメータを編集**
4. **シミュレーション実行**: `python -m rovibrational_excitation.simulation.runner my_params.py`

---

## 🎯 クイックスタート

### 1. ライブラリAPIの基本使用例

```python
import numpy as np
import rovibrational_excitation as rve

# === 物理定数の設定 ===
c_vacuum = 299792458 * 1e2 / 1e15      # 光速 [cm/fs]
debye_unit = 3.33564e-30               # デバイ単位 [C·m]
Omega01_rad_phz = 2349*2*np.pi*c_vacuum    # 振動周波数
Delta_omega_rad_phz = 25*2*np.pi*c_vacuum  # 非調和項
B_rad_phz = 0.39e-3*2*np.pi*c_vacuum       # 回転定数
Mu0_Cm = 0.3 * debye_unit              # 双極子モーメント
Potential_type = "harmonic"  # または "morse"
V_max = 2
J_max = 4

basis = rve.LinMolBasis(
    V_max=V_max, J_max=J_max, use_M=True,
    omega_rad_phz=Omega01_rad_phz,
    delta_omega_rad_phz=Delta_omega_rad_phz
)  # |v J M⟩ 直積

dip = rve.LinMolDipoleMatrix(
    basis, mu0=Mu0_Cm, potential_type=Potential_type,
    backend="numpy", dense=True
)  # CSR on GPU

mu_x = dip.mu_x  # 遅延構築、以後キャッシュ
mu_y = dip.mu_y
mu_z = dip.mu_z

# === 2. ハミルトニアンの構築 ===
H0 = rve.generate_H0_LinMol(
    basis,
    omega_rad_phz=Omega01_rad_phz,
    delta_omega_rad_phz=Delta_omega_rad_phz,
    B_rad_phz=B_rad_phz
)

# === 3. 電場の設定 ===
t = np.linspace(-200, 200, 4001)  # 時間軸 [fs]
E = rve.ElectricField(tlist=t)

E.add_dispersed_Efield(
    envelope_func=rve.fields.gaussian_fwhm,
    duration=50.0,                # ガウシアンFWHM [fs]
    t_center=0.0,                 # パルス中心時刻
    carrier_freq=2349*2*np.pi*c_vacuum,  # キャリア周波数
    amplitude=1.0,                # 振幅
    polarization=[1.0, 0.0]       # x偏光
)

# === 4. 初期状態 |v=0,J=0,M=0⟩ ===
from rovibrational_excitation.core.states import StateVector
psi0 = StateVector(basis)
psi0.set_state((0,0,0), 1.0)  # (v,J,M) = (0,0,0)
psi0.normalize()

# === 5. 時間発展計算 ===
psi_t = rve.schrodinger_propagation(
    H0, E, dip, psi0.data,
    axes="xy",              # Ex→μx, Ey→μy
    sample_stride=10,       # サンプリング間隔
    backend="numpy"         # または "cupy"
)

# 結果の解析
population = np.abs(psi_t)**2
print(f"Population shape: {population.shape}")  # (Nt, dim)
```

### 2. バッチ実行システム

#### パラメータファイル作成 (`params_CO2.py`)

```python
# 計算の説明（結果ディレクトリ名に使用）
description = "CO2_antisymm_stretch"

# === 時間軸設定 ===
t_start, t_end, dt = -200.0, 200.0, 0.1  # [fs]

# === 電場パラメータスキャン ===
duration = [50.0, 80.0]                  # ガウシアンFWHM [fs]
polarization = [
    [1, 0],                               # x偏光
    [1/2**0.5, 1j/2**0.5]                # 円偏光
]
t_center = [0.0, 100.0]                  # パルス中心 [fs]

carrier_freq = 2349*2*np.pi*1e12*1e-15   # キャリア周波数 [rad/fs]
amplitude = 1.0e9                         # 電場振幅 [V/m]

# === 分子定数 ===
V_max, J_max = 2, 4                      # 振動・回転の最大量子数
omega_rad_phz = carrier_freq * 2 * np.pi # 振動周波数
mu0_Cm = 0.3 * 3.33564e-30              # 双極子モーメント [C·m]
```

#### バッチ実行

```bash
# 4プロセス並列実行
python -m rovibrational_excitation.simulation.runner \
       examples/params_CO2.py -j 4

# ドライラン（実行せずにケース一覧表示）
python -m rovibrational_excitation.simulation.runner \
       examples/params_CO2.py --dry-run
```

#### 出力結果

```
results/
└── YYYY-MM-DD_hh-mm-ss_CO2_antisymm_stretch/
    ├── summary.csv              # 全ケースの概要
    ├── case_001/
    │   ├── result.npz           # 計算結果
    │   └── parameters.json      # パラメータ
    ├── case_002/
    │   └── ...
    └── ...
```

---

## 🔬 適用例

### CO2分子の反対称伸縮振動励起
- **分子**: CO2 (線形三原子分子)
- **励起モード**: 反対称伸縮振動 (ν₃ ≈ 2349 cm⁻¹)
- **レーザー**: フェムト秒パルス
- **解析**: 振動準位間の個体数移動

### ポンプ・プローブ実験
- **ポンプパルス**: 分子励起
- **プローブパルス**: 時間遅延後の状態探査
- **測定量**: 時間分解スペクトル、個体数動力学

---

## 🏗️ プロジェクト構造

```
rovibrational_excitation/
├── src/rovibrational_excitation/
│   ├── __init__.py          # public re-export
│   ├── core/                # 核となる計算エンジン
│   │   ├── basis/           # 基底関数クラス
│   │   │   ├── __init__.py
│   │   │   ├── base.py      # 抽象基底クラス
│   │   │   ├── linmol.py    # 線形分子基底
│   │   │   ├── twolevel.py  # 二準位系
│   │   │   └── viblad.py    # 振動はしご
│   │   ├── propagator.py    # 時間発展演算子
│   │   ├── electric_field.py # 電場オブジェクト
│   │   ├── hamiltonian.py   # ハミルトニアン構築 (DEPRECATED)
│   │   ├── states.py        # 量子状態管理
│   │   ├── _rk4_schrodinger.py # RK4-シュレーディンガー法
│   │   ├── _rk4_lvne.py     # RK4-LVNE法
│   │   └── _splitop_schrodinger.py # スプリット演算子法
│   ├── dipole/              # 双極子モーメント計算
│   │   ├── linmol/          # 線形分子双極子
│   │   │   ├── builder.py   # 双極子行列構築
│   │   │   └── cache.py     # キャッシュ管理
│   │   ├── twolevel/        # 二準位系
│   │   ├── viblad/          # 振動はしご
│   │   ├── rot/             # 回転双極子
│   │   │   ├── j.py         # J量子数関連
│   │   │   └── jm.py        # J,M量子数関連
│   │   └── vib/             # 振動双極子
│   │       ├── harmonic.py  # 調和振動子
│   │       └── morse.py     # モース振動子
│   ├── plots/               # 可視化ツール
│   │   ├── plot_electric_field.py      # 電場プロット
│   │   ├── plot_electric_field_vector.py # 電場ベクトル
│   │   └── plot_population.py          # 個体数プロット
│   └── simulation/          # シミュレーション管理
│       ├── runner.py        # バッチ実行エンジン
│       ├── manager.py       # 実行管理
│       └── config.py        # 設定管理
├── tests/                   # 単体テスト (pytest)
├── validation/              # 物理検証スクリプト
│   ├── core/                # コア物理検証
│   ├── dipole/              # 双極子行列検証
│   └── simulation/          # 統合検証
├── examples/                # 使用例
└── docs/                    # ドキュメント
```

### 検証 vs テスト

- **`tests/`**: コードの正確性を確認する単体テスト（高速、包括的）
- **`validation/`**: 科学的精度を確認する物理検証（低速、物理法則に焦点）

```bash
# 単体テスト実行
pytest tests/ -v

# 物理検証実行
python validation/core/check_core_basis.py
find validation/ -name "check_*.py" -exec python {} \;
```

---

## 🛠️ 開発環境セットアップ

```bash
# リポジトリクローン
git clone https://github.com/1160-hrk/rovibrational-excitation.git
cd rovibrational-excitation

# 仮想環境作成・有効化
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# または
.venv\Scripts\activate     # Windows

# 開発用依存関係インストール
pip install -r requirements-dev.txt

# テスト実行
pytest -v
```

### 開発用ツール
- **Black**: コードフォーマッター
- **Ruff**: 高速リンター  
- **MyPy**: 静的型チェック
- **pytest**: テストフレームワーク

設定は `pyproject.toml` に記述されています。

---

## 🤝 コントリビューション

1. **Issue報告**: バグ報告・機能要求
2. **Pull Request**: コード改善・新機能追加
3. **ドキュメント**: 使用例・チュートリアル追加

### 開発ガイドライン
- PEP8準拠のコードスタイル
- 型ヒント必須
- テストカバレッジ維持
- 詳細なdocstring

---

## 📚 参考文献

1. **量子力学**: Griffiths, "Introduction to Quantum Mechanics"
2. **分子分光学**: Herzberg, "Molecular Spectra and Molecular Structure"
3. **数値計算**: Press et al., "Numerical Recipes"

---

## 📄 ライセンス

[MIT License](LICENSE)

© 2025 Hiroki Tsusaka. All rights reserved.

---

## 📞 お問い合わせ

- **GitHub Issues**: [リポジトリ](https://github.com/1160-hrk/rovibrational-excitation)
- **Email**: プロジェクトページをご確認ください

---

*最終更新: 2025年1月* 