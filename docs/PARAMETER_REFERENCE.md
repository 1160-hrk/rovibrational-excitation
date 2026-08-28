# パラメータリファレンス (Parameter Reference)

## 概要

rovibrational-excitation パッケージでは、パラメータファイル（`.py` ファイル）を使用してシミュレーション設定を行います。このドキュメントでは、指定可能な全パラメータの詳細と使用例を説明します。

## パラメータファイルの基本構造

```python
#!/usr/bin/env python
"""
パラメータファイル例
"""
import numpy as np

# メタ情報
description = "my_simulation"

# モデル設定
basis_type = "linmol"
representation = "m_resolved"
axes = "xy"
initial_states = [0]

# 時間軸設定
t_start, t_end, dt = -50.0, 50.0, 0.1

# 物理パラメータ
V_max, J_max = 3, 5
vibrational_frequency = 2349.0
vibrational_frequency_units = "cm^-1"
anharmonic_shift = 0.0
anharmonic_shift_units = "cm^-1"
rotational_constant = 0.02
rotational_constant_units = "rad/fs"
vibration_rotation_coupling = 0.0
vibration_rotation_coupling_units = "rad/fs"
potential_type = "harmonic"
mu0_Cm = 1.0e-30

# 電場パラメータ
duration = [20.0, 30.0]  # スイープ対象
carrier_frequency = 2349.0
carrier_frequency_units = "cm^-1"
amplitude = 1e9
polarization = [1.0, 0.0]  # 固定値

# 実行設定（暗黙値なし）
backend = "numpy"
storage = "dense"
algorithm = "rk4"
nondimensional = False
renorm = False
return_traj = True
sample_stride = 1
```

## パラメータ一覧

### 1. 必須パラメータ

#### 1.1 メタ情報

| パラメータ | 型 | 必須 | 説明 | 例 |
|-----------|---|------|------|-----|
| `description` | `str` | 推奨 | シミュレーションの説明（結果ディレクトリ名に使用） | `"CO2_excitation"` |

#### 1.2 時間軸設定

| パラメータ | 型 | 必須 | 単位 | 説明 | 例 |
|-----------|---|------|------|------|-----|
| `t_start` | `float` | ✅ | fs | 時間軸の開始時刻 | `-100.0` |
| `t_end` | `float` | ✅ | fs | 時間軸の終了時刻 | `100.0` |
| `dt` | `float` | ✅ | fs | 電場のサンプリング間隔 | `0.1` |

`dt` は電場のサンプリング間隔です。RK4 / split-operator の1時間発展ステップは
左端・中点・右端を使うため `2 * dt` 進みます。したがって
`t_end - t_start` は `2 * dt` の整数倍でなければエラーになります。

#### 1.3 量子系設定

| パラメータ | 型 | 必須 | 説明 | 例 |
|-----------|---|------|------|-----|
| `basis_type` | `str` | ✅ | モデル名 | `"linmol"` |
| `V_max` | `int` | LinMol / VibLadder ✅ | 最大振動量子数 | `3` |
| `J_max` | `int` | LinMol ✅ | 最大回転量子数 | `5` |
| `representation` | `str` | LinMol ✅ | `"m_resolved"`: Mを明示、`"m_incoherent_average"`: M縮退の非干渉平均 | `"m_resolved"` |

`representation="m_incoherent_average"` は M=0 の純粋状態近似ではありません。固定直線偏光を内部 z 軸へ
合わせ、初期 J の各 M ブロックを別々に時間発展し、規格化重み
`1 / (2J+1)` で population を非干渉和します。円偏光・楕円偏光・時間依存偏光は
受け付けません。外部の直線偏光方向は任意ですが、`axes` は指定できません。

複数の `initial_states` は同じ J 内（異なる v）のコヒーレント重ね合わせだけを
許します。異なる J をまたぐ指定は、等方的な M 平均でコヒーレンスを一意に
定められないためエラーです。

#### 1.4 基本電場パラメータ

| パラメータ | 型 | 必須 | 単位 | 説明 | 例 |
|-----------|---|------|------|------|-----|
| `envelope_kind` | `str` | ✅ | - | 包絡線の種類（下表） | `"gaussian_fwhm"` |
| `duration` | `float` | ✅ | fs | 包絡線の幅（意味は種類ごとに異なる） | `20.0` |
| `t_center` | `float` | ✅ | fs | パルス中心時刻（暗黙値なし） | `0.0` |
| `modulation_kind` | `str` | ✅ | - | `"none"` または `"sinusoidal"` | `"none"` |
| `carrier_frequency` | `float` | ✅ | 次行で指定 | 搬送波周波数（通常周波数・波数・角周波数を選択可） | `2349.0` |
| `carrier_frequency_units` | `str` | ✅ | - | `carrier_frequency` の単位 | `"cm^-1"` |
| `amplitude` | `float` | ✅ | V/m | 電場振幅 | `1e9` |
| `polarization` | `list` | generated `m_resolved` ✅ | - | Jones偏光ベクトル [x, y] | `[1.0, 0.0]` |

`carrier_frequency_units` は必須です。通常周波数は `Hz`、`kHz`、`MHz`、
`GHz`、`THz`、`PHz`、波数は `cm^-1`、`cm-1`、`wavenumber`、
角周波数は `rad/s`、`rad/ps`、`rad/fs` を選べます。通常周波数と波数の
入力値には `2π` を含めません。角周波数を選んだ場合だけ入力値に `2π` が
含まれます。runner は検証済みの `Frequency` 境界で一度だけ `rad/fs` へ
正規化します。旧 `carrier_freq` は単位が曖昧なため削除され、移行エラーに
なります。

通常 runner が受け付ける包絡線は `gaussian`（`duration` は標準偏差）、
`gaussian_fwhm`（FWHM）、`lorentzian`（半値半幅）、`lorentzian_fwhm`（FWHM）です。
Voigt は2個の幅が必要なため、この単一 `duration` スキーマでは受け付けません。Voigt、
任意 callable、任意波形は、正確な `TimeGrid` を持つ `ScalarField` または
`CartesianField` として Python API に注入してください。runner は補間・再標本化しません。

`duration` は必須です。旧名 `pulse_duration` は削除済みで、自動変換せず
時間発展前に移行エラーになります。

`polarization` は生成電場を使う `m_resolved` LinMol で必須です。
TwoLevel と VibLadder は偏光自由度を持たないため指定せず、指定した場合は
適用不能キーとしてエラーになります。両モデルは内部の固定 x 成分から生成した
スカラー電場と結合し、typed scalar field に Jones 偏光を保持しません。

LinMol の `representation="m_incoherent_average"` も偏光を省略できます。
指定する場合は固定直線偏光だけを受け付け、その方向には依存しません。ただしこれは
偏光自由度がないためではなく、量子化軸を固定直線偏光へ合わせて M 縮退を
非干渉平均する、D-017 の近似によるものです。

#### 1.5 基本物理パラメータ

| パラメータ | 型 | 必須 | 単位 | 説明 | 例 |
|-----------|---|------|------|------|-----|
| `vibrational_frequency` | `float` | LinMol / VibLadder ✅ | 対応する `*_units` | 0→1振動遷移周波数 | `2349.0` |
| `vibrational_frequency_units` | `str` | LinMol / VibLadder ✅ | - | 振動周波数の単位 | `"cm^-1"` |
| `anharmonic_shift` | `float` | LinMol / VibLadder ✅ | 対応する `*_units` | 隣接遷移周波数の準位ごとの減少量 | `0.0` |
| `anharmonic_shift_units` | `str` | LinMol / VibLadder ✅ | - | 非調和シフトの単位 | `"cm^-1"` |
| `rotational_constant` | `float` | LinMol ✅ | 対応する `*_units` | 回転定数 | `0.3902` |
| `rotational_constant_units` | `str` | LinMol ✅ | - | 回転定数の単位 | `"cm^-1"` |
| `vibration_rotation_coupling` | `float` | LinMol ✅ | 対応する `*_units` | 振動-回転相互作用定数 | `0.0` |
| `vibration_rotation_coupling_units` | `str` | LinMol ✅ | - | 振動-回転相互作用定数の単位 | `"cm^-1"` |
| `potential_type` | `str` | LinMol / VibLadder ✅ | - | `"harmonic"` または `"morse"` | `"harmonic"` |
| `mu0_Cm` | `float` | 全モデル ✅ | C·m | 双極子モーメント | `1e-30` |
| `energy_gap` | `float` | TwoLevel ✅ | `energy_gap_units`で指定 | 二準位間のエネルギー差 | `1.0` |
| `energy_gap_units` | `str` | TwoLevel ✅ | - | energy_gap の単位 | `"rad/fs"` |

ゼロが正しい物理値である場合も `0.0` を明示してください。省略とゼロは区別され、
基底クラスを直接生成する場合も LinMol、VibLadder、TwoLevel の物理定数は未指定だとエラーになります。
双極子行列の直接生成では `mu0` が必須で、振動を含むモデルでは `potential_type` も必須です。

#### 1.6 初期状態

| パラメータ | 型 | 必須 | 説明 | 例 |
|-----------|---|------|------|-----|
| `initial_states` | `list[int]` | ✅ | 初期状態のインデックス（暗黙の基底状態なし） | `[0]` |

複数インデックスは、等振幅・同位相で正規化したコヒーレント重ね合わせとして扱います。インコヒーレント混合には `MixedStatePropagator` を使用します。空リストはエラーです。

### 2. オプションパラメータ

#### 2.1 電場高度設定

| パラメータ | 型 | デフォルト | 単位 | 説明 | 例 |
|-----------|---|-----------|------|------|-----|
| `gdd` | `float` | `0.0` | fs² | 群遅延分散（2次） | `1000.0` |
| `tod` | `float` | `0.0` | fs³ | 群遅延分散（3次） | `50000.0` |
| `phase_rad` | `float` | `0.0` | rad | キャリア位相 | `np.pi/4` |

#### 2.2 正弦波変調

| パラメータ | 型 | デフォルト | 説明 | 例 |
|-----------|---|-----------|------|-----|
| `modulation_kind` | `str` | 必須 | `"none"` または `"sinusoidal"` | `"sinusoidal"` |
| `amplitude_sin_mod` | `float` | - | 変調振幅 | `0.1` |
| `carrier_freq_sin_mod` | `float` | - | 既存のスペクトル変調係数（単位契約は未確定） | `0.01` |
| `phase_rad_sin_mod` | `float` | `0.0` | 変調位相 | `np.pi/2` |
| `type_mod_sin_mod` | `str` | sinusoidal時は必須 | 変調タイプ | `"phase"` or `"amplitude"` |

`modulation_kind="sinusoidal"` では `amplitude_sin_mod`、`carrier_freq_sin_mod`、
`type_mod_sin_mod` をすべて明示します。`phase_rad_sin_mod` だけは加算的な不在値
`0.0` を既定値として保持します。`modulation_kind="none"` と正弦変調用キーの併記は
エラーです。旧 `Sinusoidal_modulation` は削除済みで、自動変換しません。

#### 2.3 ハミルトニアンの定義

振動エネルギーは `vibrational_frequency` を0→1遷移周波数として受け取り、明示された単位から内部の角周波数へ一度だけ変換した後、`E_v = (omega + delta_omega)(v + 1/2) - (delta_omega / 2)(v + 1/2)^2` で定義します。

`potential_type = "morse"` の場合、`anharmonic_shift` は非ゼロ必須です。Morse準位パラメータは正規化済みの `vibrational_frequency` と `anharmonic_shift` からケースごとに計算され、`V_max <= floor(N) - 1` を満たさない入力はエラーになります。

#### 2.4 双極子行列設定

| パラメータ | 型 | 必須 | 説明 | 例 |
|-----------|---|------|------|-----|
| `backend` | `str` | ✅ | 双極子行列生成と時間発展で共通の計算バックエンド | `"numpy"`, `"cupy"` |
| `storage` | `str` | ✅ | 行列保存方式 | `"dense"`, `"csr"` |

`backend` は双極子行列生成と時間発展の両方に適用されます。CuPy 経路は密行列専用で、
`backend = "cupy"` と `storage = "csr"` の組合せは型不一致へ進む前に
エラーになります。削除済みの `dense` または `sparse` を指定した場合もエラーです。

#### 2.5 伝播設定

| パラメータ | 型 | 必須/既定値 | 説明 | 例 |
|-----------|---|-------------|------|-----|
| `axes` | `str` | `m_resolved` で必須 | 電場-双極子の軸対応。M平均では指定不可 | `"xy"`, `"zx"` |
| `algorithm` | `str` | ✅ | 時間発展法 | `"rk4"`, `"split_operator"` |
| `split_interaction` | `str` | M-resolved LinMol の split 法で必須 | split相互作用モデル | `"cartesian"`, `"helicity_projected"` |
| `renorm` | `bool` | ✅ | 各ステップで状態を再規格化するか | `False` |
| `nondimensional` | `bool` | ✅ | 無次元化して時間発展するか | `False` |
| `validate_units` | `bool` | `True` | 物理単位を検証するか | `False` |
| `verbose` | `bool` | `False` | 詳細な検証情報を表示するか | `True` |
| `return_traj` | `bool` | ✅ | 軌跡を返すか | `True` |
| `sample_stride` | `int` | ✅ | サンプリング間隔 | `1` |

`auto_timestep` と `target_accuracy` は削除済みです。指定した場合は、
入力時間格子を暗黙変更せずエラーになります。

runner は保存結果との対応を保証するため、常に物理時間 `t_p` を生成します。
`return_traj = False` の場合、`t_p` は `[t_end]`、population は `(1, n_states)` です。
`split_interaction` は `algorithm="split_operator"` かつ
`basis_type="linmol"`, `representation="m_resolved"` の場合だけ指定します。
RK4、TwoLevel、VibLadder、M平均で指定するとエラーです。
`split_interaction = "cartesian"` は RK4 と同じ実 Cartesian 電場を使用します。
`"helicity_projected"` は片方向遷移演算子とその随伴を使う明示的な近似です。
split法はスパース入力を受け付けますが、相互作用の固有ベクトルは密行列なので、
`storage = "csr"` はsplit法のスパースメモリスケーリングを意味しません。


#### 2.6 出力設定

単位名が不正な場合、値を未変換のまま継続せず `ValueError` になります。必須値の欠落、
非有限値、ゼロ偏光、モデルに不正な整数値も時間発展前にエラーになります。未知キー、
別モデルの物理パラメータ、生成電場と外部電場の混在、アルゴリズムに適用不能なキーも、
無視せず該当キーを示してエラーになります。

| パラメータ | 型 | デフォルト | 説明 | 例 |
|-----------|---|-----------|------|-----|
| `save` | `bool` | `True` | 結果を保存するか | `False` |
| `outdir` | `str` | 自動生成 | 出力ディレクトリ | `"/path/to/output"` |

### 3. スイープ制御パラメータ

#### 3.1 固定値キー (FIXED_VALUE_KEYS)

以下のキーは常に固定値として扱われます（リスト形式でもスイープ対象になりません）：

| キー | 説明 | 例 |
|-----|------|-----|
| `polarization` | 適用可能な LinMol での偏光ベクトル | `[1.0, 0.0]` |
| `initial_states` | 初期状態 | `[0, 5]` |

#### 3.2 明示的スイープ指定

キー名に `_sweep` 接尾辞を付けると明示的にスイープ対象になります：

```python
# 明示的スイープ指定
amplitude_sweep = [1e8, 5e8, 1e9]      # 3ケース → 'amplitude' として保存
duration_sweep = [20.0, 30.0, 40.0]   # 3ケース → 'duration' として保存

# 合計: 3 × 3 = 9ケース
```

#### 3.3 従来のスイープ判定

`_sweep` 接尾辞がなく、FIXED_VALUE_KEYS に含まれないキーは、リストの長さで判定されます：

```python
# 従来の判定ルール
V_max = [3, 5, 7]     # 長さ3 → スイープ対象（3ケース）
J_max = [2]           # 長さ1 → 固定値
amplitude = 1e9       # スカラー → 固定値
```

## 外部サンプル電場をPythonから注入する

Python APIでは、パルス生成パラメータの代わりに、既にサンプリング済みの
実数電場を渡せます。入口は
`rovibrational_excitation.simulation.runner.run_simulation_case` です。

```python
import numpy as np

from rovibrational_excitation.core.time import TimeGrid
from rovibrational_excitation.fields import ScalarField
from rovibrational_excitation.simulation.runner import run_simulation_case

grid = TimeGrid.from_bounds(-50.0, 50.0, 0.1)
samples_v_per_m = 1.0e8 * np.exp(-(grid.field_times_fs / 20.0) ** 2)
field = ScalarField(grid, samples_v_per_m)

params = {
    "basis_type": "twolevel",
    "energy_gap": 0.2,
    "energy_gap_units": "rad/fs",
    "mu0_Cm": 3.0e-30,
    "initial_states": [0],
    "backend": "numpy",
    "storage": "dense",
    "algorithm": "rk4",
    "return_traj": True,
    "sample_stride": 1,
    "nondimensional": False,
    "renorm": False,
    "save": False,
}
population = run_simulation_case(params, field=field)
```

TwoLevel、VibLadder、`m_incoherent_average`は`ScalarField`を使います。
`m_resolved` LinMolは次のように`CartesianField(grid, E0, E1)`を使い、
2成分の順序を`axes`（例: `"xy"`）へ対応させます。

外部注入では`t_start`、`t_end`、`dt`、`duration`、
`carrier_frequency`、`carrier_frequency_units`、`amplitude`、`polarization`などの生成用キーを同時に
指定するとエラーです。時間は`field.time_grid`だけが正本です。入力配列は
防御コピーされ、有限・実数・1次元・格子と同じ長さでなければエラーになります。
格子は有限、単調増加、等間隔、奇数長で、両端点を含む必要があります。
ライブラリは切り詰め、padding、丸め、補間、resampling、規格化を行いません。

一般の`CartesianField`からJones偏光やscalar波形を推定しません。
近似`split_interaction="helicity_projected"`を外部電場で明示的に使う場合だけ、
コンストラクタへ規格化済みの`jones_polarization`と対応する
`scalar_samples_v_per_m`を両方渡す必要があります。厳密なCartesian RK4 /
split-operatorではこの追加情報は不要です。

保存時の`result.npz["E"]`は、scalarなら`(n_samples,)`、
Cartesianなら`(n_samples, 2)`です。

## 使用例

### 基本例

```python
#!/usr/bin/env python
"""
基本的なシミュレーション設定
"""
import numpy as np

# メタ情報
description = "basic_simulation"

# 時間軸（短時間で高速計算）
t_start, t_end, dt = -20.0, 20.0, 0.1

# 量子系（小規模で高速計算）
V_max, J_max = 2, 2

# 物理パラメータ
vibrational_frequency = 2349.0
vibrational_frequency_units = "cm^-1"  # CO2 ν3 mode
mu0_Cm = 0.3 * 3.33564e-30                      # ~0.3 Debye

# 電場パラメータ
envelope_kind = "gaussian_fwhm"
modulation_kind = "none"
duration = 10.0
t_center = 0.0
carrier_frequency = 2349.0
carrier_frequency_units = "cm^-1"
amplitude = 1e9
polarization = [1.0, 0.0]  # x偏光

# 初期状態
initial_states = [0]  # 基底状態

# 計算設定
backend = "numpy"
sample_stride = 1
```

### スイープ例

```python
#!/usr/bin/env python
"""
パラメータスイープの例
"""
import numpy as np

description = "parameter_sweep"

# 基本設定
t_start, t_end, dt = -50.0, 50.0, 0.1
V_max, J_max = 3, 3
vibrational_frequency = 2349.0
vibrational_frequency_units = "cm^-1"
mu0_Cm = 0.3 * 3.33564e-30
t_center = 0.0
carrier_frequency = 2349.0
carrier_frequency_units = "cm^-1"
envelope_kind = "gaussian_fwhm"
modulation_kind = "none"

# スイープパラメータ
duration = [10.0, 20.0, 30.0]           # 3ケース
amplitude_sweep = [1e8, 5e8, 1e9]       # 3ケース → 'amplitude'として保存
polarization = [1.0, 0.0]               # 固定値（x偏光）

# 初期状態
initial_states = [0]

# 合計ケース数: 3 × 3 = 9ケース
```

### 高度な設定例

```python
#!/usr/bin/env python
"""
高度な機能を使用した設定例
"""
import numpy as np

description = "advanced_simulation"

# 時間軸
t_start, t_end, dt = -100.0, 100.0, 0.05

# 量子系（大規模計算）
V_max, J_max = 5, 10
representation = "m_resolved"
axes = "xy"

# 物理パラメータ（CO2分子）
vibrational_frequency = 2349.0
vibrational_frequency_units = "cm^-1"
anharmonic_shift = 2.349
anharmonic_shift_units = "cm^-1"
rotational_constant = 0.39
rotational_constant_units = "cm^-1"
vibration_rotation_coupling = 0.000039
vibration_rotation_coupling_units = "cm^-1"
mu0_Cm = 0.3 * 3.33564e-30
potential_type = "morse"

# 電場パラメータ（成形パルス）
# Voigtや任意波形は外部 sampled field として Python API に注入する
envelope_kind = "gaussian_fwhm"
duration = 50.0
t_center = 0.0
carrier_frequency = 2349.0
carrier_frequency_units = "cm^-1"
amplitude = 5e9
polarization = [1/np.sqrt(2), 1j/np.sqrt(2)]   # 円偏光
phase_rad = np.pi/4
gdd = 1000.0                                    # 群遅延分散
tod = 50000.0                                   # 3次分散

# 正弦波変調
modulation_kind = "sinusoidal"
amplitude_sin_mod = 0.1
carrier_freq_sin_mod = 0.01
phase_rad_sin_mod = np.pi/2
type_mod_sin_mod = "phase"

# 初期状態（複数状態の重ね合わせ）
initial_states = [0, 1, 2]

# 計算設定
backend = "cupy"        # GPU計算
storage = "csr"        # CSRスパース行列
axes = "xy"
sample_stride = 5       # メモリ節約
```

## 単位系

### 時間
- **基本単位**: fs（フェムト秒）

### 周波数
- **内部標準単位**: rad/fs
- **通常周波数入力**: 1 THz = 0.001 cycles/fs = `2π × 10^-3 rad/fs`
- **波数入力**: 1 cm⁻¹ = `2π c × 10^-13 rad/fs`（c は m/s）
- **入力規則**: `*_units` を必須指定し、通常周波数・波数には `2π` を含めない

### 電場
- **基本単位**: V/m
- **典型値**: 1e8 ～ 1e12 V/m

### 双極子モーメント
- **基本単位**: C·m
- **変換**: Debye → C·m: `μ_D × 3.33564e-30`

## パフォーマンス最適化

### 高速化のコツ

1. **小さな系から始める**: `V_max`, `J_max` を小さく設定
2. **時間軸を短く**: `t_start`, `t_end` の範囲を最小限に
3. **サンプリング間隔**: `sample_stride` を増やしてメモリ節約
4. **バックエンド選択**: CuPy（GPU）利用可能な場合は `backend="cupy"`

### メモリ節約

```python
# メモリ効率の良い設定
sample_stride = 10      # サンプリング間隔を増やす
storage = "csr"        # CSRスパース行列を使用
backend = "numpy"       # CPUで確実に動作
```

### 大規模計算

```python
# 大規模計算の設定
V_max, J_max = 10, 20   # 大きな基底
backend = "cupy"        # GPU加速
storage = "dense"      # GPU計算では密行列を使用
nproc = 8               # 並列実行
checkpoint_interval = 5 # チェックポイント頻度を上げる
```

## トラブルシューティング

### よくあるエラー

1. **パラメータ不足**:
   ```
   KeyError: 't_start'
   ```
   → 必須パラメータが不足。上記リストを確認

2. **スイープキーエラー**:
   ```
   ValueError: Parameter 'amplitude_sweep' has '_sweep' suffix but is not iterable
   ```
   → `_sweep` 接尾辞キーは必ずリストにする

3. **メモリ不足**:
   ```
   MemoryError
   ```
   → `V_max`, `J_max` を小さくするか `sample_stride` を増やす

### デバッグ方法

1. **ドライラン**: 計算を実行せずケース数確認
   ```bash
   python -m rovibrational_excitation.simulation.runner params.py --dry-run
   ```

2. **小規模テスト**: パラメータを小さくして動作確認

3. **ログ確認**: エラーメッセージとパラメータを照合

## 関連ドキュメント

- [スイープ仕様](SWEEP_SPECIFICATION.md) - パラメータスイープの詳細
- [パッケージAPI](../src/rovibrational_excitation/) - モジュール詳細
- [examples/](../examples/) - パラメータファイル例 