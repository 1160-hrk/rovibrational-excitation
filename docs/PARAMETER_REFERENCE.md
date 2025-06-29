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

# 時間軸設定
t_start, t_end, dt = -50.0, 50.0, 0.1

# 物理パラメータ
V_max, J_max = 3, 5
omega_rad_phz = 2349 * 2 * np.pi * 3e10 / 1e15

# 電場パラメータ
duration = [20.0, 30.0]  # スイープ対象
amplitude = 1e9
polarization = [1.0, 0.0]  # 固定値

# その他の設定...
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
| `dt` | `float` | ✅ | fs | 時間刻み幅 | `0.1` |

#### 1.3 量子系設定

| パラメータ | 型 | 必須 | 説明 | 例 |
|-----------|---|------|------|-----|
| `V_max` | `int` | ✅ | 最大振動量子数 | `3` |
| `J_max` | `int` | ✅ | 最大回転量子数 | `5` |
| `use_M` | `bool` | ❌ | 磁気量子数を使用するか | `True` (デフォルト) |

#### 1.4 基本電場パラメータ

| パラメータ | 型 | 必須 | 単位 | 説明 | 例 |
|-----------|---|------|------|------|-----|
| `duration` | `float` | ✅ | fs | パルス幅（FWHM） | `20.0` |
| `t_center` | `float` | ✅ | fs | パルス中心時刻 | `0.0` |
| `carrier_freq` | `float` | ✅ | rad/fs | キャリア周波数 | `0.14847` |
| `amplitude` | `float` | ✅ | V/m | 電場振幅 | `1e9` |
| `polarization` | `list` | ✅ | - | 偏光ベクトル [x, y] | `[1.0, 0.0]` |

#### 1.5 基本物理パラメータ

| パラメータ | 型 | 必須 | 単位 | 説明 | 例 |
|-----------|---|------|------|------|-----|
| `omega_rad_phz` | `float` | ✅ | rad/fs | 振動固有周波数 | `0.14847` |
| `mu0_Cm` | `float` | ✅ | C·m | 双極子モーメント | `1e-30` |

#### 1.6 初期状態

| パラメータ | 型 | 必須 | 説明 | 例 |
|-----------|---|------|------|-----|
| `initial_states` | `list[int]` | ✅ | 初期状態のインデックス | `[0]` |

### 2. オプションパラメータ

#### 2.1 電場高度設定

| パラメータ | 型 | デフォルト | 単位 | 説明 | 例 |
|-----------|---|-----------|------|------|-----|
| `envelope_func` | `callable` | `gaussian_fwhm` | - | 包絡線関数 | `gaussian_fwhm` |
| `gdd` | `float` | `0.0` | fs² | 群遅延分散（2次） | `1000.0` |
| `tod` | `float` | `0.0` | fs³ | 群遅延分散（3次） | `50000.0` |
| `phase_rad` | `float` | `0.0` | rad | キャリア位相 | `np.pi/4` |

#### 2.2 正弦波変調

| パラメータ | 型 | デフォルト | 説明 | 例 |
|-----------|---|-----------|------|-----|
| `Sinusoidal_modulation` | `bool` | `False` | 正弦波変調を使用するか | `True` |
| `amplitude_sin_mod` | `float` | - | 変調振幅 | `0.1` |
| `carrier_freq_sin_mod` | `float` | - | 変調キャリア周波数 | `0.01` |
| `phase_rad_sin_mod` | `float` | `0.0` | 変調位相 | `np.pi/2` |
| `type_mod_sin_mod` | `str` | `"phase"` | 変調タイプ | `"phase"` or `"amplitude"` |

#### 2.3 ハミルトニアンパラメータ

| パラメータ | 型 | デフォルト | 単位 | 説明 | 例 |
|-----------|---|-----------|------|------|-----|
| `delta_omega_rad_phz` | `float` | `0.0` | rad/fs | 振動非調和性 | `0.001` |
| `B_rad_phz` | `float` | `0.0` | rad/fs | 回転定数 | `0.02` |
| `alpha_rad_phz` | `float` | `0.0` | rad/fs | 振動-回転相互作用定数 | `0.0001` |
| `potential_type` | `str` | `"harmonic"` | - | ポテンシャル形式 | `"harmonic"` or `"morse"` |

#### 2.4 双極子行列設定

| パラメータ | 型 | デフォルト | 説明 | 例 |
|-----------|---|-----------|------|-----|
| `backend` | `str` | `"numpy"` | 計算バックエンド | `"numpy"` or `"cupy"` |
| `dense` | `bool` | `True` | 密行列を使用するか | `False` |

#### 2.5 伝播設定

| パラメータ | 型 | デフォルト | 説明 | 例 |
|-----------|---|-----------|------|-----|
| `axes` | `str` | `"xy"` | 電場-双極子の軸対応 | `"xy"`, `"zx"` |
| `return_traj` | `bool` | `True` | 軌跡を返すか | `False` |
| `return_time_psi` | `bool` | `True` | 時間配列も返すか | `False` |
| `sample_stride` | `int` | `1` | サンプリング間隔 | `10` |

#### 2.6 出力設定

| パラメータ | 型 | デフォルト | 説明 | 例 |
|-----------|---|-----------|------|-----|
| `save` | `bool` | `True` | 結果を保存するか | `False` |
| `outdir` | `str` | 自動生成 | 出力ディレクトリ | `"/path/to/output"` |

### 3. スイープ制御パラメータ

#### 3.1 固定値キー (FIXED_VALUE_KEYS)

以下のキーは常に固定値として扱われます（リスト形式でもスイープ対象になりません）：

| キー | 説明 | 例 |
|-----|------|-----|
| `polarization` | 偏光ベクトル | `[1.0, 0.0]` |
| `initial_states` | 初期状態 | `[0, 5]` |
| `envelope_func` | 包絡線関数 | `gaussian_fwhm` |

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
omega_rad_phz = 2349 * 2 * np.pi * 3e10 / 1e15  # CO2 ν3 mode
mu0_Cm = 0.3 * 3.33564e-30                      # ~0.3 Debye

# 電場パラメータ
duration = 10.0
t_center = 0.0
carrier_freq = omega_rad_phz
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
omega_rad_phz = 2349 * 2 * np.pi * 3e10 / 1e15
mu0_Cm = 0.3 * 3.33564e-30
t_center = 0.0
carrier_freq = omega_rad_phz

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
from rovibrational_excitation.core.electric_field import voigt_fwhm

description = "advanced_simulation"

# 時間軸
t_start, t_end, dt = -100.0, 100.0, 0.05

# 量子系（大規模計算）
V_max, J_max = 5, 10
use_M = True

# 物理パラメータ（CO2分子）
omega_rad_phz = 2349 * 2 * np.pi * 3e10 / 1e15
delta_omega_rad_phz = 0.001 * omega_rad_phz    # 非調和性
B_rad_phz = 0.39 * 2 * np.pi * 3e10 / 1e15     # 回転定数
alpha_rad_phz = 0.0001 * B_rad_phz              # 振動-回転相互作用
mu0_Cm = 0.3 * 3.33564e-30
potential_type = "morse"

# 電場パラメータ（成形パルス）
envelope_func = voigt_fwhm
duration = 50.0
t_center = 0.0
carrier_freq = omega_rad_phz
amplitude = 5e9
polarization = [1/np.sqrt(2), 1j/np.sqrt(2)]   # 円偏光
phase_rad = np.pi/4
gdd = 1000.0                                    # 群遅延分散
tod = 50000.0                                   # 3次分散

# 正弦波変調
Sinusoidal_modulation = True
amplitude_sin_mod = 0.1
carrier_freq_sin_mod = 0.01
phase_rad_sin_mod = np.pi/2
type_mod_sin_mod = "phase"

# 初期状態（複数状態の重ね合わせ）
initial_states = [0, 1, 2]

# 計算設定
backend = "cupy"        # GPU計算
dense = False           # スパース行列
axes = "xy"
sample_stride = 5       # メモリ節約
```

## 単位系

### 時間
- **基本単位**: fs（フェムト秒）
- **変換**: THz = 2π × 1000 rad/fs

### 周波数
- **基本単位**: rad/fs
- **変換**: 
  - cm⁻¹ → rad/fs: `E_cm * 2π × 3e10 / 1e15`
  - THz → rad/fs: `E_THz * 2π * 1000`

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
dense = False           # スパース行列を使用
backend = "numpy"       # CPUで確実に動作
```

### 大規模計算

```python
# 大規模計算の設定
V_max, J_max = 10, 20   # 大きな基底
backend = "cupy"        # GPU加速
dense = True            # GPU計算では密行列が高速
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