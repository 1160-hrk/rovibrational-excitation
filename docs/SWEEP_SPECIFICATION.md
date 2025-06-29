# スイープ仕様 (Sweep Specification)

## 概要

rovibrational-excitation パッケージでは、パラメータスイープを自動的に行うことができます。新しい仕様では、より明確で直感的なスイープ制御が可能になりました。

## スイープ判定ルール

パラメータがスイープ対象になるかどうかは、以下の優先順位で判定されます：

### 1. `_sweep` 接尾辞 → 明示的にスイープ対象

キー名が `_sweep` で終わる場合、明示的にスイープ対象として扱われます。**展開時には接尾辞が取り除かれたキー名で保存されます。**

```python
# 明示的なスイープ指定
amplitude_sweep = [1e8, 5e8, 1e9]          # 3ケース → 'amplitude' として保存
polarization_sweep = [[1, 0], [0, 1]]      # 2ケース → 'polarization' として保存
duration_sweep = [20.0, 30.0, 40.0, 50.0] # 4ケース → 'duration' として保存

# 合計: 3 × 2 × 4 = 24ケース
# 各ケースでは 'amplitude', 'polarization', 'duration' キーでアクセス可能
```

### 2. 固定値キー (FIXED_VALUE_KEYS) → 常に固定値

特定のキーは、リスト形式であっても常に固定値として扱われます：

```python
# これらのキーは常に固定値
polarization = [1.0, 0.0]      # x偏光（1ケースのみ）
initial_states = [0, 5]        # 初期状態（1ケースのみ）
envelope_func = gaussian_fwhm   # 包絡線関数（1ケースのみ）
```

現在の固定値キー一覧：
- `polarization` - 偏光ベクトル
- `initial_states` - 初期状態
- `envelope_func` - 包絡線関数

### 3. 通常のリスト → 長さで判定（従来通り）

上記に該当しない場合、リストの長さで判定されます：

```python
# 長さ > 1 の場合はスイープ対象
V_max = [3, 5, 7]              # 3ケース（スイープ）
amplitude = [0.1, 0.2]         # 2ケース（スイープ）

# 長さ = 1 の場合は固定値
J_max = [2]                    # 1ケース（固定値）
```

## 使用例

### 基本的な使用例

```python
# パラメータファイル例
import numpy as np

description = "sweep_example"

# 固定パラメータ
V_max, J_max = 5, 3
t_start, t_end, dt = -100.0, 100.0, 0.1
t_center = 0.0
omega_rad_phz = 1.0
mu0_Cm = 1e-30

# スイープパラメータ
duration = [20.0, 30.0, 40.0]              # 3ケース
amplitude_sweep = [1e8, 5e8, 1e9]          # 3ケース

# 固定値（FIXED_VALUE_KEYS）
polarization = [1.0, 0.0]                  # x偏光（固定）

# 合計: 3 × 3 = 9ケース
```

### 偏光をスイープしたい場合

```python
# 偏光もスイープしたい場合は _sweep 接尾辞を使用
polarization_sweep = [
    [1, 0],                    # x偏光
    [0, 1],                    # y偏光  
    [1/np.sqrt(2), 1j/np.sqrt(2)]  # 円偏光
]                              # 3ケース

duration = [20.0, 30.0]        # 2ケース

# 合計: 3 × 2 = 6ケース
```

### 複雑なスイープ例

```python
# 複数パラメータの組み合わせ
V_max = [3, 5]                 # 2ケース
duration_sweep = [20, 30, 40]  # 3ケース  
amplitude_sweep = [1e8, 1e9]   # 2ケース

# 固定値
polarization = [1.0, 0.0]      # 固定（x偏光）
J_max = 2                      # 固定
initial_states = [0]           # 固定（基底状態）

# 合計: 2 × 3 × 2 = 12ケース
```

## エラーケース

### `_sweep` 接尾辞でiterableでない場合

```python
# エラーになる例
amplitude_sweep = 1e8  # 非iterable

# ValueError: Parameter 'amplitude_sweep' has '_sweep' suffix but is not iterable
```

## 後方互換性

既存のパラメータファイルは基本的にそのまま動作しますが、一部変更が必要な場合があります：

### 変更が必要なケース

```python
# 旧仕様（問題あり）
polarization = [1.0, 0.0]  # 2ケース生成されていた

# 新仕様（修正済み）
polarization = [1.0, 0.0]  # 1ケース（固定値）

# または明示的にスイープしたい場合
polarization_sweep = [[1, 0], [0, 1]]  # 2ケース
```

## 推奨事項

1. **明示性を重視**: スイープしたいパラメータには `_sweep` 接尾辞を使用
2. **固定値の確認**: `polarization` などの特別なキーは固定値として扱われることを理解
3. **テスト実行**: パラメータファイルをテストして期待するケース数が生成されるか確認

```python
# ケース数確認例
from rovibrational_excitation.simulation.runner import _expand_cases, _load_params_file

params = _load_params_file("your_params.py")
cases = list(_expand_cases(params))
print(f"Total cases: {len(cases)}")
```

## マイグレーションガイド

既存のパラメータファイルを新仕様に移行する手順：

1. `polarization` がリストの場合、意図を確認
   - 固定値の場合：そのまま（新仕様では自動的に固定値扱い）
   - スイープの場合：`polarization_sweep` に変更

2. その他のスイープパラメータを明示的にする（オプション）
   ```python
   # 旧仕様
   amplitude = [1e8, 5e8, 1e9]
   
   # 新仕様（推奨）
   amplitude_sweep = [1e8, 5e8, 1e9]
   ```

3. テスト実行で期待するケース数が生成されることを確認 