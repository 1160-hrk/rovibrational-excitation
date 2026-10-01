# parameter sweep specification

通常 simulation の parameter file は、実行前に ordered Cartesian product へ展開されます。
この文書は開発版 `0.3.0.dev1` の実装を説明します。各 case の物理 key と unit は
[PARAMETER_REFERENCE.md](PARAMETER_REFERENCE.md) に従います。

## 最初に dry run する

```bash
rve-simulate my_params.py --dry-run --no-save
```

`--dry-run` は展開後の case 件数を報告し、propagation を実行しません。
`--no-save` も併記すると result root/case directory を作りません。保存を有効にした
dry run が directory を先に materialize する挙動は、通常実行と同じ path を確認するための
現行契約です。

## 判定順序

parameter file の各 public variable を insertion order で一度ずつ分類します。

1. `str` と `bytes` は固定値。
2. key が `_sweep` で終わる場合、nonempty sized iterable を明示 sweep とする。
3. `polarization` と `initial_states` は list でも固定値。
4. それ以外の nonempty sized iterable は sweep。
5. その他は固定値。

empty または長さを取得できない `_sweep` 値はエラーです。通常 key の empty iterable は
展開されず固定値のまま残りますが、ほとんどの物理 parameter では後段の strict
validation に失敗します。

## 明示 `_sweep` suffix

suffix は各 case で除かれます。

```python
amplitude_sweep = [1.0e8, 5.0e8]
phase_rad_sweep = [0.0, 0.5]
```

展開後の case key は `amplitude` と `phase_rad` です。元の suffix 付き key は case に
残りません。

list 自体が一つの物理値である key も、suffix を使えば明示 sweep にできます。

```python
polarization_sweep = [
    [1.0, 0.0],
    [0.0, 1.0],
]
initial_states_sweep = [
    [0],
    [1],
    [0, 1],
]
```

`initial_states=[0, 1]` は一つの coherent superposition ですが、上の
`initial_states_sweep` は3種類の初期状態 specification を比較します。

## suffix なし iterable

固定値 key 以外の list、tuple、NumPy array など、nonempty で長さを持つ iterable は
sweep dimension になります。

```python
duration = [10.0, 20.0]
amplitude = [1.0e8, 5.0e8]
```

この例は `2 * 2 = 4` cases です。明示性のため、特に list 自体が物理値になり得る場合は
`_sweep` suffix を推奨します。

### 1要素 iterable

1要素でも sweep dimension です。

```python
V_max = [3]
```

case 内では scalar `V_max=3` に正規化されます。case 数は増えませんが、`V_max` は
sweep key list、result path、declared-run provenance に残ります。「1要素だから固定」とは
扱いません。固定 scalar にしたい場合は `V_max=3` と書きます。

## 固定 list key

suffix のない次の2 key だけは list のまま固定されます。

- `polarization`: 一つの Jones vector
- `initial_states`: 一つの pure-state/coherent-superposition specification

```python
polarization = [1.0, 0.0]
initial_states = [0, 1]
```

これらを sweep したい場合は、前節の nested list と `_sweep` suffix を使います。

## Cartesian product と順序

sweep dimension は parameter file の insertion order を保持します。Cartesian product
では右端の dimension が最も速く変化します。

```python
amplitude = [1.0, 2.0]
phase_rad_sweep = [0.1, 0.2]
```

展開順は次の通りです。

```text
(amplitude=1.0, phase_rad=0.1)
(amplitude=1.0, phase_rad=0.2)
(amplitude=2.0, phase_rad=0.1)
(amplitude=2.0, phase_rad=0.2)
```

全 dimension は Cartesian product です。二つの list を element-wise に zip する mode は
ありません。値と unit の組を両方 sweep すると全組合せになるため、意図しない unit/value
pair を作らないよう注意してください。

## result path

保存時は sweep key order で nested directory を作ります。上の例では概念的に次の path
になります。

```text
amplitude_1/phase_rad_0.1/
amplitude_1/phase_rad_0.2/
amplitude_2/phase_rad_0.1/
amplitude_2/phase_rad_0.2/
```

各 case には scalar 化した値、`save`, `outdir` が渡されます。result payload と strict
loader は [RESULT_STORAGE.md](RESULT_STORAGE.md) を参照してください。

## checkpoint と resume

checkpoint schema v1 は、展開後の全 case をこの exact order のまま正規化して SHA-256
へ含めます。parameter、unit、dimension order、値、case membership のいずれかが変わると
別 run と判断し、resume 前にエラーになります。resume は同じ parameter file から全 case
を再構成・検証してから completed case を除外します。

`checkpoint_interval` は正の integer で、CLI option として指定します。

```bash
rve-simulate my_params.py --checkpoint-interval 5
rve-simulate --resume results/my_run --checkpoint-interval 5
```

checkpoint pair の保存形式と保証範囲は [RESULT_STORAGE.md](RESULT_STORAGE.md) を参照して
ください。

## error policy

次は暗黙修復せずエラーです。

- `_sweep` value が nonempty sized iterable でない
- 展開後に必須 scalar が list/array のまま残る
- suffix を除いた case key が model/schema に不適用
- 展開後 case の unknown key、欠落 unit、非対応 execution policy
- resume 時の ordered expanded-run hash 不一致

case 数・順序を確認するために private helper を import せず、CLI の
`--dry-run --no-save` を使ってください。
