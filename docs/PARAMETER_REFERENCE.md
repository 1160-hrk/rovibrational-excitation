# 通常シミュレーション parameter reference

対象はバージョン `0.3.0` の normal simulation です。最適化 YAML は
[configs/README.md](../configs/README.md)、保存 schema は
[RESULT_STORAGE.md](RESULT_STORAGE.md) を参照してください。v0.2 の parameter 名や
暗黙 default は移行しません。unknown key と model/algorithm に不適用な key は
propagation 前にエラーになります。

## 信頼できる開始点

完全な実行例は [`examples/params_template.py`](../examples/params_template.py) です。
この template は CI で保存なしの end-to-end 実行を行います。

```bash
cp examples/params_template.py my_params.py
python -m rovibrational_excitation.cli.simulate my_params.py --dry-run
python -m rovibrational_excitation.cli.simulate my_params.py --no-save
```

parameter file は Python module として実行されます。信頼できない file を読み込ませないで
ください。CLI は module の public variable を読み、値と単位 label を変更せず各 case へ
渡します。

## 二つの電場入力経路

`run_simulation_case(params, field=...)` の `field` keyword は必須です。

1. `field=None`: parameter から pulse と canonical `TimeGrid` を生成する。
2. `field=ScalarField(...)` または `field=CartesianField(...)`: Python で用意した
   sampled field を注入する。

二つの key 集合は混在できません。外部 field 経路に生成 pulse key があればエラーです。
どちらの経路も補間、padding、切り詰め、規格化、時間刻みの自動変更を行いません。

## 全経路で必要な選択

| key | 型 | 契約 |
|---|---|---|
| `basis_type` | `str` | `twolevel`, `vibladder`, `linmol`, `symtop` |
| `initial_states` | `list[int]` | 空でない basis index list。暗黙の基底状態なし |
| `backend` | `str` | `numpy` または `cupy` |
| `storage` | `str` | `dense` または `csr` |
| `algorithm` | `str` | `rk4` または `split_operator` |
| `return_traj` | `bool` | trajectory 全体か final state だけか |
| `sample_stride` | positive `int` | trajectory 出力 stride。伝播刻みは変えない |
| `nondimensional` | `bool` | 明示的な nondimensional propagation 選択 |
| `renorm` | `bool` | step ごとの wavefunction 規格化を選択 |

CuPy と CSR の組合せは非対応です。density/Liouville は NumPy dense RK4 のみです。
SymTop は NumPy RK4 のみです。CuPy RK4 と split のdevice-native経路は、
実 CUDA 数値受入れでparity・backend identity・転送量・同期済み時間を確認済みです。
小規模診断の時間は速度保証ではありません。`backend="cupy"` を要求して利用不能な
場合、NumPyへfallbackせずエラーになります。

### optional workflow key

| key | 型・default | 意味 |
|---|---|---|
| `description` | `str`, file name または `run` | batch result directory の label |
| `save` | `bool`, CLI が選択 | result を保存するか |
| `outdir` | path | direct Python call で保存する場合の case directory。batch CLI は生成する |
| `validate_units` | `bool`, `True` | solver 内部の dimension check。公開入力の必須 unit pair を省略可能にはしない |
| `verbose` | `bool`, `False` | solver/scaling diagnostic output |

## model 別の必須 parameter

全ての物理 scalar は値と専用 unit key の組で指定します。unit label は caller の値と
ともに保存され、model boundary で canonical 値へ一度だけ変換されます。

### TwoLevel

| key | 内容 |
|---|---|
| `energy_gap`, `energy_gap_units` | 二準位 energy gap。frequency または energy unit |
| `dipole_scale`, `dipole_scale_units` | 遷移双極子 scale |

TwoLevel は scalar coupling です。`representation`, `axes`, `polarization` は指定できません。

### VibLadder

| key | 内容 |
|---|---|
| `V_max` | 最大振動量子数。non-negative integer |
| `vibrational_frequency`, `vibrational_frequency_units` | 0→1 遷移 frequency |
| `anharmonic_shift`, `anharmonic_shift_units` | 隣接遷移 frequency の準位ごとの減少量 |
| `dipole_scale`, `dipole_scale_units` | 双極子 scale |
| `potential_type` | `harmonic` または `morse` |

VibLadder は scalar coupling です。Morse では `anharmonic_shift` が非ゼロでなければ
エラーです。最大 bound level は model instance の frequency と anharmonicity から
導出され、`V_max` が範囲外ならエラーです。

### LinMol

| key | 内容 |
|---|---|
| `representation` | `m_resolved` または `m_incoherent_average` |
| `V_max`, `J_max` | 最大振動・回転量子数 |
| `vibrational_frequency`, `vibrational_frequency_units` | 0→1 遷移 frequency |
| `anharmonic_shift`, `anharmonic_shift_units` | 非調和 shift |
| `rotational_constant`, `rotational_constant_units` | 回転定数 |
| `vibration_rotation_coupling`, `vibration_rotation_coupling_units` | vibration-rotation coupling |
| `dipole_scale`, `dipole_scale_units` | 双極子 scale |
| `potential_type` | `harmonic` または `morse` |

`m_resolved` では ordered Cartesian coupling を表す `axes` が必須です。2文字は
`x`, `y`, `z` から異なるものを選び、field column 0/1 に順に対応します。

`m_incoherent_average` では `axes` を指定できません。固定直線偏光を内部 z 軸へ合わせ、
初期 J の各 M block を別々に伝播し、規格化 weight `1/(2J+1)` で population を
非干渉和します。複数 `initial_states` は同じ J 内の異なる v に限られます。

Morse の非ゼロ非調和性と bound-level 制限は VibLadder と同じです。

### SymTop

| key | 内容 |
|---|---|
| `molecule` | 現在の production preset は `CH3F` |
| `nuclear_spin_isomer` | `ortho` または `para` のどちらか一つ |
| `V_max`, `J_max` | 最大振動・回転量子数 |
| `vibrational_frequency`, `vibrational_frequency_units` | 0→1 遷移 frequency |
| `anharmonic_shift`, `anharmonic_shift_units` | 非調和 shift |
| `rotational_constant_perpendicular`, `rotational_constant_perpendicular_units` | perpendicular 回転定数 |
| `rotational_constant_parallel`, `rotational_constant_parallel_units` | parallel 回転定数 |
| `vibration_rotation_coupling_perpendicular`, `vibration_rotation_coupling_perpendicular_units` | perpendicular vibration-rotation coupling |
| `vibration_rotation_coupling_parallel`, `vibration_rotation_coupling_parallel_units` | parallel vibration-rotation coupling |
| `dipole_scale`, `dipole_scale_units` | 双極子 scale |
| `potential_type` | `harmonic` または `morse` |
| `axes` | ordered Cartesian coupling axes |

SymTop は signed `|v,J,K,M>` order の rigid parallel band、Delta K=0、Cartesian
Delta M=0,+/-1 です。`molecule` と核 spin sector が許す level だけを構築します。
all-isomer pure state、split operator、CuPy、optimization は非対応です。

## generated-field 経路

`field=None` の場合、次の全 key が必須です。

| value key | unit/selector key | 契約 |
|---|---|---|
| `t_start` | `t_start_units` | field grid の開始時刻 |
| `t_end` | `t_end_units` | field grid の終了時刻 |
| `dt` | `dt_units` | field sampling interval |
| `duration` | `duration_units` | positive envelope width |
| `t_center` | `t_center_units` | pulse center |
| `carrier_frequency` | `carrier_frequency_units` | carrier frequency |
| `amplitude` | `amplitude_units` | field amplitude |
| `envelope_kind` | — | envelope selector |
| `modulation_kind` | — | `none` または `sinusoidal` |

受理する generated envelope は次の4個です。

| `envelope_kind` | `duration` の意味 |
|---|---|
| `gaussian` | standard deviation |
| `gaussian_fwhm` | full width at half maximum |
| `lorentzian` | half width at half maximum |
| `lorentzian_fwhm` | full width at half maximum |

Voigt、custom callable、任意 waveform は sampled field として注入します。

### 時間格子

全 time value に unit が必要です。内部では fs へ一度だけ変換します。

- `dt` は field sampling interval。
- propagation step は厳密に `2 * dt`。
- `t_end - t_start` は `2 * dt` の正の整数倍。
- field sample 数は奇数で `2 * n_steps + 1`。
- grid は有限、単調増加、等間隔、両 endpoint を含む。

格子・刻みの自動選択はありません。精度は caller が別の coarse/fine grid と目的の
observable を選び、明示的に convergence を評価します。

### frequency と 2 pi

`carrier_frequency_units` と model frequency unit には ordinary frequency、波数、
angular frequency を選べます。例えば `Hz`–`PHz` と `cm^-1` は 2 pi を含まない
入力、`rad/s`, `rad/ps`, `rad/fs` は 2 pi を含む angular frequency です。内部では
`rad/fs` へ一度だけ変換します。

### polarization

- generated `m_resolved` LinMol と SymTop: finite nonzero 2-component
  `polarization` が必須。
- TwoLevel/VibLadder: scalar model のため `polarization` は不適用。
- LinMol M average: 省略可能。指定する場合は固定直線偏光だけを受理し、方向を内部
  z 軸へ合わせる。円/楕円/時間依存偏光は不適用。

polarization は物理的な field 値を規格化するものではなく、Jones direction を
規格化して使用します。

### optional phase と dispersion

| key | default/条件 |
|---|---|
| `phase_rad` | `0.0` |
| `gdd`, `gdd_units` | pair 全体を省略すると厳密な zero。片方だけはエラー |
| `tod`, `tod_units` | pair 全体を省略すると厳密な zero。片方だけはエラー |

### sinusoidal modulation

`modulation_kind="none"` のとき modulation 用 key は併記できません。
`modulation_kind="sinusoidal"` では次が必要です。

| key | 契約 |
|---|---|
| `modulation_depth` | finite scalar。amplitude mode は `0 <= depth <= 1` |
| `modulation_delay`, `modulation_delay_units` | time delay pair |
| `modulation_mode` | `phase` または `amplitude` |
| `modulation_phase_rad` | optional、default `0.0` |

## external sampled-field 経路

Python API の完全な例は
[`example_external_scalar_field.py`](../examples/example_external_scalar_field.py) です。

- TwoLevel、VibLadder、LinMol M average は `ScalarField`。
- LinMol M-resolved と SymTop は `CartesianField`。
- sample は finite real で、grid と同じ length。
- signed field sample の unit は direct electric-field amplitude。intensity label は不可。
- constructor は defensive read-only copy を作る。
- generated-field key は params に含めない。

Cartesian field から Jones polarization や scalar waveform を推測しません。
LinMol の `split_interaction="helicity_projected"` を選ぶ場合だけ、field construction 時に
対応する normalized Jones ket と scalar sample を明示する必要があります。厳密
Cartesian RK4/split にはこの追加 metadata は不要です。

## split operator

LinMol `m_resolved` で `algorithm="split_operator"` を選ぶ場合、
`split_interaction` が必須です。

- `cartesian`: 実 Cartesian field から Hermitian interaction を作る厳密 route。
- `helicity_projected`: 片方向 transition operator と随伴を使う明示的近似。

RK4、scalar model、LinMol M average、SymTop へ `split_interaction` を指定すると
エラーです。CSR input を受けても interaction eigendecomposition は dense なので、
split operator の `storage="csr"` は完全な sparse-memory scaling を意味しません。

## initial state の意味

`initial_states=[i]` は basis index `i` の pure state です。複数 index は等振幅・
同位相の coherent superposition を作って規格化します。classical/incoherent mixture を
表しません。incoherent mixture は `MixedStatePropagator` と statistical weight を使います。

## 保存と return shape

- `return_traj=True`: `sample_stride` ごとの状態に加え、割り切れない場合も exact endpoint
  を必ず含む。
- `return_traj=False`: final time/state だけを返す。
- normal Python API の return は population array。
- `save=True` の result は versioned generation と manifest を通して公開。

保存済み array key、strict loader、checkpoint/resume は
[RESULT_STORAGE.md](RESULT_STORAGE.md) を参照してください。direct Python call で
`save=True` を使う場合は case `outdir` を明示してください。batch CLI は case ごとに
生成します。

## parameter sweep

parameter file の iterable 展開と `_sweep` suffix は
[SWEEP_SPECIFICATION.md](SWEEP_SPECIFICATION.md) が所有します。`polarization` と
`initial_states` は list でも固定値です。まず `--dry-run` で case 数と順序を確認して
ください。展開順序は checkpoint の declared-run provenance に含まれます。

## failure policy

次は警告付き継続ではなく、propagation 前のエラーです。

- 必須 key または unit pair の欠落
- unknown key、別 model の key、入力 route に不適用な key
- non-finite scalar、zero polarization、不正な integer/range
- unsupported backend/storage/algorithm combination
- Morse の zero anharmonicity または bound-level 超過
- external field/grid shape 不一致
- split interaction selector の欠落・不適用

library は user input を暗黙 clip、renormalize、symmetrize、resample、修復しません。
