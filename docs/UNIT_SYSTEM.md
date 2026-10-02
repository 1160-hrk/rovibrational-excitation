# 単位系と変換境界

この文書はバージョン `0.3.0` の公開入力、内部 canonical 単位、保存時の
provenance を説明する。通常シミュレーションで各キーが必要になる条件は
[PARAMETER_REFERENCE.md](PARAMETER_REFERENCE.md)、時間格子の意味は
[TIME_PROPAGATION.md](TIME_PROPAGATION.md) を参照する。

## 1. 基本方針

物理量の境界は次の順序を守る。

```text
caller value + required unit
  -> frozen validation boundary
  -> documented canonical unitへ1回だけ変換
  -> unit文字列を見ない数値kernel
```

- normal simulation の scalar physical input は value/unit pair を要求する。
- `phase_rad`、`field_dt_fs` のように名前自体が単位を固定する内部・専用引数は、
  その名前が unit contract である。
- 個数、量子数、enum、bool、規格化済み Jones vector などの無次元値に unit は付けない。
- 未知の unit、unit の欠落、boolean、array を scalar として渡すこと、非有限値は
  allocation または伝播前にエラーにする。
- unit を推測せず、未知の表記を値の変更なしで通す fallback も行わない。

## 2. 何を保存するか

normal simulation の入力 mapping と結果の parameter JSON には、caller が指定した
値と unit label を変更せず保存する。再現性のための provenance はこの組が権威である。

計算側では同じ mapping から frozen schema を作り、必要な canonical 値を1回だけ
生成する。schema によっては元の `value`/`unit` も保持するが、数値 kernel が読むのは
`angular_rad_per_fs`、`femtoseconds`、`dipole_c_m` のように単位を名前に含む属性で
ある。元の mapping を canonical 値で上書きしないため、stale unit label との二重変換を
防ぐ。

直接構築する `ScalarField` と `CartesianField` は、引数名
`samples_v_per_m` / `*_component_v_per_m` が canonical unit を固定する。この境界へ
別単位の配列を渡してはいけない。変換が必要なら、unit を受け取る field/config
boundary で先に変換する。

## 3. Canonical 単位一覧

| 量 | 入力境界の代表 | 内部 canonical |
|---|---|---|
| 時間 | value + time unit | `fs` |
| 普通周波数・波数・角周波数 | value + frequency unit | `rad/fs` |
| Hamiltonian energy | energy/frequency unit | `J` または明示的な `rad/fs` view |
| 双極子 moment | value + dipole unit | `C*m` |
| 電場 sample / peak amplitude | direct field unit、限定された境界では intensity | `V/m` |
| GDD | value + GDD unit | `fs^2` |
| TOD | value + TOD unit | `fs^3` |
| local-control gain | value + gain unit | `(V/m)^2 fs` |
| standard-Krotov penalty | value + penalty unit | `1 / ((V/m)^2 fs)` |
| spectroscopy temperature | value + exact `K` label | `K` |
| spectroscopy pressure | value + exact `Pa` label | `Pa` |
| spectroscopy optical length | value + exact `m` label | `m` |
| spectroscopy coherence time | value + exact `ps` label | `ps` |
| spectroscopy molecular mass | value + exact `kg` label | kg per molecule |
| spectroscopy grid/resolution | array/scalar + exact `cm^-1` label | formulasの固定波数境界 |

伝播準備は Hamiltonian を `J`、dipole を `C*m`、field を `V/m`、time を `fs`
として構造検証した後、既存の dimensional または nondimensional 数値表現へ投影する。

## 4. 対応 unit spelling

unit spelling は case-sensitive である。ここにない同義語を推測しない。

### Frequency

| 表現 | 対応 unit | 入力値に $2\pi$ を含むか |
|---|---|---|
| ordinary frequency | `Hz`, `kHz`, `MHz`, `GHz`, `THz`, `PHz` | 含まない |
| wavenumber | `cm^-1`, `cm-1`, `wavenumber` | 含まない |
| angular frequency | `rad/s`, `rad/ps`, `rad/fs` | 含む |

### Energy

`J`, `eV`, `meV`, `keV`, `Ry`, `Ha`, `kJ/mol`, `kcal/mol` に加え、
上の frequency/wavenumber 表現のうち `rad/s`, `GHz`, `MHz`, `kHz`, `Hz` を
除く `rad/fs`, `rad/ps`, `PHz`, `THz`, `cm^-1`, `cm-1`, `wavenumber` を
Hamiltonian energy boundary が受け付ける。

### Time and dispersion

- time: `fs`, `ps`, `ns`, `μs`, `us`, `ms`, `s`, `atomic`
- GDD: `fs^2`, `ps^2`, `ns^2`, `μs^2`, `us^2`, `ms^2`, `s^2`
- TOD: `fs^3`, `ps^3`, `ns^3`, `μs^3`, `us^3`, `ms^3`, `s^3`

GDD/TOD はそれぞれ value と unit を同時に渡す。両方を省略した場合だけ exact zero
modifier であり、片方だけの指定はエラーになる。

### Dipole

`C*m`, `C·m`, `Cm`, `D`, `Debye`, `ea0`, `e*a0`, `atomic`,
`rad/fs/(V/m)`, `rad*PHz/(V/m)` を受け付け、`C*m` へ変換する。

### Direct electric-field amplitude

`V/m`, `V/nm`, `V/Å`, `V/A`, `kV/m`, `kV/cm`, `MV/m`, `MV/cm`,
`GV/m`, `TV/m`, `atomic` を受け付け、`V/m` へ変換する。

### Cycle-averaged intensity

`W/cm^2`, `W/cm2`, `W/m^2`, `W/m2`, `MW/cm^2`, `MW/cm2`,
`GW/cm^2`, `GW/cm2`, `TW/cm^2`, `TW/cm2` を受け付ける境界では、

$$
E_{\mathrm{peak}}=\sqrt{2I\mu_0c}
$$

により peak `V/m` へ変換する。cm² の prefix conversion もこの境界で行う。

## 5. 周波数と $2\pi$

ordinary frequency $\nu$、wavenumber $\tilde\nu$、angular frequency $\omega$ は

$$
\omega=2\pi\nu=2\pi c\tilde\nu
$$

で結ばれる。`THz`、`PHz`、`cm^-1` の入力値へ $2\pi$ を入れない。
`rad/fs` などの angular unit を選んだ場合だけ値自体に $2\pi$ が含まれる。

次の3つは同じ周波数を表す。

```python
Frequency(100.0, "THz")
Frequency(0.1, "PHz")
Frequency(2 * np.pi * 0.1, "rad/fs")
```

`Frequency` は caller の `value` と `unit` を保持し、数値消費者には
`angular_rad_per_fs` を渡す。FFT consumer は明示的に `cycles_per_fs` view を使う。
carrier phase は angular frequency、FFT bin は ordinary frequency を使うため、同じ
canonical object からそれぞれを得て二重に $2\pi$ を掛けない。

TwoLevel の `energy_gap` は energy unit または frequency unit を受け付ける。
LinMol、VibLadder、SymTop の vibrational/rotational/anharmonic parameter は
frequency unit を要求する。

## 6. 電場入力ごとの unit 制約

| 入力経路 | intensity label | 規則 |
|---|---|---|
| normal generated pulse `amplitude/amplitude_units` | 許可 | cycle-averaged intensity または direct peak amplitudeを `V/m` へ変換 |
| low-level `ElectricField.add_dispersed_Efield` | 不可 | `amplitude_units` は direct amplitude のみ |
| arbitrary signed `ElectricField.add_arbitrary_Efield` | 不可 | `field_units` は direct amplitude、shape は既存 field と一致 |
| typed `ScalarField` / `CartesianField` | 不可 | constructor argument はすでに canonical `V/m` |
| GRAPE / legacy sampled field | 不可 | direct unit、実数、有限、exact grid match |
| standard Krotov sampled/generated control | 不可 | direct unit、interval count/grid を変換しない |
| local seed field | 不可 | direct unit、正の amplitude、正の segment count |

符号付き sample から cycle-averaged intensity を一意に復元できないため、array 入力へ
intensity label を認めない。入力を absolute value 化、clip、resample して修復しない。

## 7. Model と operator

normal model は次を value/unit pair として要求する。

- 全 model: `dipole_scale/dipole_scale_units`
- TwoLevel: `energy_gap/energy_gap_units`
- VibLadder: `vibrational_frequency` と `anharmonic_shift` の各 unit
- LinMol: VibLadder の2組に `rotational_constant` と
  `vibration_rotation_coupling` の各 unitを追加
- SymTop: vibrational/anharmonic、parallel/perpendicular の rotational constant
  と vibration-rotation coupling の全 unit

Morse potential は canonical anharmonic shift が zero ならエラーになる。Morse level
count は変換済み vibrational frequency と anharmonic shift から model instance ごとに
導出し、固定値や global state を使わない。

`Hamiltonian` object は行列と現在の `J` または `rad/fs` label を保持し、変換には
権威ある `CONSTANTS.HBAR` を使う。dipole object は SI accessor
`get_mu_*_SI()` を提供する。伝播 validator は raw attribute や unit-less array へ
fallback しない。

## 8. Optimization

local control の `gain` は正の field-squared-time であり、`gain_units` が必須である。

- `(V/m)^2 fs`
- `(MV/m)^2 fs`
- `(GV/m)^2 fs`
- `(TV/m)^2 fs`

standard Krotov の `lambda_a` は正の inverse-field-squared-time penalty であり、
`lambda_a_units` が必須である。

- `1 / ((V/m)^2 fs)`
- `1 / ((MV/m)^2 fs)`
- `1 / ((GV/m)^2 fs)`
- `1 / ((TV/m)^2 fs)`

gain が大きいほど local update の電場変更を強める一方、standard-Krotov penalty が
大きいほど更新を抑える。同じ数値を相互に流用してはいけない。

GRAPE と `legacy_batch_overlap` の `lambda_a`、`learning_rate`、収束 tolerance は、
離散 objective 固有の normalization を持つ Class-D quantity であり、standard Krotov
の penalty unit を割り当てない。local の `c_abs_min`、`drive_abs_min`、
`shape_floor` も物理次元が未確定である。これらを推論で改名・変換しない。

`total_fs`、`field_dt_fs`、`control_dt_fs`、`segment_size_fs`、`times_fs`、
`field_max_v_per_m` は名前に unit を固定した optimizer 専用境界である。特に local
optimizer の奇数長 grid、端点、index、slice、RK4 prefix は unit 整理のために変更しない。

## 9. Spectroscopy

`ExperimentalConditions` は以下の値/unit pair を全て要求し、現状は表記変換をせず
exact label のみ受け付ける。

- `temperature/temperature_units="K"`
- `pressure/pressure_units="Pa"`
- `optical_length/optical_length_units="m"`
- `coherence_time/coherence_time_units="ps"`
- `molecular_mass/molecular_mass_units="kg"`（1 molecule あたり）

公開 spectrum grid は `wavenumber_units="cm^-1"` が必須である。device function を
有効にする場合だけ、正の `device_resolution` と
`device_resolution_units="cm^-1"` を組で要求する。無効時にこの組を渡すことも
エラーになる。

## 10. エラーと変換の禁止事項

- unit の欠落、unknown spelling、物理量と不整合な unit はエラー。
- caller mapping の値と unit label は書き換えない。
- 1つの値を複数 boundary で再変換しない。
- signed field に intensity を使わない。
- ordinary frequency と angular frequency を名前だけで取り違えない。
- 非対応 unit から canonical 値を推測しない。
- user data を規格化、対称化、clip、resample して unit error を隠さない。
- backend や algorithm を変更して unit incompatibility を回避しない。

実装がこの文書と食い違う場合は、値・式を推測して合わせず、
[PHYSICS_CONTRACTS.md](refactoring/PHYSICS_CONTRACTS.md) と
[DECISIONS.md](refactoring/DECISIONS.md) を確認する。
