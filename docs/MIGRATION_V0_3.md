# v0.2 から v0.3 への移行

対象は正式版 `0.3.0` の API と設定です。v0.3 は
v0.2 と後方互換ではなく、互換 shim もありません。旧 script や YAML をそのまま
実行して偶然通ることを移行とは見なしません。物理値と意図を確認しながら、現行の
型・単位・時間格子へ明示的に写してください。

この文書は計算式を変更する手順ではありません。固定された式と符号は
[物理契約](refactoring/PHYSICS_CONTRACTS.md)、現在の完全な normal-simulation
schema は [parameter reference](PARAMETER_REFERENCE.md)、単位は
[unit system](UNIT_SYSTEM.md) が権威です。

## 推奨する移行順序

1. v0.2 の環境、入力、結果を読み取り専用で保存する。
2. v0.3 用に新しい環境と新しい出力ディレクトリを作る。
3. [`examples/params_template.py`](../examples/params_template.py) または
   [`configs/`](../configs/) の現行例を複製する。archive を土台にしない。
4. 旧入力から物理値を一つずつ移し、各値に対応する単位を明記する。単位を推測しない。
5. normal simulation は `rve-simulate PARAMFILE --dry-run` で case 数と順序を確認する。
6. 小さい系で実行し、population、norm、目的 observable を旧計算と比較する。
7. 精度が重要なら caller が coarse/fine grid を用意し、明示的な convergence 比較を行う。
8. v0.3 の新規 run として保存する。旧 result/checkpoint を新形式として上書きしない。

## import の変更

ルートは次の 8 名だけを公開します。

```text
__version__, ElectricField, TimeGrid, ExecutionPolicy,
PropagationProblem, PropagationOptions, PropagationResult,
run_simulation_case
```

旧 root convenience 名は owning subpackage へ移動しました。

| v0.2 の import | v0.3 の import / 対応 |
|---|---|
| root `LinMolBasis` | `rovibrational_excitation.models.linear_molecule.LinMolBasis` |
| root `LinMolDipoleMatrix` | `rovibrational_excitation.models.linear_molecule.LinMolDipoleMatrix` |
| root `Hamiltonian` | `rovibrational_excitation.core.operators.Hamiltonian` |
| root `StateVector` / `DensityMatrix` | typed propagation では `core.states.PureState`, `DensityState`, `IncoherentEnsemble` を使用 |
| root `AbsorbanceCalculator` | `rovibrational_excitation.spectroscopy.AbsorbanceCalculator` |
| root `ExperimentalConditions` | `rovibrational_excitation.spectroscopy.ExperimentalConditions` |
| root `create_calculator_from_params` | `rovibrational_excitation.spectroscopy.create_calculator_from_params` |
| removed `dipole.*` model packages | `models.linear_molecule`, `models.vib_ladder`, `models.two_level`, `models.symmetric_top` |
| removed `plots` package | owning module under `rovibrational_excitation.visualization` |
| procedural `generate_H0` / `schrodinger_propagation` | model builderと typed `PropagationProblem`、または `run_simulation_case`。一対一 shim はない |

Normal simulation を行うだけなら、まず root の
`run_simulation_case` と現行 parameter schema を使うのが最短です。低レベル伝播では
`PureState`、`DensityState`、`IncoherentEnsemble` は caller の値を暗黙に規格化・
対称化・clip しません。

## normal simulation parameter の変更

`run_simulation_case(params, field=...)` の `field` keyword は必須です。

- `field=None`: parameter から generated field を作る。
- `field=ScalarField(...)` または `field=CartesianField(...)`: caller の sample を使う。

この二経路は混在できません。外部 sample は補間、padding、切り詰め、規格化されません。

### model と物理量

| 旧 key | 現行 key |
|---|---|
| `use_M=True` | LinMol `representation="m_resolved"` |
| `use_M=False` | LinMol `representation="m_incoherent_average"` |
| `omega_rad_phz`, `vibrational_frequency_rad_per_fs` | `vibrational_frequency` + `vibrational_frequency_units` |
| `delta_omega_rad_phz`, `anharmonicity_correction_rad_per_fs` | `anharmonic_shift` + `anharmonic_shift_units` |
| `B_rad_phz`, `rotational_constant_rad_per_fs` | `rotational_constant` + `rotational_constant_units` |
| `alpha_rad_phz`, `vibration_rotation_coupling_rad_per_fs` | `vibration_rotation_coupling` + `vibration_rotation_coupling_units` |
| `mu0_Cm` | `dipole_scale` + `dipole_scale_units` |

全ての物理 scalar は値/unit の pair が必要です。frequency label は意味を持ちます。
`Hz`、`PHz`、`cm^-1` は ordinary frequency/波数で 2π を含まず、`rad/s`、
`rad/ps`、`rad/fs` は angular frequency です。数値を変えず label だけ置換しては
いけません。

TwoLevel と VibLadder は scalar coupling なので `axes` と `polarization` を受け付けません。
LinMol `m_resolved` と SymTop は ordered Cartesian `axes` が必要です。LinMol
`m_incoherent_average` は axes を受け取らず、固定直線偏光を内部 z 軸へ合わせて M block を
別々に伝播し、規格化 weight で population を非干渉和します。

Morse potential は非ゼロ `anharmonic_shift` が必要です。許容最大準位は各 model instance の
frequency と anharmonicity から導出され、固定 `N=200` や global state はありません。

### field と時間

| 旧 key / 挙動 | 現行 key / 挙動 |
|---|---|
| `envelope_func` | `envelope_kind`、または custom waveform を sampled field として注入 |
| `Sinusoidal_modulation` | `modulation_kind` |
| `carrier_freq` | `carrier_frequency` + `carrier_frequency_units` |
| `pulse_duration` | `duration` + `duration_units` |
| `amplitude_sin_mod`, `carrier_freq_sin_mod`, `phase_rad_sin_mod`, `type_mod_sin_mod` | 全て削除。下記の現行 modulation 式から明示的に再設定 |
| boolean `dense` / `sparse` | required `storage="dense"` / `storage="csr"` |
| `auto_timestep`, `target_accuracy` | 削除。正確な grid と別々の convergence 比較を caller が指定 |

Generated field は `t_start`, `t_end`, `dt`, `duration`, `t_center`,
`carrier_frequency`, `amplitude` の各値/unit pair、`envelope_kind`,
`modulation_kind` を必要とします。Sinusoidal modulation は
`theta(f) = 2*pi*tau*(f-f0) + phi0` の `modulation_depth`,
`modulation_delay/modulation_delay_units`, `modulation_phase_rad`,
`modulation_mode` を明示します。旧 `carrier_freq_sin_mod` の数値を delay として流用する
一対一変換はありません。`dt` は field sampling interval であり、伝播刻みは厳密に
`2 * dt`、sample 数は `2 * n_steps + 1` の奇数です。

実行選択も暗黙 default に頼らず、`backend`, `storage`, `algorithm`, `return_traj`,
`sample_stride`, `nondimensional`, `renorm`, `initial_states` を明示します。複数の
`initial_states` は等振幅・同位相の coherent superposition であり、混合状態ではありません。

## 最適化の変更

旧 YAML は [`examples/archives/v0_2/optimization_configs/`](../examples/archives/v0_2/optimization_configs/)
に履歴として保存されていますが、現行 CLI 入力ではありません。新しい設定は
[`configs/README.md`](../configs/README.md) と 3 個の current YAML から作ります。

### model と共通 schema

| 旧設定 | 現行設定 |
|---|---|
| `system.type: viblad` | `system.type: vibladder` |
| `omega_cm`, `delta_omega_cm` | value/unit pair の `vibrational_frequency`, `anharmonic_shift` |
| `B_cm`, `alpha_cm` | value/unit pair の `rotational_constant`, `vibration_rotation_coupling` |
| `energy_gap_cm` | `energy_gap` + `energy_gap_units` |
| `mu0`, `unit_dipole` | `dipole_scale` + `dipole_scale_units` |
| `plot.save_fig` | required `plot.enabled` |
| implicit axes | algorithm ごとの required `control_axes` |

`output.dir` と `plot.enabled` は必須です。unknown section/key と別 model 用 key はエラーです。

### 旧 Krotov と標準 Krotov は機械的に置換しない

v0.2 で Krotov と呼ばれていた batch-overlap 更新を同じ数値ロジックで再現する場合は
`algorithm: legacy_batch_overlap` を選びます。これは canonical odd field grid、
`field_dt_fs`、generated/sampled initial field を使います。legacy spectral filter はこの経路だけです。

`algorithm: krotov` は、状態 endpoint `N+1` 個と midpoint の piecewise-constant control
`N` 個を持つ標準の逐次更新です。こちらは `control_dt_fs`、`lambda_a`、
`lambda_a_units`、generated/sampled interval control を必要とします。

両者は field shape、時間点、更新順序が異なります。名前だけを変更したり、一方の sample を
他方へ resample したりすると計算ロジックが変わるため禁止です。

### optimizer ごとの明示入力

- GRAPE と `legacy_batch_overlap`: `dt_fs` は `field_dt_fs`、`sample_stride` は
  output-only の `output_stride` へ変更。field length は `2*N+1`。
- standard Krotov: `control_dt_fs` と `(N, 2)` interval controls を使用。
  `field_dt_fs` は不適用。
- Local: frozen legacy grid を保持するため `field_dt_fs` と `sample_stride` を使用。
  端点、segment index、共有境界、RK4 prefix は変更しない。
- Local の `gain` と `gain_units` は必須。推奨 unit は `(GV/m)^2 fs`。
- Local seed は required `initialization` mapping へ移す。`seed_field` は amplitude/unit と
  `max_segments`、`none` は field を注入せず zero-control fixed point なら propagation 前に停止。
- `field_max` は `field_max_v_per_m`。旧 `seed_amplitude`,
  `seed_amplitude_v_per_m`, `seed_max_segments` は top level に置かない。
- GRAPE と legacy batch の初期 field、standard Krotov の初期 control は generated/sampled を
  明示する。zero seed や shape 修復への fallback はない。

## 分光の変更

旧 root import は使わず `rovibrational_excitation.spectroscopy` から import します。
通常吸収は `AbsorbanceCalculator.standard_absorption(...)` で構築し、mOD を返す
`calculate(...)` を使います。Cartesian analyzer は
`CartesianAnalyzerProjection` と `analyzer_complex_response(...)` を使い、mOD へ変換する
前の complex molecular response を返します。

Analyzer intensity/absorbance は reference-field 測定規約が未定義なので未実装です。
`pol_det` を推測したり、analyzer response を通常吸収へ暗黙変換したりしません。

## result と checkpoint

v0.3 の normal result は `result_current.json` が immutable generation を選ぶ
result schema v1 です。checkpoint も `checkpoint_current.json` と完全な ordered expanded-run
SHA-256 を持つ schema v1 です。詳細は [result storage](RESULT_STORAGE.md) を参照してください。

- manifest のない v0.2 NPY/NPZ collection は strict loader や plotter で読まない。
- unversioned checkpoint は resume しない。
- 壊れた pointer から旧 layout へ fallback しない。
- 旧結果を新 schema として推測変換または暗黙上書きしない。
- 現時点では一般的な自動 migration tool はない。必要な結果は v0.3 入力で再計算する。

旧結果を比較資料として残す場合はディレクトリごと読み取り専用で保存し、v0.3 の出力先と
分離してください。

## 移行後の確認

```bash
rve-simulate my_params.py --dry-run
rve-simulate my_params.py --no-save
pytest -q
python scripts/smoke_examples.py
```

最低限、次を旧計算と比較します。

- basis dimension と state/index の対応
- Hamiltonian diagonal と dipole の非ゼロ要素
- field sample、endpoint、sample 数、`propagation_dt = 2 * field_dt`
- 初期/最終 population、norm、対象 observable
- optimizer の control layout と target fidelity
- sweep case 数・順序と checkpoint provenance

差が出た場合、単位、2π、coherent/incoherent、軸順、Krotov route、時間 grid を先に確認します。
式、符号、threshold、規格化を差に合わせて変更しないでください。
