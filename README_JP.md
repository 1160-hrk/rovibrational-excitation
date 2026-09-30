# rovibrational-excitation

[![CI](https://github.com/1160-hrk/rovibrational-excitation/actions/workflows/ci.yml/badge.svg)](https://github.com/1160-hrk/rovibrational-excitation/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English](README.md)

レーザー電場で駆動される振動回転量子ダイナミクスを計算する Python
ライブラリです。現在の開発版は `0.3.0.dev1` で、v0.2 との後方互換性は
意図的にありません。

明示単位付きのモデル構築、生成電場・外部サンプル電場、型付き時間発展、
最適制御、線形応答分光、バッチ実行、厳格な結果/checkpoint schema、可視化を
提供します。

## 現在の状態

検証済みの本番経路は CPU 計算です。CI では Python 3.10–3.13、CPU 全テスト、
物理参照、branch coverage、対応例、型検査、wheel の clean install を実行します。
詳細は [CI workflow](.github/workflows/ci.yml) を参照してください。

実 CUDA での実行は未検証です。低レベル RK4 は CPU と同じ計算グラフを使い、
device-native な CuPy 配列を返す実装になりましたが、実 GPU での parity と性能の
検証が必須です。split-operator の CuPy 経路には device-to-host round trip が
残っています。CuPy を要求して NumPy へ暗黙 fallback することはありません。

## 対応する物理モデル

| モデル | 結合と現在の範囲 |
|---|---|
| 二準位系 | scalar 結合。energy gap と双極子の単位を明示 |
| 振動 ladder | scalar の調和/Morse ladder。Morse は非ゼロの非調和性が必須 |
| 直線分子 | 調和/Morse 振動と回転。Cartesian M-resolved または scalar-z の M 非干渉平均 |
| 対称コマ | signed `|v,J,K,M>` 順の rigid parallel band。CH3F ortho/para filter。NumPy dense/CSR RK4 のみ |

対称コマでは axes、全物理定数と単位、核スピン異性体 sector を一つだけ明示
します。CuPy、split operator、全異性体を一つの純粋状態にする経路、最適化は
明示的にエラーになります。

## 数値計算の対応範囲

- 純粋状態の Schrödinger 時間発展は NumPy dense/CSR の RK4 に対応します。
  Numba CSR 経路では compiled RK4 loop 内で sparse matvec を実行します。
- split operator には厳密な Cartesian 相互作用と、明示的な
  helicity-projected 近似があります。相互作用 mode は必須で、fallback で選びません。
- インコヒーレント混合は専用 mixed-state propagator と規格化済み統計重みを
  使います。
- 密度行列/Liouville 時間発展は NumPy dense RK4 のみです。
- 電場 sampling 間隔は伝播刻みの半分です。RK4 電場は
  `2 * n_steps + 1` 点になります。
- 相互作用の符号は全経路で `H(t) = H0 - mu E(t)` です。

固定した規約と制限は [物理契約](docs/refactoring/PHYSICS_CONTRACTS.md) と
[Cartesian split operator の説明](docs/CARTESIAN_SPLIT_OPERATOR.md) を参照してください。

## インストール

Python 3.10 以上が必要です。

```bash
pip install rovibrational-excitation
```

現在の source checkout を使う場合:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,io,plot]"
```

optional の `gpu` extra は `cupy-cuda12x` を入れます。CUDA 環境が一致する場合
だけ選び、上記の未検証状態に注意してください。

## クイックスタート

この生成電場例は README 契約テストから直接実行されます。全ての物理量に単位を
付け、`field=None` によって生成電場経路を明示的に選びます。

```python
# README_SMOKE
import numpy as np

from rovibrational_excitation import run_simulation_case

params = {
    "basis_type": "twolevel",
    "energy_gap": 0.2,
    "energy_gap_units": "rad/fs",
    "dipole_scale": 3.0e-30,
    "dipole_scale_units": "C*m",
    "t_start": -10.0,
    "t_start_units": "fs",
    "t_end": 10.0,
    "t_end_units": "fs",
    "dt": 0.1,
    "dt_units": "fs",
    "duration": 4.0,
    "duration_units": "fs",
    "t_center": 0.0,
    "t_center_units": "fs",
    "envelope_kind": "gaussian_fwhm",
    "modulation_kind": "none",
    "carrier_frequency": 0.05,
    "carrier_frequency_units": "PHz",
    "amplitude": 1.0e8,
    "amplitude_units": "V/m",
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

population = run_simulation_case(params, field=None)
np.testing.assert_allclose(population.sum(axis=1), 1.0, rtol=1e-9, atol=1e-11)
print("final populations:", population[-1])
```

外部波形では `TimeGrid` と `ScalarField` または `CartesianField` を明示的に
構築します。sample は canonical V/m へ一度だけ変換してコピーされ、resample、
padding、規格化、暗黙修復は行いません。
[`example_external_scalar_field.py`](examples/example_external_scalar_field.py) を参照してください。

## 公開 API

package root が公開する名前は次の 8 個だけです。

```text
__version__, ElectricField, TimeGrid, ExecutionPolicy,
PropagationProblem, PropagationOptions, PropagationResult,
run_simulation_case
```

それ以外は明示的な subpackage から import します。

- `rovibrational_excitation.models` — schema とモデル構築
- `rovibrational_excitation.core` — 低レベル状態、演算子、単位、共通契約
- `rovibrational_excitation.fields` — sampled field、envelope、modulation
- `rovibrational_excitation.dynamics` — propagator と capability 契約
- `rovibrational_excitation.optimization` — GRAPE、standard Krotov、
  `legacy_batch_overlap`、local control
- `rovibrational_excitation.spectroscopy` — standard absorption と型付き complex analyzer response
- `rovibrational_excitation.visualization` — optional plot helper

旧 v0.2 の root convenience name に compatibility shim はありません。

## CLI と対応例

実行可能な parameter template を保存なしで実行します。

```bash
rve-simulate examples/params_template.py --no-save
```

3 個の対応例を一覧・実行できます。

```bash
python examples/launcher.py --list
python examples/launcher.py --run quickstart --quick
python scripts/smoke_examples.py
```

対応対象は [examples/README.md](examples/README.md) に掲載した top-level file だけです。
`examples/archives/v0_2/` は移行用の履歴資料で、実行も推測修正もしません。

最適化は 3 個の厳格な現行 config のいずれかから開始します。

```bash
rve-optimize --config configs/example_local_viblad_v3.yaml --no-plot
```

時間格子、seed/penalty/gain の必須単位、Python からの電場注入は
[configs/README.md](configs/README.md) を参照してください。

## 分光

standard absorption は型付き projection を使って mOD を返します。Cartesian
analyzer では射影した complex molecular response を返せます。analyzer の
intensity/absorbance と本番用 thermal-state constructor は未実装です。reference
field や測定規約をライブラリが推測することはなく、該当操作はエラーになります。

## 結果保存と再開

保存結果は immutable generation を作り、atomic な `result_current.json` pointer で
選択します。checkpoint も version 付きの組として保存し、resume 前に展開済み全 run
の順序まで検証します。不正、version なし、破損、別 run の data はエラーとなり、
legacy fallback や暗黙修復は行いません。詳細は
[結果保存形式](docs/RESULT_STORAGE.md) を参照してください。

## 開発時の検証

```bash
pytest -q
coverage run --data-file=/tmp/rve-coverage \
  --source=src/rovibrational_excitation --branch -m pytest -q
coverage report --data-file=/tmp/rve-coverage --show-missing --fail-under=47
ruff check --no-fix src tests examples benchmarks scripts
ruff format --check src tests examples benchmarks scripts
mypy
```

現在の local checkpoint は CPU 1500 tests pass、optional GPU 10 tests skip、
実測 branch coverage 80% です。skip された GPU test は CUDA の検証根拠ではありません。

## ドキュメント

- [ドキュメント索引](docs/README.md)
- [v0.3 移行ガイド](docs/MIGRATION_V0_3.md)
- [parameter reference](docs/PARAMETER_REFERENCE.md)
- [時間発展](docs/TIME_PROPAGATION.md)
- [単位系](docs/UNIT_SYSTEM.md)
- [sweep 仕様](docs/SWEEP_SPECIFICATION.md)
- [version/release 手順](docs/VERSION_MANAGEMENT.md)
- [変更履歴](CHANGELOG.md)

## ライセンス

[MIT](LICENSE)
