# rovibrational-excitation ドキュメント

この索引は開発版 `0.3.0.dev1` の実装と検証状態を基準にしています。v0.2 の
API とは後方互換ではありません。最初にルートの
[English README](../README.md) または [日本語 README](../README_JP.md) で、
対応モデル・数値経路・既知の制限を確認してください。

## 最短の実行手順

source checkout から環境を作り、CI と同じ小さな例を実行します。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,io,plot]"
python examples/launcher.py --run quickstart --quick
```

単位付き parameter template は保存なしで実行できます。

```bash
python -m rovibrational_excitation.cli.simulate \
  examples/params_template.py --no-save
```

外部波形を Python から注入する例は
[`example_external_scalar_field.py`](../examples/example_external_scalar_field.py)、
対応例の完全な一覧は [examples/README.md](../examples/README.md) を参照してください。
`examples/archives/v0_2/` は履歴資料であり、現行例ではありません。

## 文書の検証状態

| 文書 | 内容 | 状態 |
|---|---|---|
| [RESULT_STORAGE.md](RESULT_STORAGE.md) | result/checkpoint schema v1、atomic generation、strict loader | 現行契約 |
| [CARTESIAN_SPLIT_OPERATOR.md](CARTESIAN_SPLIT_OPERATOR.md) | 厳密 Cartesian split と helicity-projected 近似の原理 | 現行の物理契約 |
| [VERSION_MANAGEMENT.md](VERSION_MANAGEMENT.md) | final tag、CPU/実 GPU gate、明示的な release 操作 | 現行 release 契約 |
| [DOCKER_SETUP.md](DOCKER_SETUP.md) | Dev Container と Jupyter の安全な起動 | release 前に環境全体を再検証予定 |
| [PARAMETER_REFERENCE.md](PARAMETER_REFERENCE.md) | 通常 simulation parameter と generated/external field | 現行 schema 契約 |
| [SWEEP_SPECIFICATION.md](SWEEP_SPECIFICATION.md) | ordered Cartesian-product sweep と resume provenance | 現行契約 |
| [TIME_PROPAGATION.md](TIME_PROPAGATION.md) | RK4/split の説明 | 移行監査中 |
| [UNIT_SYSTEM.md](UNIT_SYSTEM.md) | 公開境界と内部単位 | 移行監査中 |

「移行監査中」の文書には旧キーや旧例が残る可能性があります。値・単位・物理規約を
推測して実行可能にせず、現時点では
[`examples/params_template.py`](../examples/params_template.py) と実装の strict
validation を優先してください。各文書は個別の契約テストを追加してから現行扱いへ
移します。

## 通常シミュレーション

通常の parameter-file 実行入口は `rve-simulate` です。

```bash
rve-simulate examples/params_template.py --dry-run
rve-simulate examples/params_template.py --no-save
```

- 生成電場では pulse の値と単位を全て明示します。
- 外部電場では canonical `TimeGrid` と `ScalarField` または
  `CartesianField` を構築します。
- `field=None` と sampled field は別の明示的な入力経路で、resample や
  fallback はありません。
- parameter sweep の case 順序は checkpoint provenance の一部です。

保存形式と検証付き読み込みは [RESULT_STORAGE.md](RESULT_STORAGE.md) を参照して
ください。

## 最適化

現行 schema として対応する YAML は `configs/` 直下の 3 個だけです。用途、時間
格子、field/control shape、seed、penalty、gain の単位は
[configs/README.md](../configs/README.md) に記載しています。

```bash
rve-optimize --config configs/example_local_viblad_v3.yaml --no-plot
```

YAML は CLI 用です。Python API では sampled field や interval control を直接
注入できます。標準 Krotov、GRAPE/legacy batch、local control の時間格子を相互に
変換したり、長さを暗黙修復したりしません。

## 分光

`rovibrational_excitation.spectroscopy` は次を分けて扱います。

- standard absorption: 型付き projection から mOD を計算
- Cartesian analyzer: 射影した complex molecular response を返す

analyzer intensity/absorbance と production thermal-state constructor は未実装です。
reference field や測定規約を推測せず、非対応操作はエラーにします。

## 固定された重要契約

- 相互作用は `H(t) = H0 - mu E(t)`。
- 電場 sampling 間隔は伝播刻みの半分。
- RK4 field grid は `2 * n_steps + 1` 点。
- normal simulation の物理 scalar は値と単位を対で指定。
- Morse potential は非ゼロの非調和性が必須。
- 複数 `initial_states` は等振幅・同位相の coherent superposition。
- incoherent mixture は専用 propagator を使用。
- 非対応 backend/algorithm/storage は fallback せずエラー。

詳細と優先順位は [物理契約](refactoring/PHYSICS_CONTRACTS.md) および
[決定記録](refactoring/DECISIONS.md) が権威です。

## CPU/CUDA 状態

検証済みの本番経路は CPU です。実 CUDA は未検証です。CuPy を要求した場合に
NumPy へ暗黙 fallback はしませんが、現在の低レベル GPU 経路には host round trip
が残ります。skip された GPU test は実 GPU の検証根拠ではありません。

## 開発者・Codex 向け

リファクタ作業はルートの [AGENTS.md](../AGENTS.md) を入口にし、次の順で読みます。

1. [PHYSICS_CONTRACTS.md](refactoring/PHYSICS_CONTRACTS.md)
2. [DECISIONS.md](refactoring/DECISIONS.md)
3. [TARGET_ARCHITECTURE.md](refactoring/TARGET_ARCHITECTURE.md)
4. [EXECUTION_PLAN.md](refactoring/EXECUTION_PLAN.md)
5. [refactoring/README.md](refactoring/README.md)

公開 API の現状と移行先は
[API_INVENTORY.md](refactoring/API_INVENTORY.md)、文書/workflow の残作業は
[DOCUMENTATION_WORKFLOW_AUDIT.md](refactoring/DOCUMENTATION_WORKFLOW_AUDIT.md) に
記録しています。

## 検証コマンド

```bash
pytest -q
python scripts/smoke_examples.py
python examples/tools/build_index.py --check
ruff check --no-fix src tests examples benchmarks scripts
ruff format --check src tests examples benchmarks scripts
mypy
```

物理参照は `tests/physics/`、公開・failure policy は `tests/contracts/`、複合経路は
`tests/integration/` が所有します。standalone の print/plot script を正しさの根拠には
しません。
