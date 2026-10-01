# テストスイート

このディレクトリは rovibrational-excitation の自動検証を所有します。正しさの根拠は
通常の unit test だけでなく、独立した物理参照、公開契約、workflow 統合、性能基準に
分かれています。リポジトリ root から実行してください。

依存関係と pytest 設定の唯一の authority は [`pyproject.toml`](../pyproject.toml) です。
追加の依存 manifest や独自の test runner は使いません。

## 環境

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,io,plot]"
```

CUDA test を実行する専用環境では `gpu` extra を追加します。この extra は
`cupy-cuda12x[ctk]` と CUDA 12 user-space components を環境内へ導入しますが、
互換 NVIDIA driver は別途必要です。CPU 上で skip された GPU test は実 CUDA の
検証根拠ではありません。

## ディレクトリ

```text
tests/
├── contracts/     # API、validation、failure policy、architecture、文書/workflow
├── physics/       # 解析解・独立実装・保存量・数値参照
├── integration/   # 複数 component を通る workflow
├── performance/   # correctness を伴う非 blocking benchmark/reference
├── unit/          # 小さい独立 component と unit conversion
└── test_*.py      # 移動前から残る収集対象の component/regression test
```

ルート直下の test を機械的に移動しません。ownership が明確になり、import と数値結果を
固定する test がある場合だけ、別 commit で移動します。

## 標準コマンド

完全な CPU suite:

```bash
pytest -q
```

物理参照と公開契約だけを確認する場合:

```bash
pytest -q tests/physics tests/contracts
```

Branch coverage は repository 内に `.coverage` を残さず測定します。

```bash
coverage run --data-file=/tmp/rve-coverage \
  --source=src/rovibrational_excitation --branch -m pytest -q
coverage report --data-file=/tmp/rve-coverage --show-missing --fail-under=47
```

Active source/test/example/tooling の品質 gate:

```bash
ruff check --no-fix src tests examples benchmarks scripts
ruff format --check src tests examples benchmarks scripts
python -m mypy --no-incremental
python scripts/smoke_examples.py
python examples/tools/build_index.py --check
```

CI の完全な job 構成と Python 3.10–3.13 matrix は
[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) が authority です。

## marker

| marker | 意味 | 使用法 |
|---|---|---|
| `physics` | analytic result、独立 reference、物理 invariant | `pytest -m physics` |
| `gpu` | 実 CuPy/CUDA device が必要 | real-GPU runner で `pytest -m gpu` |
| `performance` | runtime/memory reference。通常の全 suite にも含まれる | `pytest -m performance` |
| `slow` | 長い deterministic correctness test | 必要なら `pytest -m "not slow"` で除外 |

Marker は [`pyproject.toml`](../pyproject.toml) に登録され、unknown marker はエラーです。
CUDA release evidence は通常 CI とは別の self-hosted GPU job が所有します。

## test の役割

### `contracts/`

公開 signature、必須 key/unit、unsupported combination の明示 error、依存方向、disk
schema、documentation/workflow wiring を固定します。契約 test は物理式の独立 oracle の
代わりではありません。

### `physics/`

本番実装をコピーせず、可能な限り解析式・直接展開・独立実装から期待値を作ります。
Hamiltonian、dipole、selection rule、RK4/split、optimizer、spectroscopy の式や符号を
変更する場合は、先にここへ reference を追加します。

### `integration/`

model construction から伝播・最適化・保存まで、単体境界をまたぐ挙動を確認します。
小さい deterministic system を使い、failure を再現可能に保ちます。

### `performance/`

性能比較には固定 protocol と committed JSON artifact を使います。短い単発 timing を
correctness や高速化の根拠にしません。基準は
[`benchmarks/README.md`](../benchmarks/README.md) を参照してください。

## 追加・変更時の規則

1. 変更前の挙動と authority を特定する。
2. 数値または物理ロジックなら、production code と独立した期待値を先に追加する。
3. 正常系だけでなく、unsupported input が fallback せず失敗することも確認する。
4. random input を使う場合は seed と tolerance の由来を明記する。
5. GPU test は `gpu` marker を付け、CPU skip を pass の証拠として扱わない。
6. temporary output は pytest の `tmp_path` または `/tmp` を使い、repository に保存しない。
7. focused test の後に完全 suite、Ruff、format、`git diff --check` を実行する。
8. public contract、phase status、物理判断を変えた場合は同じ commit で文書を更新する。

式、符号、threshold、normalization、axis、time step の意味が既存の決定記録と test から
確定できない場合は、推測して test を新仕様に合わせず user に確認します。詳細は
[physics contracts](../docs/refactoring/PHYSICS_CONTRACTS.md) と
[decision log](../docs/refactoring/DECISIONS.md) を参照してください。

## 既知の環境差

- CuPy/CUDA がない環境では `gpu` test が skip されます。
- Numba の初回 JIT compile は timing から分離します。
- plot test は non-interactive backend を使います。一部の空 series では Matplotlib の
  legend warning が出ますが、error への変換や broad warning suppression はしません。
- coverage percentage は code と test の増減で変わるため、古い表をこの文書へ複製しません。
  現在の測定値と必須 floor は root [`AGENTS.md`](../AGENTS.md) に記録します。

## 履歴資料

[`TEST_CATALOG.md`](TEST_CATALOG.md)、[`TEST_STATUS_REPORT.md`](TEST_STATUS_REPORT.md)、
[`XFAIL_FIXES_REPORT.md`](XFAIL_FIXES_REPORT.md) の古い件数や XFAIL 記録は当時の移行資料です。
現在の test 数、coverage、capability の証拠として使いません。削除済み standalone runner や
validation script の disposition は
[`VALIDATION_INVENTORY.md`](../docs/refactoring/VALIDATION_INVENTORY.md) にあります。
