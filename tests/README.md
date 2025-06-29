# Rovibrational Excitation テストスイート

このディレクトリには、rovibrational-excitation パッケージの包括的なテストスイートが含まれています。

## 現在のテストカバレッジ

**全体カバレッジ: 63%** (1504行中944行がテスト済み)

### モジュール別カバレッジ詳細

#### 🟢 高カバレッジ (80%以上)
| モジュール | カバレッジ | ステータス |
|-----------|-----------|-----------|
| LinMolBasis | 100% | ✅ 完全テスト済み |
| TwoLevelBasis | 100% | ✅ 完全テスト済み |
| VibLadderBasis | 100% | ✅ 完全テスト済み |
| Harmonic Vibration | 100% | ✅ 完全テスト済み |
| Morse Vibration | 100% | ✅ 完全テスト済み |
| States | 98% | ✅ ほぼ完全 |
| VibLadder Dipole | 96% | ✅ 良好 |
| __init__.py | 87% | ✅ 良好 |
| TwoLevel Dipole | 86% | ✅ 良好 |
| Propagator | 83% | ✅ 良好 |

#### 🟡 中程度カバレッジ (50-79%)
| モジュール | カバレッジ | 主な未テスト領域 |
|-----------|-----------|-----------------|
| Hamiltonian | 67% | 非推奨機能 |
| Simulation Runner | 62% | 高度なバッチ機能 |
| Cache | 54% | HDF5保存・読み込み、高度な機能 |
| ElectricField | 53% | 分散補正、複雑変調機能 |
| LinMol Dipole | 52% | CuPyバックエンド、スパース行列 |
| LinMol Builder | 50% | GPU機能、エラーハンドリング |

#### 🔴 低カバレッジ (50%未満)
| モジュール | カバレッジ | 優先度 |
|-----------|-----------|-------|
| J Rotation Dipole | 45% | 🟡 中 |
| Split-Op Schrödinger | 38% | 🔴 高 |
| RK4 Schrödinger | 34% | 🔴 高 |
| JM Rotation Dipole | 29% | 🟡 中 |
| RK4 LVNE | 25% | 🔴 高 |

### テストカバレッジの改善目標

#### フェーズ3A: Simulation Runner (22% → 60%+)
- 最高優先度 - エンドツーエンドワークフロー
- シミュレーション設定、実行、結果解析

#### フェーズ3B: Low-Level Propagators (25-38% → 60%+)
- RK4 LVNE、RK4 Schrödinger、Split-Operator
- Numba/CuPy バックエンドテスト

#### フェーズ3C: Electric Field Advanced Features (53% → 70%+)
- 分散補正 (GDD/TOD)
- 複雑な変調機能
- 時間依存偏光

#### フェーズ3D: Dipole System Extensions (29-45% → 70%+)
- 回転双極子の高度な機能
- 振動双極子の物理的妥当性
- 選択則とコヒーレンス

### カバレッジ測定方法

```bash
# カバレッジ付きテスト実行
cd tests/
coverage run -m pytest
coverage report --show-missing

# HTML レポート生成
coverage html
# htmlcov/index.html をブラウザで開く

# 特定モジュールのカバレッジ
coverage report --include="*/propagator.py"
```

### 注意事項

- **Numba JITコンパイル**: `@njit` 装飾された関数は測定困難
- **CuPy GPU コード**: GPU実行部分は環境依存でカバレッジ測定制限あり
- **スパース行列**: scipy.sparse対応部分は条件付きテスト

## テストファイル一覧

### 基底クラステスト
- `test_basis.py` - LinMolBasis クラスの拡張テスト
- `test_basis_twolevel.py` - TwoLevelBasis クラスのテスト
- `test_basis_viblad.py` - VibLadderBasis クラスのテスト

### コア機能テスト
- `test_electric_field.py` - ElectricField クラスと関連関数のテスト
- `test_states.py` - StateVector と DensityMatrix クラスのテスト
- `test_hamiltonian.py` - ハミルトニアン生成関数のテスト（非推奨機能）

### 伝播アルゴリズムテスト
- `test_propagator.py` - 高レベル伝播関数のテスト
- `test_rk4_schrodinger.py` - RK4 Schrodinger伝播のテスト
- `test_rk4_lvne.py` - RK4 Liouville-von Neumann伝播のテスト
- `test_splitop_schrodinger.py` - Split-Operator伝播のテスト

### 統合・パフォーマンステスト
- `test_integration.py` - 複数モジュール間の統合テスト
- `test_runner.py` - シミュレーション実行システムのテスト

### 実行ツール
- `run_tests.py` - テスト実行用スクリプト

## テスト実行方法

### 1. 全テスト実行

```bash
# Pytestを使用
cd tests/
python -m pytest -v

# 実行スクリプトを使用
python run_tests.py
```

### 2. 特定のテストファイル実行

```bash
# 特定ファイル
python -m pytest test_basis.py -v

# 実行スクリプト使用
python run_tests.py basis
```

### 3. 特定のテスト関数実行

```bash
python -m pytest test_basis.py::test_linmol_initialization_parameters -v
```

### 4. パフォーマンステスト実行

```bash
# 通常のテストのみ（高速）
python -m pytest -v -m "not slow"

# すべて（低速テスト含む）
python -m pytest -v
```

## テストカテゴリ

### 基本機能テスト
- クラスの初期化とメソッド
- エラーハンドリング
- 境界条件

### 数値精度テスト
- ノルム保存
- エネルギー保存
- エルミート性

### 物理的妥当性テスト
- ポピュレーションダイナミクス
- コヒーレンス
- 量子力学的性質

### 統合テスト
- モジュール間連携
- エンドツーエンドワークフロー
- バックエンド一貫性

### パフォーマンステスト
- 大規模システム
- 長時間伝播
- メモリ効率

## 依存関係

テスト実行に必要なパッケージ：

```bash
pip install pytest numpy scipy numba
```

オプション（CuPyテスト用）：
```bash
pip install cupy
```

## テスト設定

### pytest.ini の設定例

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    slow: マークされたテストは実行時間が長い
addopts = -v --tb=short
```

### カスタムマーカー

- `@pytest.mark.slow` - 実行時間の長いテスト
- `@pytest.mark.skipif` - 条件付きスキップ

## トラブルシューティング

### よくある問題

1. **ImportError**: パッケージが見つからない
   ```bash
   # src ディレクトリを Python パスに追加
   export PYTHONPATH=$PYTHONPATH:../src
   ```

2. **Numbaのコンパイルエラー**
   ```bash
   # キャッシュクリア
   export NUMBA_DISABLE_JIT=1
   ```

3. **CuPy関連エラー**
   ```bash
   # CuPyテストをスキップ
   python -m pytest -k "not cupy"
   ```

### パフォーマンス問題

- 大規模テストがメモリ不足で失敗する場合は、システムスペックを確認
- 長時間テストは `-m "not slow"` でスキップ可能

## テスト結果の解釈

### 成功例
```
test_basis.py::test_linmol_basic ✓ PASSED
test_propagator.py::test_schrodinger_propagation ✓ PASSED
```

### 失敗例
```
test_basis.py::test_invalid_state ✗ FAILED
AssertionError: Expected ValueError not raised
```

## 継続的インテグレーション

GitHub Actionsでの自動テスト実行設定例：

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest
    - name: Run tests
      run: |
        cd tests
        python -m pytest -v -m "not slow"
```

## 貢献ガイドライン

新しいテストを追加する際は：

1. 適切なファイルに配置
2. わかりやすいテスト名を使用
3. ドキュメント文字列で説明
4. 境界条件とエラーケースを含める
5. 実行時間を考慮（長時間テストは`@pytest.mark.slow`）

## サポート

テストに関する質問や問題は、GitHubのIssueで報告してください。 