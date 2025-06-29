# カバレッジバッジ自動更新システム

## 概要

プロジェクトのREADME.mdに表示されるカバレッジバッジを自動的に最新の値に更新するシステムです。

### 🔄 自動更新の仕組み

```
テスト実行 → カバレッジ測定 → バッジ色決定 → README更新
```

### 🎨 カバレッジ率による色分け

| カバレッジ率 | 色 | 表示例 |
|-------------|---|--------|
| ≥ 90% | `brightgreen` | ![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen.svg) |
| ≥ 80% | `green` | ![Coverage](https://img.shields.io/badge/coverage-85%25-green.svg) |
| ≥ 70% | `yellowgreen` | ![Coverage](https://img.shields.io/badge/coverage-75%25-yellowgreen.svg) |
| ≥ 60% | `yellow` | ![Coverage](https://img.shields.io/badge/coverage-65%25-yellow.svg) |
| ≥ 50% | `orange` | ![Coverage](https://img.shields.io/badge/coverage-55%25-orange.svg) |
| < 50% | `red` | ![Coverage](https://img.shields.io/badge/coverage-45%25-red.svg) |

---

## 🚀 使用方法

### 方法1: 手動更新スクリプト（推奨）

最も簡単で確実な方法です。

```bash
# カバレッジを測定してバッジを更新
python scripts/update_coverage_badge.py
```

### 方法2: テスト実行+自動更新

テスト実行からバッジ更新まで一括で行います。

```bash
# テスト実行 → カバレッジ測定 → バッジ更新
./scripts/test_and_update_badge.sh
```

### 方法3: GitHub Actions（自動化）

プッシュのたびに自動でバッジを更新します。

- `.github/workflows/tests.yml` が既に設定済み
- `main`ブランチにプッシュすると自動実行
- 複数Python版でテストして、最新のカバレッジでバッジを更新

---

## 📊 現在のカバレッジ状況

**全体カバレッジ: 75%** (4088行中3068行がテスト済み)

### モジュール別カバレッジ詳細

#### 🟢 高カバレッジ (80%以上)
- **LinMol Basis**: 100% ✅ 完全テスト済み
- **TwoLevel Basis**: 100% ✅ 完全テスト済み
- **VibLadder Basis**: 100% ✅ 完全テスト済み
- **Hamiltonian**: 100% ✅ 完全テスト済み
- **States**: 98% ✅ ほぼ完全
- **VibLadder Dipole**: 94% ✅ 優秀
- **TwoLevel Dipole**: 84% ✅ 良好
- **Propagator**: 83% ✅ 良好

#### 🟡 中程度カバレッジ (50-79%)
- **BasisBase**: 72% - 抽象メソッドの一部
- **Cache**: 56% - HDF5保存・読み込み、高度な機能
- **ElectricField**: 53% - 分散補正、複雑変調機能
- **LinMolBuilder**: 50% - CuPyバックエンド、スパース行列

#### 🔴 低カバレッジ (50%未満)
- **JM Rotation Dipole**: 29% 🔴 高優先度
- **Split-Op Schrödinger**: 38% 🔴 高優先度
- **RK4 Schrödinger**: 34% 🔴 高優先度
- **RK4 LVNE**: 25% 🔴 高優先度
- **Simulation Runner**: 22% 🔴 最高優先度

---

## ⚙️ 設定とカスタマイズ

### スクリプトの動作を理解する

```python
# scripts/update_coverage_badge.py の主要部分

def get_coverage_percentage():
    """カバレッジ率を取得"""
    result = subprocess.run(
        ['coverage', 'report', '--format=total'],
        capture_output=True, text=True, check=True
    )
    return int(result.stdout.strip())

def get_badge_color(percentage):
    """カバレッジ率に応じたバッジの色を決定"""
    if percentage >= 90: return "brightgreen"
    elif percentage >= 80: return "green"
    elif percentage >= 70: return "yellowgreen"
    elif percentage >= 60: return "yellow"
    elif percentage >= 50: return "orange"
    else: return "red"

def update_readme_badge(percentage, color):
    """README.mdのバッジを更新"""
    pattern = r'coverage-\d+%25-\w+'
    new_badge = f'coverage-{percentage}%25-{color}'
    updated_content = re.sub(pattern, new_badge, content)
```

### 色の閾値をカスタマイズ

`scripts/update_coverage_badge.py`の`get_badge_color`関数を編集：

```python
def get_badge_color(percentage):
    # より厳しい基準に変更
    if percentage >= 95: return "brightgreen"
    elif percentage >= 85: return "green"
    elif percentage >= 75: return "yellowgreen"
    elif percentage >= 65: return "yellow"
    elif percentage >= 55: return "orange"
    else: return "red"
```

---

## 🛠️ トラブルシューティング

### よくある問題と解決方法

#### 1. 「coverageコマンドが見つかりません」
```bash
pip install coverage
```

#### 2. 「.coverageファイルが見つかりません」
```bash
cd tests
coverage run -m pytest
cd ..
python scripts/update_coverage_badge.py
```

#### 3. 「README.mdにカバレッジバッジが見つかりません」
README.mdに以下の形式のバッジがあることを確認：
```markdown
[![Coverage](https://img.shields.io/badge/coverage-XX%25-COLOR.svg)](リンク)
```

#### 4. GitHub Actionsでの権限エラー

リポジトリ設定で以下を確認：
- Settings → Actions → General → Workflow permissions
- "Read and write permissions" を選択

#### 5. バッジが更新されない（ブラウザキャッシュ）
- ブラウザのキャッシュをクリア
- URLs.io のキャッシュをクリア（`?v=1`をURLに追加）

---

## 🔧 GitHub Actions設定詳細

### ワークフロー構成

`.github/workflows/tests.yml`は以下を実行：

1. **マルチバージョンテスト**: Python 3.9-3.12
2. **カバレッジ測定**: Python 3.11環境で実行
3. **バッジ自動更新**: `main`ブランチのみ
4. **自動コミット**: 変更があれば自動コミット・プッシュ

### 手動での実行

```bash
# GitHub Actionsを手動トリガー
# GitHubのActionsタブから「Run workflow」をクリック
```

### ローカルで同じ処理を実行

```bash
# 複数Pythonバージョンでテスト（pyenvを使用）
for version in 3.9 3.10 3.11 3.12; do
  pyenv local $version
  python -m pip install -e .
  cd tests && coverage run -m pytest -v
  cd ..
done

# バッジを更新
python scripts/update_coverage_badge.py
```

---

## 📈 改善計画とロードマップ

### Phase 4A: Simulation Runner強化 (22% → 70%+)
- 最高優先度：エンドツーエンドワークフロー
- 設定ファイル読み込み・検証
- バッチ処理・並列実行

### Phase 4B: Low-Level Propagators (25-38% → 60%+)
- RK4 LVNE、RK4 Schrödinger、Split-Operator
- Numba/CuPy バックエンドテスト

### 目標カバレッジ

**保守的予測**: 75% → 87% (+12pt)
**楽観的予測**: 75% → 93% (+18pt)

### 継続的改善

1. **週次レビュー**: カバレッジ変動の監視
2. **四半期目標**: 段階的な品質向上
3. **年次評価**: 業界標準（90%+）の達成

---

## 📚 参考資料

### 外部サービス比較

| サービス | 利点 | 欠点 |
|---------|------|------|
| **自作スクリプト** | 完全制御、シンプル | 手動実行必要 |
| **GitHub Actions** | 自動化、統合 | 設定複雑、権限管理 |
| **Codecov** | 美しいレポート、詳細 | 外部依存、制限あり |
| **Coveralls** | 軽量、高速 | 機能限定 |

### 推奨アプローチ

1. **開発段階**: 手動スクリプト
2. **CI/CD構築**: GitHub Actions
3. **本格運用**: Codecov連携

---

## ✅ チェックリスト

### 初回設定

- [ ] `scripts/update_coverage_badge.py`の実行確認
- [ ] `scripts/test_and_update_badge.sh`の実行権限設定
- [ ] `.github/workflows/tests.yml`の動作確認
- [ ] README.mdのバッジ形式確認

### 定期メンテナンス

- [ ] 月次カバレッジレポート確認
- [ ] 低カバレッジモジュールの改善計画
- [ ] GitHub Actionsログの監視
- [ ] バッジ色の閾値見直し

### 品質保証

- [ ] 全テストの定期実行
- [ ] カバレッジ回帰の早期発見
- [ ] 新機能のテスト追加確認
- [ ] ドキュメントの最新化

---

## 🎯 まとめ

カバレッジバッジ自動更新システムにより：

✅ **現在のカバレッジ**: 75%（yellowgreen）
✅ **自動更新**: GitHub Actions対応
✅ **手動更新**: 簡単なスクリプト実行
✅ **視覚的品質管理**: 色分けによる一目瞭然の状況把握

このシステムによって、プロジェクトの品質を持続的に監視・改善できます。 