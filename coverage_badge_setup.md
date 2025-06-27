# カバレッジバッジの自動更新設定ガイド

## 現在の状況

プロジェクトのREADMEに以下のカバレッジバッジを追加しました：

```markdown
[![Coverage](https://img.shields.io/badge/coverage-50%25-yellow.svg)](tests/README.md#現在のテストカバレッジ)
```

**現在のカバレッジ: 50%** (手動更新)

## 自動更新方法

### 方法1: GitHub Actionsで自動生成

`.github/workflows/coverage.yml` を作成：

```yaml
name: Coverage

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  coverage:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install coverage pytest
    
    - name: Run tests with coverage
      run: |
        cd tests
        coverage run -m pytest
        coverage report --format=markdown > coverage_report.md
        COVERAGE_PERCENT=$(coverage report --format=total)
        echo "COVERAGE_PERCENT=$COVERAGE_PERCENT" >> $GITHUB_ENV
    
    - name: Update coverage badge
      if: github.ref == 'refs/heads/main'
      run: |
        # カバレッジ率に応じて色を決定
        if [ $COVERAGE_PERCENT -ge 80 ]; then
          COLOR="brightgreen"
        elif [ $COVERAGE_PERCENT -ge 60 ]; then
          COLOR="yellow"
        elif [ $COVERAGE_PERCENT -ge 40 ]; then
          COLOR="orange"
        else
          COLOR="red"
        fi
        
        # README.mdのバッジを更新
        sed -i "s/coverage-[0-9]*%25-[a-z]*/coverage-${COVERAGE_PERCENT}%25-${COLOR}/g" README.md
    
    - name: Commit coverage badge update
      if: github.ref == 'refs/heads/main'
      run: |
        git config --local user.email "action@github.com"
        git config --local user.name "GitHub Action"
        git add README.md
        git diff --staged --quiet || git commit -m "Update coverage badge to ${COVERAGE_PERCENT}%"
        git push
```

### 方法2: Codecov連携

1. **Codecovアカウント作成**: https://codecov.io/
2. **リポジトリ追加**: Codecovでリポジトリを有効化
3. **GitHub Actions設定**:

```yaml
name: Coverage with Codecov

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install coverage pytest
    
    - name: Run tests with coverage
      run: |
        cd tests
        coverage run -m pytest
        coverage xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: tests/coverage.xml
        flags: unittests
        name: codecov-umbrella
```

4. **README.mdのバッジを変更**:

```markdown
[![Coverage](https://codecov.io/gh/1160-hrk/rovibrational-excitation/branch/main/graph/badge.svg)](https://codecov.io/gh/1160-hrk/rovibrational-excitation)
```

### 方法3: Coveralls連携

1. **Coverallsアカウント作成**: https://coveralls.io/
2. **GitHub Actions設定**:

```yaml
- name: Upload coverage to Coveralls
  uses: coverallsapp/github-action@v2
  with:
    github-token: ${{ secrets.GITHUB_TOKEN }}
    path-to-lcov: tests/coverage.lcov
```

3. **バッジ**:
```markdown
[![Coverage Status](https://coveralls.io/repos/github/1160-hrk/rovibrational-excitation/badge.svg?branch=main)](https://coveralls.io/github/1160-hrk/rovibrational-excitation?branch=main)
```

## 推奨設定

### 段階的なアプローチ

1. **現在**: 手動更新 (50%)
2. **Phase 1**: GitHub Actionsで自動計算 + 手動確認
3. **Phase 2**: 完全自動化 (Codecov推奨)

### カバレッジ色設定

```
🟢 >= 80%: brightgreen
🟡 >= 60%: yellow  
🟠 >= 40%: orange
🔴 < 40%:  red
```

## 現在のカバレッジ詳細

| カテゴリ | カバレッジ | ファイル数 |
|---------|-----------|----------|
| 🟢 高 (80%+) | 7モジュール | LinMolBasis, TwoLevelBasis, 他 |
| 🟡 中 (50-79%) | 4モジュール | ElectricField, Cache, 他 |
| 🔴 低 (<50%) | 8モジュール | Simulation Runner, 伝播アルゴリズム |

**次の目標**: 55% (Phase 3Aでシミュレーションランナー改善後)

## 手動更新方法

現在のカバレッジ率を確認して手動更新：

```bash
# カバレッジ測定
cd tests/
coverage run -m pytest
COVERAGE=$(coverage report --format=total)
echo "Current coverage: ${COVERAGE}%"

# README.mdのバッジを手動更新
# 例: 50% → 55%
sed -i 's/coverage-50%25-yellow/coverage-55%25-yellow/g' README.md
```

## トラブルシューティング

### よくある問題

1. **GitHub Actions権限エラー**
   - Settings → Actions → General → Workflow permissions → "Read and write permissions"

2. **coverage.xmlが見つからない**
   ```bash
   coverage xml --o tests/coverage.xml
   ```

3. **バッジが更新されない**
   - ブラウザキャッシュをクリア
   - shields.ioのキャッシュクリア: `?v=1` をURLに追加

### デバッグ

```bash
# GitHub Actionsログを確認
# カバレッジレポート出力テスト
coverage report --show-missing
coverage html  # htmlcov/index.html
``` 