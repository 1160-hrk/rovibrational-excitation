# Codecov統合ガイド

このプロジェクトでは、テストカバレッジの自動化とレポート生成にCodecovを使用しています。

## セットアップ手順

### 1. Codecov.ioでのアカウント作成

1. [codecov.io](https://codecov.io/) にアクセス
2. GitHubアカウントでサインイン
3. リポジトリ `1160-hrk/rovibrational-excitation` を追加

### 2. 設定ファイル

- `codecov.yml`: Codecovの動作設定
- `.github/workflows/tests.yml`: GitHub ActionsからCodecovへのアップロード設定

### 3. 自動実行

プッシュやプルリクエストが作成されると：

1. GitHub Actionsがテストを実行
2. カバレッジデータを生成（`coverage.xml`）
3. Codecovにアップロード
4. レポートとバッジが自動更新

## Codecovの機能

### カバレッジレポート

- **プロジェクト全体**: https://codecov.io/gh/1160-hrk/rovibrational-excitation
- **ファイル別詳細**: 各ファイルの行単位でのカバレッジ
- **履歴グラフ**: 時系列でのカバレッジ変遷

### プルリクエスト統合

- **自動コメント**: PRでのカバレッジ変化を自動通知
- **Diff カバレッジ**: 変更箇所のカバレッジ状況
- **CIステータス**: カバレッジ基準を満たしているかチェック

### バッジ

```markdown
[![Coverage](https://codecov.io/gh/1160-hrk/rovibrational-excitation/branch/main/graph/badge.svg)](https://codecov.io/gh/1160-hrk/rovibrational-excitation)
```

## ローカルでのカバレッジ確認

```bash
# テスト実行 + カバレッジ
cd tests/
coverage run --source=../src -m pytest -v
coverage report

# HTML レポート生成
coverage html
open htmlcov/index.html

# XML レポート生成 (Codecov用)
coverage xml
```

## カバレッジ設定

`codecov.yml` での主な設定：

- **target**: 目標カバレッジ（auto = 現在値維持）
- **threshold**: 許容下降幅（1%）
- **ignore**: 対象外ディレクトリ
- **precision**: 小数点以下桁数（2桁）

## トラブルシューティング

### アップロードエラー

- トークンが不要な場合がほとんど（パブリックリポジトリ）
- エラーが発生した場合は `fail_ci_if_error: false` で継続

### カバレッジが0%と表示される

- `coverage.xml` のパスを確認
- `--source` パラメータが正しいか確認（`src/rovibrational_excitation`）
- テストファイルが実際に実行されているか確認
- 作業ディレクトリを統一する（ルートディレクトリから実行）

### "Unusable report" エラー

よくある原因と解決方法：

1. **ソースパスの不一致**:
   ```yaml
   # ❌ 間違い
   coverage run --source=../src -m pytest
   
   # ✅ 正しい  
   coverage run --source=src/rovibrational_excitation -m pytest tests/
   ```

2. **ファイルパスの問題**:
   ```yaml
   # Codecov v5では明示的にファイルを指定
   uses: codecov/codecov-action@v5
   with:
     file: ./coverage.xml  # 必須
     verbose: true         # デバッグ用
   ```

3. **codecov.yml での修正設定**:
   ```yaml
   fixes:
     - "src/rovibrational_excitation/::"
   ```

### プライベートリポジトリの場合

1. Codecovでトークンを生成
2. GitHub Secretsに `CODECOV_TOKEN` として追加
3. ワークフローで使用：

```yaml
- name: Upload coverage to Codecov
  uses: codecov/codecov-action@v3
  with:
    token: ${{ secrets.CODECOV_TOKEN }}
    file: ./coverage.xml
```

## 利点

- **自動化**: 手動でのバッジ更新が不要
- **詳細分析**: ファイル・関数単位での詳細レポート
- **PR統合**: コードレビュー時にカバレッジ変化を確認
- **履歴管理**: 時系列でのカバレッジ推移を追跡
- **可視化**: 美しいグラフとレポート 