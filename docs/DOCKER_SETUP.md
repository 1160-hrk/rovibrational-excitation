# Docker 開発環境セットアップガイド

このドキュメントでは、`rovibrational-excitation`プロジェクトの改善されたDocker開発環境について説明します。

## 🎯 主な改善点

### ✅ 解決された問題
- **VSCodeのPython認識**: Dockerでインストールされたpython3.12との完全同期
- **依存関係の事前インストール**: ビルド時に全パッケージをインストール
- **Jupyterサポート**: 対話的開発のための完全なJupyter Lab環境
- **ファイル編集権限**: すべてのファイルをDocker環境内で編集可能
- **Build & Release**: build, twineによるパッケージング環境

### 🔧 技術仕様
- **ベースイメージ**: `python:3.12-slim`
- **開発ユーザー**: `devuser` (sudo権限付き)
- **Jupyterポート**: 8888 (自動転送)
- **作業ディレクトリ**: `/workspace`

## 🚀 クイックスタート

### 1. VSCode Dev Container（推奨）

```bash
# リポジトリをクローン
git clone <repository-url>
cd rovibrational-excitation

# VSCodeで開く
code .

# VSCodeでコマンドパレット (Ctrl+Shift+P) を開き、
# "Dev Containers: Reopen in Container" を選択
```

## 📊 Jupyter Lab の使用

### コンテナ内でJupyter起動

```bash
# スクリプトを使用（推奨）
./scripts/start_jupyter.sh
```

### ブラウザでアクセス
- URL: `http://localhost:8888`
- トークン: 不要（開発環境用設定）

### 対話的開発の例

```python
# Jupyter Cellで実行
import numpy as np
import matplotlib.pyplot as plt
from rovibrational_excitation.core import propagator

# 振動回転励起シミュレーションの例
# examples/example_twolevel_excitation.py を参照
```

## 🛠 開発ワークフロー

### コード品質チェック

```bash
# コード静的解析
ruff check src/ tests/ examples/
mypy src/

# フォーマット
black src/ tests/ examples/
ruff check --fix src/ tests/ examples/
```

### テスト実行

```bash
# 全テスト実行
python -m pytest tests/ -v

# カバレッジ付きテスト
coverage run -m pytest tests/
coverage html
```

### パッケージビルド

```bash
# ビルド
python -m build

# 公開（テスト環境）
twine upload --repository testpypi dist/*

# 公開（本番環境）
twine upload dist/*
```

## 📁 ディレクトリ構造

```
/workspace/
├── src/rovibrational_excitation/  # メインソースコード
├── examples/                      # 使用例・デモ
├── tests/                         # テストコード
├── docs/                          # ドキュメント
├── results/                       # シミュレーション結果
├── notebooks/                     # Jupyter ノートブック (新規作成)
└── scripts/                       # 開発用スクリプト
```

## 🔧 Makefile コマンド

```bash
make help          # 使用可能なコマンド一覧
make build         # Dockerイメージビルド
make clean         # Dockerリソース清理
make jupyter       # Jupyter起動コマンド表示
make test          # テスト実行コマンド表示
make lint          # コード品質チェックコマンド表示
make format        # フォーマットコマンド表示
```

## 🐛 トラブルシューティング

### よくある問題

#### 1. Pythonパスの認識問題
```bash
# VSCode内でPythonインタープリターを確認
which python
# 出力: /usr/local/bin/python

# VSCodeのPython設定確認
# Ctrl+Shift+P → "Python: Select Interpreter"
# /usr/local/bin/python を選択
```

#### 2. Jupyterポートアクセス問題
```bash
# ポート転送確認
docker ps
# PORTS列で "0.0.0.0:8888->8888/tcp" を確認

# ファイアウォール確認（macOS）
sudo lsof -i :8888
```

#### 3. ファイル権限問題
```bash
# コンテナ内でファイル権限確認
ls -la /workspace/

# 所有者変更（必要に応じて）
sudo chown -R devuser:devuser /workspace/
```

#### 4. パッケージインストール問題
```bash
# 依存関係再インストール
pip install --no-cache-dir -r requirements-dev.txt

# キャッシュクリア
pip cache purge
```

## 🔒 セキュリティ注意事項

⚠️ **開発環境専用設定**
- Jupyterのトークン認証を無効化
- CORS制限を緩和
- root権限でのJupyter実行を許可

**本番環境では使用しないでください！**

## 📚 参考資料

- [VSCode Dev Containers Documentation](https://code.visualstudio.com/docs/devcontainers/containers)
- [Jupyter Lab Documentation](https://jupyterlab.readthedocs.io/)
- [Docker Multi-stage Builds](https://docs.docker.com/develop/dev-best-practices/)

## 🤝 貢献

Docker環境の改善提案やバグ報告は、GitHubのIssueまでお願いします。 