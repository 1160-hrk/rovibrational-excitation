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

### ローカル専用で起動

```bash
# スクリプトを使用（推奨）
./scripts/start_jupyter.sh
```

既定では `127.0.0.1:8888` にだけbindし、Jupyter標準のtoken認証とXSRF
保護を維持します。端末に表示されたtoken付きURLを使用してください。

### 明示的に外部へbindする場合

外部bindが必要な隔離済み開発環境でのみ指定します。

```bash
RVE_JUPYTER_HOST=0.0.0.0 ./scripts/start_jupyter.sh
```

この場合も認証やXSRFを無効化しません。ポート公開範囲とJupyter tokenを
別途管理してください。

### 対話的開発の例

```bash
python examples/launcher.py --list
python examples/launcher.py --run quickstart --quick
```

## 🛠 開発ワークフロー

### コード品質チェック

```bash
ruff check --no-fix src tests examples benchmarks scripts
ruff format --check src tests examples benchmarks scripts
mypy
python scripts/smoke_examples.py
```

### テスト実行

```bash
pytest -q
coverage run --data-file=/tmp/rve-coverage --source=src/rovibrational_excitation --branch -m pytest -q
coverage report --data-file=/tmp/rve-coverage --show-missing --fail-under=47
```

### パッケージビルド

```bash
python -m build
python -m twine check dist/*
```

公開は手動のTwine uploadでは行いません。実GPUを含むrelease gateは
[バージョン管理とリリース](VERSION_MANAGEMENT.md)を参照してください。

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