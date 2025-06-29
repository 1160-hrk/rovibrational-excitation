# バージョン管理ガイド

## 概要

このプロジェクトでは、**pyproject.tomlのバージョンとGitHubタグを一致させる**バージョン管理システムを採用しています。

### 🎯 基本原則

1. **pyproject.toml** がマスターバージョン
2. **Gitタグ** は `v{version}` 形式（例: `v0.1.4`）
3. **自動化** によるバージョン不整合の防止
4. **セマンティックバージョニング** の採用

---

## 🚀 リリースプロセス

### 方法1: 自動スクリプト使用（推奨）

```bash
# パッチリリース（バグ修正）
python scripts/release.py 0.1.5 "Bug fixes and improvements"

# マイナーリリース（新機能）
python scripts/release.py 0.2.0 "New features added"

# メジャーリリース（破壊的変更）
python scripts/release.py 1.0.0 "Major release with breaking changes"

# ドライラン（実際の変更なし）
python scripts/release.py 0.1.5 --dry-run
```

### 方法2: 手動リリース

```bash
# 1. pyproject.tomlのバージョンを更新
# version = "0.1.5"

# 2. 変更をコミット
git add pyproject.toml
git commit -m "Bump version to 0.1.5"

# 3. タグ作成
git tag -a v0.1.5 -m "Release version 0.1.5"

# 4. タグをプッシュ（GitHub Actionsトリガー）
git push origin v0.1.5
```

---

## 🔧 自動化システム

### GitHub Actions

**タグプッシュ時の自動処理**:
1. ✅ バージョン整合性チェック
2. 🏗️ パッケージビルド
3. 📝 GitHub Releaseを自動作成
4. 📦 PyPIへ自動公開

### バージョン整合性チェック

```yaml
# .github/workflows/release.yml
- name: Verify version consistency
  run: |
    TAG_VERSION=${GITHUB_REF#refs/tags/v}
    PYPROJECT_VERSION=$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])")
    if [ "$TAG_VERSION" != "$PYPROJECT_VERSION" ]; then
      echo "❌ バージョン不一致"
      exit 1
    fi
```

---

## 📋 セマンティックバージョニング

### バージョン番号の意味

```
v1.2.3
│ │ │
│ │ └─ PATCH: バグ修正
│ └─── MINOR: 新機能追加（後方互換性あり）
└───── MAJOR: 破壊的変更
```

### 具体例

| 変更内容 | 現在 | 新バージョン | 種類 |
|---------|------|-------------|------|
| バグ修正 | 0.1.4 | 0.1.5 | PATCH |
| 新機能追加 | 0.1.5 | 0.2.0 | MINOR |
| API変更 | 0.2.0 | 1.0.0 | MAJOR |

---

## 🛠️ 開発ワークフロー

### 1. 日常開発

```bash
# 開発ブランチで作業
git checkout -b feature/new-feature
# ... 開発 ...
git commit -m "Add new feature"
git push origin feature/new-feature
```

### 2. リリース準備

```bash
# mainブランチに戻る
git checkout main
git pull origin main

# CHANGELOGを更新
# CHANGELOG.md の [Unreleased] セクションを更新

# リリース実行
python scripts/release.py 0.1.5 "Release description"
```

### 3. リリース後確認

```bash
# GitHub Actionsの実行確認
# https://github.com/owner/repo/actions

# PyPIの確認
# https://pypi.org/project/rovibrational-excitation/

# インストールテスト
pip install rovibrational-excitation==0.1.5
```

---

## 📊 バージョン管理コマンド

### 現在のバージョン確認

```bash
# pyproject.tomlから
python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"

# インストール済みパッケージから
python -c "import rovibrational_excitation; print(rovibrational_excitation.__version__)"

# Gitタグから
git describe --tags --abbrev=0
```

### バージョン履歴

```bash
# 全タグ表示
git tag -l

# 最新タグの詳細
git show $(git describe --tags --abbrev=0)

# CHANGELOGの確認
cat CHANGELOG.md
```

---

## 🔍 トラブルシューティング

### よくある問題と解決方法

#### 1. バージョン不整合エラー

**問題**: タグとpyproject.tomlのバージョンが一致しない

```bash
❌ バージョン不一致: Tag=0.1.5, pyproject.toml=0.1.4
```

**解決方法**:
```bash
# pyproject.tomlを修正
python scripts/release.py 0.1.5 --dry-run  # 確認
# 正しいバージョンで再実行
```

#### 2. タグが既に存在する

**問題**: 同じタグが既に存在

```bash
fatal: tag 'v0.1.5' already exists
```

**解決方法**:
```bash
# 既存タグを削除（注意）
git tag -d v0.1.5
git push origin :refs/tags/v0.1.5

# または新しいバージョン番号を使用
python scripts/release.py 0.1.6 "Fix version conflict"
```

#### 3. GitHub Actions失敗

**問題**: 自動リリースが失敗

**確認事項**:
- [ ] PYPI_API_TOKEN の設定
- [ ] リポジトリの権限設定
- [ ] ワークフローファイルの構文

**解決方法**:
```bash
# ローカルでビルドテスト
python -m build
twine check dist/*

# 手動でPyPIアップロード
twine upload dist/* --verbose
```

#### 4. 権限エラー

**問題**: GitHubへのプッシュ権限がない

**解決方法**:
```bash
# SSH設定の確認
ssh -T git@github.com

# HTTPSの場合はトークン確認
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

---

## 📚 参考資料

### 公式ドキュメント

- [Semantic Versioning](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)
- [PyPA Build](https://build.pypa.io/)
- [GitHub Actions](https://docs.github.com/en/actions)

### プロジェクト内リンク

- [CHANGELOG.md](../CHANGELOG.md) - 変更履歴
- [pyproject.toml](../pyproject.toml) - パッケージ設定
- [.github/workflows/release.yml](../.github/workflows/release.yml) - リリースワークフロー
- [scripts/release.py](../scripts/release.py) - リリーススクリプト

### ベストプラクティス

1. **頻繁な小さなリリース** より良い
2. **CHANGELOG.md** を常に最新に保つ
3. **ドライラン** で事前確認
4. **テスト** 後にリリース
5. **バックアップ** の確保

---

## ✅ チェックリスト

### リリース前チェック

- [ ] 全テストがパス
- [ ] CHANGELOG.md が更新済み
- [ ] ドキュメントが最新
- [ ] バージョン番号が適切
- [ ] 破壊的変更がある場合はマイグレーションガイド作成

### リリース後チェック

- [ ] GitHub Release が作成されている
- [ ] PyPI に公開されている
- [ ] インストールテストが成功
- [ ] ドキュメントサイトが更新されている（該当する場合）

### 緊急リリース時

- [ ] 問題の特定と修正
- [ ] 最小限の変更でパッチバージョンアップ
- [ ] 迅速なテストと検証
- [ ] ユーザーへの適切な通知

---

## 💡 よくある質問（FAQ）

### Q: 開発版のバージョンはどうする？

A: `0.1.5.dev0` や `0.1.5-alpha.1` などのプレリリース版を使用してください。

### Q: 複数のメンテナが同時にリリースしたら？

A: Git の競合解決と同様に、最初にマージされたものが優先されます。

### Q: PyPI にアップロードできない場合は？

A: PYPI_API_TOKEN の設定確認と、パッケージ名の重複チェックを行ってください。

### Q: バージョンを戻したい場合は？

A: 新しいバージョンでロールバックすることを推奨します（例: 0.1.5 で問題があれば 0.1.6 で修正）。

---

このガイドに関する質問や改善提案は、GitHub Issues でお知らせください。 