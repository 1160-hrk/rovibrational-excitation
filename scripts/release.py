#!/usr/bin/env python3
"""
バージョン管理とリリーススクリプト
=================================
pyproject.tomlのバージョン更新とGitタグ作成を一括で行います。

使用例:
    python scripts/release.py 0.1.5 "Bug fixes and performance improvements"
    python scripts/release.py 0.2.0 --minor "New features added"
    python scripts/release.py 1.0.0 --major "Major release"
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import tomllib


def read_current_version():
    """pyproject.tomlから現在のバージョンを読み取り"""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        raise FileNotFoundError("pyproject.toml が見つかりません")

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    return data["project"]["version"]


def update_version_in_pyproject(new_version):
    """pyproject.tomlのバージョンを更新"""
    pyproject_path = Path("pyproject.toml")

    # ファイルを読み込み
    with open(pyproject_path, encoding="utf-8") as f:
        content = f.read()

    # バージョン行を置換
    pattern = r'version = "[^"]*"'
    replacement = f'version = "{new_version}"'

    new_content = re.sub(pattern, replacement, content)

    # ファイルに書き戻し
    with open(pyproject_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"✅ pyproject.toml のバージョンを {new_version} に更新しました")


def validate_version_format(version):
    """セマンティックバージョニング形式の検証"""
    pattern = r"^\d+\.\d+\.\d+(?:-[a-zA-Z0-9.-]+)?(?:\+[a-zA-Z0-9.-]+)?$"
    if not re.match(pattern, version):
        raise ValueError(f"無効なバージョン形式: {version}")


def run_git_command(cmd, capture_output=True):
    """Gitコマンドを実行"""
    try:
        result = subprocess.run(
            cmd.split(), capture_output=capture_output, text=True, check=True
        )
        return result.stdout.strip() if capture_output else None
    except subprocess.CalledProcessError as e:
        print(f"❌ Gitコマンドエラー: {e}")
        return None


def check_git_status():
    """Git作業ディレクトリの状態確認"""
    status = run_git_command("git status --porcelain")
    if status:
        print("⚠️  未コミットの変更があります:")
        print(status)
        response = input("続行しますか？ (y/N): ")
        if response.lower() != "y":
            print("中止しました")
            sys.exit(1)


def create_git_tag(version, message):
    """Gitタグを作成してプッシュ"""
    tag_name = f"v{version}"

    # タグ作成
    cmd = f"git tag -a {tag_name} -m '{message}'"
    if run_git_command(cmd, capture_output=False) is None:
        return False

    print(f"✅ タグ {tag_name} を作成しました")

    # タグをプッシュ
    if run_git_command(f"git push origin {tag_name}", capture_output=False) is None:
        return False

    print(f"✅ タグ {tag_name} をリモートにプッシュしました")
    return True


def get_version_increment_type(current_version, new_version):
    """バージョンの増分タイプを判定"""
    current_parts = [int(x) for x in current_version.split(".")]
    new_parts = [int(x) for x in new_version.split(".")]

    if new_parts[0] > current_parts[0]:
        return "major"
    elif new_parts[1] > current_parts[1]:
        return "minor"
    elif new_parts[2] > current_parts[2]:
        return "patch"
    else:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="バージョン管理とリリース")
    parser.add_argument("version", help="新しいバージョン (例: 0.1.5)")
    parser.add_argument("message", nargs="?", help="リリースメッセージ")
    parser.add_argument(
        "--dry-run", action="store_true", help="実際の変更を行わずに確認のみ"
    )
    parser.add_argument("--major", action="store_true", help="メジャーリリース")
    parser.add_argument("--minor", action="store_true", help="マイナーリリース")
    parser.add_argument("--patch", action="store_true", help="パッチリリース")

    args = parser.parse_args()

    try:
        # 現在のバージョン取得
        current_version = read_current_version()
        print(f"現在のバージョン: {current_version}")

        # 新バージョンの検証
        validate_version_format(args.version)
        print(f"新しいバージョン: {args.version}")

        # バージョンタイプ判定
        increment_type = get_version_increment_type(current_version, args.version)
        print(f"バージョン増分タイプ: {increment_type}")

        # リリースメッセージの決定
        if args.message:
            release_message = args.message
        else:
            type_messages = {
                "major": "Major release with breaking changes",
                "minor": "Minor release with new features",
                "patch": "Patch release with bug fixes",
                "unknown": "Version update",
            }
            release_message = type_messages.get(increment_type, "Release update")

        print(f"リリースメッセージ: {release_message}")

        if args.dry_run:
            print("\n🔍 ドライラン - 実際の変更は行いません")
            print(f"   pyproject.toml: {current_version} → {args.version}")
            print(f"   Git tag: v{args.version}")
            print(f"   Message: {release_message}")
            return

        # 実際の処理開始
        print("\n🚀 リリースプロセスを開始します...")

        # Git状態確認
        check_git_status()

        # pyproject.tomlのバージョン更新
        update_version_in_pyproject(args.version)

        # 変更をコミット
        run_git_command("git add pyproject.toml", capture_output=False)
        commit_msg = f"Bump version to {args.version}"
        run_git_command(f"git commit -m '{commit_msg}'", capture_output=False)
        print(f"✅ バージョン更新をコミットしました: {commit_msg}")

        # タグ作成とプッシュ
        if create_git_tag(args.version, release_message):
            print(f"\n🎉 リリース {args.version} が完了しました！")
            print("   GitHub Actions により自動的にリリースとPyPI公開が行われます")
        else:
            print("❌ タグ作成に失敗しました")
            sys.exit(1)

    except Exception as e:
        print(f"❌ エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
