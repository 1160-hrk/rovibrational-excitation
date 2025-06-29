#!/usr/bin/env python3
"""
カバレッジバッジ自動更新スクリプト

使用方法:
    python scripts/update_coverage_badge.py
    
要求:
    - tests/ディレクトリに.coverageファイルが存在する
    - coverageパッケージがインストール済み
"""

import os
import sys
import re
import subprocess
from pathlib import Path

def get_coverage_percentage():
    """カバレッジ率を取得"""
    try:
        # tests/ディレクトリに移動してカバレッジを測定
        os.chdir('tests')
        result = subprocess.run(
            ['coverage', 'report', '--format=total'],
            capture_output=True, text=True, check=True
        )
        return int(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"カバレッジ測定エラー: {e}")
        print("先にテストを実行してください: cd tests && coverage run -m pytest")
        return None
    except FileNotFoundError:
        print("coverageコマンドが見つかりません。インストールしてください: pip install coverage")
        return None
    finally:
        os.chdir('..')

def get_badge_color(percentage):
    """カバレッジ率に応じたバッジの色を決定"""
    if percentage >= 90:
        return "brightgreen"
    elif percentage >= 80:
        return "green"
    elif percentage >= 70:
        return "yellowgreen"
    elif percentage >= 60:
        return "yellow"
    elif percentage >= 50:
        return "orange"
    else:
        return "red"

def update_readme_badge(percentage, color):
    """README.mdのバッジとテキストを更新"""
    readme_path = Path("README.md")
    
    if not readme_path.exists():
        print("README.mdが見つかりません")
        return False
    
    # README.mdを読み込み
    content = readme_path.read_text(encoding='utf-8')
    
    # 1. バッジを更新
    badge_pattern = r'coverage-\d+%25-\w+'
    new_badge = f'coverage-{percentage}%25-{color}'
    
    if re.search(badge_pattern, content):
        content = re.sub(badge_pattern, new_badge, content)
        print(f"✅ カバレッジバッジを {percentage}% ({color}) に更新")
    else:
        print("⚠️ カバレッジバッジが見つかりません")
    
    # 2. 本文中のカバレッジテキストを更新
    text_patterns = [
        r'\*\*\d+% code coverage\*\*',  # **XX% code coverage**
        r'with \*\*\d+% code coverage\*\*',  # with **XX% code coverage**
        r'全体カバレッジ[：:]\s*\d+%',  # 全体カバレッジ: XX%
    ]
    
    for pattern in text_patterns:
        if re.search(pattern, content):
            if 'code coverage' in pattern:
                new_text = f'**{percentage}% code coverage**'
                content = re.sub(r'\*\*\d+% code coverage\*\*', new_text, content)
                print(f"✅ 本文のカバレッジテキストを {percentage}% に更新")
            elif '全体カバレッジ' in pattern:
                new_text = f'全体カバレッジ: {percentage}%'
                content = re.sub(r'全体カバレッジ[：:]\s*\d+%', new_text, content)
                print(f"✅ 日本語カバレッジテキストを {percentage}% に更新")
    
    # ファイルに書き戻し
    readme_path.write_text(content, encoding='utf-8')
    return True

def main():
    """メイン処理"""
    print("🔍 カバレッジ率を測定中...")
    
    # カバレッジ率を取得
    percentage = get_coverage_percentage()
    if percentage is None:
        sys.exit(1)
    
    print(f"📊 現在のカバレッジ: {percentage}%")
    
    # バッジの色を決定
    color = get_badge_color(percentage)
    print(f"🎨 バッジの色: {color}")
    
    # README.mdを更新
    if update_readme_badge(percentage, color):
        print("\n✨ カバレッジバッジの更新が完了しました！")
        print(f"   新しいバッジ: coverage-{percentage}%25-{color}")
        
        # 詳細な情報を表示
        print("\n📈 カバレッジ詳細:")
        os.chdir('tests')
        subprocess.run(['coverage', 'report', '--show-missing'], check=False)
        
    else:
        print("\n❌ バッジの更新に失敗しました")
        sys.exit(1)

if __name__ == "__main__":
    main() 