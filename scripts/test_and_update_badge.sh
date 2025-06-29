#!/bin/bash
# カバレッジテスト実行 + バッジ自動更新スクリプト

set -e  # エラーで停止

echo "🧪 カバレッジテストを実行中..."
echo "=================================="

# testsディレクトリに移動
cd tests

# カバレッジ付きでテスト実行
echo "▶ pytest実行中..."
coverage run -m pytest -v

# カバレッジレポート表示
echo ""  
echo "📊 カバレッジレポート:"
echo "===================="
coverage report --show-missing

# 元のディレクトリに戻る
cd ..

# バッジを更新
echo ""
echo "🔄 カバレッジバッジを更新中..."
echo "=========================="
python scripts/update_coverage_badge.py

echo ""
echo "✅ 完了！"
echo "プロジェクトのREADME.mdのカバレッジバッジが最新の値に更新されました。" 