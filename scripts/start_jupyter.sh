#!/bin/bash
# Jupyter Lab 起動スクリプト
# rovibrational-excitation 開発環境用

echo "🔬 Rovibrational Excitation - Jupyter Lab 起動中..."
echo "================================================"

# 作業ディレクトリの確認
if [ ! -f "pyproject.toml" ]; then
    echo "❌ エラー: プロジェクトルートディレクトリで実行してください"
    exit 1
fi

# 必要なディレクトリの作成
mkdir -p results notebooks

# Jupyter Lab設定の確認
if [ ! -d "$HOME/.jupyter" ]; then
    echo "📝 Jupyter設定ディレクトリを作成中..."
    mkdir -p $HOME/.jupyter
    
    # 基本設定ファイルの作成
    cat > $HOME/.jupyter/jupyter_lab_config.py << 'EOF'
# Jupyter Lab 設定ファイル
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True
c.ServerApp.token = ''
c.ServerApp.password = ''

# ワークスペース設定
c.ServerApp.root_dir = '/workspace'
c.ServerApp.preferred_dir = '/workspace'

# セキュリティ設定（開発環境用）
c.ServerApp.disable_check_xsrf = True
c.ServerApp.allow_origin = '*'
EOF
fi

echo "🚀 Jupyter Lab を起動中..."
echo "📍 アクセス URL: http://localhost:8888"
echo "📁 作業ディレクトリ: /workspace"
echo ""
echo "💡 使用方法:"
echo "   - examples/ フォルダで既存の例を確認"
echo "   - notebooks/ フォルダで新しいノートブックを作成"
echo "   - src/ フォルダでソースコードを編集"
echo ""
echo "🛑 終了するには Ctrl+C を押してください"
echo "================================================"

# Jupyter Lab の起動
exec jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --ServerApp.token='' \
    --ServerApp.password='' \
    --ServerApp.disable_check_xsrf=True \
    --ServerApp.allow_origin='*' 