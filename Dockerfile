# ───────────── dev stage ─────────────
FROM python:3.12-slim

# 作業ディレクトリの設定
WORKDIR /workspace

# システムパッケージの更新とビルドツールのインストール
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      python3-dev \
      git \
      curl \
      vim \
      sudo \
 && rm -rf /var/lib/apt/lists/*

# pipのアップグレード
RUN pip install --upgrade pip setuptools wheel

# 依存関係ファイルをコピー
COPY requirements.txt requirements-dev.txt ./

# Python依存関係のインストール（開発環境用）
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -r requirements-dev.txt

# 開発ユーザーの作成（ファイル編集権限のため）
RUN useradd -m -s /bin/bash -G sudo devuser \
 && echo "devuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Jupyter設定ディレクトリの作成
RUN mkdir -p /home/devuser/.jupyter

# Jupyter設定ファイルの作成
RUN echo "c.ServerApp.ip = '0.0.0.0'" > /home/devuser/.jupyter/jupyter_server_config.py \
 && echo "c.ServerApp.port = 8888" >> /home/devuser/.jupyter/jupyter_server_config.py \
 && echo "c.ServerApp.open_browser = False" >> /home/devuser/.jupyter/jupyter_server_config.py \
 && echo "c.ServerApp.allow_root = True" >> /home/devuser/.jupyter/jupyter_server_config.py \
 && echo "c.ServerApp.token = ''" >> /home/devuser/.jupyter/jupyter_server_config.py \
 && chown -R devuser:devuser /home/devuser/.jupyter

# ワークスペースの権限設定
RUN chown -R devuser:devuser /workspace

# ユーザーの切り替え
USER devuser

# Jupyterポートの露出
EXPOSE 8888

# 起動維持（開発環境用）
ENTRYPOINT ["tail", "-f", "/dev/null"]
