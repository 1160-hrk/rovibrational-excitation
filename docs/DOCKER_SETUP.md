# Dev Container 開発環境

このガイドは開発版 `0.3.0.dev1` の `Dockerfile` と
`.devcontainer/devcontainer.json` に対応する。これは開発環境であり、production
image や simulation deployment の仕様ではない。

## 構成

- base image: `python:3.12-slim`
- workspace: `/workspace`
- non-root user: `devuser`
- Python dependency source of truth: `pyproject.toml`
- installed extras: `dev`, `io`, `plot`
- interactive tools: `jupyter`, `ipykernel`
- forwarded port: 8888

Dockerfile は Jupyter の user/global configuration を作らない。bind address、port、
root directory は起動スクリプトが明示し、token/password と XSRF は Jupyter の標準
設定に任せる。

## VS Code Dev Container

前提は Docker と VS Code の Dev Containers extension である。

```bash
git clone <repository-url>
cd rovibrational-excitation
code .
```

VS Code の command palette から
`Dev Containers: Reopen in Container` を選ぶ。設定は repository root を
`/workspace` へ bind mount し、`devuser` と
`/usr/local/bin/python` を使用する。

image build は package 本体と `.[dev,io,plot]` を editable install する。host source
を bind mount した後も editable path は `/workspace` を指す。requirements file は
container dependency authority ではない。

## Jupyter Lab

container terminal で次を実行する。

```bash
./scripts/start_jupyter.sh
```

既定値は次のとおり。

- bind: `127.0.0.1`
- port: `8888`
- root directory: repository root
- browser auto-open: disabled
- authentication: Jupyter standard token/password policy
- XSRF protection: enabled

terminal に表示された token 付き URL を使う。Dockerfile、Dev Container 設定、
launcher のいずれも token/password を空にせず、wildcard CORS を設定しない。

port は明示的に変更できる。

```bash
RVE_JUPYTER_PORT=8890 ./scripts/start_jupyter.sh
```

隔離済み環境で外部 bind が必要な場合だけ host を変更する。

```bash
RVE_JUPYTER_HOST=0.0.0.0 ./scripts/start_jupyter.sh
```

この指定でも認証と XSRF は無効にならない。`0.0.0.0` を使う場合は Docker/host
firewall の公開範囲と token 管理を利用者が確認する。

## Container smoke

Docker daemon を利用できる clean checkout では、次の1コマンドで image と runtime
境界を検証する。

```bash
scripts/smoke_container.sh
```

この script は image を `--pull` 付きでbuildし、image単体から `devuser`、
`/workspace`、package import、Jupyter executableを確認する。続いてrepositoryを
`/workspace/project` へread-only bind mountし、兄弟の
`/workspace/notebooks` にJupyterだけが使用する一時writable mountを分離して、
非root processを起動する。clean checkoutに空の`notebooks/`が存在しなくても
read-only mount内にdirectoryを作らない。localhostの一時portから既知のtest token付き
`/api/contents` が成功し、tokenなしrequestがHTTP 200にならないことを確認する。
Docker CLIまたはdaemonがなければskipせず失敗する。

通常CIの `container-smoke` jobとtag releaseの `container-validation` jobは同じ
scriptを実行する。これによりDockerfileとlauncherの別々の再実装をworkflow内に
持たない。VS Code UIのattach操作とPorts view自体は自動化対象外である。

## 開発コマンド

repository root で実行する。

```bash
pytest -q
ruff check --no-fix src tests examples benchmarks scripts
ruff format --check src tests examples benchmarks scripts
mypy
python scripts/smoke_examples.py
python examples/tools/build_index.py --check
```

branch coverage は repository-local database を作らずに測定する。

```bash
coverage run --data-file=/tmp/rve-coverage \
  --source=src/rovibrational_excitation --branch -m pytest -q
coverage report --data-file=/tmp/rve-coverage --show-missing --fail-under=47
```

distribution の検証は次のとおり。

```bash
python -m build
python -m twine check dist/*
```

publish、commit、tag、push はこれらのコマンドでは行わない。final release gate は
[VERSION_MANAGEMENT.md](VERSION_MANAGEMENT.md) を参照する。

## Supported examples

```bash
python examples/launcher.py --list
python examples/launcher.py --run quickstart --quick
python -m rovibrational_excitation.cli.simulate \
  examples/params_template.py --no-save
```

`examples/archives/v0_2/` は現行例ではない。結果、coverage、build、cache は runtime
artifact であり、repository 構造の一部として作成を前提にしない。

## Troubleshooting

### Python interpreter

```bash
which python
python --version
python -c "import rovibrational_excitation; print(rovibrational_excitation.__version__)"
```

`which python` は `/usr/local/bin/python` を示す。import に失敗する場合は container
build log の editable-install step を確認し、Dev Container を rebuild する。

### Jupyter を起動できない

```bash
bash -n scripts/start_jupyter.sh
command -v jupyter
./scripts/start_jupyter.sh
```

port error の場合は `RVE_JUPYTER_PORT` を 1–65535 の整数で指定する。launcher は
repository root を script 自身の位置から解決するため、呼出し時の current directory
には依存しない。

### Port forwarding

VS Code の Ports view で 8888 の forwarding を確認する。Dev Container は
`forwardPorts` を宣言するが、Jupyter 自体は既定で localhost にのみ bind する。
外部公開が必要でない限り `RVE_JUPYTER_HOST` は変更しない。

### File ownership

通常は Dev Container の `remoteUser=devuser` と bind-mount UID 調整を使用する。
repository 全体へ再帰的な `chown` や permission 緩和を安易に行わない。

## 検証状態

この checkpoint では Dockerfile の安全既定、Dev Container JSON、launcher と
container smokeのshell syntax、repository test contracts、通常/release workflow
wiringを検証している。この実行環境には Docker CLI/daemon がないため、ここでは
clean image build とHTTP smokeを実行していない。GitHubの `container-smoke` 成功を
実行証拠とし、VS Code attachとPorts viewはrelease前に人が確認する。未実行やqueued
jobを成功扱いしない。

- [VS Code Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers)
- [Jupyter Server security](https://jupyter-server.readthedocs.io/en/latest/operators/security.html)
- [Docker build best practices](https://docs.docker.com/build/building/best-practices/)
