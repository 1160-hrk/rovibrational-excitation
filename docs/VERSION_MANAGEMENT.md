# バージョン管理とリリース

Last verified: 2026-09-30

## 原則

- pyproject.toml の project.version が唯一のパッケージバージョンです。
- 公開タグは最終版だけを受理し、形式は vX.Y.Z です。
- .dev、alpha、beta、rc等のタグは公開workflowで拒否します。
- ローカルスクリプトはcommit、tag、push、publishを行いません。
- 公開前にCPU品質ゲートと実CUDAゲートの両方が必須です。
- PyPI公開成功後にだけGitHub Releaseを作成します。

現在の開発版は 0.3.0.dev1 です。正式版へ進む場合の対象は 0.3.0
ですが、Phase 8の受入条件を満たすまではタグを作成しません。

## ローカル準備

### 読み取り専用の確認

~~~bash
python scripts/release.py 0.3.0 --dry-run
~~~

このコマンドは現在版から対象版への遷移だけを検証し、ファイル、Git、
タグ、外部サービスを変更しません。

### バージョン更新とローカルゲート

cleanなworktreeで次を実行します。

~~~bash
python scripts/release.py 0.3.0 --apply
~~~

applyは次を順に行います。

1. worktreeがcleanであることを確認する。
2. pyproject.tomlのversionだけを更新する。
3. Ruff lint/format、strict mypy、全pytestを実行する。
4. 3個のsupported exampleとparams_template.pyを実行する。
5. sdist/wheelをbuildし、Twineで検証する。
6. 失敗時はpyproject.tomlを元の内容へ戻す。

成功してもrelease受入完了ではなく、commitやtagも作りません。完了表示は
pyproject.tomlとCHANGELOGの明示的レビュー・commitに加え、通常CI、accepted
real-CUDA evidence、hosted container smoke、Dev Containers UI、PyPI Trusted
Publisherの全条件をtag前に要求します。

## タグを作成する前の条件

- Phase 8のAPI、README、migration note、backend matrixが確定している。
- CHANGELOGが対象版を説明している。
- worktreeがcleanで、通常CIのrequired jobが成功している。
- self-hosted GPU runnerに labels
  [self-hosted, linux, x64, gpu] が設定され、Node 24 Actionsに必要なrunner
  version 2.327.1以上である。
- `gpu` extra が `cupy-cuda12x[ctk]` を介してCUDA 12 user-space runtime・
  header・libraryを導入し、runner hostは互換NVIDIA driverだけを提供する。
  WSL2ではLinux NVIDIA driverを追加しない。
- `.github/workflows/cuda-validation.yml` をrelease候補commitに対して手動実行し、
  schema-v1の `status: pass` artifactをレビューしている。
- 通常CIの `container-smoke` が同じcommitで成功し、VS Code Dev Containersの
  attachとPorts viewを手動確認している。
- GitHubのprotected environment `pypi`を設定し、PyPI Trusted Publisherに
  owner `1160-hrk`、repository `rovibrational-excitation`、workflow
  `release.yml`、environment `pypi`を完全一致で登録している。API token secretは
  使用しない。
- 対象版の成果物がPyPIにまだ存在しない。

ローカルでレビュー済みのversion commitをpushした後、明示的にannotated tagを
作成してpushします。タグ操作はrelease.pyの責務ではありません。

~~~bash
git tag -a v0.3.0 -m "Release 0.3.0"
git push origin v0.3.0
~~~

公開済みタグやPyPI版は上書き・削除せず、新しいpatch版で訂正します。

## GitHub Actionsの公開ゲート

.github/workflows/release.yml は v* pushで開始しますが、以下のすべてを
通らなければ公開しません。

1. タグとpyproject.tomlが同じ最終X.Y.Z版であること。
2. Ruff、mypy、全CPUテスト、supported example/template smoke。
3. 実CUDA deviceの存在確認。
4. trustedなNumPy/CuPy parity referenceと全gpu marker test。
5. 5経路の数値・backend・転送量・同期済み時間を記録するschema-v1 CUDA証拠。
6. clean build、非root package import、認証付きJupyter HTTPを行うcontainer検証。
7. sdist/wheel build、Twine検査、clean wheel install/import。
8. PyPI公開。
9. PyPI成功後、CUDA証拠JSONを添付したGitHub Release作成。

GPU jobはself-hosted GPU runner専用です。runnerがない、またはversionが2.327.1
未満の場合にskipやCPU fallbackはせず、releaseは待機または失敗します。通常CPU
CIでのGPU skipはリリース
証拠にはなりません。タグ前の同じ検証はActions画面または次で明示実行します。

~~~bash
gh workflow run cuda-validation.yml --ref refactor/v0.3
~~~

生成された `real-cuda-evidence-<commit>` artifactのsource commit、device、
`status`、全5 caseを確認します。タグworkflowは同じレコーダーを再実行する
ため、事前artifactだけで公開gateを省略することはありません。

## 失敗時

### ローカルapplyが失敗した

スクリプトはpyproject.tomlを復元します。表示された最初の失敗ゲートを修正し、
cleanなworktreeから再実行します。distはignoredな生成物であり、公開入力には
GitHub Actionsがタグcommitから再生成した成果物だけを使います。

### タグとversionが一致しない

release workflowは外部公開前に停止します。既にpushした誤タグをそのまま
使わず、状況を確認してから新しい正しいリリース手順を決めます。公開操作を
自動的にやり直すスクリプトはありません。

### GPU jobが開始しない

[self-hosted, linux, x64, gpu] の全labelを持つrunnerがonlineか確認します。
GPU gateを削除したりskip扱いにして公開してはいけません。

### PyPI公開が失敗した

GitHub Releaseはまだ作成されません。protected environment、PyPI Trusted
Publisherのowner/repository/workflow/environment完全一致、OIDC permission、既存
versionを確認します。API token fallbackは追加しません。同じversionが既に公開済み
なら再アップロードせず、新しいversionを用意します。

## バージョン確認

~~~bash
python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"
python -c "import rovibrational_excitation; print(rovibrational_excitation.__version__)"
git tag -l
~~~

## 関連ファイル

- [v0.3 migration guide](MIGRATION_V0_3.md)
- [CHANGELOG.md](../CHANGELOG.md)
- [pyproject.toml](../pyproject.toml)
- [release workflow](../.github/workflows/release.yml)
- [PyPI OIDC configuration](https://docs.github.com/en/actions/how-tos/secure-your-work/security-harden-deployments/oidc-in-pypi)
- [manual real-CUDA workflow](../.github/workflows/cuda-validation.yml)
- [real-CUDA evidence recorder](../benchmarks/run_cuda_evidence.py)
- [container smoke](../scripts/smoke_container.sh)
- [local release preparation](../scripts/release.py)
- [documentation/workflow audit](refactoring/DOCUMENTATION_WORKFLOW_AUDIT.md)
- [Phase 8 release-readiness audit](refactoring/PHASE8_RELEASE_READINESS_AUDIT.md)
