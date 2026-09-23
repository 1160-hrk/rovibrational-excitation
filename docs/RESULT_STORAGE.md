# 結果・checkpointの保存形式（開発版 v0.3）

この文書は現在の `refactor/v0.3` ブランチの通常シミュレーション結果と
batch checkpointを説明します。旧 README のコード例全体はまだ移行中です。

## 新しい結果ディレクトリ

```text
case_dir/
├── result_current.json
└── .result_generations/
    └── <32文字の世代ID>/
        ├── result.npz
        ├── parameters.json
        ├── result_manifest.json
        └── regime_analysis.json  # 該当するときだけ
```

一回の計算結果は一つの世代ディレクトリに完成させます。manifest v1 が
NPZ の配列名・型・形、単位、各 payload の SHA-256 を記録し、読み込み前に
検証します。最後に `result_current.json` を原子的に置き換えます。
このポインタの `publication_schema_version=1` は manifest の
`schema_version=1` やパッケージのバージョンとは別の番号です。

通常は固定名の `case_dir/result.npz` を直接開かず、検証済みの結果を
次のように取得してください。

```python
from pathlib import Path
from rovibrational_excitation.io.result_schema import load_simulation_result

saved = load_simulation_result(Path("case_dir"))
population = saved.arrays["pop"]
field_v_per_m = saved.arrays["E"]
parameters = saved.parameters
```

個々の NPZ/JSON ファイルが必要なら
`resolve_result_directory(Path("case_dir"))` で現在の世代を一度だけ
解決し、その戻り値の下のファイルを扱います。読み込み中にポインタが
切り替わっても、一度解決した世代の組を読み続けられます。

## 旧データと失敗時

ポインタのない既存の manifest v1 直置き結果は、同じ strict loader で
読み込めます。一方、旧直置き結果があるディレクトリを新形式で暗黙に
上書きすることは拒否します。必要なら今後の明示的な移行手順を用います。
ポインタが存在するのに壊れている場合は旧ファイルへ戻らず、エラーに
します。manifest のない旧結果も推測して読み込みません。

計算途中で書き込みや公開が失敗した場合、公開ポインタは旧世代を指した
ままです。完了済みの旧世代も自動削除しません。失敗・中断で未公開の世代が
残ることがあり、自動 GC や復旧はまだありません。manifest のハッシュは
ファイル間の整合性検査であり、入力した Hamiltonian や双極子などの
完全な科学的 provenance を保証するものではありません。電源断後の
ディレクトリエントリの永続性もまだ保証していません。

## checkpointの一括公開

```text
run_dir/
├── checkpoint_current.json
└── .checkpoint_generations/
    └── <32文字の世代ID>/
        ├── checkpoint.json
        └── failed_cases.json
```

checkpointと失敗ケース一覧は同じ世代へ書き終えてから、
`checkpoint_current.json`を原子的に切り替えます。途中でいずれかのJSON
またはポインタの保存に失敗した場合、以前の完全な組が選択されたままです。
旧形式の直置き`checkpoint.json`は読み込み可能で、次の正常な保存時に
世代形式へ移行します。ポインタがあるのに壊れている場合は旧ファイルへ
戻らず、保存による暗黙修復も行いません。

checkpoint payload自体はまだ厳格にversion管理・構造検証していません。
`load_checkpoint`の従来のエラー表示と`None`返却、MD5ケース識別子、
保存間隔、重複排除、再開対象判定は維持しています。MD5値はケースの重複を
判定する識別子であり、現在の入力や結果が同じことを証明するprovenanceでは
ありません。旧世代と未公開世代の自動GC、電源断時のdirectory fsync、
並行writerの調停も未実装です。

数値計算・出力配列の値は、これらの保存形式変更では変えていません。
