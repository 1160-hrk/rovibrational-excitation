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

## standalone可視化

`visualization.plot_electric_field`、`plot_electric_field_vector`、
`plot_population`のresult-directory入口も同じstrict loaderを使います。
電場は`result.npz`の`t_E`と`E`、populationは`t_p`と`pop`を使用します。
通常の電場plotは1成分scalarと2成分Cartesianを受理しますが、vector plotは
厳密に2成分Cartesianだけを受理します。populationは時間を第0軸に持つ2次元
配列でなければなりません。各データの第0軸長は対応する時間軸と一致する必要が
あります。

manifestのない旧`tlist.npy`、`Efield_real.npy`、`Efield_vector.npy`、
`population.npy`だけのディレクトリは推測して読みません。破損、未知schema、
不完全な公開世代、shape不整合は`ResultFormatError`になります。plotterは
エラーをprintしてreturnへ変換しません。図のseries、filename、全stateを描く
現在のpopulation挙動、`show()`と`savefig()`の順序はこの読込移行では変更して
いません。

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
ポインタがあるのに壊れている場合は旧ファイルへ戻らず、保存による暗黙修復も
行いません。

`checkpoint.json`は独立した`checkpoint_schema_version=1`を持ちます。
既知のfield、型、件数、case hash、`failed_cases.json`との完全一致を
読み込み時に検証します。破損・未知version・不完全な組は
`CheckpointFormatError`になり、警告表示と`None`への変換はしません。

保存時には、sweep展開後の全caseを実行順のまま正規化し、SHA-256を
`run_provenance`へ保存します。`outdir`、`save`、`error`だけは
runtime fieldとして除外します。resumeでは保存済み`params.py`から同じ
全caseを再構成し、SHA-256と各完了・失敗caseの所属を確認した後にだけ
実行済みcaseを除外します。入力値、単位、model、field、algorithm、backend、
storage、sweep構成や順序が変わっていれば、case実行前に停止します。

従来のMD5はcase単位の重複判定用として計算式を変えずに残します。
unversioned checkpointは不足している全run provenanceを推測できないため、
暗黙に読み込み・upgradeせず、明示的なmigrationまたは新しいrunを要求します。
詳しいfieldと保証範囲は
[checkpoint schema v1](refactoring/PHASE7_CHECKPOINT_SCHEMA_V1.md)を
参照してください。

このSHA-256は宣言された全caseの一致を保証するもので、package source、
依存関係、生成後のHamiltonian・双極子・電場・結果配列までhashするものでは
ありません。旧世代と未公開世代の自動GC、電源断時のdirectory fsync、
並行writerの調停も未実装です。数値計算・出力配列、checkpoint cadence、
有効な同一runのresume結果は、この保存境界変更では変えていません。
