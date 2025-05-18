import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

#!/usr/bin/env python
"""
scripts/check_runner.py
=======================
* examples/params_CO2_antisymm.py を読み込み
* simulation.runner を呼び出して 1 ケースだけ実行
* 結果フォルダ／出力ファイルの有無をチェック
"""

from __future__ import annotations
import os, sys, shutil, tempfile, time, importlib.util
from rovibrational_excitation.simulation import runner

import numpy as np
# %%
# ---------------------------------------------------------------
# 対象パラメータファイル
PARAM_FILE = os.path.join(
    os.path.abspath(os.path.join(os.path.join(os.pardir, os.pardir), "examples")),
    "params_CO2_AntiSymm.py"
    )

params = runner._load_params(PARAM_FILE)
p_dict = runner._serializable_params(params)

root = runner._make_root(params.description)
shutil.copy(PARAM_FILE, os.path.join(root, "params.py"))
params_iter, params_not_iter = runner._params_iter_not_iter(p_dict)
cases_dict, paths = runner._case_paths(root, params_iter)
inputs = [di.update(**params_not_iter, outdir=path) for di, path in zip(cases_dict, paths)]

# %%
# ---------------------------------------------------------------
# 一時コピーを作り，gauss_widths / polarizations / delays を 1 要素に絞る
tmpdir = tempfile.mkdtemp()
tmp_param = os.path.join(tmpdir, "param.py")

with open(PARAM_FILE, "r", encoding="utf-8") as src, \
     open(tmp_param, "w", encoding="utf-8") as dst:
    code = src.read()
    # ↓ リストを 1 つだけ残す
    code = code.replace("gauss_widths = [50.0, 80.0]",
                        "gauss_widths = [50.0]")
    code = code.replace("delays = [0.0, 100.0, 200.0]",
                        "delays = [0.0]")
    dst.write(code)

print("✔  Copied param file to", tmp_param)

# ---------------------------------------------------------------
# 実行

t0 = time.perf_counter()
runner.run_all(tmp_param, parallel=False)
dt = time.perf_counter() - t0

print(f"✔  runner.run_all finished in {dt:.2f} s")

# ---------------------------------------------------------------
# 最新の results フォルダを取得
results_root = sorted(
    (os.path.join("results", d) for d in os.listdir("results")),
    key=os.path.getmtime
)[-1]

expect_files = [
    "summary.csv",
    "duration_50.0/polarization_[1, 0]/t_center_0.0/rho_t.npy",
]

missing = [f for f in expect_files if not os.path.exists(os.path.join(results_root, f))]
if missing:
    print("✗  Missing files:", missing)
    sys.exit(1)

rho = np.load(os.path.join(results_root, expect_files[1]))
print("✔  rho_t.npy shape:", rho.shape)
print("\nAll runner checks passed!")

# ---------------------------------------------------------------
# 後始末（tempfile と生成フォルダを残したい場合はコメントアウト）
shutil.rmtree(tmpdir)
