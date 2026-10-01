#!/usr/bin/env python
"""
scripts/check_runner.py
=======================
* examples/params_CO2_antisymm.py を読み込み
* simulation.runner を呼び出して 1 ケースだけ実行
* 結果フォルダ／出力ファイルの有無をチェック
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from rovibrational_excitation.simulation import runner
# %%
# ---------------------------------------------------------------
# 対象パラメータファイル（絶対パスに解決）
PARAM_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "params_CO2_AntiSymm.py"))

# %%
# ---------------------------------------------------------------

tmpdir = tempfile.mkdtemp()
tmp_param = os.path.join(tmpdir, "param.py")

# ---------------------------------------------------------------
# 実行

t0 = time.perf_counter()
results = runner.run_all(PARAM_FILE, save=True)
dt = time.perf_counter() - t0

print(f"✔  runner.run_all finished in {dt:.2f} s")
