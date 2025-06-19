import sys
import os
import tempfile
import shutil
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import numpy as np
import pytest
from rovibrational_excitation.simulation import runner

def test_json_safe_basic():
    # 複素数
    c = 1+2j
    safe = runner._json_safe(c)
    assert safe["__complex__"] and safe["r"] == 1 and safe["i"] == 2
    # ndarray
    arr = np.array([1,2,3])
    safe_arr = runner._json_safe(arr)
    assert safe_arr == [1,2,3]
    # dict, list
    d = {"a": 1+1j, "b": [2+2j, 3]}
    safe_d = runner._json_safe(d)
    assert "a" in safe_d and "b" in safe_d

def test_expand_cases():
    # sweepなし
    base = {"a": 1, "b": "x"}
    cases = list(runner._expand_cases(base))
    assert len(cases) == 1
    # sweepあり
    base2 = {"a": [1,2], "b": "x"}
    cases2 = list(runner._expand_cases(base2))
    assert len(cases2) == 2

def test_run_all_dryrun():
    params = {"description": "test", "a": [1,2], "b": "x"}
    # dry_run=Trueで実行（ファイル出力なし）
    runner.run_all(params, dry_run=True, save=False) 

# --- ファイル出力まで確認するテスト ---
def test_run_all_file_output():
    # 物理計算を伴わない最小限のパラメータ
    params = {
        "description": "testfile",
        "t_start": 0.0,
        "t_end": 0.1,
        "dt": 0.1,
        "duration": 0.1,
        "t_center": 0.05,
        "carrier_freq": 0.0,
        "amplitude": 1.0,
        "polarization": [1.0, 0.0],
        "V_max": 0,
        "J_max": 0,
        "omega_rad_phz": 1.0,
        "mu0_Cm": 1.0,
        "initial_states": [0],
        "outdir": None,  # runner側で自動生成
        "save": True
    }
    # 一時ディレクトリをresultsに見立てる
    with tempfile.TemporaryDirectory() as tmpdir:
        cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            results = runner.run_all(params, nproc=1, save=True, dry_run=False)
            # 出力ディレクトリを探索
            found = False
            for root, dirs, files in os.walk("results"):
                if "result.npz" in files and "parameters.json" in files:
                    found = True
                    break
            assert found, "result.npzとparameters.jsonが出力されていること"
        finally:
            os.chdir(cwd) 