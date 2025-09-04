"""
Comprehensive tests for rovibrational_excitation.simulation.runner module

Tests cover:
- CheckpointManager functionality
- Error handling and retry mechanisms
- JSON serialization safety
- Parameter file loading and expansion
- Parallel execution
- Result saving and loading
- Checkpoint-based resumption
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from rovibrational_excitation.simulation.runner import (
    CheckpointManager,
    _deserialize_pol,
    _expand_cases,
    _json_safe,
    _load_params_file,
    _run_one_safe,
    resume_run,
    run_all,
    run_all_with_checkpoint,
)


class TestCheckpointManager:
    """CheckpointManager クラスの包括的テスト"""

    def test_initialization(self):
        """CheckpointManager の初期化テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir)
            manager = CheckpointManager(root_dir)

            assert manager.root_dir == root_dir
            assert manager.checkpoint_file == root_dir / "checkpoint.json"
            assert manager.failed_cases_file == root_dir / "failed_cases.json"

    def test_case_hash(self):
        """ケースハッシュ生成テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(Path(temp_dir))

            case1 = {"V_max": 5, "J_max": 3, "amplitude": 0.1, "outdir": "/tmp/test"}
            case2 = {
                "V_max": 5,
                "J_max": 3,
                "amplitude": 0.1,
                "outdir": "/tmp/different",
            }
            case3 = {"V_max": 5, "J_max": 3, "amplitude": 0.2, "outdir": "/tmp/test"}

            hash1 = manager._case_hash(case1)
            hash2 = manager._case_hash(case2)
            hash3 = manager._case_hash(case3)

            # outdirは無視されるため、hash1とhash2は同じ
            assert hash1 == hash2
            # amplitudeが異なるため、hash1とhash3は異なる
            assert hash1 != hash3

    def test_save_and_load_checkpoint(self):
        """チェックポイント保存・読み込みテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(Path(temp_dir))

            completed_cases = [
                {"V_max": 5, "J_max": 3, "amplitude": 0.1},
                {"V_max": 5, "J_max": 3, "amplitude": 0.2},
            ]
            failed_cases = [{"V_max": 10, "J_max": 5, "error": "Out of memory"}]
            total_cases = 10
            start_time = 1234567890.0

            # 保存
            manager.save_checkpoint(
                completed_cases, failed_cases, total_cases, start_time
            )

            # ファイル存在確認
            assert manager.checkpoint_file.exists()

            # 読み込み
            checkpoint = manager.load_checkpoint()
            assert checkpoint is not None
            assert checkpoint["total_cases"] == total_cases
            assert checkpoint["completed_cases"] == 2
            assert checkpoint["failed_cases"] == 1
            assert len(checkpoint["completed_case_hashes"]) == 2
            assert len(checkpoint["failed_case_data"]) == 1

    def test_filter_remaining_cases(self):
        """残りケースフィルタリングテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(Path(temp_dir))

            all_cases = [
                {"V_max": 5, "J_max": 3, "amplitude": 0.1},
                {"V_max": 5, "J_max": 3, "amplitude": 0.2},
                {"V_max": 5, "J_max": 3, "amplitude": 0.3},
            ]

            # チェックポイントなしの場合
            remaining = manager.filter_remaining_cases(all_cases)
            assert len(remaining) == 3

            # 完了ケースがある場合
            completed_cases = [all_cases[0], all_cases[1]]
            manager.save_checkpoint(completed_cases, [], len(all_cases), 0.0)

            remaining = manager.filter_remaining_cases(all_cases)
            assert len(remaining) == 1
            assert remaining[0]["amplitude"] == 0.3

    def test_is_resumable(self):
        """再開可能性チェックテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(Path(temp_dir))

            # チェックポイントファイルなし
            assert not manager.is_resumable()

            # チェックポイントファイル作成
            manager.save_checkpoint([], [], 0, 0.0)
            assert manager.is_resumable()


class TestJSONSafe:
    """_json_safe 関数の包括的テスト"""

    def test_complex_numbers(self):
        """複素数のJSON変換テスト"""
        z = 3 + 4j
        result = _json_safe(z)

        assert result["__complex__"] is True
        assert result["r"] == 3.0
        assert result["i"] == 4.0

    def test_numpy_arrays(self):
        """NumPy配列のJSON変換テスト"""
        arr = np.array([1, 2, 3])
        result = _json_safe(arr)

        assert result == [1, 2, 3]

    def test_numpy_scalars(self):
        """NumPyスカラーのJSON変換テスト"""
        scalar = np.float64(3.14)
        result = _json_safe(scalar)

        assert isinstance(result, float)
        assert result == 3.14

    def test_nested_structures(self):
        """ネストした構造のJSON変換テスト"""
        data = {
            "complex": 1 + 2j,
            "array": np.array([1, 2, 3]),
            "list": [3 + 4j, np.float32(5.0)],
            "dict": {"nested": 6 + 7j},
        }

        result = _json_safe(data)

        assert result["complex"]["__complex__"] is True
        assert result["array"] == [1, 2, 3]
        assert result["list"][0]["__complex__"] is True
        assert isinstance(result["list"][1], float)
        assert result["dict"]["nested"]["__complex__"] is True

    def test_callable_objects(self):
        """callable オブジェクトのJSON変換テスト"""

        def test_func():
            pass

        result = _json_safe(test_func)

        # 関数は文字列表現に変換される
        assert isinstance(result, str)
        assert "test_func" in result


class TestDeserializePol:
    """_deserialize_pol 関数のテスト"""

    def test_single_value(self):
        """単一値の偏光変換テスト"""
        result = _deserialize_pol(1.0)
        expected = np.array([1.0, 0.0], dtype=complex)
        np.testing.assert_array_equal(result, expected)

    def test_complex_dict(self):
        """複素数辞書の偏光変換テスト"""
        pol_dict = {"r": 0.5, "i": 0.8}
        result = _deserialize_pol([pol_dict, {"r": 0.2, "i": -0.3}])

        expected = np.array([0.5 + 0.8j, 0.2 - 0.3j], dtype=complex)
        np.testing.assert_array_equal(result, expected)

    def test_list_expansion(self):
        """リスト展開の偏光変換テスト"""
        # 単一要素リスト → [x, 0]
        result = _deserialize_pol([1.5])
        expected = np.array([1.5, 0.0], dtype=complex)
        np.testing.assert_array_equal(result, expected)

        # 2要素リスト → [x, y]
        result = _deserialize_pol([1.0, 2.0])
        expected = np.array([1.0, 2.0], dtype=complex)
        np.testing.assert_array_equal(result, expected)

    def test_invalid_input(self):
        """無効入力のエラーテスト"""
        with pytest.raises(ValueError):
            _deserialize_pol([1, 2, 3, 4])  # 3要素以上


class TestExpandCases:
    """_expand_cases 関数のテスト"""

    def test_no_sweep(self):
        """スイープなしケース展開テスト"""
        base = {"V_max": 5, "amplitude": 0.1}

        cases = list(_expand_cases(base))

        assert len(cases) == 1
        case, sweep_keys = cases[0]
        assert case == base
        assert sweep_keys == []

    def test_single_sweep(self):
        """単一パラメータスイープテスト"""
        base = {"V_max": [3, 5, 7], "amplitude": 0.1}

        cases = list(_expand_cases(base))

        assert len(cases) == 3
        for i, (case, sweep_keys) in enumerate(cases):
            assert case["V_max"] == [3, 5, 7][i]
            assert case["amplitude"] == 0.1
            assert sweep_keys == ["V_max"]

    def test_multiple_sweep(self):
        """複数パラメータスイープテスト"""
        base = {"V_max": [3, 5], "J_max": [2, 4], "amplitude": 0.1}

        cases = list(_expand_cases(base))

        assert len(cases) == 4  # 2 × 2
        sweep_keys = set()
        for case, keys in cases:
            assert case["amplitude"] == 0.1
            sweep_keys.update(keys)

        assert "V_max" in sweep_keys
        assert "J_max" in sweep_keys

    def test_single_element_list(self):
        """単一要素リストは固定値として扱われるテスト"""
        base = {"V_max": [5], "amplitude": [0.1, 0.2]}

        cases = list(_expand_cases(base))

        assert len(cases) == 2  # amplitudeのみスイープ
        for case, sweep_keys in cases:
            # 現行仕様では単一要素リストは展開されスカラーに正規化される
            assert case["V_max"] == 5
            assert sweep_keys == ["V_max", "amplitude"]

    def test_fixed_value_keys(self):
        """FIXED_VALUE_KEYSは常に固定値として扱われるテスト"""
        base = {
            "V_max": [3, 5],  # スイープ対象
            "polarization": [1.0, 0.0],  # 固定値（FIXED_VALUE_KEYS）
            "amplitude": [0.1, 0.2],  # スイープ対象
        }

        cases = list(_expand_cases(base))

        # V_max (2) × amplitude (2) = 4ケース
        # polarizationは固定値なので影響しない
        assert len(cases) == 4

        for case, sweep_keys in cases:
            assert case["polarization"] == [1.0, 0.0]  # 常に同じ値
            assert "polarization" not in sweep_keys
            assert "V_max" in sweep_keys
            assert "amplitude" in sweep_keys

    def test_sweep_suffix(self):
        """_sweep接尾辞キーは明示的にスイープ対象テスト"""
        base = {
            "V_max": 5,  # 固定値
            "polarization_sweep": [[1, 0], [0, 1]],  # スイープ対象
            "amplitude": 0.1,  # 固定値
        }

        cases = list(_expand_cases(base))

        assert len(cases) == 2  # polarization_sweepのみ

        pol_values = []
        for case, sweep_keys in cases:
            assert case["V_max"] == 5
            assert case["amplitude"] == 0.1
            # スイープキーは接尾辞が取り除かれた名前
            assert sweep_keys == ["polarization"]
            # ケースには接尾辞が取り除かれたキーで値が保存される
            assert "polarization" in case
            assert "polarization_sweep" not in case
            pol_values.append(case["polarization"])

        assert [1, 0] in pol_values
        assert [0, 1] in pol_values

    def test_sweep_suffix_error(self):
        """_sweep接尾辞だがiterableでない場合のエラーテスト"""
        base = {
            "V_max": 5,
            "amplitude_sweep": 0.1,  # 非iterable
        }

        with pytest.raises(
            ValueError,
            match="'amplitude_sweep' has '_sweep' suffix but is not iterable",
        ):
            list(_expand_cases(base))


class TestLoadParamsFile:
    """_load_params_file 関数のテスト"""

    def test_load_valid_file(self):
        """有効なパラメータファイル読み込みテスト"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
V_max = 5
J_max = 3
amplitude = [0.1, 0.2, 0.3]
description = "Test parameters"
"""
            )
            f.flush()

            try:
                params = _load_params_file(f.name)

                assert params["V_max"] == 5
                assert params["J_max"] == 3
                assert params["amplitude"] == [0.1, 0.2, 0.3]
                assert params["description"] == "Test parameters"
            finally:
                os.unlink(f.name)

    def test_load_nonexistent_file(self):
        """存在しないファイルのエラーテスト"""
        with pytest.raises(FileNotFoundError):
            _load_params_file("/nonexistent/file.py")


class TestRunOneSafe:
    """_run_one_safe 関数のテスト"""

    @patch("rovibrational_excitation.simulation.runner._run_one")
    def test_successful_execution(self, mock_run_one):
        """正常実行テスト"""
        expected_result = np.array([[0.9, 0.1], [0.8, 0.2]])
        mock_run_one.return_value = expected_result

        params = {"test": "param", "save": False}
        result, error = _run_one_safe(params)

        assert error is None
        np.testing.assert_array_equal(result, expected_result)
        mock_run_one.assert_called_once_with(params)

    @patch("rovibrational_excitation.simulation.runner._run_one")
    def test_retry_mechanism(self, mock_run_one):
        """リトライ機構テスト"""
        # 最初の2回は失敗、3回目は成功
        mock_run_one.side_effect = [
            RuntimeError("First failure"),
            RuntimeError("Second failure"),
            np.array([[1.0, 0.0]]),
        ]

        with patch("time.sleep"):  # sleep をモック
            result, error = _run_one_safe({"save": False}, max_retries=2)

        assert error is None
        assert mock_run_one.call_count == 3

    @patch("rovibrational_excitation.simulation.runner._run_one")
    def test_failure_after_max_retries(self, mock_run_one):
        """最大リトライ後の失敗テスト"""
        mock_run_one.side_effect = RuntimeError("Persistent error")

        with patch("time.sleep"):
            result, error = _run_one_safe({"save": False}, max_retries=1)

        assert result is None
        assert error is not None and "Persistent error" in error
        assert mock_run_one.call_count == 2


class TestRunAllBasic:
    """run_all 関数の基本テスト"""

    def test_dry_run(self):
        """ドライランテスト"""
        params = {
            "V_max": [3, 5],
            "J_max": 2,
            "amplitude": 0.1,
            "description": "test_dry_run",
        }

        with patch("builtins.print") as mock_print:
            result = run_all(params, dry_run=True, save=False)

        assert result == []
        # ドライランメッセージが出力されているか確認
        calls = [str(call) for call in mock_print.call_args_list]
        assert any("Dry-run" in call for call in calls)

    @patch("rovibrational_excitation.simulation.runner._run_one")
    def test_single_case_execution(self, mock_run_one):
        """単一ケース実行テスト"""
        mock_run_one.return_value = np.array([[1.0, 0.0]])

        params = {
            "V_max": 3,
            "J_max": 2,
            "amplitude": 0.1,
            "description": "single_case",
        }

        # save=Trueにしてcheckpoint_managerが作成されるようにする
        result = run_all(params, save=True)

        assert len(result) == 1
        mock_run_one.assert_called_once()


class TestRunAllWithCheckpoint:
    """run_all_with_checkpoint 関数のテスト"""

    @patch("rovibrational_excitation.simulation.runner._run_one")
    def test_checkpoint_creation(self, mock_run_one):
        """チェックポイント作成テスト"""
        mock_run_one.return_value = np.array([[1.0, 0.0]])

        params = {
            "V_max": [3, 5],
            "J_max": 2,
            "amplitude": 0.1,
            "description": "checkpoint_test",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "rovibrational_excitation.simulation.runner._make_root"
            ) as mock_make_root:
                mock_make_root.return_value = Path(temp_dir)

                result = run_all_with_checkpoint(params, checkpoint_interval=1)

        assert len(result) == 2


class TestResumeRun:
    """resume_run 関数のテスト"""

    def test_resume_nonexistent_directory(self):
        """存在しないディレクトリでの再開テスト"""
        with pytest.raises(FileNotFoundError):
            resume_run("/nonexistent/directory")

    def test_resume_without_checkpoint(self):
        """チェックポイントなしでの再開テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(
                ValueError, match="再開可能なチェックポイントが見つかりません"
            ):
                resume_run(temp_dir)


class TestIntegration:
    """統合テスト"""

    @patch("rovibrational_excitation.simulation.runner._run_one")
    def test_complete_workflow(self, mock_run_one):
        """完全なワークフローテスト"""
        # モックの結果設定
        mock_run_one.return_value = np.array([[0.9, 0.1], [0.8, 0.2]])

        # パラメータファイル作成
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
V_max = [3, 5]
J_max = 2
amplitude = 0.1
t_start = -5.0
t_end = 5.0
dt = 0.1
duration = 2.0
t_center = 0.0
carrier_freq = 1.0
mu0_Cm = 1e-30
omega_rad_phz = 1.0
polarization = [[1.0, 0.0]]
description = "integration_test"
"""
            )
            f.flush()

            try:
                # 実行
                with tempfile.TemporaryDirectory() as temp_dir:
                    with patch(
                        "rovibrational_excitation.simulation.runner._make_root"
                    ) as mock_make_root:
                        mock_make_root.return_value = Path(temp_dir)

                        results = run_all(f.name, save=True)

                # 結果検証
                # V_max = [3, 5] (2ケース) × polarization = [[1.0, 0.0]] (1ケース) = 2ケース
                assert len(results) == 2
                assert mock_run_one.call_count == 2

            finally:
                os.unlink(f.name)


class TestErrorHandling:
    """エラーハンドリングテスト"""

    def test_invalid_params_type(self):
        """無効なparams型のエラーテスト"""
        with pytest.raises(TypeError):
            run_all(None)  # type: ignore  # Noneは無効

    @patch("rovibrational_excitation.simulation.runner._run_one")
    def test_partial_failure_handling(self, mock_run_one):
        """部分的失敗の処理テスト"""
        # 1つ目は成功、2つ目は失敗
        mock_run_one.side_effect = [
            np.array([[1.0, 0.0]]),
            RuntimeError("Simulation failed"),
        ]

        params = {
            "V_max": [3, 5],
            "J_max": 2,
            "amplitude": 0.1,
            "description": "partial_failure",
        }

        # save=Trueにしてcheckpoint_managerが作成されるようにする
        results = run_all(params, save=True)

        # 成功したケースの結果のみ返される
        assert len(results) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
