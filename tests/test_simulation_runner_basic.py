"""
Basic tests for rovibrational_excitation.simulation.runner module

Focus on core functionality with reliable tests:
- CheckpointManager basic operations
- JSON serialization
- Parameter expansion
- Basic utility functions
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from rovibrational_excitation.simulation.runner import (
    CheckpointManager,
    _deserialize_pol,
    _expand_cases,
    _json_safe,
)


class TestCheckpointManager:
    """CheckpointManager クラスの基本テスト"""

    def test_initialization(self):
        """CheckpointManager の初期化テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir)
            manager = CheckpointManager(root_dir)

            assert manager.root_dir == root_dir
            assert manager.checkpoint_file == root_dir / "checkpoint.json"
            assert manager.failed_cases_file == root_dir / "failed_cases.json"

    def test_case_hash_consistency(self):
        """ケースハッシュの一貫性テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(Path(temp_dir))

            case1 = {"V_max": 5, "J_max": 3, "amplitude": 0.1}
            case2 = {"V_max": 5, "J_max": 3, "amplitude": 0.1}
            case3 = {"V_max": 5, "J_max": 3, "amplitude": 0.2}

            hash1 = manager._case_hash(case1)
            hash2 = manager._case_hash(case2)
            hash3 = manager._case_hash(case3)

            # 同じパラメータは同じハッシュ
            assert hash1 == hash2
            # 異なるパラメータは異なるハッシュ
            assert hash1 != hash3

    def test_checkpoint_save_load(self):
        """チェックポイント保存・読み込みテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(Path(temp_dir))

            completed_cases = [{"V_max": 5, "amplitude": 0.1}]
            failed_cases = []
            total_cases = 3
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
            assert checkpoint["completed_cases"] == 1

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
    """_json_safe 関数のテスト"""

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
        assert abs(result - 3.14) < 1e-10

    def test_nested_structures(self):
        """ネストした構造のJSON変換テスト"""
        data = {"complex": 1 + 2j, "array": np.array([1, 2]), "simple": "text"}

        result = _json_safe(data)

        assert result["complex"]["__complex__"] is True
        assert result["array"] == [1, 2]
        assert result["simple"] == "text"

    def test_basic_types(self):
        """基本型のJSON変換テスト"""
        data = {
            "string": "hello",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "none": None,
            "list": [1, 2, 3],
        }

        result = _json_safe(data)

        # 基本型はそのまま
        assert result == data


class TestDeserializePol:
    """_deserialize_pol 関数のテスト"""

    def test_single_value(self):
        """単一値の偏光変換テスト"""
        result = _deserialize_pol(1.0)
        expected = np.array([1.0, 0.0], dtype=complex)
        np.testing.assert_array_equal(result, expected)

    def test_two_element_list(self):
        """2要素リストの偏光変換テスト"""
        result = _deserialize_pol([1.0, 2.0])
        expected = np.array([1.0, 2.0], dtype=complex)
        np.testing.assert_array_equal(result, expected)

    def test_single_element_list(self):
        """単一要素リストの偏光変換テスト"""
        result = _deserialize_pol([1.5])
        expected = np.array([1.5, 0.0], dtype=complex)
        np.testing.assert_array_equal(result, expected)

    def test_complex_dict(self):
        """複素数辞書の偏光変換テスト"""
        pol_dict = {"r": 0.5, "i": 0.8}
        result = _deserialize_pol([pol_dict])

        expected = np.array([0.5 + 0.8j, 0.0], dtype=complex)
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
        base = {"V_max": [3, 5], "amplitude": 0.1}

        cases = list(_expand_cases(base))

        assert len(cases) == 2
        v_max_values = [case["V_max"] for case, _ in cases]
        assert 3 in v_max_values
        assert 5 in v_max_values

        # 全てのケースでamplitudeは固定
        for case, sweep_keys in cases:
            assert case["amplitude"] == 0.1
            assert sweep_keys == ["V_max"]

    def test_multiple_sweep(self):
        """複数パラメータスイープテスト"""
        base = {"V_max": [3, 5], "amplitude": [0.1, 0.2], "J_max": 2}

        cases = list(_expand_cases(base))

        # 2 × 2 = 4ケース
        assert len(cases) == 4

        # 全ケースでJ_maxは固定
        for case, sweep_keys in cases:
            assert case["J_max"] == 2
            # V_maxとamplitudeがスイープ対象
            assert "V_max" in sweep_keys
            assert "amplitude" in sweep_keys


class TestRunnerUtilities:
    """Runner ユーティリティ関数のテスト"""

    def test_json_roundtrip(self):
        """JSON変換の往復テスト"""
        original = {
            "complex": 1 + 2j,
            "array": np.array([1.0, 2.0, 3.0]),
            "scalar": np.float32(3.14),
            "nested": {"inner": 5 + 6j},
        }

        # JSON変換
        json_safe = _json_safe(original)

        # JSON文字列化→復元が可能であることを確認
        json_str = json.dumps(json_safe)
        restored = json.loads(json_str)

        # 基本構造が保持されていることを確認
        assert "complex" in restored
        assert "array" in restored
        assert "scalar" in restored
        assert "nested" in restored

    def test_polarization_roundtrip(self):
        """偏光データの往復テスト"""
        original_pols = [
            1.0,
            [1.0, 0.5],
            [{"r": 0.5, "i": 0.8}],
        ]

        for pol in original_pols:
            result = _deserialize_pol(pol)
            assert result.shape == (2,)
            assert result.dtype == complex

    def test_case_expansion_completeness(self):
        """ケース展開の完全性テスト"""
        base = {"V_max": [2, 3, 4], "amplitude": [0.1, 0.2], "fixed_param": "constant"}

        cases = list(_expand_cases(base))

        # 3 × 2 = 6ケース
        assert len(cases) == 6

        # 全ての組み合わせが含まれることを確認
        v_max_values = set()
        amplitude_values = set()

        for case, sweep_keys in cases:
            v_max_values.add(case["V_max"])
            amplitude_values.add(case["amplitude"])
            assert case["fixed_param"] == "constant"
            assert set(sweep_keys) == {"V_max", "amplitude"}

        assert v_max_values == {2, 3, 4}
        assert amplitude_values == {0.1, 0.2}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
