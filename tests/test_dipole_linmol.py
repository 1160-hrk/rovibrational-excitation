import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import tempfile

import numpy as np
import pytest

from rovibrational_excitation.core.basis import LinMolBasis
from rovibrational_excitation.dipole.linmol import LinMolDipoleMatrix
from rovibrational_excitation.dipole.linmol.builder import build_mu

# CuPyが利用可能か判定
try:
    import cupy  # noqa: F401

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

try:
    import h5py  # noqa: F401

    HAS_HDF5 = True
except ImportError:
    HAS_HDF5 = False


class TestLinMolDipoleMatrix:
    """LinMolDipoleMatrixクラスのテスト"""

    def test_initialization_default(self):
        """デフォルト初期化のテスト"""
        basis = LinMolBasis(V_max=1, J_max=1, use_M=False)
        dipole = LinMolDipoleMatrix(basis)

        assert dipole.basis is basis
        assert dipole.mu0 == 1.0
        assert dipole.potential_type == "harmonic"
        assert dipole.backend == "numpy"
        assert dipole.dense
        assert len(dipole._cache) == 0

    def test_initialization_custom(self):
        """カスタム初期化のテスト"""
        basis = LinMolBasis(V_max=2, J_max=2, use_M=True)
        dipole = LinMolDipoleMatrix(
            basis, mu0=0.5, potential_type="morse", backend="numpy", dense=False
        )

        assert dipole.mu0 == 0.5
        assert dipole.potential_type == "morse"
        assert dipole.backend == "numpy"
        assert not dipole.dense

    def test_mu_x_property(self):
        """mu_xプロパティのテスト"""
        basis = LinMolBasis(V_max=1, J_max=1, use_M=False)
        dipole = LinMolDipoleMatrix(basis)

        mu_x = dipole.mu_x
        assert mu_x.shape == (basis.size(), basis.size())
        assert mu_x.dtype == np.complex128

        # キャッシュされているか確認
        assert ("x", True) in dipole._cache

        # 同じオブジェクトが返されるか確認
        mu_x2 = dipole.mu_x
        assert mu_x is mu_x2

    def test_mu_y_property(self):
        """mu_yプロパティのテスト"""
        basis = LinMolBasis(V_max=1, J_max=1, use_M=False)
        dipole = LinMolDipoleMatrix(basis)

        mu_y = dipole.mu_y
        assert mu_y.shape == (basis.size(), basis.size())
        assert mu_y.dtype == np.complex128

    def test_mu_z_property(self):
        """mu_zプロパティのテスト"""
        basis = LinMolBasis(V_max=1, J_max=1, use_M=False)
        dipole = LinMolDipoleMatrix(basis)

        mu_z = dipole.mu_z
        assert mu_z.shape == (basis.size(), basis.size())
        assert mu_z.dtype == np.complex128

    def test_mu_method_with_axis(self):
        """mu()メソッドの軸指定テスト"""
        basis = LinMolBasis(V_max=1, J_max=1, use_M=False)
        dipole = LinMolDipoleMatrix(basis)

        mu_x = dipole.mu("x")
        mu_y = dipole.mu("y")
        mu_z = dipole.mu("z")

        assert mu_x.shape == (basis.size(), basis.size())
        assert mu_y.shape == (basis.size(), basis.size())
        assert mu_z.shape == (basis.size(), basis.size())

        # use_M=False（全てM=0）の場合の物理的期待値：
        # μx, μy: ΔM=±1が必要だが、M=0のみなのでゼロ行列
        # μz: ΔM=0が許されるので非ゼロ要素を持つ
        assert np.allclose(mu_x, 0)  # μxはゼロ行列
        assert np.allclose(mu_y, 0)  # μyはゼロ行列
        assert np.any(mu_z != 0)  # μzは非ゼロ要素を持つ

    def test_mu_method_dense_override(self):
        """mu()メソッドのdense上書きテスト"""
        basis = LinMolBasis(V_max=1, J_max=1, use_M=False)
        dipole = LinMolDipoleMatrix(basis, dense=True)

        # まず通常のdense版を取得してキャッシュに登録
        dipole.mu("x", dense=True)

        # 次にsparse指定で取得
        mu_sparse = dipole.mu("x", dense=False)

        # スパース行列かどうかを確認
        try:
            import scipy.sparse as sp

            assert sp.issparse(mu_sparse)

            # キャッシュに両方の形式が保存されているか確認
            assert ("x", True) in dipole._cache  # dense
            assert ("x", False) in dipole._cache  # sparse
        except ImportError:
            pytest.skip("scipy not available")

    def test_stacked_method(self):
        """stacked()メソッドのテスト"""
        basis = LinMolBasis(V_max=1, J_max=1, use_M=False)
        dipole = LinMolDipoleMatrix(basis)

        # デフォルト（xyz順）
        stacked = dipole.stacked()
        assert stacked.shape == (3, basis.size(), basis.size())

        # カスタム順序
        stacked_xz = dipole.stacked("xz")
        assert stacked_xz.shape == (2, basis.size(), basis.size())

        # 正しい順序で積み重ねられているか確認
        np.testing.assert_array_equal(stacked[0], dipole.mu_x)
        np.testing.assert_array_equal(stacked[1], dipole.mu_y)
        np.testing.assert_array_equal(stacked[2], dipole.mu_z)

    def test_potential_types(self):
        """異なるpotential_typeのテスト（use_M=Trueで確認）"""
        # M量子数がある場合で比較（より多くの非ゼロ要素）
        basis = LinMolBasis(V_max=2, J_max=1, use_M=True)

        dipole_harm = LinMolDipoleMatrix(basis, potential_type="harmonic")
        dipole_morse = LinMolDipoleMatrix(basis, potential_type="morse")

        mu_harm = dipole_harm.mu_x
        mu_morse = dipole_morse.mu_x

        # 同じ形状だが値は異なる（M量子数がある場合）
        assert mu_harm.shape == mu_morse.shape
        # 非ゼロ要素がある場合は値が異なることが期待される
        if np.any(mu_harm != 0) and np.any(mu_morse != 0):
            assert not np.allclose(mu_harm, mu_morse)
        else:
            # 両方ともゼロ行列の場合はスキップ
            pytest.skip("Both matrices are zero, cannot compare potential types")

    def test_mu0_scaling(self):
        """mu0スケーリングのテスト"""
        basis = LinMolBasis(V_max=1, J_max=1, use_M=False)

        dipole1 = LinMolDipoleMatrix(basis, mu0=1.0)
        dipole2 = LinMolDipoleMatrix(basis, mu0=2.0)

        mu1 = dipole1.mu_x
        mu2 = dipole2.mu_x

        # mu0=2.0の場合は2倍になっているはず
        np.testing.assert_array_almost_equal(mu2, 2.0 * mu1)

    def test_different_basis_sizes(self):
        """異なる基底サイズでのテスト"""
        basis_small = LinMolBasis(V_max=1, J_max=1, use_M=False)
        basis_large = LinMolBasis(V_max=3, J_max=2, use_M=False)

        dipole_small = LinMolDipoleMatrix(basis_small)
        dipole_large = LinMolDipoleMatrix(basis_large)

        mu_small = dipole_small.mu_x
        mu_large = dipole_large.mu_x

        assert mu_small.shape == (basis_small.size(), basis_small.size())
        assert mu_large.shape == (basis_large.size(), basis_large.size())
        assert mu_large.shape[0] > mu_small.shape[0]

    def test_with_M_quantum_number(self):
        """M量子数ありの基底でのテスト"""
        basis = LinMolBasis(V_max=1, J_max=1, use_M=True)
        dipole = LinMolDipoleMatrix(basis)

        mu_x = dipole.mu_x
        mu_y = dipole.mu_y
        mu_z = dipole.mu_z

        # M量子数ありの場合のサイズ確認
        expected_size = basis.size()
        assert mu_x.shape == (expected_size, expected_size)
        assert mu_y.shape == (expected_size, expected_size)
        assert mu_z.shape == (expected_size, expected_size)

        # x, y軸の要素は複素数になることがある
        assert mu_x.dtype == np.complex128
        assert mu_y.dtype == np.complex128
        assert mu_z.dtype == np.complex128

    def test_hermiticity_properties(self):
        """エルミート性のテスト"""
        basis = LinMolBasis(V_max=2, J_max=1, use_M=True)
        dipole = LinMolDipoleMatrix(basis)

        mu_x = dipole.mu_x
        mu_y = dipole.mu_y
        mu_z = dipole.mu_z

        # 双極子行列はエルミートであるべき
        np.testing.assert_array_almost_equal(mu_x, mu_x.conj().T, decimal=10)
        np.testing.assert_array_almost_equal(mu_y, mu_y.conj().T, decimal=10)
        np.testing.assert_array_almost_equal(mu_z, mu_z.conj().T, decimal=10)

    @pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")
    def test_cupy_backend(self):
        """CuPyバックエンドのテスト"""
        basis = LinMolBasis(V_max=1, J_max=1, use_M=False)
        dipole = LinMolDipoleMatrix(basis, backend="cupy")

        mu_x = dipole.mu_x

        # CuPy配列かどうか確認
        assert hasattr(mu_x, "device")  # CuPy特有の属性
        assert mu_x.shape == (basis.size(), basis.size())

    @pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")
    def test_numpy_vs_cupy_consistency(self):
        """NumPyとCuPyの一貫性テスト"""
        basis = LinMolBasis(V_max=1, J_max=1, use_M=False)

        dipole_np = LinMolDipoleMatrix(basis, backend="numpy")
        dipole_cp = LinMolDipoleMatrix(basis, backend="cupy")

        mu_np = dipole_np.mu_x
        mu_cp = dipole_cp.mu_x

        # 結果が一致するかテスト（CuPyが利用可能な場合のみ）
        if HAS_CUPY:
            import cupy as cp_local

            np.testing.assert_array_almost_equal(
                mu_np, cp_local.asnumpy(mu_cp), decimal=10
            )

    @pytest.mark.skipif(not HAS_HDF5, reason="h5py not available")
    def test_hdf5_save_load(self):
        """HDF5保存・読込のテスト"""
        basis = LinMolBasis(V_max=1, J_max=1, use_M=False)
        dipole_orig = LinMolDipoleMatrix(
            basis, mu0=0.5, potential_type="harmonic", dense=True
        )

        # 行列を生成してキャッシュに載せる
        mu_x_orig = dipole_orig.mu_x
        mu_y_orig = dipole_orig.mu_y

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_path = f.name

        try:
            # 保存
            dipole_orig.to_hdf5(temp_path)

            # 読込
            dipole_loaded = LinMolDipoleMatrix.from_hdf5(temp_path, basis=basis)

            # 属性の確認
            assert dipole_loaded.mu0 == 0.5
            assert dipole_loaded.potential_type == "harmonic"
            assert dipole_loaded.dense

            # 行列の確認
            np.testing.assert_array_equal(dipole_loaded.mu_x, mu_x_orig)
            np.testing.assert_array_equal(dipole_loaded.mu_y, mu_y_orig)

        finally:
            os.unlink(temp_path)

    def test_repr_string(self):
        """文字列表現のテスト"""
        basis = LinMolBasis(V_max=1, J_max=1, use_M=False)
        dipole = LinMolDipoleMatrix(
            basis, mu0=0.3, potential_type="morse", dense=False
        )
        
        # reprにはクラス名、mu0、potential_type、backend、denseが含まれる
        repr_str = repr(dipole)
        assert "LinMolDipoleMatrix" in repr_str
        assert "mu0=0.3" in repr_str
        assert "potential_type='morse'" in repr_str
        assert "backend='numpy'" in repr_str
        assert "dense=False" in repr_str
        
        # キャッシュが機能していることを確認
        mu_x1 = dipole.mu_x
        mu_x2 = dipole.mu_x
        assert mu_x1 is mu_x2


class TestBuildMuFunction:
    """build_mu関数の直接テスト"""

    def test_build_mu_basic(self):
        """基本的なbuild_muテスト"""
        basis = LinMolBasis(V_max=1, J_max=1, use_M=False)

        mu_x = build_mu(basis, "x", mu0=1.0)

        assert mu_x.shape == (basis.size(), basis.size())
        assert mu_x.dtype == np.complex128

    def test_build_mu_all_axes(self):
        """全軸のbuild_muテスト"""
        basis = LinMolBasis(V_max=1, J_max=1, use_M=False)

        mu_x = build_mu(basis, "x", mu0=1.0)
        mu_y = build_mu(basis, "y", mu0=1.0)
        mu_z = build_mu(basis, "z", mu0=1.0)

        # 形状は同じ
        assert mu_x.shape == mu_y.shape == mu_z.shape

        # use_M=False（全てM=0）の場合の物理的期待値：
        # μx, μy: ΔM=±1が必要だが、M=0のみなのでゼロ行列
        # μz: ΔM=0が許されるので非ゼロ要素を持つ
        assert np.allclose(mu_x, 0)  # μxはゼロ行列
        assert np.allclose(mu_y, 0)  # μyはゼロ行列
        assert np.any(mu_z != 0)  # μzは非ゼロ要素を持つ

    def test_build_mu_potential_types(self):
        """異なるpotential_typeのテスト（use_M=Trueで確認）"""
        # M量子数がある場合で比較（より多くの非ゼロ要素）
        basis = LinMolBasis(V_max=2, J_max=1, use_M=True)

        mu_harm = build_mu(basis, "x", mu0=1.0, potential_type="harmonic")
        mu_morse = build_mu(basis, "x", mu0=1.0, potential_type="morse")

        assert mu_harm.shape == mu_morse.shape
        # 非ゼロ要素がある場合は値が異なることが期待される
        if np.any(mu_harm != 0) and np.any(mu_morse != 0):
            assert not np.allclose(mu_harm, mu_morse)
        else:
            # 両方ともゼロ行列の場合はスキップ
            pytest.skip("Both matrices are zero, cannot compare potential types")

    def test_build_mu_dense_vs_sparse(self):
        """dense vs sparseの比較テスト"""
        basis = LinMolBasis(V_max=2, J_max=1, use_M=False)

        mu_dense = build_mu(basis, "x", mu0=1.0, dense=True)
        mu_sparse = build_mu(basis, "x", mu0=1.0, dense=False)

        # dense版は通常のnumpy配列
        assert isinstance(mu_dense, np.ndarray)

        # sparse版はスパース行列
        try:
            import scipy.sparse as sp

            assert sp.issparse(mu_sparse)

            # 値は同じ
            np.testing.assert_array_almost_equal(mu_dense, mu_sparse.toarray())
        except ImportError:
            pytest.skip("scipy not available")

    def test_build_mu_mu0_scaling(self):
        """mu0スケーリングのテスト"""
        basis = LinMolBasis(V_max=1, J_max=1, use_M=False)

        mu1 = build_mu(basis, "x", mu0=1.0)
        mu2 = build_mu(basis, "x", mu0=2.5)

        # 比例関係になっているはず
        np.testing.assert_array_almost_equal(mu2, 2.5 * mu1)

    def test_build_mu_invalid_axis(self):
        """無効な軸指定のエラーテスト"""
        basis = LinMolBasis(V_max=1, J_max=1, use_M=False)

        with pytest.raises(ValueError, match="axis must be x, y or z"):
            # 型チェッカー回避のためキャスト
            build_mu(basis, "invalid", mu0=1.0)  # type: ignore

    def test_build_mu_invalid_potential(self):
        """無効なpotential_typeのエラーテスト"""
        basis = LinMolBasis(V_max=1, J_max=1, use_M=False)

        with pytest.raises(
            ValueError, match="potential_type must be harmonic or morse"
        ):
            # 型チェッカー回避のためキャスト
            build_mu(basis, "x", mu0=1.0, potential_type="invalid")  # type: ignore

    @pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")
    def test_build_mu_cupy_backend(self):
        """CuPyバックエンドのテスト"""
        basis = LinMolBasis(V_max=1, J_max=1, use_M=False)

        mu_cupy = build_mu(basis, "x", mu0=1.0, backend="cupy")

        # CuPy配列かどうか確認
        assert hasattr(mu_cupy, "device")
        assert mu_cupy.shape == (basis.size(), basis.size())

    def test_build_mu_case_insensitive(self):
        """大文字小文字非依存のテスト"""
        basis = LinMolBasis(V_max=1, J_max=1, use_M=False)

        # 型チェッカー回避のため実行時に文字列操作
        axis_upper = "x".upper()
        mu_x = build_mu(basis, axis_upper, mu0=1.0)  # type: ignore
        mu_x_lower = build_mu(basis, "x", mu0=1.0)

        np.testing.assert_array_equal(mu_x, mu_x_lower)


class TestPhysicalProperties:
    """物理的性質のテスト"""

    def test_selection_rules(self):
        """選択則のテスト"""
        basis = LinMolBasis(V_max=2, J_max=2, use_M=True)
        dipole = LinMolDipoleMatrix(basis)

        mu_x = dipole.mu_x
        mu_y = dipole.mu_y
        mu_z = dipole.mu_z

        # 非ゼロ要素の確認（選択則を満たす遷移のみ）
        for i in range(basis.size()):
            for j in range(basis.size()):
                state_i = basis.get_state(i)
                state_j = basis.get_state(j)

                if len(state_i) == 3:  # M量子数あり
                    v_i, J_i, M_i = state_i
                    v_j, J_j, M_j = state_j

                    # 振動選択則: Δv = ±1 (harmonic)
                    if abs(v_i - v_j) > 1:
                        assert mu_x[i, j] == 0
                        assert mu_y[i, j] == 0
                        assert mu_z[i, j] == 0

                    # 回転選択則: ΔJ = ±1
                    if abs(J_i - J_j) != 1:
                        assert mu_x[i, j] == 0
                        assert mu_y[i, j] == 0
                        assert mu_z[i, j] == 0

                    # 磁気選択則
                    # μz: ΔM = 0
                    if M_i != M_j:
                        assert mu_z[i, j] == 0

                    # μx, μy: ΔM = ±1
                    if abs(M_i - M_j) != 1:
                        assert mu_x[i, j] == 0
                        assert mu_y[i, j] == 0
