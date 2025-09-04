import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import pytest

from rovibrational_excitation.core.basis import LinMolBasis
from rovibrational_excitation.core.electric_field import ElectricField, gaussian_fwhm
from rovibrational_excitation.core.nondimensional import (
    NondimensionalizationScales,
    analyze_regime,
    dimensionalize_wavefunction,
    get_physical_time,
    nondimensionalize_system,
)
from rovibrational_excitation.dipole.linmol import LinMolDipoleMatrix
from rovibrational_excitation.core.units.constants import CONSTANTS


def test_nondimensionalization_scales():
    """NondimensionalizationScalesクラスのテスト"""
    scales = NondimensionalizationScales(
        E0=1e-20,  # J
        mu0=1e-30,  # C·m
        Efield0=1e8,  # V/m
        t0=1e-15,  # s
        lambda_coupling=0.5,
    )
    
    assert scales.E0 == 1e-20
    assert scales.mu0 == 1e-30
    assert scales.Efield0 == 1e8
    assert scales.t0 == 1e-15
    assert scales.lambda_coupling == 0.5
    
    # repr test
    repr_str = repr(scales)
    assert "NondimensionalizationScales" in repr_str
    assert "E0=" in repr_str
    assert "λ=" in repr_str


def test_nondimensionalize_system_basic():
    """基本的な無次元化システムのテスト"""
    # 簡単なシステム設定
    tlist = np.linspace(-10, 10, 201)
    efield = ElectricField(tlist)
    efield.add_dispersed_Efield(
        gaussian_fwhm,
        duration=5.0,
        t_center=0.0,
        carrier_freq=0.1,
        amplitude=1e8,
        polarization=np.array([1.0, 0.0]),
    )
    
    # ダミーのハミルトニアンと双極子行列
    dim = 3
    # エネルギー単位（J）のハミルトニアン
    _HBAR = 1.054571817e-34
    H0_freq = np.diag([0.0, 0.1, 0.2])  # rad/fs
    H0 = H0_freq * _HBAR / 1e-15  # rad/fs → J
    mu_x = np.array([[0, 1e-30, 0], [1e-30, 0, 1e-30], [0, 1e-30, 0]])  # C·m
    mu_y = np.zeros_like(mu_x)
    
    # 無次元化実行
    (
        H0_prime,
        mu_x_prime,
        mu_y_prime,
        Efield_prime,
        tlist_prime,
        dt_prime,
        scales,
    ) = nondimensionalize_system(H0, mu_x, mu_y, efield,
                                 H0_units="energy", time_units="fs")
    
    # 形状チェック
    assert H0_prime.shape == H0.shape
    assert mu_x_prime.shape == mu_x.shape
    assert mu_y_prime.shape == mu_y.shape
    assert Efield_prime.shape == efield.get_Efield().shape
    assert len(tlist_prime) == len(tlist)
    
    # スケールがポジティブ
    assert scales.E0 > 0
    assert scales.mu0 > 0
    assert scales.Efield0 > 0
    assert scales.t0 > 0
    assert scales.lambda_coupling >= 0
    
    # 無次元化チェック（範囲が適切か）
    assert np.max(np.abs(H0_prime)) <= 1.1  # 多少の余裕
    assert np.max(np.abs(mu_x_prime)) <= 1.1
    assert np.max(np.abs(Efield_prime)) <= 1.1


def test_analyze_regime():
    """物理レジーム分析のテスト"""
    # 弱結合
    scales_weak = NondimensionalizationScales(1e-20, 1e-30, 1e8, 1e-15, 0.05)
    regime_weak = analyze_regime(scales_weak)
    assert regime_weak["regime"] == "weak_coupling"
    assert regime_weak["lambda"] == 0.05
    
    # 中間結合
    scales_intermediate = NondimensionalizationScales(1e-20, 1e-30, 1e8, 1e-15, 0.5)
    regime_intermediate = analyze_regime(scales_intermediate)
    assert regime_intermediate["regime"] == "intermediate_coupling"
    assert regime_intermediate["lambda"] == 0.5
    
    # 強結合
    scales_strong = NondimensionalizationScales(1e-20, 1e-30, 1e8, 1e-15, 2.0)
    regime_strong = analyze_regime(scales_strong)
    assert regime_strong["regime"] == "strong_coupling"
    assert regime_strong["lambda"] == 2.0
    
    # 返り値の型チェック
    for regime in [regime_weak, regime_intermediate, regime_strong]:
        assert "regime" in regime
        assert "lambda" in regime
        assert "description" in regime
        assert "energy_scale_eV" in regime
        assert "time_scale_fs" in regime


def test_get_physical_time():
    """無次元時間を物理時間に変換するテスト"""
    scales = NondimensionalizationScales(1e-20, 1e-30, 1e8, 1e-15, 0.5)
    tau = np.array([0, 1, 2, 3])  # 無次元時間
    
    t_physical = get_physical_time(tau, scales)
    
    # 単位変換チェック
    expected = tau * scales.t0 * 1e15  # s → fs
    np.testing.assert_array_almost_equal(t_physical, expected)


def test_dimensionalize_wavefunction():
    """波動関数の次元化テスト"""
    scales = NondimensionalizationScales(1e-20, 1e-30, 1e8, 1e-15, 0.5)
    psi_prime = np.array([1.0, 0.0, 0.0], dtype=complex)
    
    psi = dimensionalize_wavefunction(psi_prime, scales)
    
    # 正規化は保持される
    np.testing.assert_array_almost_equal(psi, psi_prime)
    assert np.abs(np.linalg.norm(psi) - 1.0) < 1e-12


def test_nondimensionalize_with_realistic_system():
    """現実的なシステムでの無次元化テスト"""
    # CO2の無次元振動子
    basis = LinMolBasis(V_max=3, J_max=5, use_M=False)
    
    # 時間軸
    tlist = np.linspace(-50, 50, 1001)
    efield = ElectricField(tlist)
    efield.add_dispersed_Efield(
        gaussian_fwhm,
        duration=20.0,
        t_center=0.0,
        carrier_freq=0.159,
        amplitude=1e9,
        polarization=np.array([1.0, 0.0]),
    )
    
    # ハミルトニアン（エネルギー単位）
    H0 = basis.generate_H0(omega_rad_phz=0.159, B_rad_phz=3.9e-5, return_energy_units=True)
    
    # 双極子行列
    dip = LinMolDipoleMatrix(basis, mu0=0.3e-29, backend="numpy", dense=True)
    
    # 無次元化
    (
        H0_prime,
        mu_x_prime,
        mu_y_prime,
        Efield_prime,
        tlist_prime,
        dt_prime,
        scales,
    ) = nondimensionalize_system(H0.get_matrix(units="J"), dip.mu_x, dip.mu_y, efield,
                                 H0_units="energy", time_units="fs")
    
    # 物理レジーム分析
    regime_info = analyze_regime(scales)
    
    # 基本チェック
    assert H0_prime.shape == H0.shape
    assert len(tlist_prime) == len(tlist)
    assert regime_info["lambda"] > 0
    assert regime_info["regime"] in ["weak_coupling", "intermediate_coupling", "strong_coupling"]
    
    # エネルギースケールが妥当か（eV範囲）
    energy_eV = regime_info["energy_scale_eV"]
    assert 0.001 < energy_eV < 100  # meV ~ 100eV の範囲
    
    # 時間スケールが妥当か（fs範囲）
    time_fs = regime_info["time_scale_fs"]
    assert 0.01 < time_fs < 10000  # 0.01fs ~ 10ps の範囲


def test_edge_cases():
    """エッジケースのテスト"""
    # ゼロ電場
    tlist = np.linspace(-5, 5, 101)
    efield_zero = ElectricField(tlist)
    # 電場を追加しない（ゼロのまま）
    
    dim = 2
    # エネルギー単位（J）に変換
    _HBAR = 1.054571817e-34
    H0_freq = np.diag([0.0, 0.1])  # rad/fs
    H0 = H0_freq * _HBAR / 1e-15  # rad/fs → J
    mu_x = np.array([[0, 1e-30], [1e-30, 0]])
    mu_y = np.zeros_like(mu_x)
    
    # デフォルト値が使われるはず
    (
        H0_prime,
        mu_x_prime,
        mu_y_prime,
        Efield_prime,
        tlist_prime,
        dt_prime,
        scales,
    ) = nondimensionalize_system(H0, mu_x, mu_y, efield_zero,
                                 H0_units="energy", time_units="fs")
    
    # デフォルト値チェック
    # 現行仕様ではデフォルト電場スケールは 1e8 V/m
    assert scales.Efield0 == 1e8  # デフォルト電場スケール
    assert np.all(Efield_prime == 0)  # ゼロ電場
    
    # ゼロ双極子
    mu_x_zero = np.zeros_like(mu_x)
    mu_y_zero = np.zeros_like(mu_y)
    
    (
        H0_prime,
        mu_x_prime,
        mu_y_prime,
        Efield_prime,
        tlist_prime,
        dt_prime,
        scales,
    ) = nondimensionalize_system(H0, mu_x_zero, mu_y_zero, efield_zero,
                                 H0_units="energy", time_units="fs")
    
    assert scales.mu0 == CONSTANTS.DEBYE_TO_CM  # デフォルト双極子スケール
    assert np.all(mu_x_prime == 0)
    assert np.all(mu_y_prime == 0)


if __name__ == "__main__":
    pytest.main([__file__]) 
