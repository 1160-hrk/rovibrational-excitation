import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import pytest

from rovibrational_excitation.core.basis import LinMolBasis
from rovibrational_excitation.core.electric_field import (
    ElectricField,
    ZeroField,
    gaussian_fwhm,
)
from rovibrational_excitation.core.nondimensional import (
    NondimensionalizationScales,
    analyze_regime,
    dimensionalize_wavefunction,
    get_physical_time,
    nondimensionalize_system,
)
from rovibrational_excitation.dipole.linmol import LinMolDipoleMatrix


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
    ) = nondimensionalize_system(
        H0, mu_x, mu_y, efield, H0_units="energy", time_units="fs"
    )

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
    """Diagnostics avoid model-independent weak/strong thresholds."""
    field_free = NondimensionalizationScales(
        1e-20,
        1e-30,
        None,
        1e-15,
        0.0,
        interaction_energy=0.0,
        physical_coupling_ratio=0.0,
    )
    gapless = NondimensionalizationScales(
        1e-20,
        1e-30,
        1e8,
        1e-15,
        1.0,
        free_energy_span=0.0,
        interaction_energy=1e-20,
        physical_coupling_ratio=None,
    )
    classified_only_by_data = NondimensionalizationScales(
        1e-20,
        1e-30,
        1e8,
        1e-15,
        0.5,
        interaction_energy=5e-21,
        physical_coupling_ratio=0.5,
    )

    diagnostics = [
        analyze_regime(field_free),
        analyze_regime(gapless),
        analyze_regime(classified_only_by_data),
    ]
    assert [item["regime"] for item in diagnostics] == [
        "field_free",
        "gapless_driven",
        "unclassified",
    ]
    for item in diagnostics:
        assert "numerical_coupling_coefficient" in item
        assert "physical_coupling_ratio" in item
        assert "reference_energy" in item


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
    basis = LinMolBasis(
        V_max=3, J_max=5, use_M=True, omega=1.0, B=0.001, alpha=0.0, delta_omega=0.0
    )

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
    H0 = basis.generate_H0(
        omega_rad_phz=0.159, B_rad_phz=3.9e-5, return_energy_units=True
    )

    # 双極子行列
    dip = LinMolDipoleMatrix(
        basis, mu0=0.3e-29, backend="numpy", dense=True, potential_type="harmonic"
    )

    # 無次元化
    (
        H0_prime,
        mu_x_prime,
        mu_y_prime,
        Efield_prime,
        tlist_prime,
        dt_prime,
        scales,
    ) = nondimensionalize_system(
        H0.get_matrix(units="J"),
        dip.mu_x,
        dip.mu_y,
        efield,
        H0_units="energy",
        time_units="fs",
    )

    # 物理レジーム分析
    regime_info = analyze_regime(scales)

    # 基本チェック
    assert H0_prime.shape == H0.shape
    assert len(tlist_prime) == len(tlist)
    assert regime_info["numerical_coupling_coefficient"] > 0
    assert regime_info["regime"] == "unclassified"

    # エネルギースケールが妥当か（eV範囲）
    energy_eV = regime_info["energy_scale_eV"]
    assert 0.001 < energy_eV < 100  # meV ~ 100eV の範囲

    # 時間スケールが妥当か（fs範囲）
    time_fs = regime_info["time_scale_fs"]
    assert 0.01 < time_fs < 10000  # 0.01fs ~ 10ps の範囲


def test_edge_cases():
    """Zero scales are explicit inactive states, never invented defaults."""
    tlist = np.linspace(-5, 5, 101)
    ambiguous_zero = ElectricField(tlist)
    zero_field = ZeroField(tlist)

    hbar = 1.054571817e-34
    h0 = np.diag([0.0, 0.1]) * hbar / 1e-15
    mu_x = np.array([[0, 1e-30], [1e-30, 0]])
    mu_y = np.zeros_like(mu_x)

    with pytest.raises(ValueError, match="ZeroField"):
        nondimensionalize_system(
            h0, mu_x, mu_y, ambiguous_zero, H0_units="energy", time_units="fs"
        )

    *_, field_prime, _, _, scales = nondimensionalize_system(
        h0, mu_x, mu_y, zero_field, H0_units="energy", time_units="fs"
    )
    assert scales.Efield0 is None
    np.testing.assert_array_equal(field_prime, np.zeros((tlist.size, 2)))

    zero_dipole = np.zeros_like(mu_x)
    _, mu_x_prime, mu_y_prime, _, _, _, scales = nondimensionalize_system(
        h0, zero_dipole, zero_dipole, zero_field, H0_units="energy", time_units="fs"
    )
    assert scales.mu0 is None
    np.testing.assert_array_equal(mu_x_prime, zero_dipole)
    np.testing.assert_array_equal(mu_y_prime, zero_dipole)


if __name__ == "__main__":
    pytest.main([__file__])
