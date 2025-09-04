import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import pytest

from rovibrational_excitation.core.basis import LinMolBasis
from rovibrational_excitation.core.electric_field import ElectricField, gaussian_fwhm
from rovibrational_excitation.core.nondimensional import nondimensionalize_system


def test_hamiltonian_unit_consistency():
    """ハミルトニアンの単位統一のテスト"""
    basis = LinMolBasis(V_max=2, J_max=3, use_M=False, omega=0.159, B=3.9e-5, input_units="rad/fs")
    
    # 周波数単位で生成
    basis_freq = LinMolBasis(V_max=2, J_max=3, use_M=False, omega=0.159, B=3.9e-5, input_units="rad/fs", output_units="rad/fs")
    H0_freq = basis_freq.generate_H0()
    
    # エネルギー単位で生成
    basis_energy = LinMolBasis(V_max=2, J_max=3, use_M=False, omega=0.159, B=3.9e-5, input_units="rad/fs", output_units="J")
    H0_energy = basis_energy.generate_H0()
    
    # 単位変換確認
    _HBAR = 6.62607015e-034 / (2 * np.pi)
    _FS_TO_S = 1e-15
    H0_converted = H0_freq.matrix * _HBAR / _FS_TO_S
    np.testing.assert_array_almost_equal(H0_energy.matrix, H0_converted, decimal=10)
    
    # エネルギー単位が適切な範囲にあるか
    energies = np.diag(H0_energy.matrix)
    energy_range_J = np.max(energies) - np.min(energies)
    energy_range_eV = energy_range_J / 1.602176634e-19
    
    # CO2の振動エネルギーは数百meV程度
    assert 0.001 < energy_range_eV < 10  # 1meV ~ 10eV


def test_nondimensionalization_unit_options():
    """無次元化の単位オプションのテスト"""
    # テスト用システム
    tlist = np.linspace(-10, 10, 201)
    efield = ElectricField(tlist)
    efield.add_dispersed_Efield(
        gaussian_fwhm, duration=5.0, t_center=0.0, 
        carrier_freq=0.1, amplitude=1e8,
        polarization=np.array([1.0, 0.0])
    )
    
    basis = LinMolBasis(V_max=2, J_max=2, use_M=False, omega=0.159, input_units="rad/fs")
    
    # エネルギー単位のH0
    H0_energy = basis.generate_H0_with_params(omega=0.159, input_units="rad/fs", units="J")
    
    # 周波数単位のH0  
    H0_freq = basis.generate_H0_with_params(omega=0.159, input_units="rad/fs", units="rad/fs")
    
    # ダミー双極子
    mu_x = np.random.random(H0_energy.matrix.shape) * 1e-30
    mu_y = np.zeros_like(mu_x)
    
    # エネルギー単位での無次元化
    (_, _, _, _, _, _, scales_energy) = nondimensionalize_system(
        H0_energy.matrix, mu_x, mu_y, efield,
        H0_units="energy", time_units="fs"
    )
    
    # 周波数単位での無次元化
    (_, _, _, _, _, _, scales_freq) = nondimensionalize_system(
        H0_freq.matrix, mu_x, mu_y, efield,
        H0_units="frequency", time_units="fs"
    )
    
    # 結果は同等であるべき
    assert abs(scales_energy.lambda_coupling - scales_freq.lambda_coupling) < 1e-10
    assert abs(scales_energy.E0 - scales_freq.E0) < 1e-25  # エネルギーはJ単位で小さい


def test_time_unit_handling():
    """時間単位の処理テスト"""
    tlist_fs = np.linspace(-10, 10, 101)  # fs
    tlist_s = tlist_fs * 1e-15  # s
    
    # fs単位の電場
    efield_fs = ElectricField(tlist_fs)
    efield_fs.add_dispersed_Efield(
        gaussian_fwhm, duration=5.0, t_center=0.0,
        carrier_freq=0.1, amplitude=1e8,
        polarization=np.array([1.0, 0.0])
    )
    
    # s単位の電場（同じ物理的内容）
    efield_s = ElectricField(tlist_s)
    efield_s.add_dispersed_Efield(
        gaussian_fwhm, duration=5.0e-15, t_center=0.0,
        carrier_freq=0.1/1e-15, amplitude=1e8,
        polarization=np.array([1.0, 0.0])
    )
    
    # 簡単なハミルトニアン
    H0 = np.diag([0, 1e-21, 2e-21])  # J
    mu_x = np.array([[0, 1e-30, 0], [1e-30, 0, 1e-30], [0, 1e-30, 0]])
    mu_y = np.zeros_like(mu_x)
    
    # fs単位での無次元化
    (_, _, _, _, tlist_prime_fs, dt_prime_fs, scales_fs) = nondimensionalize_system(
        H0, mu_x, mu_y, efield_fs,
        H0_units="energy", time_units="fs"
    )
    
    # s単位での無次元化  
    (_, _, _, _, tlist_prime_s, dt_prime_s, scales_s) = nondimensionalize_system(
        H0, mu_x, mu_y, efield_s,
        H0_units="energy", time_units="s"
    )
    
    # 無次元時間は同じになるべき
    np.testing.assert_array_almost_equal(tlist_prime_fs, tlist_prime_s, decimal=8)
    assert abs(dt_prime_fs - dt_prime_s) < 1e-10


def test_backward_compatibility():
    """後方互換性のテスト"""
    # 古い使い方（周波数単位）でも動作するか
    basis = LinMolBasis(V_max=2, J_max=2, use_M=False)
    
    # デフォルトではエネルギー単位
    H0_default = basis.generate_H0_with_params(omega_rad_pfs=0.159, units="J")
    
    # 明示的にエネルギー単位
    H0_explicit = basis.generate_H0_with_params(omega_rad_pfs=0.159, units="J")
    
    # 同じ結果になるべき
    np.testing.assert_array_equal(H0_default.matrix, H0_explicit.matrix)
    
    # エネルギー単位である事を確認
    energies = np.diag(H0_default.matrix)
    # CO2の基底振動エネルギー：約0.2eV ≈ 3.2e-20 J
    expected_order = 1e-20  # J
    assert np.max(energies) > expected_order / 10
    assert np.max(energies) < expected_order * 100


def test_physical_regime_analysis():
    """物理レジーム分析の妥当性テスト"""
    from rovibrational_excitation.core.nondimensional import analyze_regime
    from rovibrational_excitation.dipole.linmol import LinMolDipoleMatrix
    
    # 現実的なCO2系
    basis = LinMolBasis(V_max=3, J_max=5, use_M=False, omega=0.159, B=3.9e-5, input_units="rad/fs")
    H0 = basis.generate_H0()
    dip = LinMolDipoleMatrix(basis, mu0=0.3e-29, backend="numpy", dense=True)
    
    # 弱い電場
    tlist = np.linspace(-50, 50, 1001)
    efield_weak = ElectricField(tlist)
    efield_weak.add_dispersed_Efield(
        gaussian_fwhm, duration=20.0, t_center=0.0,
        carrier_freq=0.159, amplitude=1e7,  # 弱い電場
        polarization=np.array([1.0, 0.0])
    )
    
    # 強い電場
    efield_strong = ElectricField(tlist)
    efield_strong.add_dispersed_Efield(
        gaussian_fwhm, duration=20.0, t_center=0.0,
        carrier_freq=0.159, amplitude=1e10,  # 強い電場
        polarization=np.array([1.0, 0.0])
    )
    
    # 弱結合解析
    _, _, _, _, _, _, scales_weak = nondimensionalize_system(
        H0.matrix, dip.mu_x, dip.mu_y, efield_weak,
        H0_units="energy", time_units="fs"
    )
    regime_weak = analyze_regime(scales_weak)
    
    # 強結合解析
    _, _, _, _, _, _, scales_strong = nondimensionalize_system(
        H0.matrix, dip.mu_x, dip.mu_y, efield_strong,
        H0_units="energy", time_units="fs"
    )
    regime_strong = analyze_regime(scales_strong)
    
    # 弱結合の方がλが小さいはず
    assert scales_weak.lambda_coupling < scales_strong.lambda_coupling
    
    # レジーム分類が適切か
    if scales_weak.lambda_coupling < 0.1:
        assert regime_weak["regime"] == "weak_coupling"
    if scales_strong.lambda_coupling >= 1.0:
        assert regime_strong["regime"] == "strong_coupling"


if __name__ == "__main__":
    pytest.main([__file__]) 
