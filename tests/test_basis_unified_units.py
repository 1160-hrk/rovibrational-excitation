import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import pytest

from rovibrational_excitation.core.basis import LinMolBasis, TwoLevelBasis, VibLadderBasis


def test_linmol_basis_units():
    """LinMolBasisの単位統一テスト"""
    basis = LinMolBasis(
        V_max=2, 
        J_max=2, 
        use_M=False,
        omega=0.159,
        B=3.9e-5,
        input_units="rad/fs",
        output_units="J"
    )
    
    # エネルギー単位（デフォルト）
    H0_energy = basis.generate_H0()
    
    # 周波数単位
    basis_freq = LinMolBasis(
        V_max=2,
        J_max=2,
        use_M=False,
        omega=0.159,
        B=3.9e-5,
        input_units="rad/fs",
        output_units="rad/fs",
    )
    H0_freq = basis_freq.generate_H0()
    
    # エネルギー単位 vs 周波数単位の変換チェック
    _HBAR = 6.62607015e-034 / (2 * np.pi)
    _FS_TO_S = 1e-15
    H0_converted = H0_freq.matrix * _HBAR / _FS_TO_S
    np.testing.assert_array_almost_equal(H0_energy.matrix, H0_converted, decimal=10)
    
    # エネルギーが適切な範囲
    energies = np.diag(H0_energy.matrix)
    energy_range_eV = (np.max(energies) - np.min(energies)) / 1.602176634e-19
    assert 0.001 < energy_range_eV < 10  # 1meV ~ 10eV


def test_twolevel_basis_units():
    """TwoLevelBasisの単位統一テスト"""
    energy_gap_J = 1e-20  # J
    
    # エネルギー単位（デフォルトJ）
    basis_energy = TwoLevelBasis(energy_gap=energy_gap_J, input_units="J")
    H0_energy = basis_energy.generate_H0()

    # 周波数単位の入力
    freq_gap_rad_pfs = 0.159  # rad/fs
    basis_from_freq = TwoLevelBasis(energy_gap=freq_gap_rad_pfs, input_units="rad/fs", output_units="J")
    H0_from_freq = basis_from_freq.generate_H0()
    
    # 周波数単位で出力
    basis_freq_out = TwoLevelBasis(energy_gap=energy_gap_J, input_units="J", output_units="rad/fs")
    H0_freq_out = basis_freq_out.generate_H0()
    
    # 単位変換の確認
    _HBAR = 6.62607015e-034 / (2 * np.pi)
    expected_energy_from_freq = freq_gap_rad_pfs * _HBAR / 1e-15
    assert abs(H0_from_freq.matrix[1, 1] - expected_energy_from_freq) < 1e-25
    
    expected_freq_from_energy = energy_gap_J / _HBAR * 1e-15
    assert abs(H0_freq_out.matrix[1, 1] - expected_freq_from_energy) < 1e-10


def test_viblad_basis_units():
    """VibLadderBasisの単位統一テスト"""
    basis = VibLadderBasis(V_max=3, omega=0.159, delta_omega=0.001, input_units="rad/fs", output_units="J")
    
    # エネルギー単位（デフォルト）
    H0_energy = basis.generate_H0()
    
    # 周波数単位
    basis_freq = VibLadderBasis(V_max=3, omega=0.159, delta_omega=0.001, input_units="rad/fs", output_units="rad/fs")
    H0_freq = basis_freq.generate_H0()
    
    # エネルギー単位 vs 周波数単位の変換チェック
    _HBAR = 6.62607015e-034 / (2 * np.pi)
    _FS_TO_S = 1e-15
    H0_converted = H0_freq.matrix * _HBAR / _FS_TO_S
    np.testing.assert_array_almost_equal(H0_energy.matrix, H0_converted, decimal=10)
    
    # 振動エネルギーの妥当性チェック
    energies = np.diag(H0_energy.matrix)
    energy_spacing = energies[1] - energies[0]  # 基本振動間隔
    energy_spacing_eV = energy_spacing / 1.602176634e-19
    assert 0.01 < energy_spacing_eV < 1  # 10meV ~ 1eV


def test_all_basis_consistency():
    """すべてのbasisで単位が一貫しているかテスト"""
    # 同じ周波数でのエネルギー比較
    omega = 0.159  # rad/fs
    
    # LinMolBasis
    linmol = LinMolBasis(V_max=1, J_max=0, use_M=False, omega=omega, input_units="rad/fs", output_units="J")
    H0_linmol = linmol.generate_H0()
    
    # VibLadderBasis  
    viblad = VibLadderBasis(V_max=1, omega=omega, input_units="rad/fs", output_units="J")
    H0_viblad = viblad.generate_H0()
    
    # TwoLevelBasis（周波数入力）
    twolevel = TwoLevelBasis(energy_gap=omega, input_units="rad/fs", output_units="J")
    H0_twolevel = twolevel.generate_H0()
    
    # 基本振動エネルギー（ℏω）の比較
    _HBAR = 6.62607015e-034 / (2 * np.pi)
    _FS_TO_S = 1e-15
    expected_energy = omega * _HBAR / _FS_TO_S
    
    # LinMolBasisとVibLadderBasisの0→1遷移エネルギー
    linmol_spacing = H0_linmol.matrix[1, 1] - H0_linmol.matrix[0, 0]
    viblad_spacing = H0_viblad.matrix[1, 1] - H0_viblad.matrix[0, 0]
    
    # TwoLevelBasisのエネルギーギャップ
    twolevel_gap = H0_twolevel.matrix[1, 1] - H0_twolevel.matrix[0, 0]
    
    # すべて同じになるべき
    assert abs(linmol_spacing - expected_energy) < 1e-25
    assert abs(viblad_spacing - expected_energy) < 1e-25
    assert abs(twolevel_gap - expected_energy) < 1e-25
    
    # 相互比較
    assert abs(linmol_spacing - viblad_spacing) < 1e-25
    assert abs(linmol_spacing - twolevel_gap) < 1e-25


def test_backward_compatibility():
    """後方互換性のテスト"""
    # 従来の使用方法が動作するか
    
    # LinMolBasis
    linmol = LinMolBasis(V_max=2, J_max=1, use_M=False)
    H0_old = linmol.generate_H0_with_params(omega_rad_pfs=0.159, units="J")
    linmol_new = LinMolBasis(V_max=2, J_max=1, use_M=False, omega=0.159, input_units="rad/fs", output_units="J")
    H0_new = linmol_new.generate_H0()
    np.testing.assert_allclose(H0_old.matrix, H0_new.matrix)
    
    # TwoLevelBasis
    twolevel = TwoLevelBasis()
    H0_old = twolevel.generate_H0_with_params(energy_gap=1.0, energy_gap_units="rad/fs", units="J")
    twolevel_new = TwoLevelBasis(energy_gap=1.0, input_units="rad/fs", output_units="J")
    H0_new = twolevel_new.generate_H0()
    np.testing.assert_allclose(H0_old.matrix, H0_new.matrix)
    
    # VibLadderBasis
    viblad = VibLadderBasis(V_max=2)
    H0_old = viblad.generate_H0_with_params(omega=0.159, input_units="rad/fs", units="J")
    viblad_new = VibLadderBasis(V_max=2, omega=0.159, input_units="rad/fs", output_units="J")
    H0_new = viblad_new.generate_H0()
    np.testing.assert_allclose(H0_old.matrix, H0_new.matrix)


def test_physical_scales():
    """物理的なスケールの妥当性テスト"""
    # CO2分子系の現実的なパラメータ
    omega_co2 = 0.159  # rad/fs （CO2のω1振動）
    B_co2 = 3.9e-5      # rad/fs （CO2の回転定数）
    
    # LinMolBasis
    basis = LinMolBasis(V_max=3, J_max=5, use_M=False, omega=omega_co2, B=B_co2, input_units="rad/fs", output_units="J")
    H0 = basis.generate_H0()
    
    energies = np.diag(H0.matrix)
    energy_range_eV = (np.max(energies) - np.min(energies)) / 1.602176634e-19
    
    # CO2の振動・回転エネルギーの現実的な範囲
    assert 0.01 < energy_range_eV < 10  # 10meV ~ 10eV
    
    # 基本振動エネルギーの確認（約0.2eV）
    vib_energy_eV = (energies[basis.get_index((1, 0))] - energies[0]) / 1.602176634e-19
    assert 0.1 < vib_energy_eV < 0.5  # 100meV ~ 500meV


def test_hbar_consistency():
    """ハミルトニアン生成でのhbar値の一貫性テスト"""
    omega = 1.0  # rad/fs
    _HBAR = 6.62607015e-034 / (2 * np.pi)  # J⋅s
    _FS_TO_S = 1e-15  # fs → s conversion factor
    expected_energy = omega * _HBAR / _FS_TO_S  # J
    
    # すべてのbasisで同じhbar値が使われているかテスト
    
    # LinMolBasis
    linmol = LinMolBasis(V_max=1, J_max=0, use_M=False, omega=omega, input_units="rad/fs", output_units="J")
    H0_linmol = linmol.generate_H0()
    linmol_energy = H0_linmol.matrix[1, 1] - H0_linmol.matrix[0, 0]
    
    # VibLadderBasis
    viblad = VibLadderBasis(V_max=1, omega=omega, input_units="rad/fs", output_units="J")
    H0_viblad = viblad.generate_H0()
    viblad_energy = H0_viblad.matrix[1, 1] - H0_viblad.matrix[0, 0]
    
    # TwoLevelBasis
    twolevel = TwoLevelBasis(energy_gap=omega, input_units="rad/fs", output_units="J")
    H0_twolevel = twolevel.generate_H0()
    twolevel_energy = H0_twolevel.matrix[1, 1] - H0_twolevel.matrix[0, 0]
    
    # すべて期待値と一致するかテスト
    assert abs(linmol_energy - expected_energy) < 1e-25
    assert abs(viblad_energy - expected_energy) < 1e-25
    assert abs(twolevel_energy - expected_energy) < 1e-25


if __name__ == "__main__":
    pytest.main([__file__]) 