import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import pytest

from rovibrational_excitation.core.basis import LinMolBasis, TwoLevelBasis, VibLadderBasis


def test_linmol_basis_units():
    """LinMolBasisの単位統一テスト"""
    basis = LinMolBasis(V_max=2, J_max=2, use_M=False)
    
    # エネルギー単位（デフォルト）
    H0_energy = basis.generate_H0(omega_rad_pfs=0.159, B_rad_pfs=3.9e-5)
    
    # 明示的にエネルギー単位
    H0_energy_explicit = basis.generate_H0(
        omega_rad_pfs=0.159, B_rad_pfs=3.9e-5, return_energy_units=True
    )
    
    # 周波数単位
    H0_freq = basis.generate_H0(
        omega_rad_pfs=0.159, B_rad_pfs=3.9e-5, return_energy_units=False
    )
    
    # デフォルトと明示的エネルギー単位は同じ
    np.testing.assert_array_equal(H0_energy.matrix, H0_energy_explicit.matrix)
    
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
    basis = TwoLevelBasis()
    
    # エネルギー単位（デフォルト）
    energy_gap = 1e-20  # J
    H0_energy = basis.generate_H0(energy_gap=energy_gap)
    
    # 明示的にエネルギー単位
    H0_energy_explicit = basis.generate_H0(
        energy_gap=energy_gap, energy_gap_units="energy", return_energy_units=True
    )
    
    # 周波数単位の入力
    freq_gap = 0.159  # rad/fs
    H0_from_freq = basis.generate_H0(
        energy_gap=freq_gap, energy_gap_units="frequency", return_energy_units=True
    )
    
    # 周波数単位で出力
    H0_freq_out = basis.generate_H0(
        energy_gap=energy_gap, energy_gap_units="energy", return_energy_units=False
    )
    
    # デフォルトと明示的エネルギー単位は同じ
    np.testing.assert_array_equal(H0_energy.matrix, H0_energy_explicit.matrix)
    
    # 単位変換の確認
    _HBAR = 6.62607015e-034 / (2 * np.pi)
    expected_energy_from_freq = freq_gap * _HBAR / 1e-15
    assert abs(H0_from_freq.matrix[1, 1] - expected_energy_from_freq) < 1e-25
    
    expected_freq_from_energy = energy_gap / _HBAR * 1e-15
    assert abs(H0_freq_out.matrix[1, 1] - expected_freq_from_energy) < 1e-10


def test_viblad_basis_units():
    """VibLadderBasisの単位統一テスト"""
    basis = VibLadderBasis(V_max=3, omega_rad_pfs=0.159, delta_omega_rad_pfs=0.001)
    
    # エネルギー単位（デフォルト）
    H0_energy = basis.generate_H0()
    
    # 明示的にエネルギー単位
    H0_energy_explicit = basis.generate_H0(return_energy_units=True)
    
    # 周波数単位
    H0_freq = basis.generate_H0(return_energy_units=False)
    
    # デフォルトと明示的エネルギー単位は同じ
    np.testing.assert_array_equal(H0_energy.matrix, H0_energy_explicit.matrix)
    
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
    linmol = LinMolBasis(V_max=1, J_max=0, use_M=False)
    H0_linmol = linmol.generate_H0(omega_rad_pfs=omega, return_energy_units=True)
    
    # VibLadderBasis  
    viblad = VibLadderBasis(V_max=1, omega_rad_pfs=omega)
    H0_viblad = viblad.generate_H0(return_energy_units=True)
    
    # TwoLevelBasis（周波数入力）
    twolevel = TwoLevelBasis()
    H0_twolevel = twolevel.generate_H0(
        energy_gap=omega, energy_gap_units="frequency", return_energy_units=True
    )
    
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
    H0_old = linmol.generate_H0(omega_rad_pfs=0.159)  # 従来の呼び出し
    H0_new = linmol.generate_H0(omega_rad_pfs=0.159, return_energy_units=True)
    np.testing.assert_array_equal(H0_old.matrix, H0_new.matrix)
    
    # TwoLevelBasis
    twolevel = TwoLevelBasis()
    H0_old = twolevel.generate_H0(energy_gap=1e-20)  # 従来の呼び出し
    H0_new = twolevel.generate_H0(energy_gap=1e-20, return_energy_units=True)
    np.testing.assert_array_equal(H0_old.matrix, H0_new.matrix)
    
    # VibLadderBasis
    viblad = VibLadderBasis(V_max=2, omega_rad_pfs=0.159)
    H0_old = viblad.generate_H0()  # 従来の呼び出し
    H0_new = viblad.generate_H0(return_energy_units=True)
    np.testing.assert_array_equal(H0_old.matrix, H0_new.matrix)


def test_physical_scales():
    """物理的なスケールの妥当性テスト"""
    # CO2分子系の現実的なパラメータ
    omega_co2 = 0.159  # rad/fs （CO2のω1振動）
    B_co2 = 3.9e-5      # rad/fs （CO2の回転定数）
    
    # LinMolBasis
    basis = LinMolBasis(V_max=3, J_max=5, use_M=False)
    H0 = basis.generate_H0(omega_rad_pfs=omega_co2, B_rad_pfs=B_co2, return_energy_units=True)
    
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
    linmol = LinMolBasis(V_max=1, J_max=0, use_M=False)
    H0_linmol = linmol.generate_H0(omega_rad_pfs=omega, return_energy_units=True)
    linmol_energy = H0_linmol.matrix[1, 1] - H0_linmol.matrix[0, 0]
    
    # VibLadderBasis
    viblad = VibLadderBasis(V_max=1, omega_rad_pfs=omega)
    H0_viblad = viblad.generate_H0(return_energy_units=True)
    viblad_energy = H0_viblad.matrix[1, 1] - H0_viblad.matrix[0, 0]
    
    # TwoLevelBasis
    twolevel = TwoLevelBasis()
    H0_twolevel = twolevel.generate_H0(
        energy_gap=omega, energy_gap_units="frequency", return_energy_units=True
    )
    twolevel_energy = H0_twolevel.matrix[1, 1] - H0_twolevel.matrix[0, 0]
    
    # すべて期待値と一致するかテスト
    assert abs(linmol_energy - expected_energy) < 1e-25
    assert abs(viblad_energy - expected_energy) < 1e-25
    assert abs(twolevel_energy - expected_energy) < 1e-25


if __name__ == "__main__":
    pytest.main([__file__]) 