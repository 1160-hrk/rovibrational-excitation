#!/usr/bin/env python
"""
Check script for new simulation systems.
Test propagator compatibility with new dipole matrix classes.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
from rovibrational_excitation.core.basis import TwoLevelBasis, VibLadderBasis
from rovibrational_excitation.core.states import StateVector
from rovibrational_excitation.core.electric_field import ElectricField, gaussian
from rovibrational_excitation.core.propagator import schrodinger_propagation
from rovibrational_excitation.dipole.twolevel import TwoLevelDipoleMatrix
from rovibrational_excitation.dipole.viblad import VibLadderDipoleMatrix


def check_twolevel_compatibility():
    """二準位系とpropagatorの互換性をチェック"""
    print("=== 二準位系 × Propagator 互換性チェック ===")
    
    # 基底とdipole matrix
    basis = TwoLevelBasis()
    H0 = basis.generate_H0(energy_gap=1.0)
    dipole_matrix = TwoLevelDipoleMatrix(basis, mu0=1.0)
    
    print(f"基底サイズ: {basis.size()}")
    print(f"ハミルトニアン: \n{H0}")
    print(f"双極子行列 mu_x: \n{dipole_matrix.mu_x}")
    
    # 初期状態
    state = StateVector(basis)
    state.set_state(0, 1.0)
    psi0 = state.data
    
    # 簡単な電場
    ti, tf = 0.0, 10.0
    dt4Efield = 0.1
    time4Efield = np.arange(ti, tf + 2*dt4Efield, dt4Efield)
    
    Efield = ElectricField(tlist=time4Efield)
    Efield.add_dispersed_Efield(
        envelope_func=gaussian,
        duration=5.0,
        t_center=5.0,
        carrier_freq=1.0/(2*np.pi),
        amplitude=0.1,
        polarization=np.array([1, 0]),  # 2要素ベクトル (Ex, Ey)
        const_polarisation=False,
    )
    
    try:
        # propagatorを実行
        print("時間発展計算を開始...")
        time4psi, psi_t = schrodinger_propagation(
            H0=H0,
            Efield=Efield,
            dipole_matrix=dipole_matrix,
            psi0=psi0,
            axes="xy",  # Exとμx、Eyとμyをカップリング
            return_traj=True,
            return_time_psi=True,
            sample_stride=2
        )
        print(f"✓ 成功! 時間ステップ数: {len(time4psi)}")
        print(f"  最終状態占有率: |0⟩={np.abs(psi_t[-1, 0])**2:.4f}, |1⟩={np.abs(psi_t[-1, 1])**2:.4f}")
        return True
        
    except Exception as e:
        print(f"✗ 失敗: {e}")
        return False


def check_viblad_compatibility():
    """振動ラダー系とpropagatorの互換性をチェック"""
    print("\n=== 振動ラダー系 × Propagator 互換性チェック ===")
    
    # 基底とdipole matrix
    basis = VibLadderBasis(V_max=3, omega_rad_phz=1.0)
    H0 = basis.generate_H0()
    dipole_matrix = VibLadderDipoleMatrix(basis, mu0=1e-30, potential_type="harmonic")
    
    print(f"基底サイズ: {basis.size()}")
    print(f"エネルギー準位: {np.diag(H0)}")
    print(f"双極子行列 mu_z の非ゼロ要素数: {np.count_nonzero(dipole_matrix.mu_z)}")
    
    # 初期状態
    state = StateVector(basis)
    state.set_state((0,), 1.0)
    psi0 = state.data
    
    # 簡単な電場
    ti, tf = 0.0, 20.0
    dt4Efield = 0.1
    time4Efield = np.arange(ti, tf + 2*dt4Efield, dt4Efield)
    
    Efield = ElectricField(tlist=time4Efield)
    Efield.add_dispersed_Efield(
        envelope_func=gaussian,
        duration=10.0,
        t_center=10.0,
        carrier_freq=1.0/(2*np.pi),
        amplitude=1e10,
        polarization=np.array([0, 1]),  # 2要素ベクトル (Ex, Ey) - Eyを使用
        const_polarisation=False,
    )
    
    try:
        # propagatorを実行
        print("時間発展計算を開始...")
        time4psi, psi_t = schrodinger_propagation(
            H0=H0,
            Efield=Efield,
            dipole_matrix=dipole_matrix,
            psi0=psi0,
            axes="zy",  # Eyとμzをカップリングさせるためzyを使用
            return_traj=True,
            return_time_psi=True,
            sample_stride=3
        )
        print(f"✓ 成功! 時間ステップ数: {len(time4psi)}")
        
        # 各振動状態の最終占有率
        final_populations = np.abs(psi_t[-1, :])**2
        print("  最終振動状態占有率:")
        for v in range(min(4, basis.size())):
            print(f"    |v={v}⟩: {final_populations[v]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 失敗: {e}")
        return False


def check_dipole_matrix_attributes():
    """dipole matrixクラスが必要な属性を持っているかチェック"""
    print("\n=== Dipole Matrix 属性チェック ===")
    
    # 二準位系
    basis_2level = TwoLevelBasis()
    dipole_2level = TwoLevelDipoleMatrix(basis_2level, mu0=1.0)
    
    print("TwoLevelDipoleMatrix:")
    for attr in ['mu_x', 'mu_y', 'mu_z']:
        has_attr = hasattr(dipole_2level, attr)
        if has_attr:
            matrix = getattr(dipole_2level, attr)
            print(f"  ✓ {attr}: shape={matrix.shape}, dtype={matrix.dtype}")
        else:
            print(f"  ✗ {attr}: 属性なし")
    
    # 振動ラダー系
    basis_viblad = VibLadderBasis(V_max=2)
    dipole_viblad = VibLadderDipoleMatrix(basis_viblad, mu0=1.0, potential_type="harmonic")
    
    print("\nVibLadderDipoleMatrix:")
    for attr in ['mu_x', 'mu_y', 'mu_z']:
        has_attr = hasattr(dipole_viblad, attr)
        if has_attr:
            matrix = getattr(dipole_viblad, attr)
            nnz = np.count_nonzero(matrix)
            print(f"  ✓ {attr}: shape={matrix.shape}, dtype={matrix.dtype}, nnz={nnz}")
        else:
            print(f"  ✗ {attr}: 属性なし")


def main():
    """メイン実行関数"""
    print("新しいシミュレーションシステムの互換性チェック")
    print("=" * 60)
    
    # 属性チェック
    check_dipole_matrix_attributes()
    
    # 実際のシミュレーション互換性チェック
    success_2level = check_twolevel_compatibility()
    success_viblad = check_viblad_compatibility()
    
    print("\n" + "=" * 60)
    print("結果:")
    print(f"  二準位系: {'✓ 成功' if success_2level else '✗ 失敗'}")
    print(f"  振動ラダー系: {'✓ 成功' if success_viblad else '✗ 失敗'}")
    
    if success_2level and success_viblad:
        print("\n🎉 全てのシステムがpropagatorと互換性があります！")
        print("   examples/ のシミュレーションも実行できるはずです。")
    else:
        print("\n⚠️  一部のシステムで問題があります。")
        print("   propagatorの修正が必要かもしれません。")


if __name__ == "__main__":
    main() 