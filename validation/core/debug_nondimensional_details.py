import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
import time

import matplotlib.pyplot as plt
import numpy as np

from rovibrational_excitation.core.basis import LinMolBasis
from rovibrational_excitation.core.electric_field import ElectricField, gaussian
from rovibrational_excitation.core.hamiltonian import generate_H0_LinMol
from rovibrational_excitation.core.nondimensionalize import nondimensionalize_system, get_physical_time
from rovibrational_excitation.core.propagator import schrodinger_propagation, _prepare_args
from rovibrational_excitation.core.basis import StateVector
from rovibrational_excitation.dipole.linmol.cache import LinMolDipoleMatrix


def detailed_nondimensional_debug():
    """無次元化の詳細デバッグ"""
    print("=" * 70)
    print("無次元化システムの詳細デバッグ分析")
    print("=" * 70)
    
    # システム設定（検証しやすい小さなシステム）
    V_max, J_max = 1, 1
    omega01, domega, mu0_cm = 1.0, 0.01, 1e-30
    axes = "xy"
    
    # 基底とハミルトニアン
    basis = LinMolBasis(V_max, J_max)
    H0 = generate_H0_LinMol(
        basis,
        omega_rad_phz=omega01,
        delta_omega_rad_phz=domega,
        B_rad_phz=0.01,
    )
    
    # 双極子行列
    dipole_matrix = LinMolDipoleMatrix(
        basis,
        mu0=mu0_cm,
        potential_type="harmonic",
        backend="numpy",
        dense=True,
    )
    
    # 初期状態
    state = StateVector(basis)
    state.set_state((0, 0, 0), 1)
    psi0 = state.data
    
    # 電場設定（簡単な設定）
    ti, tf = 0.0, 100
    dt4Efield = 0.02
    time4Efield = np.arange(ti, tf + 2 * dt4Efield, dt4Efield)
    
    duration = 20
    tc = (time4Efield[-1] + time4Efield[0]) / 2
    amplitude = 1e9
    polarization = np.array([1, 0])
    
    Efield = ElectricField(tlist_fs=time4Efield)
    Efield.add_dispersed_Efield(
        envelope_func=gaussian,
        duration=duration,
        t_center=tc,
        carrier_freq=omega01 / (2 * np.pi),
        amplitude=amplitude,
        polarization=polarization,
        const_polarisation=True,
    )
    
    print("\n🔍 1. 元の物理パラメータ")
    print(f"   H0 対角成分: {np.diag(H0)}")
    print(f"   H0 エネルギー差: {np.diff(np.diag(H0))}")
    print(f"   双極子行列最大値: {np.max(np.abs(dipole_matrix.mu_x)):.3e}")
    print(f"   電場最大値: {np.max(np.abs(Efield.get_Efield())):.3e}")
    print(f"   時間範囲: {time4Efield[0]:.1f} - {time4Efield[-1]:.1f} fs")
    print(f"   時間ステップ: {Efield.dt:.3f} fs")
    
    # 無次元化の実行
    print("\n🔬 2. 無次元化プロセス")
    (
        H0_prime,
        mu_x_prime,
        mu_y_prime,
        Efield_prime,
        tlist_prime,
        dt_prime,
        scales,
    ) = nondimensionalize_system(
        H0, dipole_matrix.mu_x, dipole_matrix.mu_y, Efield,
        H0_units="energy", time_units="fs"
    )
    
    print(f"\n📏 スケールファクター:")
    print(f"   E0 = {scales.E0:.3e} J")
    print(f"   mu0 = {scales.mu0:.3e} C·m")
    print(f"   Efield0 = {scales.Efield0:.3e} V/m")
    print(f"   t0 = {scales.t0:.3e} s = {scales.t0 * 1e15:.3f} fs")
    print(f"   λ = {scales.lambda_coupling:.6f}")
    
    print(f"\n🎯 無次元化された値:")
    print(f"   H0' 対角成分: {np.diag(H0_prime)}")
    print(f"   H0' エネルギー差: {np.diff(np.diag(H0_prime))}")
    print(f"   mu_x' 最大値: {np.max(np.abs(mu_x_prime)):.6f}")
    print(f"   Efield' 最大値: {np.max(np.abs(Efield_prime)):.6f}")
    print(f"   t' 範囲: {tlist_prime[0]:.3f} - {tlist_prime[-1]:.3f}")
    print(f"   dt' = {dt_prime:.6f}")
    
    # 時間変換の確認
    print(f"\n⏰ 3. 時間変換の確認")
    time_back_to_fs = get_physical_time(tlist_prime, scales)
    print(f"   元の時間: {time4Efield[0]:.3f} - {time4Efield[-1]:.3f} fs")
    print(f"   逆変換時間: {time_back_to_fs[0]:.3f} - {time_back_to_fs[-1]:.3f} fs")
    print(f"   時間差: max={np.max(np.abs(time4Efield - time_back_to_fs)):.3e} fs")
    
    # propagatorでの_prepare_argsの確認
    print(f"\n🔧 4. propagator内での前処理確認")
    try:
        # 次元ありの場合
        H0_prep, mu_a_prep, mu_b_prep, Ex_prep, Ey_prep, dt_prep, steps_prep = _prepare_args(
            H0, Efield, dipole_matrix, axes=axes
        )
        print(f"   次元あり - mu_a 最大値: {np.max(np.abs(mu_a_prep)):.3e}")
        print(f"   次元あり - Ex 最大値: {np.max(np.abs(Ex_prep)):.3e}")
        print(f"   次元あり - dt: {dt_prep:.6f}")
        print(f"   次元あり - steps: {steps_prep}")
    except Exception as e:
        print(f"   次元あり前処理エラー: {e}")
    
    # 5. 実際の計算実行と詳細比較
    print(f"\n🚀 5. 実際の時間発展計算")
    sample_stride = 10
    
    # 次元ありでの計算
    print("   次元ありでの計算...")
    try:
        time_dimensional, psi_dimensional = schrodinger_propagation(
            H0=H0,
            Efield=Efield,
            dipole_matrix=dipole_matrix,
            psi0=psi0,
            axes=axes,
            return_traj=True,
            return_time_psi=True,
            sample_stride=sample_stride,
            nondimensional=False,
        )
        print(f"     出力時間範囲: {time_dimensional[0]:.3f} - {time_dimensional[-1]:.3f} fs")
        print(f"     出力時間ステップ: {np.mean(np.diff(time_dimensional)):.6f} fs")
        print(f"     波動関数形状: {psi_dimensional.shape}")
    except Exception as e:
        print(f"     次元あり計算エラー: {e}")
        return
        
    # 無次元化での計算
    print("   無次元化での計算...")
    try:
        time_nondimensional, psi_nondimensional = schrodinger_propagation(
            H0=H0,
            Efield=Efield,
            dipole_matrix=dipole_matrix,
            psi0=psi0,
            axes=axes,
            return_traj=True,
            return_time_psi=True,
            sample_stride=sample_stride,
            nondimensional=True,
        )
        print(f"     出力時間範囲: {time_nondimensional[0]:.3f} - {time_nondimensional[-1]:.3f} fs")
        print(f"     出力時間ステップ: {np.mean(np.diff(time_nondimensional)):.6f} fs")
        print(f"     波動関数形状: {psi_nondimensional.shape}")
    except Exception as e:
        print(f"     無次元化計算エラー: {e}")
        return
    
    # 6. 詳細な比較
    print(f"\n📊 6. 詳細比較")
    
    # デフォルト値を設定
    time_diff = np.array([0])
    prob_diff = np.array([0])
    prob_dimensional = np.abs(psi_dimensional)**2
    prob_nondimensional = np.abs(psi_nondimensional)**2
    
    # 時間配列の比較
    if time_dimensional.shape == time_nondimensional.shape:
        time_diff = np.abs(time_dimensional - time_nondimensional)
        print(f"   時間配列の差:")
        print(f"     最大差: {np.max(time_diff):.3e} fs")
        print(f"     平均差: {np.mean(time_diff):.3e} fs")
        print(f"     最大相対差: {np.max(time_diff / np.abs(time_dimensional)):.3e}")
        
        # 時間差が大きい場所の詳細
        max_idx = np.argmax(time_diff)
        print(f"     最大差発生位置: index={max_idx}")
        print(f"       次元あり: {time_dimensional[max_idx]:.6f} fs")
        print(f"       無次元化: {time_nondimensional[max_idx]:.6f} fs")
        print(f"       差: {time_diff[max_idx]:.6f} fs")
    else:
        print(f"   時間配列の形状が異なります: {time_dimensional.shape} vs {time_nondimensional.shape}")
    
    # 波動関数の比較
    if psi_dimensional.shape == psi_nondimensional.shape:
        psi_diff = np.abs(psi_dimensional - psi_nondimensional)
        prob_diff = np.abs(prob_dimensional - prob_nondimensional)
        
        print(f"   波動関数の差:")
        print(f"     振幅最大差: {np.max(psi_diff):.3e}")
        print(f"     存在確率最大差: {np.max(prob_diff):.3e}")
        print(f"     存在確率相対差: {np.max(prob_diff / (prob_dimensional + 1e-16)):.3e}")
        
        # 最大差発生位置の詳細
        max_prob_idx = np.unravel_index(np.argmax(prob_diff), prob_diff.shape)
        print(f"     最大存在確率差発生位置: {max_prob_idx}")
        print(f"       次元あり: {prob_dimensional[max_prob_idx]:.6e}")
        print(f"       無次元化: {prob_nondimensional[max_prob_idx]:.6e}")
        print(f"       差: {prob_diff[max_prob_idx]:.6e}")
    else:
        print(f"   波動関数の形状が異なります: {psi_dimensional.shape} vs {psi_nondimensional.shape}")
    
    # 7. 理論値との比較（簡単なケース）
    print(f"\n🧮 7. 理論値との比較（基底状態のみ）")
    ground_pop_dim = prob_dimensional[:, 0]
    ground_pop_nondim = prob_nondimensional[:, 0]
    
    print(f"   基底状態存在確率:")
    print(f"     初期値 - 次元あり: {ground_pop_dim[0]:.6f}")
    print(f"     初期値 - 無次元化: {ground_pop_nondim[0]:.6f}")
    print(f"     最終値 - 次元あり: {ground_pop_dim[-1]:.6f}")
    print(f"     最終値 - 無次元化: {ground_pop_nondim[-1]:.6f}")
    print(f"     最大差: {np.max(np.abs(ground_pop_dim - ground_pop_nondim)):.3e}")
    
    # 8. 結論
    print(f"\n📋 8. 分析結論")
    time_problem = np.max(time_diff) > 1e-10 if time_dimensional.shape == time_nondimensional.shape else True
    prob_problem = np.max(prob_diff) > 1e-10 if psi_dimensional.shape == psi_nondimensional.shape else True
    
    if time_problem:
        print("   ❌ 時間配列に大きな差異があります")
        print("      → get_physical_time関数またはスケールファクターに問題の可能性")
    
    if prob_problem:
        print("   ❌ 存在確率に大きな差異があります") 
        print("      → 無次元化された方程式の実装に問題の可能性")
        
    if not time_problem and not prob_problem:
        print("   ✅ 無次元化は正常に動作しています")
    
    return {
        'scales': scales,
        'time_dimensional': time_dimensional,
        'time_nondimensional': time_nondimensional,
        'psi_dimensional': psi_dimensional,
        'psi_nondimensional': psi_nondimensional,
    }


def analyze_scale_factor_accuracy():
    """スケールファクター精度の詳細分析"""
    print("\n" + "="*50)
    print("スケールファクター精度分析")
    print("="*50)
    
    # 手動でスケールファクターを計算して比較
    V_max, J_max = 1, 1
    omega01, domega, mu0_cm = 1.0, 0.01, 1e-30
    
    basis = LinMolBasis(V_max, J_max)
    H0 = generate_H0_LinMol(basis, omega_rad_phz=omega01, delta_omega_rad_phz=domega, B_rad_phz=0.01)
    
    dipole_matrix = LinMolDipoleMatrix(basis, mu0=mu0_cm, potential_type="harmonic", backend="numpy", dense=True)
    
    # 簡単な電場
    time4Efield = np.linspace(0, 100, 1001)
    amplitude = 1e9
    Efield = ElectricField(tlist_fs=time4Efield)
    Efield.add_dispersed_Efield(
        envelope_func=gaussian,
        duration=20,
        t_center=50,
        carrier_freq=omega01 / (2 * np.pi),
        amplitude=amplitude,
        polarization=np.array([1, 0]),
        const_polarisation=True,
    )
    
    # 手動計算
    print("🔧 手動スケールファクター計算:")
    
    # エネルギースケール
    H0_diag = np.diag(H0)
    energy_diffs = []
    for i in range(len(H0_diag)):
        for j in range(i+1, len(H0_diag)):
            energy_diffs.append(abs(H0_diag[i] - H0_diag[j]))
    E0_manual = max(energy_diffs) if energy_diffs else H0_diag[0]
    print(f"   E0 (手動): {E0_manual:.3e} J")
    
    # 双極子スケール
    mu_x = dipole_matrix.mu_x
    mu_offdiag = mu_x.copy()
    np.fill_diagonal(mu_offdiag, 0)
    mu0_manual = np.max(np.abs(mu_offdiag))
    print(f"   mu0 (手動): {mu0_manual:.3e} C·m")
    
    # 電場スケール
    Efield0_manual = np.max(np.abs(Efield.get_Efield()))
    print(f"   Efield0 (手動): {Efield0_manual:.3e} V/m")
    
    # 時間スケール
    _HBAR = 1.054571817e-34
    t0_manual = _HBAR / E0_manual
    print(f"   t0 (手動): {t0_manual:.3e} s = {t0_manual * 1e15:.3f} fs")
    
    # 結合強度
    lambda_manual = (Efield0_manual * mu0_manual) / E0_manual
    print(f"   λ (手動): {lambda_manual:.6f}")
    
    # 自動計算と比較
    print("\n🤖 自動計算との比較:")
    (_, _, _, _, _, _, scales_auto) = nondimensionalize_system(
        H0, dipole_matrix.mu_x, dipole_matrix.mu_y, Efield,
        H0_units="energy", time_units="fs"
    )
    
    print(f"   E0 差: {abs(E0_manual - scales_auto.E0):.3e} J")
    print(f"   mu0 差: {abs(mu0_manual - scales_auto.mu0):.3e} C·m")
    print(f"   Efield0 差: {abs(Efield0_manual - scales_auto.Efield0):.3e} V/m")
    print(f"   t0 差: {abs(t0_manual - scales_auto.t0):.3e} s")
    print(f"   λ 差: {abs(lambda_manual - scales_auto.lambda_coupling):.3e}")
    
    if all([
        abs(E0_manual - scales_auto.E0) < 1e-20,
        abs(mu0_manual - scales_auto.mu0) < 1e-35,
        abs(Efield0_manual - scales_auto.Efield0) < 1e5,
        abs(t0_manual - scales_auto.t0) < 1e-20,
        abs(lambda_manual - scales_auto.lambda_coupling) < 1e-10
    ]):
        print("   ✅ スケールファクター計算は正確です")
    else:
        print("   ❌ スケールファクター計算に誤差があります")


if __name__ == "__main__":
    result = detailed_nondimensional_debug()
    analyze_scale_factor_accuracy()
    
    print("\n" + "="*70)
    print("デバッグ完了")
    print("="*70) 