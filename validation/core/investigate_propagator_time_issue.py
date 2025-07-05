import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
import numpy as np

from rovibrational_excitation.core.basis import LinMolBasis
from rovibrational_excitation.core.electric_field import ElectricField, gaussian
from rovibrational_excitation.core.hamiltonian import generate_H0_LinMol
from rovibrational_excitation.core.nondimensionalize import nondimensionalize_system
from rovibrational_excitation.core.propagator import _prepare_args
from rovibrational_excitation.core.basis import StateVector
from rovibrational_excitation.dipole.linmol.cache import LinMolDipoleMatrix


def investigate_time_issue():
    """時間発展長さの問題を詳しく調査"""
    print("=" * 70)
    print("propagatorでの時間発展長さ問題の調査")
    print("=" * 70)
    
    # システム設定
    V_max, J_max = 1, 1
    omega01, domega, mu0_cm = 1.0, 0.01, 1e-30
    
    basis = LinMolBasis(V_max, J_max)
    H0 = generate_H0_LinMol(
        basis,
        omega_rad_phz=omega01,
        delta_omega_rad_phz=domega,
        B_rad_phz=0.01,
    )
    
    dipole_matrix = LinMolDipoleMatrix(
        basis,
        mu0=mu0_cm,
        potential_type="harmonic",
        backend="numpy",
        dense=True,
    )
    
    # 電場設定
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
    
    print("🔍 1. 入力電場の詳細")
    print(f"   時間範囲: {time4Efield[0]:.3f} - {time4Efield[-1]:.3f} fs")
    print(f"   時間ステップ: {Efield.dt:.3f} fs")
    print(f"   時間点数: {len(time4Efield)}")
    print(f"   電場形状: {Efield.get_Efield().shape}")
    
    # 無次元化実行
    print(f"\n🔬 2. 無次元化の詳細")
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
    
    print(f"   無次元時間範囲: {tlist_prime[0]:.6f} - {tlist_prime[-1]:.6f}")
    print(f"   無次元dt: {dt_prime:.6f}")
    print(f"   無次元時間点数: {len(tlist_prime)}")
    print(f"   無次元電場形状: {Efield_prime.shape}")
    
    # _prepare_argsでの処理確認
    print(f"\n🔧 3. _prepare_args での処理")
    
    # 次元ありの場合
    H0_prep, mu_a_prep, mu_b_prep, Ex_prep, Ey_prep, dt_prep, steps_prep = _prepare_args(
        H0, Efield, dipole_matrix, axes="xy"
    )
    print(f"   次元あり:")
    print(f"     Ex形状: {Ex_prep.shape}")
    print(f"     dt: {dt_prep:.6f} fs")
    print(f"     steps: {steps_prep}")
    print(f"     計算される時間長: {steps_prep * dt_prep:.3f} fs")
    
    # 無次元化での準備（手動計算）
    Ex_prime = Efield_prime[:, 0]
    Ey_prime = Efield_prime[:, 1]
    steps_prime = (len(Ex_prime) - 1) // 2
    
    print(f"\n   無次元化:")
    print(f"     Ex'形状: {Ex_prime.shape}")
    print(f"     dt': {dt_prime:.6f}")
    print(f"     steps': {steps_prime}")
    print(f"     計算される無次元時間長: {steps_prime * dt_prime:.6f}")
    print(f"     物理時間換算: {steps_prime * dt_prime * scales.t0 * 1e15:.3f} fs")
    
    # sample_strideの影響確認
    print(f"\n📊 4. sample_strideの影響")
    sample_stride = 10
    
    # 次元ありの出力時間点数
    output_steps_dimensional = steps_prep // sample_stride + 1
    time_length_dimensional = (output_steps_dimensional - 1) * dt_prep * sample_stride
    
    print(f"   次元あり (stride={sample_stride}):")
    print(f"     出力ステップ数: {output_steps_dimensional}")
    print(f"     出力時間長: {time_length_dimensional:.3f} fs")
    
    # 無次元化の出力時間点数
    output_steps_nondimensional = steps_prime // sample_stride + 1
    time_length_nondimensional = (output_steps_nondimensional - 1) * dt_prime * sample_stride * scales.t0 * 1e15
    
    print(f"   無次元化 (stride={sample_stride}):")
    print(f"     出力ステップ数: {output_steps_nondimensional}")
    print(f"     出力時間長: {time_length_nondimensional:.3f} fs")
    
    # 問題の特定
    print(f"\n🎯 5. 問題の特定")
    if steps_prep != steps_prime:
        print(f"   ❌ steps計算に差異: {steps_prep} vs {steps_prime}")
        print(f"      原因: 電場配列の長さまたは計算方法の違い")
    
    if abs(time_length_dimensional - time_length_nondimensional) > 1:
        print(f"   ❌ 出力時間長に差異: {time_length_dimensional:.3f} vs {time_length_nondimensional:.3f} fs")
        print(f"      差異: {abs(time_length_dimensional - time_length_nondimensional):.3f} fs")
    
    # 詳細な時間配列の比較
    print(f"\n🔍 6. 詳細な時間配列の比較")
    
    # 次元ありの時間配列生成（propagatorの方法に合わせる）
    time_dimensional_manual = np.arange(0, output_steps_dimensional * dt_prep * sample_stride, dt_prep * sample_stride)
    
    # 無次元化の時間配列生成
    from rovibrational_excitation.core.nondimensionalize import get_physical_time
    time_nondimensional_manual = get_physical_time(
        np.arange(0, output_steps_nondimensional) * dt_prime * sample_stride, scales
    )
    
    print(f"   次元あり時間配列:")
    print(f"     範囲: {time_dimensional_manual[0]:.3f} - {time_dimensional_manual[-1]:.3f} fs")
    print(f"     点数: {len(time_dimensional_manual)}")
    
    print(f"   無次元化時間配列:")
    print(f"     範囲: {time_nondimensional_manual[0]:.3f} - {time_nondimensional_manual[-1]:.3f} fs")
    print(f"     点数: {len(time_nondimensional_manual)}")
    
    if len(time_dimensional_manual) != len(time_nondimensional_manual):
        print(f"   ❌ 時間配列の長さが異なります")
    
    if abs(time_dimensional_manual[-1] - time_nondimensional_manual[-1]) > 1:
        print(f"   ❌ 最終時間に差異: {abs(time_dimensional_manual[-1] - time_nondimensional_manual[-1]):.3f} fs")
    
    return {
        'steps_dimensional': steps_prep,
        'steps_nondimensional': steps_prime,
        'time_length_dimensional': time_length_dimensional,
        'time_length_nondimensional': time_length_nondimensional,
        'time_arrays_match': len(time_dimensional_manual) == len(time_nondimensional_manual),
    }


if __name__ == "__main__":
    result = investigate_time_issue()
    
    print("\n" + "=" * 70)
    print("調査完了")
    print("=" * 70) 