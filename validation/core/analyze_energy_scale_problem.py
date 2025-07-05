import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
import numpy as np

from rovibrational_excitation.core.basis import LinMolBasis
from rovibrational_excitation.core.electric_field import ElectricField, gaussian
from rovibrational_excitation.core.hamiltonian import generate_H0_LinMol
from rovibrational_excitation.dipole.linmol.cache import LinMolDipoleMatrix


def analyze_energy_scale_issue():
    """エネルギースケール問題の詳細分析"""
    print("=" * 70)
    print("エネルギースケール問題の詳細分析")
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
    
    print("🔍 1. パラメータ分析")
    print(f"   omega01 = {omega01} PHz")
    print(f"   domega = {domega} PHz")
    print(f"   B_rad_phz = 0.01 PHz")
    
    # ハミルトニアンの詳細分析
    print(f"\n📊 2. ハミルトニアン詳細")
    H0_diag = np.diag(H0)
    print(f"   H0 対角成分 [J]: {H0_diag}")
    print(f"   H0 対角成分 [eV]: {H0_diag / 1.602176634e-19}")
    
    # エネルギー差の詳細
    print(f"\n⚡ 3. エネルギー差詳細")
    energy_diffs = []
    for i in range(len(H0_diag)):
        for j in range(i+1, len(H0_diag)):
            diff = abs(H0_diag[i] - H0_diag[j])
            energy_diffs.append(diff)
            print(f"   |E_{i} - E_{j}|: {diff:.3e} J = {diff/1.602176634e-19:.3e} eV")
    
    max_diff = max(energy_diffs)
    print(f"   最大エネルギー差: {max_diff:.3e} J = {max_diff/1.602176634e-19:.3e} eV")
    
    # 物理的な意味の解釈
    print(f"\n🎯 4. 物理的解釈")
    _HBAR = 1.054571817e-34
    t_max = _HBAR / max_diff
    print(f"   最大エネルギー差による時間スケール: {t_max:.3e} s = {t_max * 1e15:.3e} fs")
    
    # 典型的な分子系との比較
    print(f"\n📚 5. 典型的な分子系との比較")
    typical_vib = 0.1  # eV (典型的な振動エネルギー)
    typical_rot = 0.001  # eV (典型的な回転エネルギー)
    
    current_max_eV = max_diff / 1.602176634e-19
    print(f"   現在の最大エネルギー差: {current_max_eV:.3e} eV")
    print(f"   典型的な振動エネルギー: {typical_vib} eV")
    print(f"   典型的な回転エネルギー: {typical_rot} eV")
    print(f"   比率 (現在/典型振動): {current_max_eV / typical_vib:.3e}")
    print(f"   比率 (現在/典型回転): {current_max_eV / typical_rot:.3e}")
    
    # 問題の根本原因
    print(f"\n🔍 6. 問題の根本原因")
    print(f"   現在のシステムのエネルギー差は典型的な分子系より{typical_vib/current_max_eV:.0e}倍小さい")
    print(f"   これは以下の原因による可能性があります：")
    print(f"   - omega01 = {omega01} PHz が小さすぎる（典型的には数十〜数百PHz）")
    print(f"   - domega = {domega} PHz が小さすぎる")
    print(f"   - B_rad_phz = 0.01 PHz が小さすぎる")
    
    # 推奨修正案
    print(f"\n💡 7. 推奨修正案")
    print(f"   Option 1: より現実的なパラメータを使用")
    print(f"     omega01 = 100 PHz (約3000 cm⁻¹)")
    print(f"     domega = 10 PHz")
    print(f"     B_rad_phz = 1 PHz")
    
    print(f"\n   Option 2: 無次元化の改良")
    print(f"     - エネルギー差が小さすぎる場合の特別な処理")
    print(f"     - 時間スケールに上限を設ける")
    print(f"     - 異なるスケーリング戦略の使用")
    
    # 修正案の効果予測
    print(f"\n📈 8. 修正案の効果予測")
    
    # Option 1のテスト
    omega01_new = 100.0  # より現実的な値
    domega_new = 10.0
    B_new = 1.0
    
    basis_new = LinMolBasis(V_max, J_max)
    H0_new = generate_H0_LinMol(
        basis_new,
        omega_rad_phz=omega01_new,
        delta_omega_rad_phz=domega_new,
        B_rad_phz=B_new,
    )
    
    H0_diag_new = np.diag(H0_new)
    energy_diffs_new = []
    for i in range(len(H0_diag_new)):
        for j in range(i+1, len(H0_diag_new)):
            energy_diffs_new.append(abs(H0_diag_new[i] - H0_diag_new[j]))
    
    max_diff_new = max(energy_diffs_new)
    t_new = _HBAR / max_diff_new
    
    print(f"   新パラメータでの最大エネルギー差: {max_diff_new:.3e} J")
    print(f"   新パラメータでの時間スケール: {t_new * 1e15:.3f} fs")
    print(f"   改善率: {t_max / t_new:.3e} (時間スケールが短くなる)")
    
    return {
        'current_max_diff': max_diff,
        'current_time_scale': t_max,
        'new_max_diff': max_diff_new,
        'new_time_scale': t_new,
        'improvement_ratio': t_max / t_new
    }


def test_alternative_scaling_strategy():
    """代替スケーリング戦略のテスト"""
    print("\n" + "=" * 70)
    print("代替スケーリング戦略のテスト")
    print("=" * 70)
    
    # 現在のシステム
    V_max, J_max = 1, 1
    omega01, domega, mu0_cm = 1.0, 0.01, 1e-30
    
    basis = LinMolBasis(V_max, J_max)
    H0 = generate_H0_LinMol(
        basis,
        omega_rad_phz=omega01,
        delta_omega_rad_phz=domega,
        B_rad_phz=0.01,
    )
    
    H0_diag = np.diag(H0)
    _HBAR = 1.054571817e-34
    
    print("🔧 代替スケーリング戦略:")
    
    # Strategy 1: 最大エネルギー値をスケールとして使用
    E0_strategy1 = np.max(np.abs(H0_diag))
    t0_strategy1 = _HBAR / E0_strategy1
    print(f"   Strategy 1 (最大エネルギー値): E0={E0_strategy1:.3e} J, t0={t0_strategy1*1e15:.3f} fs")
    
    # Strategy 2: エネルギー幅をスケールとして使用
    E0_strategy2 = np.max(H0_diag) - np.min(H0_diag)
    t0_strategy2 = _HBAR / E0_strategy2 if E0_strategy2 > 0 else float('inf')
    print(f"   Strategy 2 (エネルギー幅): E0={E0_strategy2:.3e} J, t0={t0_strategy2*1e15:.3f} fs")
    
    # Strategy 3: 時間スケール上限を設定
    t0_max = 1000  # fs (上限)
    E0_strategy3 = _HBAR / (t0_max * 1e-15)
    print(f"   Strategy 3 (時間上限{t0_max}fs): E0={E0_strategy3:.3e} J")
    
    # Strategy 4: 物理的な代表エネルギーを使用
    E0_strategy4 = 0.001 * 1.602176634e-19  # 1 meV
    t0_strategy4 = _HBAR / E0_strategy4
    print(f"   Strategy 4 (代表1meV): E0={E0_strategy4:.3e} J, t0={t0_strategy4*1e15:.3f} fs")
    
    print(f"\n💡 推奨: Strategy 3 または 4 が実用的")
    
    return {
        'strategy1': (E0_strategy1, t0_strategy1),
        'strategy2': (E0_strategy2, t0_strategy2),
        'strategy3': (E0_strategy3, t0_max * 1e-15),
        'strategy4': (E0_strategy4, t0_strategy4),
    }


if __name__ == "__main__":
    # エネルギースケール問題の分析
    analysis_result = analyze_energy_scale_issue()
    
    # 代替戦略のテスト
    strategy_result = test_alternative_scaling_strategy()
    
    print("\n" + "=" * 70)
    print("分析完了")
    print("=" * 70) 