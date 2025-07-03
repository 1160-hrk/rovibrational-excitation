import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import time

import matplotlib.pyplot as plt
import numpy as np

from rovibrational_excitation.core.basis import LinMolBasis
from rovibrational_excitation.core.electric_field import ElectricField, gaussian_fwhm
from rovibrational_excitation.core.propagator import schrodinger_propagation
from rovibrational_excitation.core.basis import StateVector
from rovibrational_excitation.dipole.linmol import LinMolDipoleMatrix
from rovibrational_excitation.dipole.vib.morse import omega01_domega_to_N

V_max, J_max = 1, 1  # スパース効果を確認するため大きなサイズに
omega01, domega, mu0_cm = 1.0, 0.01, 1e-30
omega01_domega_to_N(omega01=omega01, domega=domega)
axes = "zx"

basis = LinMolBasis(V_max, J_max)
H0 = basis.generate_H0(
    omega_rad_pfs=omega01,
    delta_omega_rad_pfs=domega,
    B_rad_pfs=0.01,
    units="J"  # エネルギー単位で取得
)

dipole_matrix = LinMolDipoleMatrix(
    basis,  # ← 1. で用意した基底
    mu0=mu0_cm,  # 縮尺係数（a.u. → C·m など）     *任意*
    potential_type="harmonic",  #  'harmonic' / 'morse'
    backend="numpy",  #  'numpy' か 'cupy'  (GPU のときは 'cupy')
    dense=True,  #  False→CSR sparse でメモリ節約
    units="C*m"  # 単位を明示的に指定
)

state = StateVector(basis)
state.set_state((0, 0, 0), 1)

ti, tf = 0.0, 1000  # スパース効果をテストするため短縮
dt4Efield = 0.01
time4Efield = np.arange(ti, tf + 2 * dt4Efield, dt4Efield)

duration = 100
tc = (time4Efield[-1] + time4Efield[0]) / 2
envelope = np.exp(-((time4Efield - tc) ** 2) / (2 * duration**2))
num_rabi_cycles = 1
# amplitude = num_rabi_cycles / (duration*np.sqrt(2*np.pi)) * 2*np.pi
amplitude = 3e10
polarization = np.array([1, 0])
Efield = ElectricField(tlist=time4Efield)
Efield.add_dispersed_Efield(
    envelope_func=gaussian_fwhm,
    duration=duration,
    t_center=tc,
    carrier_freq=omega01 / (2 * np.pi),
    amplitude=amplitude,
    polarization=polarization,
    const_polarisation=False,
)

psi0 = state.data

if __name__ == "__main__":
    start = time.time()
    print("start")
    sample_stride = 10  # メモリ使用量を1/100に削減
    time4psi, psi_t = schrodinger_propagation(
        hamiltonian=H0,  # Hamiltonianオブジェクト
        Efield=Efield,
        dipole_matrix=dipole_matrix,
        psi0=psi0,
        axes=axes,
        return_traj=True,
        return_time_psi=True,
        sample_stride=sample_stride,
    )
    runtime = time.time() - start
    print("end")
    print("time:", runtime)

    fig, ax = plt.subplots(
        3,
        1,
        figsize=(6, 6),
        sharex=True,
        gridspec_kw={"hspace": 0.0, "height_ratios": [0.2, 0.2, 1]},
    )
    Efield_data = Efield.get_Efield()
    ax[0].plot(time4Efield, Efield_data[:, 0], label=r"$E_x$")
    ax[1].plot(time4Efield, Efield_data[:, 1], label=r"$E_y$")
    for i, st in enumerate(basis.basis):
        ax[2].plot(
            time4psi,
            np.abs(psi_t[:, i]) ** 2,
            label=r"$|\psi_{" f"{','.join([str(int(s)) for s in st])}" r"}|^2$",
        )
    ax[2].plot(time4psi, np.sum(np.abs(psi_t[:, :] ** 2), axis=1), ls=":", c="k")
    ax[2].set_xlabel("Time")
    ax[2].set_ylabel("Probability")
    ax[2].legend(loc=6)
    plt.savefig("propagator_check_result.png", dpi=300, bbox_inches="tight")
    print("プロットを propagator_check_result.png に保存しました")
