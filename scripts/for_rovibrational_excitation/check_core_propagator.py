import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from rovibrational_excitation.core.propagator import schrodinger_propagation
from rovibrational_excitation.core.electric_field import ElectricField, gaussian
from linmol_dipole.cache import LinMolDipoleMatrix
from vib_tdms.morse import omega01_domega_to_N
from rovibrational_excitation.core.basis import LinMolBasis
from rovibrational_excitation.core.hamiltonian import generate_H0_LinMol
from rovibrational_excitation.core.states import StateVector
import numpy as np
import matplotlib.pyplot as plt
import time

V_max, J_max = 1, 1
omega01, domega, mu0_cm = 1.0, 0.01, 1e-30
omega01_domega_to_N(omega01=omega01, domega=domega)
axes = "xy"

basis = LinMolBasis(V_max, J_max)
H0 = generate_H0_LinMol(basis)

dipole_matrix = LinMolDipoleMatrix(
    basis,              # ← 1. で用意した基底
    mu0=mu0_cm,           # 縮尺係数（a.u. → C·m など）     *任意*
    potential_type="harmonic",   #  'harmonic' / 'morse'
    backend="numpy",    #  'numpy' か 'cupy'  (GPU のときは 'cupy')
    dense=True          #  True→ndarray / False→CSR sparse
)

state = StateVector(basis)
state.set_state((0, 0, 0), 1)

ti, tf = 0.0, 1000
dt4Efield = 0.01
time4Efield = np.arange(ti, tf + 3*dt4Efield, dt4Efield)

duration = 100
tc = (time4Efield[-1] + time4Efield[0]) / 2
envelope = np.exp(-(time4Efield-tc)**2 / (2 * duration**2))
num_rabi_cycles = 1
# amplitude = num_rabi_cycles / (duration*np.sqrt(2*np.pi)) * 2*np.pi 
amplitude = 3e9
polarization = np.array(
    [1, -1j]
)
Efield = ElectricField(tlist=time4Efield)
Efield.add_Efield_disp(
    envelope_func=gaussian,
    duration=duration,
    t_center=tc,
    carrier_freq=3/(2*np.pi),
    amplitude=amplitude,
    polarization=polarization
    )

psi0 = state.data

start = time.time()
print("start")
sample_stride = 1
results = schrodinger_propagation(
    H0=H0,
    Efield=Efield,
    dipole_matrix=dipole_matrix,
    psi0=psi0,
    axes=axes,
    return_traj=True,
    sample_stride=sample_stride
    )
time4psi = time4Efield[::2][:-1:sample_stride]
runtime = time.time() - start
print("end")
print("time:", runtime)

# %%
fig, ax = plt.subplots(
    3, 1,
    figsize=(6, 6),
    sharex=True,
    gridspec_kw={"hspace": 0.0, "height_ratios": [0.2, 0.2,  1]}
    )
ax[0].plot(time4Efield, Efield.Efield[:, 0], label=r"$E_x$")   
ax[1].plot(time4Efield, Efield.Efield[:, 1], label=r"$E_y$")
for i, st in enumerate(basis.basis):
    ax[2].plot(time4psi, np.abs(results[:-1, i])**2, label=r"$|\psi_{"f"{','.join([str(int(s)) for s in st])}"r"}|^2$")
ax[2].plot(time4psi, np.sum(np.abs(results[:-1, :]**2), axis=1), ls=':', c='k')
ax[2].set_xlabel("Time")
ax[2].set_ylabel("Probability")
ax[2].legend(loc=6)