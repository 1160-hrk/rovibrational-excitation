import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from rovibrational_excitation.core._rk4_schrodinger import rk4_schrodinger_traj
import numpy as np
import matplotlib.pyplot as plt
import time

ti, tf = 0.0, 100000
dt4Efield = 0.01
dt = dt4Efield * 2
time4Efield = np.arange(ti, tf + 3*dt4Efield, dt4Efield)
time4psi = time4Efield[:-1:2]
steps = len(time4psi)
print(len(time4psi), len(time4Efield))
omega0 = 1
H0 = np.array(
    [
        [0.0, 0.0],
        [0.0, omega0]
    ], dtype=np.complex128
)

mu_x = np.array(
    [
        [0.0, 1.0],
        [1.0, 0.0]
    ], dtype=np.complex128
)
mu_y = np.array(
    [
        [0.0, 1.0*1j],
        [1.0*(-1j), 0.0]
    ], dtype=np.complex128
)

duration = 10000
tc = (time4Efield[-1] + time4Efield[0]) / 2
envelope = np.exp(-(time4Efield-tc)**2 / (2 * duration**2))
num_rabi_cycles = 1
amplitude = num_rabi_cycles / (duration*np.sqrt(2*np.pi)) * 2*np.pi
Efield_x = amplitude * envelope * np.cos(omega0 * (time4Efield-tc))
Efield_x = Efield_x.astype(np.float64)
Efield_y = np.zeros_like(Efield_x, dtype=np.float64)

psi0 = np.array([1, 0], dtype=np.complex128)
psi0 = psi0.reshape((len(psi0), 1))
print(psi0.shape)

start = time.time()
print("start")
results = rk4_schrodinger_traj(
    H0,
    mu_x,
    mu_y,
    Efield_x,
    Efield_y,
    psi0,
    dt,
    steps,
)
runtime = time.time() - start
print("end")
print("time:", runtime)

fig, ax = plt.subplots(
    2, 1,
    figsize=(6, 6),
    sharex=True,
    gridspec_kw={"hspace": 0.0, "height_ratios": [0.2, 1]}
    )
ax[0].plot(time4Efield, Efield_x, label="|psi_0|^2")   
ax[1].plot(time4psi, np.abs(results[:-1, 0])**2, label=r"$|\psi_0|^2$")
ax[1].plot(time4psi, np.abs(results[:-1, 1])**2, label=r"$|\psi_1|^2$")
ax[1].plot(time4psi, np.abs(results[:-1, 0])**2 + np.abs(results[:-1, 1])**2, label=r"$|\psi|^2$")
ax[1].set_xlabel("Time")
ax[1].set_ylabel("Probability")
ax[1].legend(loc=6)