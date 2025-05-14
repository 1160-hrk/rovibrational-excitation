# シュレディンガー・リウヴィル時間発展
# propagator.py
import numpy as np
from _rk4_lvne import rk4_lvne_traj, rk4_lvne
from _rk4_schrodinger import rk4_schrodinger_traj, rk4_schrodinger

def schrodinger_propagation(H0, Efield, mu_x, mu_y, psi0, return_traj=True, sample_stride=1):
    time4Efield = Efield.tlist
    dt = (time4Efield[1] - time4Efield[0]) * 2
    steps = len(Efield.tlist) // 2
    Efield_vec = Efield.get_vecotr_field()
    params = [H0,
        mu_x,
        mu_y,
        Efield_vec[0],
        Efield_vec[1],
        psi0,
        dt,
        steps,]
    if return_traj:
        params.append(sample_stride)
        result = rk4_schrodinger_traj(*params)
    else:
        result = rk4_schrodinger(*params)
    return result

def liouville_propagation(H0, Efield, rho0, mu_x, mu_y, return_traj=True, sample_stride=1):
    time4Efield = Efield.tlist
    dt = (time4Efield[1] - time4Efield[0]) * 2
    steps = len(Efield.tlist) // 2
    Efield_vec = Efield.get_vecotr_field()
    params = [
        H0,
        mu_x,
        mu_y,
        Efield_vec[0],
        Efield_vec[1],
        rho0,
        dt,
        steps,
        ]
    if return_traj:
        params.append(sample_stride)
        result = rk4_lvne_traj(*params)
    else:
        result = rk4_lvne(*params)
    return result