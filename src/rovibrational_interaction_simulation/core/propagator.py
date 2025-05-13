# シュレディンガー・リウヴィル時間発展
# propagator.py
import numpy as np
from _rk4_lvne import rk4_lvne_traj
from _rk4_schrodinger import rk4_schrodinger_traj

def schrodinger_propagation(H0, Efield, mu_x, mu_y, psi0):
    time4Efield = Efield.tlist
    dt = (time4Efield[1] - time4Efield[0]) * 2
    steps = len(Efield.tlist) // 2
    Efield_vec = Efield.get_vecotr_field()
    phi_traj = rk4_schrodinger_traj(
        H0,
        mu_x,
        mu_y,
        Efield_vec[0],
        Efield_vec[1],
        psi0,
        dt,
        steps,
    )
    return phi_traj

def liouville_propagation(H0, Efield, rho0, mu_x, mu_y):
    time4Efield = Efield.tlist
    dt = (time4Efield[1] - time4Efield[0]) * 2
    steps = len(Efield.tlist) // 2
    Efield_vec = Efield.get_vecotr_field()
    rho_traj = rk4_lvne_traj(
        H0,
        mu_x,
        mu_y,
        Efield_vec[0],
        Efield_vec[1],
        rho0,
        dt,
        steps,
    )
    return rho_traj