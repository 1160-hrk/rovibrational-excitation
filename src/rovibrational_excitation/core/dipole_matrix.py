from typing import Callable
import numpy as np
from .basis import LinMolBasis
from ._tdm_rot import tdm_j, tdm_jm_dict
from ._tdm_vib import tdm_v_dict, omega01_domega_to_N

PLANCK_CONSTANT = 6.62607015e-19  # J*fs
DIRAC_CONSTANT = PLANCK_CONSTANT/(2*np.pi)  # J*fs

class LinMolDipoleMatrix:
    def __init__(self, basis:LinMolBasis, mu0_cm, omega01, domega, potential_type="harmonic"):
        self.basis = basis
        self.mu0_cm = mu0_cm
        self.omega01 = omega01
        self.domega = domega
        self.potential_type = potential_type
        self.use_M = self.basis.use_M
        self._generatie_dipole_matrix()
    def _generatie_dipole_matrix(self):
        params = dict(
            basis=self.basis,
            omega01=self.omega01,
            domega=self.domega,
            mu0_cm=self.mu0_cm,
            potential_type=self.potential_type
            )
        if self.use_M:
            p = params.copy()
            p['axis'] = 'x'
            self.mu_x = generate_dipole_matrix_vjm(**p)
            p['axis'] = 'y'
            self.mu_y = generate_dipole_matrix_vjm(**p)
            p['axis'] = 'z'
            self.mu_z = generate_dipole_matrix_vjm(**p)
        else:
            self.mu_x = generate_dipole_matrix_vj(**p)
            self.mu_y = self.mu_x.copy()
            self.mu_z = self.mu_x.copy()
    def __repr__(self):
        return f"LinMolDipoleMatrix(mu0_cm={self.mu0_cm}, omega01={self.omega01}, domega={self.domega}, potential_type={self.potential_type})"
            
def generate_dipole_matrix_j(
    basis: LinMolBasis
    ) -> np.ndarray:
    """
    遷移双極子行列を生成（数値）
    tdm_func: quanta1, quanta2 -> 値 を返す関数（numpyベース）
    """
    size = basis.J_max + 1
    mu = np.zeros((size, size), dtype=np.complex128)
    for i, q1 in enumerate(basis.j_array):
        for j, q2 in enumerate(basis.j_array):
            mu[i, j] = tdm_j(q1, q2)
    return mu

def generate_dipole_matrix_jm(
    basis: LinMolBasis,
    axis='z'
    ) -> np.ndarray:
    """
    遷移双極子行列を生成（数値）
    tdm_func: quanta1, quanta2 -> 値 を返す関数（numpyベース）
    """
    size = (basis.J_max+1)**2
    mu = np.zeros((size, size), dtype=np.complex128)
    basis_jm = basis.basis[:size, 1:]
    tdm_func = tdm_jm_dict[axis]
    for i, q1 in enumerate(basis_jm):
        for j, q2 in enumerate(basis_jm):
            mu[i, j] = tdm_func(q1, q2)
    return mu

def generate_dipole_matrix_vj(
    basis: LinMolBasis,
    omega01:float = 1.0,
    domega:float = 0.0,
    mu0_cm:float = 1.0,
    potential_type='harmonic',
    ) -> np.ndarray:
    """
    遷移双極子行列を生成（単位：PHz/(V/m)）
    tdm_func: quanta1, quanta2 -> 値 を返す関数（numpyベース）
    """
    size = basis.size()
    mu0 = mu0_cm / DIRAC_CONSTANT
    mu = np.zeros((size, size), dtype=np.complex128)
    V = basis.V_max + 1
    tdm_func_v = tdm_v_dict[potential_type]
    dm_j = generate_dipole_matrix_j(basis)
    if domega == 0.0:
        tdm_func_v = tdm_v_dict['harmonic']
    elif potential_type == 'morse':
        omega01_domega_to_N(omega01, domega)
    for i in range(V):
        for j in range(i, V):
            mu[i*size:(i+1)*size, j*size:(j+1)*size] = tdm_func_v(i, j)*dm_j
    mu += mu.T.conj()
    return mu * mu0

def generate_dipole_matrix_vjm(
    basis: LinMolBasis,
    omega01:float = 1.0,
    domega:float = 0.0,
    mu0_cm:float = 1.0,
    axis='z',
    potential_type='harmonic',
    ) -> np.ndarray:
    """
    遷移双極子行列を生成（単位：PHz/(V/m)）
    tdm_func: quanta1, quanta2 -> 値 を返す関数（numpyベース）
    """
    size = basis.size()
    mu0 = mu0_cm / DIRAC_CONSTANT
    mu = np.zeros((size, size), dtype=np.complex128)
    V = basis.V_max + 1
    JJ = (basis.J_max + 1)**2
    dm_jm = generate_dipole_matrix_jm(basis, axis)
    tdm_func_v = tdm_v_dict[potential_type]
    if domega == 0.0:
        tdm_func_v = tdm_v_dict['harmonic']
    elif potential_type == 'morse':
        omega01_domega_to_N(omega01, domega)
    for i in range(V):
        for j in range(i, V):
            mu[i*JJ:(i+1)*JJ, j*JJ:(j+1)*JJ] = tdm_func_v(i, j) * dm_jm
    mu += mu.T.conj()
    return mu * mu0