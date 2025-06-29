"""
Linear molecule basis (vibration + rotation + magnetic quantum numbers).
"""

import numpy as np

from .base import BasisBase


class LinMolBasis(BasisBase):
    """
    振動(V), 回転(J), 磁気(M)量子数の直積空間における基底の生成と管理を行うクラス。
    """

    def __init__(
        self,
        V_max: int,
        J_max: int,
        use_M: bool = True,
        omega_rad_phz: float = 1.0,
        delta_omega_rad_phz: float = 0.0,
    ):
        self.V_max = V_max
        self.J_max = J_max
        self.use_M = use_M
        self.basis = self._generate_basis()
        self.V_array = self.basis[:, 0]
        self.J_array = self.basis[:, 1]
        if self.use_M:
            self.M_array = self.basis[:, 2]
        self.index_map = {tuple(state): i for i, state in enumerate(self.basis)}
        self.omega_rad_phz = omega_rad_phz
        self.delta_omega_rad_phz = delta_omega_rad_phz

    def _generate_basis(self):
        """
        V, J, MもしくはV, J の全ての組み合わせからなる基底を生成。
        Returns
        -------
        list of list: 各要素が [V, J, M]または[V, J] のリスト
        """
        basis = []
        for V in range(self.V_max + 1):
            for J in range(self.J_max + 1):
                if self.use_M:
                    for M in range(-J, J + 1):
                        basis.append([V, J, M])
                else:
                    basis.append([V, J])
        return np.array(basis)

    def get_index(self, state):
        """
        量子数からインデックスを取得
        """
        if hasattr(state, "__iter__"):
            if not isinstance(state, tuple):
                state = tuple(state)
        result = self.index_map.get(state, None)
        if result is None:
            raise ValueError(f"State {state} not found in basis")
        return result

    def get_state(self, index):
        """
        インデックスから量子状態を取得
        """
        return self.basis[index]

    def size(self):
        """
        全基底のサイズ（次元数）を返す
        """
        return len(self.basis)

    def generate_H0(
        self,
        omega_rad_phz=None,
        delta_omega_rad_phz=None,
        B_rad_phz=1.0,
        alpha_rad_phz=0.0,
        **kwargs,
    ):
        """
        分子の自由ハミルトニアン H0 を生成（単位：rad * PHz）
        E(V, J) = ω*(V+1/2) - Δω*(V+1/2)**2 + (B - α*(V+1/2))*J*(J+1)

        Parameters
        ----------
        omega_rad_phz : float, optional
            振動固有周波数（rad/fs）。Noneの場合、初期化時の値を使用。
        delta_omega_rad_phz : float, optional
            振動の非調和性補正項（rad/fs）。Noneの場合、初期化時の値を使用。
        B_rad_phz : float
            回転定数（rad/fs）
        alpha_rad_phz : float
            振動-回転相互作用定数（rad/fs）

        Returns
        -------
        np.ndarray
            Hamiltonian matrix of shape (size, size)
        """
        # Use instance values if not provided
        if omega_rad_phz is None:
            omega_rad_phz = self.omega_rad_phz
        if delta_omega_rad_phz is None:
            delta_omega_rad_phz = self.delta_omega_rad_phz

        vterm = self.V_array + 0.5
        jterm = self.J_array * (self.J_array + 1)
        energy = omega_rad_phz * vterm - delta_omega_rad_phz * vterm**2
        energy += (B_rad_phz - alpha_rad_phz * vterm) * jterm
        H0 = np.diag(energy)
        return H0

    def get_border_indices_j(self):
        if self.use_M:
            inds = (
                np.tile(np.arange(self.J_max + 1) ** 2, (self.V_max + 1, 1))
                + np.arange(self.V_max + 1).reshape((self.V_max + 1, 1))
                * (self.J_max + 1) ** 2
            )
            return inds.flatten()
        else:
            raise ValueError(
                "M is not defined, so each index is the border of J number."
            )

    def get_border_indices_v(self):
        if self.use_M:
            inds = np.arange(0, self.size(), (self.J_max + 1) ** 2)
        else:
            inds = np.arange(0, self.size(), self.J_max + 1)
        return inds

    def __repr__(self):
        return f"LinMolBasis(V_max={self.V_max}, J_max={self.J_max}, use_M={self.use_M}, size={self.size()})"
