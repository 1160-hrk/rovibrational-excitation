"""
Linear molecule basis (vibration + rotation + magnetic quantum numbers).
"""

import numpy as np

from .base import BasisBase
from .hamiltonian import Hamiltonian


class LinMolBasis(BasisBase):
    """
    振動(V), 回転(J), 磁気(M)量子数の直積空間における基底の生成と管理を行うクラス。
    """

    def __init__(
        self,
        V_max: int,
        J_max: int,
        use_M: bool = True,
        omega_rad_pfs: float = 1.0,
        delta_omega_rad_pfs: float = 0.0,
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
        self.omega_rad_pfs = omega_rad_pfs
        self.delta_omega_rad_pfs = delta_omega_rad_pfs

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
        omega_rad_pfs=None,
        delta_omega_rad_pfs=None,
        B_rad_pfs=1.0,
        alpha_rad_pfs=0.0,
        units="J",
        **kwargs,
    ) -> Hamiltonian:
        """
        分子の自由ハミルトニアン H0 を生成

        Parameters
        ----------
        omega_rad_pfs : float, optional
            振動固有周波数（rad/fs）。Noneの場合、初期化時の値を使用。
        delta_omega_rad_pfs : float, optional
            振動の非調和性補正項（rad/fs）。Noneの場合、初期化時の値を使用。
        B_rad_pfs : float
            回転定数（rad/fs）
        alpha_rad_pfs : float
            振動-回転相互作用定数（rad/fs）
        units : {"J", "rad/fs"}, optional
            返すハミルトニアンの単位。デフォルトは"J"（エネルギー単位）

        Returns
        -------
        Hamiltonian
            ハミルトニアンオブジェクト（単位情報付き）
        """
        # Use instance values if not provided
        if omega_rad_pfs is None:
            omega_rad_pfs = self.omega_rad_pfs
        if delta_omega_rad_pfs is None:
            delta_omega_rad_pfs = self.delta_omega_rad_pfs

        vterm = self.V_array + 0.5
        jterm = self.J_array * (self.J_array + 1)
        energy_freq = omega_rad_pfs * vterm - delta_omega_rad_pfs * vterm**2
        energy_freq += (B_rad_pfs - alpha_rad_pfs * vterm) * jterm
        
        # Create Hamiltonian in frequency units first
        H0_matrix = np.diag(energy_freq)
        
        # Create basis info for debugging
        basis_info = {
            "basis_type": "LinMol",
            "V_max": self.V_max,
            "J_max": self.J_max,
            "use_M": self.use_M,
            "size": self.size(),
            "omega_rad_pfs": omega_rad_pfs,
            "delta_omega_rad_pfs": delta_omega_rad_pfs,
            "B_rad_pfs": B_rad_pfs,
            "alpha_rad_pfs": alpha_rad_pfs,
        }
        
        # Create Hamiltonian object in rad/fs
        hamiltonian = Hamiltonian(H0_matrix, "rad/fs", basis_info)
        
        # Convert to requested units
        if units == "J":
            return hamiltonian.to_energy_units()
        else:
            return hamiltonian

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
