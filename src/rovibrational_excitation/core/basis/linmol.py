"""
Linear molecule basis (vibration + rotation + magnetic quantum numbers).
"""

import numpy as np

from .base import BasisBase
from .hamiltonian import Hamiltonian
from rovibrational_excitation.core.units.converters import converter

class LinMolBasis(BasisBase):
    """
    振動(V), 回転(J), 磁気(M)量子数の直積空間における基底の生成と管理を行うクラス。
    
    Parameters
    ----------
    V_max : int
        最大振動量子数
    J_max : int
        最大回転量子数
    use_M : bool, optional
        磁気量子数Mを含めるかどうか
    omega : float, optional
        振動周波数（input_unitsで指定した単位）
    B : float, optional
        回転定数（input_unitsで指定した単位）
    alpha : float, optional
        振動-回転相互作用定数（input_unitsで指定した単位）
    delta_omega : float, optional
        振動の非調和性補正項（input_unitsで指定した単位）
    input_units : str, optional
        入力パラメータの単位（"rad/fs", "cm^-1", "THz", "eV"など）
    output_units : str, optional
        出力ハミルトニアンの単位（"J" または "rad/fs"）
    """

    def __init__(
        self,
        V_max: int,
        J_max: int,
        use_M: bool = True,
        omega: float | None = None,
        B: float | None = None,
        alpha: float = 0.0,
        delta_omega: float = 0.0,
        input_units: str = "rad/fs",
        output_units: str = "J",
        # 後方互換性のための古いパラメータ名
        omega_rad_pfs: float | None = None,
        delta_omega_rad_pfs: float | None = None,
    ):
        self.V_max = V_max
        self.J_max = J_max
        self.use_M = use_M
        self.input_units = input_units
        self.output_units = output_units
        
        # 後方互換性: 古いパラメータ名が使われた場合
        if omega_rad_pfs is not None:
            omega = omega_rad_pfs
            input_units = "rad/fs"
        if delta_omega_rad_pfs is not None:
            delta_omega = delta_omega_rad_pfs
            input_units = "rad/fs"
        
        # 物理パラメータの単位変換と保存
        if input_units in converter.get_supported_units("frequency"):
            conv = converter.convert_frequency(1.0, input_units, "rad/fs")
            self.omega_rad_pfs = omega * conv if omega is not None else 1.0
            self.B_rad_pfs = B * conv if B is not None else 0.001
            self.alpha_rad_pfs = alpha * conv
            self.delta_omega_rad_pfs = delta_omega * conv
        elif input_units in converter.get_supported_units("energy"):
            # エネルギー単位からrad/fsへの変換
            energy_conv = converter.convert_energy(1.0, input_units, "J")
            self.omega_rad_pfs = (omega * energy_conv / Hamiltonian._HBAR * 1e-15 
                                  if omega is not None else 1.0)
            self.B_rad_pfs = (B * energy_conv / Hamiltonian._HBAR * 1e-15 
                              if B is not None else 1.0)
            self.alpha_rad_pfs = alpha * energy_conv / Hamiltonian._HBAR * 1e-15
            self.delta_omega_rad_pfs = delta_omega * energy_conv / Hamiltonian._HBAR * 1e-15
        else:
            raise ValueError(
                f"Unsupported input_units '{input_units}'.\n"
                f"Supported frequency units: {list(converter.get_supported_units('frequency'))}\n"
                f"Supported energy units: {list(converter.get_supported_units('energy'))}"
            )
        
        # 基底の生成
        self.basis = self._generate_basis()
        self.V_array = self.basis[:, 0]
        self.J_array = self.basis[:, 1]
        if self.use_M:
            self.M_array = self.basis[:, 2]
        self.index_map = {tuple(state): i for i, state in enumerate(self.basis)}

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

    def generate_H0(self, **kwargs) -> Hamiltonian:
        """
        保存された物理パラメータから分子の自由ハミルトニアン H0 を生成

        Returns
        -------
        Hamiltonian
            ハミルトニアンオブジェクト（output_unitsで指定した単位）
        """
        # 後方互換性: kwargsが渡された場合はgenerate_H0_with_paramsに委譲
        if kwargs:
            return self.generate_H0_with_params(**kwargs)
            
        # 内部のrad/fs単位のパラメータを使用
        vterm = self.V_array + 0.5
        jterm = self.J_array * (self.J_array + 1)
        energy_freq = (self.omega_rad_pfs + self.delta_omega_rad_pfs) * vterm - self.delta_omega_rad_pfs/2 * vterm**2
        energy_freq += (self.B_rad_pfs - self.alpha_rad_pfs * vterm) * jterm
        
        # Create Hamiltonian in frequency units first
        H0_matrix = np.diag(energy_freq)
        
        # Create basis info for debugging
        basis_info = {
            "basis_type": "LinMol",
            "V_max": self.V_max,
            "J_max": self.J_max,
            "use_M": self.use_M,
            "size": self.size(),
            "omega_rad_pfs": self.omega_rad_pfs,
            "delta_omega_rad_pfs": self.delta_omega_rad_pfs,
            "B_rad_pfs": self.B_rad_pfs,
            "alpha_rad_pfs": self.alpha_rad_pfs,
            "input_units": self.input_units,
            "output_units": self.output_units,
        }
        
        # Create Hamiltonian object in rad/fs
        hamiltonian = Hamiltonian(H0_matrix, "rad/fs", basis_info)
        
        # Convert to requested units
        if self.output_units == "J":
            return hamiltonian.to_energy_units()
        else:
            return hamiltonian

    def generate_H0_with_params(
        self,
        omega=None,
        delta_omega=None,
        B=None,
        alpha=None,
        units=None,
        input_units=None,
        # 後方互換性のための古いパラメータ名
        omega_rad_pfs=None,
        delta_omega_rad_pfs=None,
        B_rad_pfs=None,
        alpha_rad_pfs=None,
        **kwargs,
    ) -> Hamiltonian:
        """
        一時的にパラメータを変更してハミルトニアンを生成（後方互換性）

        Parameters
        ----------
        omega : float, optional
            振動固有周波数（input_unitsで指定した単位）
        delta_omega : float, optional
            振動の非調和性補正項（input_unitsで指定した単位）
        B : float, optional
            回転定数（input_unitsで指定した単位）
        alpha : float, optional
            振動-回転相互作用定数（input_unitsで指定した単位）
        units : {"J", "rad/fs"}, optional
            返すハミルトニアンの単位
        input_units : str, optional
            入力パラメータの単位

        Returns
        -------
        Hamiltonian
            ハミルトニアンオブジェクト（単位情報付き）
        """
        # 後方互換性: 古いパラメータ名の処理
        if omega_rad_pfs is not None:
            omega = omega_rad_pfs
            input_units = "rad/fs"
        if delta_omega_rad_pfs is not None:
            delta_omega = delta_omega_rad_pfs
            input_units = "rad/fs"
        if B_rad_pfs is not None:
            B = B_rad_pfs
            input_units = "rad/fs"
        if alpha_rad_pfs is not None:
            alpha = alpha_rad_pfs
            input_units = "rad/fs"
            
        # 単位の決定
        if input_units is None:
            input_units = self.input_units
        if units is None:
            units = self.output_units
            
        # 単位変換の準備
        if input_units not in converter.get_supported_units("frequency"):
            raise ValueError(
                f"Unsupported input_units '{input_units}'.\n"
                f"Supported units: {list(converter.get_supported_units('frequency'))}"
            )

        conv = converter.convert_frequency(1.0, input_units, "rad/fs")

        # パラメータの決定（指定されなければインスタンスの値を使用）
        omega_rad_pfs = (omega * conv if omega is not None 
                         else self.omega_rad_pfs)
        delta_omega_rad_pfs = (delta_omega * conv if delta_omega is not None 
                               else self.delta_omega_rad_pfs)
        B_rad_pfs = (B * conv if B is not None 
                     else self.B_rad_pfs)
        alpha_rad_pfs = (alpha * conv if alpha is not None 
                         else self.alpha_rad_pfs)

        vterm = self.V_array + 0.5
        jterm = self.J_array * (self.J_array + 1)
        energy_freq = (omega_rad_pfs+delta_omega_rad_pfs) * vterm - delta_omega_rad_pfs/2 * vterm**2
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
            "input_units": input_units,
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
        params = []
        params.append(f"V_max={self.V_max}")
        params.append(f"J_max={self.J_max}")
        params.append(f"use_M={self.use_M}")
        if hasattr(self, 'omega_rad_pfs'):
            params.append(f"ω={self.omega_rad_pfs:.3f} rad/fs")
        if hasattr(self, 'B_rad_pfs'):
            params.append(f"B={self.B_rad_pfs:.3f} rad/fs")
        params.append(f"size={self.size()}")
        return f"LinMolBasis({', '.join(params)})"
