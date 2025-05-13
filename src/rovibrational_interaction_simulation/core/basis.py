# 基底とインデックス変換
# basis.py
import numpy as np

class VJMBasis:
    """
    振動(V), 回転(J), 磁気(M)量子数の直積空間における基底の生成と管理を行うクラス。
    """
    def __init__(self, V_max, J_max):
        self.V_max = V_max
        self.J_max = J_max
        self.basis = self._generate_basis()
        self.v_array = self.basis[:, 0]
        self.j_array = self.basis[:, 1]
        self.m_array = self.basis[:, 2]
        self.index_map = {tuple(state): i for i, state in enumerate(self.basis)}

    def _generate_basis(self):
        """
        V, J, M の全ての組み合わせからなる基底を生成。
        Returns
        -------
        list of list: 各要素が [V, J, M] のリスト
        """
        basis = []
        for V in range(self.V_max + 1):
            for J in range(self.J_max + 1):
                for M in range(-J, J + 1):
                    basis.append([V, J, M])
        return np.array(basis)

    def get_index(self, V, J, M):
        """
        量子数からインデックスを取得
        """
        return self.index_map.get((V, J, M), None)

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

    def __repr__(self):
        return f"VJMBasis(V_max={self.V_max}, J_max={self.J_max}, size={self.size()})"


class VJBasis:
    """
    振動(V), 回転(J)量子数の直積空間における基底の生成と管理を行うクラス。
    """
    def __init__(self, V_max, J_max):
        self.V_max = V_max
        self.J_max = J_max
        self.basis = self._generate_basis()
        self.index_map = {tuple(state): i for i, state in enumerate(self.basis)}

    def _generate_basis(self):
        """
        V, J の全ての組み合わせからなる基底を生成。
        Returns
        -------
        list of list: 各要素が [V, J] のリスト
        """
        basis = []
        for V in range(self.V_max + 1):
            for J in range(self.J_max + 1):
                basis.append([V, J])
        return basis
    def get_index(self, V, J):
        """
        量子数からインデックスを取得
        """
        return self.index_map.get((V, J), None)
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
    def __repr__(self):
        return f"VJBasis(V_max={self.V_max}, J_max={self.J_max}, size={self.size()})"