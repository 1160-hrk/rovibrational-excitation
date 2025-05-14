# 基底とインデックス変換
# basis.py
import numpy as np

class LinMolBasis:
    """
    振動(V), 回転(J), 磁気(M)量子数の直積空間における基底の生成と管理を行うクラス。
    """
    def __init__(self, V_max:int, J_max:int, use_M:bool = True):
        self.V_max = V_max
        self.J_max = J_max
        self.use_M = use_M
        self.basis = self._generate_basis()
        self.v_array = self.basis[:, 0]
        self.j_array = self.basis[:, 1]
        self.m_array = self.basis[:, 2]
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
        if hasattr(state, '__iter__'):
            if not isinstance(state, tuple):
                state = tuple(state)
        return self.index_map.get(state, None)

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

    def get_border_indices_j(self):
        if self.use_M:
            inds = np.matlib.repmat(np.arange(self.J_max+1)**2, self.V_max+1, 1) + np.arange(self.V_max+1).reshape((self.V_max+1, 1))*(self.J_max+1)**2
            return inds.flatten()
        else:
            raise ValueError('M is not defined, so each index is the border of J number.')
    
    def get_border_indices_v(self):
        if self.use_M:
            inds = np.arange(0, self.size(), (self.J_max+1)**2)
        else:
            inds = np.arange(0, self.size(), self.J_max+1)
        return inds
        
    def __repr__(self):
        return f"VJMBasis(V_max={self.V_max}, J_max={self.J_max}, size={self.size()})"
