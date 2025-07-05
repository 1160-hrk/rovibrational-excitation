# 状態ベクトル・密度行列定義
# states.py
import numpy as np


class StateVector:
    """
    状態ベクトル（純粋状態）を表すクラス。
    """

    def __init__(self, basis):
        self.basis = basis
        self.data = np.zeros((basis.size(), 1), dtype=np.complex128)

    def set_state(self, state, amplitude=1.0):
        index = self.basis.get_index(state)
        if index is not None:
            self.data[:] = 0  # 他をゼロに
            self.data[index] = amplitude
        else:
            raise ValueError("指定された量子数は基底に存在しません。")

    def norm(self):
        return np.linalg.norm(self.data)

    def normalize(self):
        n = self.norm()
        if n != 0:
            self.data /= n

    def copy(self):
        new_state = StateVector(self.basis)
        new_state.data = self.data.copy()
        return new_state

    def __repr__(self):
        return f"StateVector(norm={self.norm():.3f})"


class DensityMatrix:
    """
    密度行列（混合状態）を表すクラス。
    """

    def __init__(self, basis):
        self.basis = basis
        size = basis.size()
        self.data = np.zeros((size, size), dtype=np.complex128)

    def set_diagonal(self, populations):
        if len(populations) != self.basis.size():
            raise ValueError("ポピュレーションの長さが基底サイズと一致しません。")
        self.data[:, :] = 0
        np.fill_diagonal(self.data, populations)

    def set_pure_state(self, state_vector):
        v = state_vector.data
        self.data = np.outer(v, v.conj())

    def trace(self):
        return np.trace(self.data)

    def normalize(self):
        tr = self.trace()
        if tr != 0:
            self.data /= tr

    def copy(self):
        new_dm = DensityMatrix(self.basis)
        new_dm.data = self.data.copy()
        return new_dm

    def __repr__(self):
        return f"DensityMatrix(trace={self.trace():.3f})"
