import sys
import os
import tracemalloc
import psutil
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from rovibrational_excitation.core.basis import LinMolBasis
from rovibrational_excitation.dipole.linmol.cache import LinMolDipoleMatrix


def get_memory_usage():
    """現在のメモリ使用量を取得（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def analyze_sparsity(matrix, name):
    """行列のスパース性を分析"""
    if sp.issparse(matrix):
        total_elements = matrix.shape[0] * matrix.shape[1]
        nonzero_elements = matrix.nnz
        sparsity = (total_elements - nonzero_elements) / total_elements * 100
        print(f"{name}: {matrix.shape}, スパース率: {sparsity:.1f}%, 非零要素: {nonzero_elements}")
        print(f"  メモリサイズ: {matrix.data.nbytes/1024/1024:.4f} MB (dense: {total_elements*16/1024/1024:.4f} MB)")
    else:
        total_elements = matrix.shape[0] * matrix.shape[1]
        nonzero_elements = np.count_nonzero(matrix)
        sparsity = (total_elements - nonzero_elements) / total_elements * 100
        print(f"{name}: {matrix.shape}, スパース率: {sparsity:.1f}%, 非零要素: {nonzero_elements}")
        print(f"  メモリサイズ: {matrix.nbytes/1024/1024:.4f} MB")


def compare_eigendecomposition(V_max, J_max):
    """dense vs sparse 固有値分解の比較"""
    
    print(f"\n=== V_max={V_max}, J_max={J_max} ===")
    
    # 基底の作成
    basis = LinMolBasis(V_max, J_max, use_M=True)
    dim = len(basis.basis)
    print(f"基底次元: {dim}")
    
    # 偏光ベクトル
    pol = np.array([1.0+0.0j, 1.0j], dtype=np.complex128)
    
    # === Dense 行列での処理 ===
    print("\n--- Dense 行列 ---")
    tracemalloc.start()
    _ = get_memory_usage()
    
    dipole_dense = LinMolDipoleMatrix(
        basis, mu0=1e-30, potential_type="harmonic",
        backend="numpy", dense=True
    )
    
    mu_x_dense = dipole_dense.mu_x
    mu_y_dense = dipole_dense.mu_y
    
    analyze_sparsity(mu_x_dense, "μ_x (dense)")
    analyze_sparsity(mu_y_dense, "μ_y (dense)")
    
    M_raw_dense = pol[0] * mu_x_dense + pol[1] * mu_y_dense
    A_dense = 0.5 * (M_raw_dense + M_raw_dense.conj().T)
    
    analyze_sparsity(A_dense, "A (dense)")
    
    mem_before_eigen_dense = get_memory_usage()
    eigvals_dense, U_dense = np.linalg.eigh(A_dense)
    mem_after_eigen_dense = get_memory_usage()
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"Dense 固有値分解メモリ増加: {mem_after_eigen_dense - mem_before_eigen_dense:.1f} MB")
    print(f"Dense ピークメモリ: {peak/1024/1024:.1f} MB")
    
    # === Sparse 行列での処理 ===
    print("\n--- Sparse 行列 ---")
    tracemalloc.start()
    _ = get_memory_usage()
    
    dipole_sparse = LinMolDipoleMatrix(
        basis, mu0=1e-30, potential_type="harmonic",
        backend="numpy", dense=False
    )
    
    mu_x_sparse = dipole_sparse.mu_x
    mu_y_sparse = dipole_sparse.mu_y
    
    analyze_sparsity(mu_x_sparse, "μ_x (sparse)")
    analyze_sparsity(mu_y_sparse, "μ_y (sparse)")
    
    # スパース行列の演算
    M_raw_sparse = pol[0] * mu_x_sparse + pol[1] * mu_y_sparse
    A_sparse = 0.5 * (M_raw_sparse + M_raw_sparse.getH())  # .getH() は共役転置
    
    analyze_sparsity(A_sparse, "A (sparse)")
    
    mem_before_eigen_sparse = get_memory_usage()
    
    # スパース固有値分解（全固有値が必要な場合）
    if dim <= 100:  # 小さい場合は全固有値を計算
        try:
            # スパース行列でも全固有値が必要な場合はdenseに変換
            A_dense_from_sparse = A_sparse.toarray()
            eigvals_sparse, U_sparse = np.linalg.eigh(A_dense_from_sparse)
            sparse_method = "toarray + eigh"
        except MemoryError:
            # メモリ不足の場合は部分固有値分解
            k = min(dim-1, 10)  # 最大10個の固有値
            eigvals_sparse, U_sparse = spla.eigsh(A_sparse, k=k, which='LA')
            sparse_method = f"eigsh (k={k})"
    else:
        # 大きい場合は部分固有値分解
        k = min(dim-1, 20)
        eigvals_sparse, U_sparse = spla.eigsh(A_sparse, k=k, which='LA')
        sparse_method = f"eigsh (k={k})"
    
    mem_after_eigen_sparse = get_memory_usage()
    
    current_sparse, peak_sparse = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"Sparse 固有値分解方法: {sparse_method}")
    print(f"Sparse 固有値分解メモリ増加: {mem_after_eigen_sparse - mem_before_eigen_sparse:.1f} MB")
    print(f"Sparse ピークメモリ: {peak_sparse/1024/1024:.1f} MB")
    
    # 結果の比較
    print("\n--- 比較結果 ---")
    print(f"メモリ節約: {(mem_after_eigen_dense - mem_before_eigen_dense) - (mem_after_eigen_sparse - mem_before_eigen_sparse):.1f} MB")
    
    if len(eigvals_dense) == len(eigvals_sparse):
        max_diff = np.max(np.abs(np.sort(eigvals_dense) - np.sort(eigvals_sparse)))
        print(f"固有値の最大誤差: {max_diff:.2e}")
    
    return {
        'dim': dim,
        'dense_memory': mem_after_eigen_dense - mem_before_eigen_dense,
        'sparse_memory': mem_after_eigen_sparse - mem_before_eigen_sparse,
        'sparse_method': sparse_method
    }


if __name__ == "__main__":
    print("Sparse vs Dense 固有値分解の比較")
    print("=" * 60)
    
    # 異なるサイズでテスト
    test_cases = [
        (1, 1),   # 8次元
        (2, 2),   # 27次元  
        (3, 3),   # 64次元
        (2, 4),   # 75次元
        (4, 4),   # 125次元
    ]
    
    results = []
    for V_max, J_max in test_cases:
        try:
            result = compare_eigendecomposition(V_max, J_max)
            results.append(result)
        except Exception as e:
            print(f"V_max={V_max}, J_max={J_max}: Error - {e}")
    
    # 結果の要約
    print("\n" + "=" * 60)
    print("結果要約:")
    print("次元 | Dense メモリ | Sparse メモリ | 節約量 | Sparse 方法")
    print("-" * 60)
    for result in results:
        savings = result['dense_memory'] - result['sparse_memory']
        print(f"{result['dim']:4d} | {result['dense_memory']:10.1f} MB | {result['sparse_memory']:11.1f} MB | {savings:6.1f} MB | {result['sparse_method']}") 