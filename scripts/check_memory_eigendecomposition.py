import sys
import os
import tracemalloc
import psutil
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from rovibrational_excitation.core.basis import LinMolBasis
from rovibrational_excitation.dipole.linmol.cache import LinMolDipoleMatrix


def get_memory_usage():
    """現在のメモリ使用量を取得（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def check_eigendecomposition_memory(V_max, J_max):
    """固有値分解のメモリ使用量をチェック"""
    
    print(f"\n=== V_max={V_max}, J_max={J_max} ===")
    
    # メモリ使用量の追跡開始
    tracemalloc.start()
    mem_start = get_memory_usage()
    print(f"開始時メモリ: {mem_start:.1f} MB")
    
    # 1. 基底の作成
    basis = LinMolBasis(V_max, J_max, use_M=True)
    dim = len(basis.basis)
    print(f"基底次元: {dim}")
    
    mem_after_basis = get_memory_usage()
    print(f"基底作成後メモリ: {mem_after_basis:.1f} MB (+{mem_after_basis-mem_start:.1f} MB)")
    
    # 2. 双極子行列の作成
    dipole_matrix = LinMolDipoleMatrix(
        basis,
        mu0=1e-30,
        potential_type="harmonic",
        backend="numpy",
        dense=True
    )
    
    # μ_x, μ_y を取得
    mu_x = dipole_matrix.mu_x
    mu_y = dipole_matrix.mu_y
    
    mem_after_dipole = get_memory_usage()
    print(f"双極子行列作成後メモリ: {mem_after_dipole:.1f} MB (+{mem_after_dipole-mem_after_basis:.1f} MB)")
    
    # 各行列のサイズを確認
    matrix_size_mb = mu_x.nbytes / 1024 / 1024
    print(f"μ_x行列サイズ: {mu_x.shape}, {matrix_size_mb:.2f} MB")
    print(f"μ_y行列サイズ: {mu_y.shape}, {mu_y.nbytes/1024/1024:.2f} MB")
    
    # 3. 偏光ベクトルで組み合わせ
    pol = np.array([1.0+0.0j, 1.0j], dtype=np.complex128)  # 円偏光
    M_raw = pol[0] * mu_x + pol[1] * mu_y
    
    mem_after_combination = get_memory_usage()
    print(f"行列組み合わせ後メモリ: {mem_after_combination:.1f} MB (+{mem_after_combination-mem_after_dipole:.1f} MB)")
    print(f"M_raw行列サイズ: {M_raw.shape}, {M_raw.nbytes/1024/1024:.2f} MB")
    
    # 4. エルミート化
    A = 0.5 * (M_raw + M_raw.conj().T)
    
    mem_after_hermitian = get_memory_usage()
    print(f"エルミート化後メモリ: {mem_after_hermitian:.1f} MB (+{mem_after_hermitian-mem_after_combination:.1f} MB)")
    print(f"A行列サイズ: {A.shape}, {A.nbytes/1024/1024:.2f} MB")
    
    # 5. 固有値分解
    print("固有値分解開始...")
    eigvals, U = np.linalg.eigh(A)
    
    mem_after_eigen = get_memory_usage()
    print(f"固有値分解後メモリ: {mem_after_eigen:.1f} MB (+{mem_after_eigen-mem_after_hermitian:.1f} MB)")
    print(f"固有値配列サイズ: {eigvals.shape}, {eigvals.nbytes/1024/1024:.4f} MB")
    print(f"固有ベクトル行列サイズ: {U.shape}, {U.nbytes/1024/1024:.2f} MB")
    
    # U_H = U.conj().T を作成
    U_H = U.conj().T
    
    mem_after_uh = get_memory_usage()
    print(f"U_H作成後メモリ: {mem_after_uh:.1f} MB (+{mem_after_uh-mem_after_eigen:.1f} MB)")
    print(f"U_H行列サイズ: {U_H.shape}, {U_H.nbytes/1024/1024:.2f} MB")
    
    # トレースマロックでの詳細情報
    current, peak = tracemalloc.get_traced_memory()
    print(f"\nTracemalloc:")
    print(f"  現在のメモリ使用量: {current/1024/1024:.1f} MB")
    print(f"  ピークメモリ使用量: {peak/1024/1024:.1f} MB")
    
    tracemalloc.stop()
    
    # 総メモリ使用量
    total_increase = mem_after_uh - mem_start
    print(f"\n総メモリ増加量: {total_increase:.1f} MB")
    
    return {
        'dim': dim,
        'total_memory_mb': total_increase,
        'matrix_size_mb': matrix_size_mb,
        'eigen_memory_mb': mem_after_eigen - mem_after_hermitian
    }


if __name__ == "__main__":
    print("固有値分解のメモリ使用量チェック")
    print("=" * 50)
    
    # 異なるサイズでテスト
    test_cases = [
        (1, 1),   # 8次元
        (2, 2),   # 18次元
        (3, 3),   # 32次元
        (2, 4),   # 30次元
        (4, 4),   # 50次元
    ]
    
    results = []
    for V_max, J_max in test_cases:
        try:
            result = check_eigendecomposition_memory(V_max, J_max)
            results.append(result)
        except MemoryError as e:
            print(f"V_max={V_max}, J_max={J_max}: MemoryError - {e}")
            break
        except Exception as e:
            print(f"V_max={V_max}, J_max={J_max}: Error - {e}")
    
    # 結果の要約
    print("\n" + "=" * 50)
    print("結果要約:")
    print("基底次元 | 総メモリ増加 | 行列サイズ | 固有値分解増加")
    print("-" * 50)
    for result in results:
        print(f"{result['dim']:8d} | {result['total_memory_mb']:10.1f} MB | {result['matrix_size_mb']:8.2f} MB | {result['eigen_memory_mb']:12.1f} MB") 