import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from rovibrational_excitation.dipole.rot.jm import tdm_jm_x
from rovibrational_excitation.dipole.vib.harmonic import tdm_vib_harm
from rovibrational_excitation.core.basis import LinMolBasis

print('期待される遷移 [0,0,0] ↔ [1,1,±1]:')
rot_plus = tdm_jm_x(0, 0, 1, 1)
rot_minus = tdm_jm_x(0, 0, 1, -1)
vib = tdm_vib_harm(0, 1)

print(f'J=0,M=0 → J=1,M=+1 (x): {rot_plus}')
print(f'J=0,M=0 → J=1,M=-1 (x): {rot_minus}')
print(f'v=0 → v=1: {vib}')
print(f'Total transition [0,0,0] → [1,1,+1]: {rot_plus * vib}')
print(f'Total transition [0,0,0] → [1,1,-1]: {rot_minus * vib}')

print('\n基底インデックス確認:')
basis = LinMolBasis(1, 1)
for i, state in enumerate(basis.basis):
    print(f'{i}: {state}')

print('\n手動で双極子行列を計算:')
import numpy as np

dim = len(basis.basis)
mu_x_manual = np.zeros((dim, dim), dtype=complex)

for i, (v1, J1, M1) in enumerate(basis.basis):
    for j, (v2, J2, M2) in enumerate(basis.basis):
        # 選択則チェック
        if abs(v1 - v2) == 1 and abs(J1 - J2) <= 1 and abs(M1 - M2) <= 1:
            rot = tdm_jm_x(J1, M1, J2, M2)
            vib = tdm_vib_harm(v1, v2)
            mu_x_manual[i, j] = rot * vib
            if rot * vib != 0:
                print(f'  [{v1},{J1},{M1}] → [{v2},{J2},{M2}]: rot={rot:.3f}, vib={vib:.1f}, total={rot*vib:.3f}')

print(f'\n手動計算の非零要素数: {(mu_x_manual != 0).sum()}') 