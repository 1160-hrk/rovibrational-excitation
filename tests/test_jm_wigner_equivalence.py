import numpy as np
import pytest

from rovibrational_excitation.dipole.rot.jm import (
    tdm_jm_x as tdm_x_analytic,
    tdm_jm_y as tdm_y_analytic,
    tdm_jm_z as tdm_z_analytic,
)

from rovibrational_excitation.dipole.rot.jm_wigner import (
    tdm_jm_x_wigner as tdm_x_wigner,
    tdm_jm_y_wigner as tdm_y_wigner,
    tdm_jm_z_wigner as tdm_z_wigner,
)


@pytest.mark.parametrize("J_max", [0, 1, 2, 3, 4])
def test_jm_wigner_equivalence(J_max):
    """Compare analytic and Wigner-3j implementations for multiple J, M pairs."""
    tol = 1e-12
    for J1 in range(J_max + 1):
        for dJ in (-1, +1):
            J2 = J1 + dJ
            if J2 < 0:
                continue
            for M1 in range(-J1, J1 + 1):
                for dM in (-1, 0, +1):
                    M2 = M1 + dM
                    if abs(M2) > J2:
                        continue

                    # x-component --------------------------------------------------
                    val_a = tdm_x_analytic(J1, M1, J2, M2)
                    val_b = tdm_x_wigner(J1, M1, J2, M2)
                    assert np.allclose(val_a, val_b, atol=tol, rtol=0), (
                        f"Mismatch in μx for J1={J1},M1={M1}→J2={J2},M2={M2}: "
                        f"analytic={val_a}, wigner={val_b}"
                    )

                    # y-component --------------------------------------------------
                    val_a = tdm_y_analytic(J1, M1, J2, M2)
                    val_b = tdm_y_wigner(J1, M1, J2, M2)
                    assert np.allclose(val_a, val_b, atol=tol, rtol=0), (
                        f"Mismatch in μy for J1={J1},M1={M1}→J2={J2},M2={M2}: "
                        f"analytic={val_a}, wigner={val_b}"
                    )

                    # z-component --------------------------------------------------
                    val_a = tdm_z_analytic(J1, M1, J2, M2)
                    val_b = tdm_z_wigner(J1, M1, J2, M2)
                    assert np.allclose(val_a, val_b, atol=tol, rtol=0), (
                        f"Mismatch in μz for J1={J1},M1={M1}→J2={J2},M2={M2}: "
                        f"analytic={val_a}, wigner={val_b}"
                    ) 