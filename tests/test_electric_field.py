import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import numpy as np
import pytest
from rovibrational_excitation.core.electric_field import ElectricField

def dummy_envelope(tlist, t_center, duration):
    return np.ones_like(tlist)

def test_electricfield_basic():
    tlist = np.linspace(0, 10, 11)
    ef = ElectricField(tlist)
    # 初期化
    assert ef.Efield.shape == (11, 2)
    ef.init_Efield()
    assert np.all(ef.Efield == 0)
    # add_dispersed_Efield
    ef.add_dispersed_Efield(dummy_envelope, duration=5, t_center=5, carrier_freq=0.0, amplitude=1.0, polarization=np.array([1.0,0.0]), const_polarisation=True)
    assert np.any(ef.Efield[:,0] != 0)
    # get_scalar_and_pol
    scalar, pol = ef.get_scalar_and_pol()
    assert scalar.shape[0] == tlist.shape[0]
    assert pol.shape == (2,)
    # get_Efield_spectrum
    freq, E_freq = ef.get_Efield_spectrum()
    assert freq.shape[0] == E_freq.shape[0]
    # エラー: polarization shape
    with pytest.raises(ValueError):
        ef.add_dispersed_Efield(dummy_envelope, duration=5, t_center=5, carrier_freq=0.0, amplitude=1.0, polarization=np.array([1.0,0.0,0.0]))
    # エラー: get_scalar_and_pol（可変偏光時）
    ef2 = ElectricField(tlist)
    with pytest.raises(ValueError):
        ef2.get_scalar_and_pol() 