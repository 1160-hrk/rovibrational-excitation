import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import numpy as np
import pytest

from rovibrational_excitation.core.electric_field import (
    ElectricField,
    apply_dispersion,
    gaussian,
    gaussian_fwhm,
    lorentzian,
    lorentzian_fwhm,
    voigt,
    voigt_fwhm,
)


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
    ef.add_dispersed_Efield(
        dummy_envelope,
        duration=5,
        t_center=5,
        carrier_freq=0.0,
        amplitude=1.0,
        polarization=np.array([1.0, 0.0]),
        const_polarisation=True,
    )
    assert np.any(ef.Efield[:, 0] != 0)
    # get_scalar_and_pol
    scalar, pol = ef.get_scalar_and_pol()
    assert scalar.shape[0] == tlist.shape[0]
    assert pol.shape == (2,)
    # get_Efield_spectrum
    freq, E_freq = ef.get_Efield_spectrum()
    assert freq.shape[0] == E_freq.shape[0]
    # エラー: polarization shape
    with pytest.raises(ValueError):
        ef.add_dispersed_Efield(
            dummy_envelope,
            duration=5,
            t_center=5,
            carrier_freq=0.0,
            amplitude=1.0,
            polarization=np.array([1.0, 0.0, 0.0]),
        )
    # エラー: get_scalar_and_pol（可変偏光時）
    ef2 = ElectricField(tlist)
    with pytest.raises(ValueError):
        ef2.get_scalar_and_pol()


def test_electricfield_initialization():
    """初期化のテスト"""
    tlist = np.linspace(-5, 5, 101)
    ef = ElectricField(tlist)

    # 属性の確認
    assert ef.dt == (tlist[1] - tlist[0])
    assert ef.dt_state == ef.dt * 2
    assert ef.steps_state == len(tlist) // 2
    assert ef.Efield.shape == (101, 2)
    assert np.all(ef.Efield == 0)
    assert ef.add_history == []
    assert ef._constant_pol is None
    assert ef._scalar_field is None


def test_electricfield_multiple_pulses():
    """複数パルスの追加テスト"""
    tlist = np.linspace(-10, 10, 201)
    ef = ElectricField(tlist)

    # 第1パルス（x偏光）
    ef.add_dispersed_Efield(
        gaussian_fwhm,
        duration=2.0,
        t_center=-2.0,
        carrier_freq=1.0,
        amplitude=1.0,
        polarization=np.array([1.0, 0.0]),
        const_polarisation=True,
    )

    # 第2パルス（同じ偏光）
    ef.add_dispersed_Efield(
        gaussian_fwhm,
        duration=2.0,
        t_center=2.0,
        carrier_freq=1.0,
        amplitude=0.5,
        polarization=np.array([1.0, 0.0]),
        const_polarisation=True,
    )

    # 偏光は一定のまま
    scalar, pol = ef.get_scalar_and_pol()
    np.testing.assert_array_almost_equal(pol, [1.0, 0.0])


def test_electricfield_variable_polarization():
    """可変偏光のテスト"""
    tlist = np.linspace(-10, 10, 201)
    ef = ElectricField(tlist)

    # 第1パルス（x偏光）
    ef.add_dispersed_Efield(
        gaussian_fwhm,
        duration=2.0,
        t_center=-2.0,
        carrier_freq=1.0,
        amplitude=1.0,
        polarization=np.array([1.0, 0.0]),
    )

    # 第2パルス（y偏光） - 偏光が変わる
    ef.add_dispersed_Efield(
        gaussian_fwhm,
        duration=2.0,
        t_center=2.0,
        carrier_freq=1.0,
        amplitude=1.0,
        polarization=np.array([0.0, 1.0]),
    )

    # 偏光が可変になっている
    assert ef._constant_pol is False
    with pytest.raises(ValueError):
        ef.get_scalar_and_pol()


def test_electricfield_dispersion():
    """分散効果のテスト"""
    tlist = np.linspace(-10, 10, 201)
    ef = ElectricField(tlist)

    # GDD, TODありのパルス
    ef.add_dispersed_Efield(
        gaussian_fwhm,
        duration=2.0,
        t_center=0.0,
        carrier_freq=1.0,
        amplitude=1.0,
        polarization=np.array([1.0, 0.0]),
        gdd=0.1,
        tod=0.01,
        const_polarisation=True,
    )

    # パルスが追加されている
    assert np.any(ef.Efield != 0)

    # スペクトル確認
    freq, E_freq = ef.get_Efield_spectrum()
    assert freq.shape[0] == E_freq.shape[0]


def test_electricfield_modulation():
    """変調機能のテスト"""
    tlist = np.linspace(-10, 10, 201)
    ef = ElectricField(tlist)

    # ベースパルス
    ef.add_dispersed_Efield(
        gaussian_fwhm,
        duration=2.0,
        t_center=0.0,
        carrier_freq=1.0,
        amplitude=1.0,
        polarization=np.array([1.0, 0.0]),
        const_polarisation=True,
    )

    # 正弦波変調を適用
    ef.apply_sinusoidal_mod(
        center_freq=1.0,
        amplitude=0.1,
        carrier_freq=0.5,
        phase_rad=0.0,
        type_mod="phase",
    )

    # 変調後もパルスが存在
    assert np.any(ef.Efield != 0)


def test_electricfield_arbitrary_field():
    """任意電場追加のテスト"""
    tlist = np.linspace(-5, 5, 101)
    ef = ElectricField(tlist)

    # 任意の電場を作成
    arbitrary_field = np.zeros((101, 2))
    arbitrary_field[:, 0] = np.sin(2 * np.pi * tlist)  # x成分
    arbitrary_field[:, 1] = np.cos(2 * np.pi * tlist)  # y成分

    # 追加
    ef.add_arbitrary_Efield(arbitrary_field)

    # 形状確認
    assert ef.Efield.shape == arbitrary_field.shape
    np.testing.assert_array_equal(ef.Efield, arbitrary_field)

    # 形状不一致エラー
    wrong_shape_field = np.zeros((50, 2))
    with pytest.raises(ValueError):
        ef.add_arbitrary_Efield(wrong_shape_field)


def test_envelope_functions():
    """包絡線関数のテスト"""
    x = np.linspace(-5, 5, 101)
    xc, sigma, gamma = 0.0, 1.0, 0.5
    fwhm = 2.0

    # Gaussian
    gauss = gaussian(x, xc, sigma)
    assert gauss.max() == 1.0  # 中心で最大
    assert len(gauss) == len(x)

    # Gaussian FWHM
    gauss_fwhm = gaussian_fwhm(x, xc, fwhm)
    assert gauss_fwhm.max() == 1.0

    # Lorentzian
    lorentz = lorentzian(x, xc, gamma)
    assert lorentz.max() == 1.0

    # Lorentzian FWHM
    lorentz_fwhm = lorentzian_fwhm(x, xc, fwhm)
    assert lorentz_fwhm.max() == 1.0

    # Voigt
    voigt_profile = voigt(x, xc, sigma, gamma)
    assert len(voigt_profile) == len(x)

    # Voigt FWHM
    voigt_fwhm_profile = voigt_fwhm(x, xc, fwhm, fwhm)
    assert len(voigt_fwhm_profile) == len(x)


def test_apply_dispersion():
    """分散適用関数のテスト"""
    tlist = np.linspace(-5, 5, 101)

    # 複素電場を作成
    Efield = np.exp(1j * 2 * np.pi * 1.0 * tlist).reshape(-1, 1)

    # 分散を適用
    Efield_disp = apply_dispersion(tlist, Efield, center_freq=1.0, gdd=0.1, tod=0.01)

    # 形状が保持される
    assert Efield_disp.shape == Efield.shape

    # 分散により波形が変化
    assert not np.allclose(Efield, Efield_disp)


def test_electricfield_spectrum_analysis():
    """スペクトル解析のテスト"""
    tlist = np.linspace(-10, 10, 1001)  # 高分解能
    ef = ElectricField(tlist)

    # 既知周波数のパルス
    carrier_freq = 2.0
    ef.add_dispersed_Efield(
        gaussian_fwhm,
        duration=1.0,
        t_center=0.0,
        carrier_freq=carrier_freq,
        amplitude=1.0,
        polarization=np.array([1.0, 0.0]),
        const_polarisation=True,
    )

    # スペクトル取得
    freq, E_freq = ef.get_Efield_spectrum()

    # キャリア周波数付近でピークを持つ
    power_spectrum = np.abs(E_freq[:, 0]) ** 2
    peak_freq_idx = np.argmax(power_spectrum)
    peak_freq = freq[peak_freq_idx]

    # キャリア周波数の近くにピークがある（許容誤差あり）
    assert abs(peak_freq - carrier_freq) < 0.1


def test_electricfield_edge_cases():
    """エッジケースのテスト"""
    # 短い時間軸
    tlist_short = np.linspace(0, 1, 3)
    ef_short = ElectricField(tlist_short)
    assert ef_short.Efield.shape == (3, 2)

    # 長い時間軸
    tlist_long = np.linspace(-100, 100, 10001)
    ef_long = ElectricField(tlist_long)
    assert ef_long.Efield.shape == (10001, 2)

    # ゼロ振幅
    ef_zero = ElectricField(np.linspace(-1, 1, 11))
    ef_zero.add_dispersed_Efield(
        gaussian_fwhm,
        duration=1.0,
        t_center=0.0,
        carrier_freq=1.0,
        amplitude=0.0,
        polarization=np.array([1.0, 0.0]),
        const_polarisation=True,
    )
    # ゼロ振幅でも偏光情報は設定される
    scalar, pol = ef_zero.get_scalar_and_pol()
    np.testing.assert_array_almost_equal(pol, [1.0, 0.0])
