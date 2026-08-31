import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import numpy as np
import pytest

from rovibrational_excitation.fields import (
    ElectricField,
    apply_dispersion,
    apply_sinusoidal_mod,
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
    ef = ElectricField(tlist, time_units="fs")
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
    ef2 = ElectricField(tlist, time_units="fs")
    with pytest.raises(ValueError):
        ef2.get_scalar_and_pol()


def test_electricfield_initialization():
    """初期化のテスト"""
    tlist = np.linspace(-5, 5, 101)
    with pytest.raises(TypeError, match="time_units"):
        ElectricField(tlist)
    with pytest.raises(TypeError, match="field_units"):
        ElectricField(tlist, time_units="fs", field_units="V/m")

    ef = ElectricField(tlist, time_units="fs")

    # 属性の確認
    assert ef.dt == (tlist[1] - tlist[0])
    assert ef.dt_state == ef.dt * 2
    assert ef.steps_state == len(tlist) // 2
    assert ef.Efield.shape == (101, 2)
    assert np.all(ef.Efield == 0)
    assert ef.add_history == []
    assert ef._constant_pol is None
    assert ef._scalar_field is None

    seconds = ElectricField(tlist * 1.0e-15, time_units="s")
    np.testing.assert_allclose(seconds.get_time_SI(), ef.get_time_SI())
    with pytest.raises(ValueError, match="Unknown time unit"):
        ElectricField(tlist, time_units="fortnight")

    with pytest.raises(TypeError, match="field_units"):
        ef.get_field_scale_info()
    scale_info = ef.get_field_scale_info(field_units="MV/cm")
    assert scale_info == {
        "scale_V_per_m": 0.0,
        "scale_MV_per_cm": 0.0,
        "scale_GV_per_m": 0.0,
        "scale_in_requested_units": 0.0,
        "requested_units": "MV/cm",
    }
    with pytest.raises(ValueError, match="electric-field amplitude"):
        ef.get_field_scale_info(field_units="W/cm^2")


def test_electricfield_multiple_pulses():
    """複数パルスの追加テスト"""
    tlist = np.linspace(-10, 10, 201)
    ef = ElectricField(tlist, time_units="fs")

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
    ef = ElectricField(tlist, time_units="fs")

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
    ef = ElectricField(tlist, time_units="fs")

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
    ef = ElectricField(tlist, time_units="fs")

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
        modulation_depth=0.1,
        delay_fs=0.5,
        phase_rad=0.0,
        mode="phase",
    )

    # 変調後もパルスが存在
    assert np.any(ef.Efield != 0)


def test_electricfield_arbitrary_field():
    """任意電場追加のテスト"""
    tlist = np.linspace(-5, 5, 101)
    ef = ElectricField(tlist, time_units="fs")

    # 任意の電場を作成
    arbitrary_field = np.zeros((101, 2))
    arbitrary_field[:, 0] = np.sin(2 * np.pi * tlist)  # x成分
    arbitrary_field[:, 1] = np.cos(2 * np.pi * tlist)  # y成分

    # 追加
    with pytest.raises(TypeError, match="field_units"):
        ef.add_arbitrary_Efield(arbitrary_field)

    ef.add_arbitrary_Efield(arbitrary_field, field_units="V/m")

    # 形状確認
    assert ef.Efield.shape == arbitrary_field.shape
    np.testing.assert_array_equal(ef.Efield, arbitrary_field)

    # 形状不一致エラー
    wrong_shape_field = np.zeros((50, 2))
    with pytest.raises(ValueError):
        ef.add_arbitrary_Efield(wrong_shape_field, field_units="V/m")


def test_arbitrary_field_units_convert_once_to_v_per_m():
    tlist = np.linspace(-5.0, 5.0, 101)
    field_v_per_m = np.column_stack(
        (
            2.0e8 * np.sin(2 * np.pi * tlist),
            3.0e8 * np.cos(2 * np.pi * tlist),
        )
    )
    canonical = ElectricField(tlist, time_units="fs")
    converted = ElectricField(tlist, time_units="fs")

    canonical.add_arbitrary_Efield(field_v_per_m, field_units="V/m")
    converted.add_arbitrary_Efield(field_v_per_m / 1.0e8, field_units="MV/cm")

    np.testing.assert_allclose(converted.get_Efield_SI(), canonical.get_Efield_SI())
    for invalid_units in ("tesla", "W/cm^2"):
        with pytest.raises(ValueError, match="field_units must be"):
            ElectricField(tlist, time_units="fs").add_arbitrary_Efield(
                field_v_per_m,
                field_units=invalid_units,
            )


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


def test_dispersion_uses_physical_gdd_and_tod_taylor_coefficients():
    tlist = np.linspace(-6.0, 6.0, 121)
    field = np.exp(-((tlist / 2.0) ** 2)).astype(np.complex128)
    center_freq, gdd_fs2, tod_fs3 = 0.11, 4.0, 9.0

    freq = np.fft.fftfreq(tlist.size, d=tlist[1] - tlist[0])
    delta_omega = 2.0 * np.pi * (freq - center_freq)
    phase = 0.5 * gdd_fs2 * delta_omega**2 + tod_fs3 * delta_omega**3 / 6.0
    expected = np.fft.ifft(np.fft.fft(field) * np.exp(-1j * phase))
    actual = apply_dispersion(tlist, field, center_freq, gdd_fs2, tod_fs3)

    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-15)


def test_sinusoidal_phase_modulation_matches_physical_delay_formula():
    tlist = np.linspace(-20.0, 20.0, 401)
    field = np.exp(-((tlist / 4.0) ** 2))
    center_freq = 0.12
    depth = 0.7
    delay_fs = 13.0
    phase_rad = 0.3

    freq = np.fft.rfftfreq(tlist.size, d=tlist[1] - tlist[0])
    argument = 2.0 * np.pi * delay_fs * (freq - center_freq) + phase_rad
    multiplier = np.exp(-1j * depth * np.sin(argument))
    expected = np.fft.irfft(np.fft.rfft(field) * multiplier, n=tlist.size)
    actual = apply_sinusoidal_mod(
        tlist,
        field,
        center_freq,
        depth,
        delay_fs,
        phase_rad,
        "phase",
    )

    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-15)


@pytest.mark.parametrize("mode", ["phase", "amplitude"])
def test_zero_sinusoidal_modulation_depth_is_identity(mode):
    tlist = np.linspace(-4.0, 4.0, 81)
    field = np.column_stack((np.cos(tlist), np.sin(tlist)))
    actual = apply_sinusoidal_mod(tlist, field, 0.1, 0.0, 10.0, 0.2, mode)
    np.testing.assert_array_equal(actual, field)


def test_sinusoidal_amplitude_modulation_matches_depth_formula():
    tlist = np.linspace(-8.0, 8.0, 161)
    field = np.exp(-((tlist / 3.0) ** 2))
    freq = np.fft.rfftfreq(tlist.size, d=tlist[1] - tlist[0])
    argument = 2.0 * np.pi * 7.0 * (freq - 0.08) + 0.4
    multiplier = 1.0 + 0.6 * np.sin(argument)
    expected = np.fft.irfft(np.fft.rfft(field) * multiplier, n=tlist.size)
    actual = apply_sinusoidal_mod(tlist, field, 0.08, 0.6, 7.0, 0.4, "amplitude")
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-15)


@pytest.mark.parametrize("depth", [-0.01, 1.01])
def test_sinusoidal_amplitude_modulation_rejects_depth_outside_unit_interval(depth):
    tlist = np.linspace(-1.0, 1.0, 21)
    field = np.ones(tlist.size)
    with pytest.raises(ValueError, match="0 <= depth <= 1"):
        apply_sinusoidal_mod(tlist, field, 0.1, depth, 1.0, mode="amplitude")


def test_electricfield_spectrum_analysis():
    """スペクトル解析のテスト"""
    tlist = np.linspace(-10, 10, 1001)  # 高分解能
    ef = ElectricField(tlist, time_units="fs")

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
    ef_short = ElectricField(tlist_short, time_units="fs")
    assert ef_short.Efield.shape == (3, 2)

    # 長い時間軸
    tlist_long = np.linspace(-100, 100, 10001)
    ef_long = ElectricField(tlist_long, time_units="fs")
    assert ef_long.Efield.shape == (10001, 2)

    # ゼロ振幅
    ef_zero = ElectricField(np.linspace(-1, 1, 11), time_units="fs")
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
