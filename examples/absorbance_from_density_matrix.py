#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 17:01:02 2023.

@author: hirokitsusaka
"""

import numpy as np
import functions_for_TDSE_and_LvNE2 as fun

h_dirac = 1.055*10**(-34)  # ディラック定数 [J*s]
ee = 1.601*10**(-19)  # 素電荷 [C]
c = 2.998*10**(8)  # 光速 [m/s]
eps = 8.854*10**(-12)  # 真空の誘電率 [F/m]
kb = 1.380649*10**(-23)  # ボルツマン定数 [J/K]

# opt_len = 1*1e-3
# pressure = 3e4
# temperature = 300
# dens_num = pressure/(kb*temperature)
m_CO2 = 44e-3/6.023e23

def prepare_variables(Nv, Nj, use_projection_number=True, T2=500,
                      m_mol=m_CO2, temp=300, l=1e-3, p=3e4,
                      pol=[1, 0], pol_rad_is_same=True, pol_rad=[1, 0],
                      Wavenumber=np.arange(2100, 2400, 0.01),
                      make_2d=False, use_wn_v=False):
    """Prepare the variables to calculate the absorbance spectrum.

    Parameters
    ----------
    Nv : int
        the number of vibrational level (not the highest number).
    Nj : int
        the number of rotational level (not the highest number).
    use_projection_number : bool, optional
        if using a projection quantum number m or not.
        The default is True.
    T2 : double, optional
        relaxation time of coherence [ps].
        The default is 200.
    pol : list, optional
        polarization vector of laser.
        The default is [1, 0].
    pol_rad_is_same : bool, optional
        whether the polarization of the radiation light is same with that of
        interaction or not.
    pol_rad : list, optional
        polarization of the radiation light.
    Wavenumber : numpy.ndarray, optional
        X axis of the absorbance spectrum.
        The default is np.arange(2000, 2400, 0.01).
    make_2d : bool, optional
        if execute the function 'calculate absorbance',
        please set this to 'True'
        The default is False.

    Returns
    -------
    None.

    """
    global gamma_coh, temperature, opt_len, pressure, dens_num, mass_mole,\
        N_level, omega_vj_vpjp_mat, tdm_x_mat, tdm_y_mat,\
        tdm_mat, tdm_mat_con, ind_tdm_mat, wavenumber, omega,\
        made_2d, omega_2d, one_over_1j_omega_m_omega_vj_vpjp, rho_dia_pm1hop
    gamma_coh = 1/(T2*1e-12)
    opt_len = l
    pressure = p
    temperature = temp
    dens_num = pressure/(kb*temperature)
    mass_mole = m_mol
    if use_projection_number:
        N_level = Nv*Nj**2
        energy_array = fun.Make_Rovibrational_Energy_array(
            Nv, Nj, use_projection_number=use_projection_number,
            use_wn_v=use_wn_v)
        energy_array_vstack = np.tile(energy_array, (N_level, 1))
        omega_vj_vpjp_mat = ((energy_array_vstack-energy_array_vstack.T)/h_dirac
                             - 1j * gamma_coh)
        rho_dia_pm1hop = np.zeros((N_level, N_level))
        for vi in range(Nv):
            for vj in range(Nv):
                if abs(vi-vj) < 2:
                    rho_dia_pm1hop[
                        vi*Nj**2:(vi+1)*Nj**2, vj*Nj**2:(vj+1)*Nj**2
                        ] = (np.ones((Nj**2, Nj**2)))
        tdm_x_mat = fun.tdm_factor_matrix_VJM(
            Nv, Nj,
            use_projection_number=True, axis='x') * fun.tdm0
        tdm_y_mat = fun.tdm_factor_matrix_VJM(
            Nv, Nj,
            use_projection_number=True, axis='y') * fun.tdm0
        tdm_mat = tdm_x_mat * pol[0] + tdm_y_mat * pol[1]
        tdm_mat_con = (
            tdm_x_mat * np.conjugate(pol_rad[0])
            + tdm_y_mat * np.conjugate(pol_rad[1])
            )
    else:
        N_level = Nv*Nj
        energy_array = fun.Make_Rovibrational_Energy_array(
            Nv, Nj, use_projection_number=use_projection_number,
            use_wn_v=use_wn_v)
        energy_array_vstack = np.tile(energy_array, (N_level, 1))
        omega_vj_vpjp_mat = ((energy_array_vstack-energy_array_vstack.T)/h_dirac
                             - 1j * gamma_coh)
        rho_dia_pm1hop = np.zeros((N_level, N_level))
        for vi in range(Nv):
            for vj in range(Nv):
                if abs(vi-vj) < 2:
                    rho_dia_pm1hop[
                        vi*Nj:(vi+1)*Nj, vj*Nj:(vj+1)*Nj
                        ] = (np.ones((Nj, Nj)))
        tdm_mat = fun.tdm_factor_matrix_VJM(
            Nv, Nj,
            use_projection_number=False, axis='x') * fun.tdm0
        tdm_mat_con = tdm_mat
    ind_tdm_mat = np.array(np.where(tdm_mat != 0))
    wavenumber = Wavenumber
    omega = 2*np.pi*c*1e2*wavenumber
    made_2d = False
    if make_2d is True:
        omega = np.reshape(omega, (len(omega), 1))
        omega_2d = omega @ np.ones((1, ind_tdm_mat.shape[1]))
        one_over_1j_omega_m_omega_vj_vpjp = 1/(
            1j*(omega_2d + omega_vj_vpjp_mat[tuple(ind_tdm_mat)])
            )
        made_2d = True
    return None


def absorbance_spectrum(rho):
    """Calculate the absorbance spectrum from density matrix.

    Parameters
    ----------
    rho : numpy.ndarray or cupy.array
        density matrix.

    Returns
    -------
    absorbance spectrum.

    """
    if made_2d is True:
        # resp_lin_per_mole = np.zeros(len(wavenumber), dtype=np.complex128)
        rho = rho * rho_dia_pm1hop
        rho_after_pr_int = (tdm_mat@rho-rho@tdm_mat)
        intensity_resp_lin_per_mole = (
            1j/h_dirac
            * tdm_mat_con[tuple(np.flip(ind_tdm_mat, axis=0))]
            * rho_after_pr_int[tuple(ind_tdm_mat)]
            )
        resp_lin_per_mole_2d = (
            -one_over_1j_omega_m_omega_vj_vpjp * intensity_resp_lin_per_mole
            )
        print(resp_lin_per_mole_2d.shape)
        print(intensity_resp_lin_per_mole.shape)
        resp_lin_per_mole = np.sum(resp_lin_per_mole_2d, axis=1)
        absorbance = (2*opt_len*omega[:, 0]/c
                      * (np.sqrt(1+resp_lin_per_mole/eps*dens_num/3)).imag
                      )
        absorbance *= np.log10(np.exp(1)) * 1000
    else:
        print(
            'Please set True for the value of make_2d for the function,\
 prepare_variables.')
        return None
    return absorbance


def absorbance_spectrum_for_loop(rho):
    """Calculate the absorbance spectrum from density matrix.

    Parameters
    ----------
    rho : numpy.ndarray or cupy.array
        density matrix.

    Returns
    -------
    absorbance spectrum.

    """
    rho = rho * rho_dia_pm1hop
    resp_lin_per_mole = np.zeros(len(wavenumber), dtype=np.complex128)
    rho_after_pr_int = (tdm_mat@rho-rho@tdm_mat_con)
    # for ind_vj_vpjp in tqdm(ind_tdm_mat.T):
    for ind_vj_vpjp in ind_tdm_mat.T:
        resp_lin_per_mole += -1j/h_dirac*(
            (tdm_mat_con[tuple((ind_vj_vpjp))]
             * rho_after_pr_int[tuple(np.flip(ind_vj_vpjp))])
            )/(1j*(omega+omega_vj_vpjp_mat[tuple(ind_vj_vpjp)]))
    absorbance = (2*opt_len*omega/c
                  * (np.sqrt(1+resp_lin_per_mole/eps*dens_num/3)).imag
                  )
    absorbance *= np.log10(np.exp(1)) * 1000
    return absorbance


def absorbance_spectrum_w_doppler_broadening(rho):
    """Calculate the absorbance spectrum from density matrix.

    Parameters
    ----------
    rho : numpy.ndarray or cupy.array
        density matrix.

    Returns
    -------
    absorbance spectrum.

    """
    rho = rho * rho_dia_pm1hop
    resp_lin_per_mole = np.zeros(len(wavenumber), dtype=np.complex128)
    rho_after_pr_int = (tdm_mat@rho-rho@tdm_mat_con)
    # for ind_vj_vpjp in tqdm(ind_tdm_mat.T):
    for ind_vj_vpjp in ind_tdm_mat.T:
        lr = -1j/h_dirac*(
            (tdm_mat_con[tuple((ind_vj_vpjp))]
             * rho_after_pr_int[tuple(np.flip(ind_vj_vpjp))])
            )/(1j*(omega+omega_vj_vpjp_mat[tuple(ind_vj_vpjp)]))
        print(len(lr))
        lr = convolution_w_doppler(
            omega, lr,
            np.real(omega_vj_vpjp_mat[tuple(ind_vj_vpjp)]), temperature, mass_mole)
        resp_lin_per_mole += lr
    absorbance = (2*opt_len*omega/c
                  * (np.sqrt(1+resp_lin_per_mole/eps*dens_num/3)).imag
                  )
    absorbance *= np.log10(np.exp(1)) * 1000
    return absorbance


def PFID_spectrum_for_loop(rho):
    """Calculate the absorbance spectrum from density matrix.

    Parameters
    ----------
    rho : numpy.ndarray or cupy.array
        density matrix.

    Returns
    -------
    absorbance spectrum.

    """
    # rho = rho * rho_dia_pm1hop
    resp_lin_per_mole = np.zeros(len(wavenumber), dtype=np.complex128)
    rho_after_pr_int = rho
    # for ind_vj_vpjp in tqdm(ind_tdm_mat.T):
    for ind_vj_vpjp in ind_tdm_mat.T:
        resp_lin_per_mole += -(
            (tdm_mat_con[tuple((ind_vj_vpjp))]
             * rho_after_pr_int[tuple(np.flip(ind_vj_vpjp))])
            )/(1j*(omega+omega_vj_vpjp_mat[tuple(ind_vj_vpjp)]))
    absorbance = (2*opt_len*omega/c
                  * (np.sqrt(1+resp_lin_per_mole/eps*dens_num/3)).imag
                  )
    absorbance *= np.log10(np.exp(1)) * 1000
    return absorbance


def radiation_spectrum_for_loop(rho):
    """Calculate the radiation spectrum from density matrix.

    e.g.) You can use this function to calculate the PFID.
            1. calculate the interaction of probe.
            rho_after_pr_int = (tdm_mat@rho-rho@tdm_mat)
            2. calculate the time evolution of rho_after_pr_int(density matrix)
            with laser pulse.
            3. use this function and you get the spectrum of PFID.

    Parameters
    ----------
    rho : numpy.ndarray or cupy.array
        density matrix.

    Returns
    -------
    absorbance spectrum.

    """
    rho = rho * rho_dia_pm1hop
    resp_lin_per_mole = np.zeros(len(wavenumber), dtype=np.complex128)
    for ind_vj_vpjp in ind_tdm_mat.T:
        resp_lin_per_mole += -(
            (tdm_mat_con[tuple(np.flip(ind_vj_vpjp))]
             * rho[tuple(ind_vj_vpjp)])
            )/(1j*(omega+omega_vj_vpjp_mat[tuple(ind_vj_vpjp)]))
    absorbance = (2*opt_len*omega/c
                  * (np.sqrt(1+resp_lin_per_mole/eps*dens_num/3)).imag
                  )
    absorbance *= np.log10(np.exp(1)) * 1000
    return absorbance


def absorbance_spectrum_from_rho_and_mu(
        rho, mu, H0, wn, T2=100):
    """Calculate the absorbance spectrum from density matrix.

    Input variable is density matrix, transition dipole moment matrix,
    eigen Hamiltonian, wavenumber.

    Parameters
    ----------
    rho : TYPE
        Density matrix of the system
        you would like to calculate the absorbancespectrum.
    mu : TYPE
        The matrix of transition dipole moment of the system.
        The size is same with rho. The unit is [C*m].
    H0 : np.ndarray
        Eigen Hamiltonian of the system. the size is same with rho.
        The unit is [J].
    wn : np.ndarray
        bottom axis of absorbance spectrum. The unit is [cm^{-1}].
    T2 : double, optional
        Relaxation time of coherence. The unit is [ps].
        The default is 100.

    Returns
    -------
    absorbance : np.ndarray.
        Absorbance spectrum. The size is same with wn. The unit is[mOD].

    """
    energy_array = np.diag(H0)
    energy_array_vstack = np.tile(energy_array, (N_level, 1))
    omega_ij = ((energy_array_vstack-energy_array_vstack.T)/h_dirac
                - 1j/T2)
    omega = wn * 2*np.pi * c * 1e2
    ind_mu = np.array(np.where(mu != 0))
    resp_lin_per_mole = np.zeros(len(wavenumber), dtype=np.complex128)
    rho = rho * rho_dia_pm1hop
    rho_after_pr_int = (mu@rho-rho@mu)
    # for ind_vj_vpjp in tqdm(ind_tdm_mat.T):
    for ind_ij in ind_mu.T:
        resp_lin_per_mole += -1j/h_dirac*(
            (mu[tuple((ind_ij))]
             * rho_after_pr_int[tuple(np.flip(ind_ij))])
            )/(1j*(omega+omega_ij[tuple(ind_ij)]))
    absorbance = (2*opt_len*omega/c
                  * (np.sqrt(1+resp_lin_per_mole/eps*dens_num/3)).imag
                  )
    absorbance *= np.log10(np.exp(1)) * 1000
    return absorbance


def doppler(k, k0, temp_K, mass_kg):
    """Return normalized gaussian function whose line width is doppler width.

    Integrated value is 1.

    Parameters
    ----------
    k : numpy.array
        horizontal axis, which is typically wavenumber axis data.
    k0 : float
        center value of gaussian function,
        which is typically central wavenumber.
    temp_K : float
        Temperature of the gas in Kelvin.
    mass_kg : float
        mass of a molecule in kg.

    Returns
    -------
    y : numpy.array
        gaussian function, which describe the doppler broadening.

    """
    dk = np.sqrt(kb*temp_K/mass_kg)/c*k0
    y = np.sqrt(1/(2*np.pi))/dk*np.exp(-k**2/(2*dk**2))
    return y

def sinc(k, dk):
    """Return normalized sinc function. 1st root is dk/2.

    Integrated value is 1.

    Parameters
    ----------
    k : numpy.array
        horizontal axis, which is typically wavenumber axis data.
    dk : float
        Double the value of the 1st root of sinc square function,
        which typically means the resolusion.

    Returns
    -------
    y : numpy.array
        sinc function, which is device function of FTIR.

    """
    y = np.sin(2*np.pi*k/dk)/(np.pi*k)
    return y


def sinc_square(k, dk):
    """Return square of normalized sinc function. 1st root is dk/2.

    Integrated value is 1.

    Parameters
    ----------
    k : numpy.array
        horizontal axis, which is typically wavenumber axis data.
    dk : float
        Double the value of the 1st root of sinc function,
        which typically means the resolusion.

    Returns
    -------
    y : numpy.array
        sinc function, which is device function of FTIR.

    """
    y = (np.sin(2*np.pi*k/dk)/(np.pi*k))**2 * dk/2
    return y


def convolution_w_doppler(x, y, k0, temp_K, mass_kg):
    """Calculate the convolution of y data with doppler function.

    Parameters
    ----------
    x : numpy.array
        horizontal axis, which is typically wavenumber axis data.
    y : numpy.array
        vertical axis.
    k0 : float
        center value of gaussian function,
        which is typically central wavenumber.
    temp_K : float
        Temperature of the gas in Kelvin.
    mass_kg : float
        mass of a molecule in kg.

    Returns
    -------
    y_conv : numpy.array
        The y data after convolution.

    """
    dx = np.average(x[1:]-x[:-1])
    dk = np.sqrt(kb*temp_K/mass_kg)/c*k0
    x_dop = np.arange(-3*dk, 3*dk, dx)
    print(len(x_dop))
    # print(f'dk:{dk},dx:{dx}')
    # device_function = sinc(wavenumber_device, k_res)
    dop = doppler(x_dop, k0, temp_K, mass_kg)
    if len(x_dop) < 15:
        y_conv = y
    else:
        y_conv = np.convolve(y, dop, mode='same') * dx
    return y_conv


def convolution_w_sinc(x, y, dk):
    """Calculate the convolution of y data with sinc function.

    Parameters
    ----------
    x : numpy.array
        horizontal axis, which is typically wavenumber axis data.
    y : numpy.array
        vertical axis.
    dk : float
        Double the value of the first root of sinc function.

    Returns
    -------
    y_conv : numpy.array
        The y data after convolution.

    """
    dx = np.average(x[1:]-x[:-1])
    x_device = np.arange(-5*dk, 5*dk, dx)
    # device_function = sinc(wavenumber_device, k_res)
    device_function = sinc(x_device, dk)
    y_conv = np.convolve(y, device_function, mode='same') * dx
    return y_conv


def convolution_w_sinc_square(x, y, dk):
    """Calculate the convolution of y data with sinc function.

    Parameters
    ----------
    x : numpy.array
        horizontal axis, which is typically wavenumber axis data.
    y : numpy.array
        vertical axis.
    dk : float
        Double the value of the first root of sinc function.

    Returns
    -------
    y_conv : numpy.array
        The y data after convolution.

    """
    dx = np.average(x[1:]-x[:-1])
    x_device = np.arange(-5*dk, 5*dk, dx)
    # device_function = sinc(wavenumber_device, k_res)
    device_function = sinc_square(x_device, dk)
    y_conv = np.convolve(y, device_function, mode='same') * dx
    return y_conv

