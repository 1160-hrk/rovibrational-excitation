import os
import shutil
import itertools
from datetime import datetime
import importlib.util
import json
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count


def load_params(params_path):
    spec = importlib.util.spec_from_file_location("params", params_path)
    params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params)
    return params


def extract_serializable_params(params):
    allowed_types = (str, int, float, bool, list, dict, tuple, type(None), np.ndarray, np.generic, np.float64, np.int64, np.complex128)
    param_dict = {}
    for k in dir(params):
        if k.startswith("__"):
            continue
        val = getattr(params, k)
        if isinstance(val, allowed_types):
            if hasattr(val, "__iter__"):
                val = np.array(val)
            param_dict[k] = val
    return param_dict


def serialize_polarization(pol):
    return [{'real': x.real, 'imag': x.imag} for x in pol]


def deserialize_polarization(pol_serialized):
    return np.array([complex(p['real'], p['imag']) for p in pol_serialized])


def make_result_root(description='Sim'):
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    folder = f'{now}_{description}'
    path = os.path.join('results', folder)
    os.makedirs(path, exist_ok=True)
    return path


def run_simulation_one_case(gauss_width, polarization_serialized, delay, outdir, param_dict):
    from core.basis import VJMBasis
    from core.states import StateVector, DensityMatrix
    from core.hamiltonian import generate_free_hamiltonian, generate_dipole_matrix
    from core.electric_field import ElectricField
    from core.propagator import liouville_propagation

    print(f"Running: width={gauss_width}, pol={polarization_serialized}, delay={delay}")

    # --- パラメータ展開 ---
    p = param_dict
    pol = deserialize_polarization(polarization_serialized)

    # --- 時間軸作成 ---
    tlist = np.linspace(p['t_start'] + delay, p['t_end'] + delay, p['t_points'])
    envelope = lambda t: np.exp(-((t - delay)/gauss_width)**2)

    # --- 電場生成 ---
    Efield = ElectricField(
        tlist=tlist,
        envelope_func=envelope,
        carrier_freq=p['carrier_freq'],
        amplitude=p['amplitude'],
        polarization=pol,
        gdd=p['gdd'],
        tod=p['tod']
    )

    # --- 基底と状態生成 ---
    basis = VJMBasis(p['V_max'], p['J_max'])
    sv = StateVector(basis)
    sv.set_state(0, 0, 0)
    sv.normalize()
    dm = DensityMatrix(basis)
    dm.set_pure_state(sv)

    # --- ハミルトニアン生成 ---
    H0 = generate_free_hamiltonian(
        basis,
        h=p['h'],
        omega=p['omega'],
        delta_omega=p['delta_omega'],
        B=p['B'],
        alpha=p['alpha']
    )
    mu_x = generate_dipole_matrix(basis, mu0 = p['dipole_constant'], axis='x')
    mu_y = generate_dipole_matrix(basis, mu0 = p['dipole_constant'], axis='y')
    # --- 時間発展 ---
    rho_t = liouville_propagation(H0, Efield, dm.rho, tlist, mu_x, mu_y)
    # rho_t = schrodinger_propagation(H0, Efield, sv.psi, tlist, mu_x, mu_y)

    # --- 保存 ---
    np.save(os.path.join(outdir, 'rho_t.npy'), rho_t)
    np.save(os.path.join(outdir, 'tlist.npy'), tlist)
    np.save(os.path.join(outdir, 'population.npy'), np.real(np.diagonal(rho_t, axis1=1, axis2=2)))
    np.save(os.path.join(outdir, 'Efield_real.npy'), np.real(Efield.field))
    np.save(os.path.join(outdir, 'Efield_vector.npy'), Efield.vector)
    with open(os.path.join(outdir, 'parameters.json'), 'w') as f:
        json.dump(p, f, indent=2, default=str)

    return np.real(np.diagonal(rho_t, axis1=1, axis2=2))


def run_all(params_path):
    params = load_params(params_path)
    result_root = make_result_root(params.description)
    shutil.copy(params_path, os.path.join(result_root, 'params.py'))

    summary_rows = []
    param_dict = extract_serializable_params(params)

    for gw, pol, dly in itertools.product(
        params.gauss_widths, params.polarizations, params.delays
    ):
        relpath = f'gauss_width_{gw}/pol_{pol}/delay_{dly}'
        outdir = os.path.join(result_root, relpath)
        os.makedirs(outdir, exist_ok=True)
        pol_serialized = serialize_polarization(pol)
        population = run_simulation_one_case(gw, pol_serialized, dly, outdir, param_dict)
        summary_rows.append({
            "gauss_width": gw,
            "polarization": str(pol),
            "delay": dly,
            "final_population_sum": float(np.sum(population[-1]))
        })

    df = pd.DataFrame(summary_rows)
    df.to_csv(os.path.join(result_root, 'summary.csv'), index=False)


def run_all_parallel(params_path):
    params = load_params(params_path)
    result_root = make_result_root(params.description)
    shutil.copy(params_path, os.path.join(result_root, 'params.py'))

    all_cases = list(itertools.product(
        params.gauss_widths, params.polarizations, params.delays
    ))

    param_dict = extract_serializable_params(params)

    inputs = []
    for gw, pol, dly in all_cases:
        relpath = f'gauss_width_{gw}/pol_{pol}/delay_{dly}'
        outdir = os.path.join(result_root, relpath)
        os.makedirs(outdir, exist_ok=True)
        pol_serialized = serialize_polarization(pol)
        inputs.append((gw, pol_serialized, dly, outdir, param_dict))

    with Pool(processes=cpu_count()) as pool:
        results = pool.starmap(run_simulation_one_case, inputs)

    summary_rows = []
    for (gw, pol, dly), pop in zip(all_cases, results):
        summary_rows.append({
            "gauss_width": gw,
            "polarization": str(pol),
            "delay": dly,
            "final_population_sum": float(np.sum(pop[-1]))
        })

    df = pd.DataFrame(summary_rows)
    df.to_csv(os.path.join(result_root, 'summary.csv'), index=False)
