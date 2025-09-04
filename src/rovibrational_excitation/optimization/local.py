from __future__ import annotations

from typing import TypedDict, Any

import numpy as np

from rovibrational_excitation.core.electric_field import ElectricField
from rovibrational_excitation.core.propagation import SchrodingerPropagator
from rovibrational_excitation.core.propagation.utils import cm_to_rad_phz

DEFAULT_PARAMS = {
    "control_axes": "xy",
    "gain": 1.0,
    "field_max": 1e12,
    "use_sin2_shape": False,
    "segment_size_steps": None,
    "segment_size_fs": 1,
    "seed_amplitude": 1e3,
    "seed_max_segments": 5,
    "c_abs_min": 1e-1,
    "shape_floor": 1e-2,
    "lookahead_enable": False,
    "lookahead_fraction": 0.5,

    "eval_mode": "weights",
    "weight_mode": "by_v",
    "weight_v_power": 2.0,
    "weight_target_factor": 1.0,
    "normalize_weights": True,
    "weight_reverse": False,
    "use_one_hot_target_in_weights": False,
    "drive_abs_min": 1e-18,

    "custom_weights": None,
    "custom_weights_dict": None,
    
    "propagator_func": None,
}

class RunResult(TypedDict, total=False):
    efield: ElectricField
    time: np.ndarray
    psi_traj: np.ndarray
    metrics: dict
    tlist: np.ndarray
    field_data: np.ndarray
    target_idx: int


def _build_segments_and_tlist(time_total: float, dt: float, seg_steps: int | None, seg_fs: float | None):
    if seg_steps is not None and seg_steps > 0:
        steps = int((seg_steps // 2) * 2)
    elif seg_fs is not None and seg_fs > 0:
        steps = int((seg_fs / dt // 2) * 2)
    else:
        raise ValueError("segment_size_steps または segment_size_fs を指定してください")
    segments_number = int(np.ceil(time_total / dt / steps))
    n_field_steps = segments_number * steps + 1
    tlist = np.arange(0.0, dt * (n_field_steps + 1), dt)
    segments = [(int(s * steps), int((s + 1) * steps)) for s in range(segments_number)]
    return segments, tlist


def _build_weights_for_basis(basis, *, mode: str, normalize: bool, custom: np.ndarray | None,
                             custom_dict: dict | None, v_power: float, one_hot_target_idx: int | None,
                             reverse: bool) -> np.ndarray:
    dim = basis.size()
    if one_hot_target_idx is not None:
        w = np.zeros(dim, dtype=float)
        if 0 <= int(one_hot_target_idx) < dim:
            w[int(one_hot_target_idx)] = 1.0
    elif mode == "custom":
        if custom is not None:
            w = np.asarray(custom, dtype=float)
            if w.shape[0] != dim:
                raise ValueError("custom_weights の長さが基底次元と一致しません")
        elif custom_dict is not None:
            w = np.zeros(dim, dtype=float)
            for i, key in enumerate(basis.basis):
                w[i] = float(custom_dict.get(tuple(key), 0.0))
        else:
            raise ValueError("custom_weights / custom_dict が未指定です")
    else:
        w = np.zeros(dim, dtype=float)
        for i, state in enumerate(basis.basis):
            v = int(state[0])
            if mode == "by_v_power":
                w[i] = float(v) ** float(v_power)
            else:
                w[i] = float(v)
    if normalize:
        wmax = float(np.max(w))
        if wmax > 0:
            w = w / wmax
    if one_hot_target_idx is None and reverse:
        if normalize:
            w = 1.0 - w
        else:
            wmax = float(np.max(w))
            w = (wmax - w)
    return w


def run_local_optimization(*, basis, hamiltonian, dipole, states: dict[str, Any], time_cfg: dict, params: dict) -> RunResult:
    initial_state = tuple(states["initial"])
    target_state = tuple(states["target"]) if states.get("target") is not None else None

    initial_idx = basis.get_index(initial_state)
    target_idx = basis.get_index(target_state) if target_state is not None else None

    time_total = float(time_cfg["total_fs"])
    dt = float(time_cfg["dt_fs"])
    sample_stride = int(time_cfg.get("sample_stride", 1))

    seg_steps = params.get("segment_size_steps", DEFAULT_PARAMS["segment_size_steps"])
    seg_fs = params.get("segment_size_fs", DEFAULT_PARAMS["segment_size_fs"])
    segments, tlist = _build_segments_and_tlist(time_total, dt, seg_steps, seg_fs)
    n_steps = len(tlist)

    propagator = SchrodingerPropagator(backend="numpy", validate_units=True, renorm=True)

    psi_initial = np.zeros(basis.size(), dtype=complex)
    psi_initial[initial_idx] = 1.0
    psi_target = np.zeros(basis.size(), dtype=complex)
    if target_idx is not None:
        psi_target[target_idx] = 1.0

    mu_x_si = dipole.get_mu_x_SI(); mu_y_si = dipole.get_mu_y_SI(); mu_z_si = dipole.get_mu_z_SI()
    if hasattr(mu_x_si, 'toarray'): mu_x_si = mu_x_si.toarray()
    if hasattr(mu_y_si, 'toarray'): mu_y_si = mu_y_si.toarray()
    if hasattr(mu_z_si, 'toarray'): mu_z_si = mu_z_si.toarray()
    mu_x_p = cm_to_rad_phz(mu_x_si); mu_y_p = cm_to_rad_phz(mu_y_si); mu_z_p = cm_to_rad_phz(mu_z_si)

    control_axes = str(params.get("control_axes", DEFAULT_PARAMS["control_axes"])).lower()
    if len(control_axes) != 2 or any(c not in 'xyz' for c in control_axes):
        control_axes = 'xy'
    mu_map = {'x': mu_x_p, 'y': mu_y_p, 'z': mu_z_p}
    mu_eff_x = mu_map[control_axes[0]]
    mu_eff_y = mu_map[control_axes[1]]

    gain = float(params.get("gain", DEFAULT_PARAMS["gain"]))
    field_max = float(params.get("field_max", DEFAULT_PARAMS["field_max"]))
    use_sin2_shape = bool(params.get("use_sin2_shape", DEFAULT_PARAMS["use_sin2_shape"]))
    seed_amplitude = float(params.get("seed_amplitude", DEFAULT_PARAMS["seed_amplitude"]))
    seed_max_segments = int(params.get("seed_max_segments", DEFAULT_PARAMS["seed_max_segments"]))
    c_abs_min = float(params.get("c_abs_min", DEFAULT_PARAMS["c_abs_min"]))
    shape_floor = float(params.get("shape_floor", DEFAULT_PARAMS["shape_floor"]))
    lookahead_enable = bool(params.get("lookahead_enable", DEFAULT_PARAMS["lookahead_enable"]))
    lookahead_fraction = float(params.get("lookahead_fraction", DEFAULT_PARAMS["lookahead_fraction"]))

    eval_mode = str(params.get("eval_mode", DEFAULT_PARAMS["eval_mode"])).lower()
    weight_mode = str(params.get("weight_mode", DEFAULT_PARAMS["weight_mode"])).lower()
    weight_v_power = float(params.get("weight_v_power", DEFAULT_PARAMS["weight_v_power"]))
    weight_target_factor = float(params.get("weight_target_factor", DEFAULT_PARAMS["weight_target_factor"]))
    normalize_weights = bool(params.get("normalize_weights", DEFAULT_PARAMS["normalize_weights"]))
    weight_reverse = bool(params.get("weight_reverse", DEFAULT_PARAMS["weight_reverse"]))
    custom_weights = params.get("custom_weights", DEFAULT_PARAMS["custom_weights"])
    custom_weights_dict = params.get("custom_weights_dict", DEFAULT_PARAMS["custom_weights_dict"])
    use_one_hot_target_in_weights = bool(params.get("use_one_hot_target_in_weights", DEFAULT_PARAMS["use_one_hot_target_in_weights"]))
    drive_abs_min = float(params.get("drive_abs_min", DEFAULT_PARAMS["drive_abs_min"]))
    propagator_func = params.get("propagator_func", DEFAULT_PARAMS["propagator_func"])

    segments_arr = segments
    full_field = np.zeros((n_steps, 2), dtype=float)

    psi_curr = psi_initial.copy()
    seed_left = seed_max_segments

    # Precompute eigenvalues for lookahead if available
    try:
        eigenvalues = hamiltonian.get_eigenvalues()
    except Exception:
        eigenvalues = None

    # Prepare evaluation operator (weights mode)
    if eval_mode == "weights":
        one_hot_idx = target_idx if use_one_hot_target_in_weights else None
        mode_is_reverse = weight_mode.endswith("_reverse")
        base_mode = weight_mode.replace("_reverse", "")
        reverse_flag = bool(weight_reverse) or mode_is_reverse
        A_diag = _build_weights_for_basis(
            basis,
            mode=base_mode,
            normalize=normalize_weights,
            custom=np.asarray(custom_weights, dtype=float) if custom_weights is not None else None,
            custom_dict=dict(custom_weights_dict) if custom_weights_dict is not None else None,
            v_power=weight_v_power,
            one_hot_target_idx=one_hot_idx,
            reverse=reverse_flag,
        )
        if one_hot_idx is None and target_idx is not None:
            try:
                w_before = float(A_diag[target_idx])
                A_diag[target_idx] = w_before * float(weight_target_factor)
            except Exception:
                pass
    else:
        A_diag = None

    def shape_function(t: np.ndarray, T: float) -> np.ndarray:
        return np.sin(np.pi * t / T) ** 2

    for (start, end) in segments_arr:
        mid = (start + end) // 2
        S = shape_function(tlist, tlist[-1])[mid] if use_sin2_shape else 1.0
        S_eff = max(S, shape_floor) if use_sin2_shape else 1.0

        psi_ref = psi_curr
        if lookahead_enable and lookahead_fraction > 0 and eigenvalues is not None and end - start >= 1:
            tau = lookahead_fraction * float(tlist[end - 1] - tlist[start])
            if tau > 0:
                phase = np.exp(-1j * eigenvalues * tau)
                psi_ref = psi_curr * phase

        if eval_mode == "weights" and A_diag is not None:
            mu_x_psi = -mu_eff_x @ psi_ref
            mu_y_psi = -mu_eff_y @ psi_ref
            A_mu_x_psi = A_diag * mu_x_psi
            A_mu_y_psi = A_diag * mu_y_psi
            term_x = complex(np.vdot(psi_ref, A_mu_x_psi))
            term_y = complex(np.vdot(psi_ref, A_mu_y_psi))
            im_x = float(np.imag(term_x))
            im_y = float(np.imag(term_y))
            ex = float(gain * S * im_x)
            ey = float(gain * S * im_y)
            if (abs(im_x) < drive_abs_min and abs(im_y) < drive_abs_min and seed_left > 0):
                ex = seed_amplitude * S_eff
                ey = seed_amplitude * S_eff
                seed_left -= 1
        else:
            c = complex(np.vdot(psi_target, psi_ref)) if target_idx is not None else 0.0
            d_x = complex(np.vdot(psi_target, (-mu_eff_x @ psi_ref))) if target_idx is not None else 0.0
            d_y = complex(np.vdot(psi_target, (-mu_eff_y @ psi_ref))) if target_idx is not None else 0.0
            val_x = float(np.imag(np.conj(c) * d_x))
            val_y = float(np.imag(np.conj(c) * d_y))
            ex = float(gain * S * val_x)
            ey = float(gain * S * val_y)
            if abs(c) < c_abs_min and seed_left > 0:
                sx = 1.0 if (abs(d_x) == 0.0) else (1.0 if (np.real(d_x) >= 0.0) else -1.0)
                sy = 1.0 if (abs(d_y) == 0.0) else (1.0 if (np.real(d_y) >= 0.0) else -1.0)
                ex = sx * seed_amplitude * S_eff
                ey = sy * seed_amplitude * S_eff
                seed_left -= 1

        ex = float(np.clip(ex, -field_max, field_max))
        ey = float(np.clip(ey, -field_max, field_max))

        full_field[start + 1:end + 1, 0] = ex
        full_field[start + 1:end + 1, 1] = ey

        ef_seg = ElectricField(tlist=tlist[start:end + 1])
        ef_seg.add_arbitrary_Efield(full_field[start:end + 1, :])
        result = propagator.propagate(
            hamiltonian=hamiltonian,
            efield=ef_seg,
            dipole_matrix=dipole,
            initial_state=psi_curr,
            axes=control_axes,
            return_traj=True,
            return_time_psi=True,
            sample_stride=sample_stride,
            algorithm="rk4",
            sparse=True,
            propagator_func=propagator_func,
        )
        psi_traj_seg = result[1]
        psi_curr = psi_traj_seg[-1]

    ef_total = ElectricField(tlist=tlist)
    ef_total.add_arbitrary_Efield(full_field)

    result_full = propagator.propagate(
        hamiltonian=hamiltonian,
        efield=ef_total,
        dipole_matrix=dipole,
        initial_state=psi_initial,
        axes=control_axes,
        return_traj=True,
        return_time_psi=True,
        sample_stride=sample_stride,
        algorithm="rk4",
        sparse=True,
        propagator_func=propagator_func,
    )
    time_full, psi_traj_full = result_full[0], result_full[1]

    def _fidelity(psi_final: np.ndarray, idx: int | None) -> float:
        if idx is None:
            return 0.0
        return float(np.abs(psi_final[idx]) ** 2)

    fidelity = _fidelity(psi_traj_full[-1], target_idx)

    # optional: running cost J_a ~ ∫ S|E|^2 dt with λ=1/gain for display
    try:
        field_penalty = 1.0 / max(gain, 1e-30)
        S_run = np.sin(np.pi * tlist / tlist[-1]) ** 2 if use_sin2_shape else np.ones_like(tlist)
        E2 = full_field[:, 0] ** 2 + full_field[:, 1] ** 2
        dt_field = float(tlist[1] - tlist[0])
        running_cost = float(field_penalty) * float(np.sum(S_run * E2) * dt_field)
    except Exception:
        running_cost = None

    return RunResult(
        efield=ef_total,
        time=time_full,
        psi_traj=psi_traj_full,
        metrics={"fidelity": fidelity, "running_cost": running_cost},
        tlist=tlist,
        field_data=full_field,
        target_idx=target_idx if target_idx is not None else -1,
    )


