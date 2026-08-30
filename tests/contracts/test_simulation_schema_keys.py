"""Strict-key contracts for normal-simulation configuration."""

import numpy as np
import pytest

from rovibrational_excitation.simulation.config import load_params_file
from rovibrational_excitation.simulation.validation import (
    SimulationConfigurationError,
    validate_simulation_case,
)


def _twolevel_case(**overrides):
    params = {
        "basis_type": "twolevel",
        "energy_gap": 0.2,
        "energy_gap_units": "rad/fs",
        "dipole_scale": 3.0e-30,
        "dipole_scale_units": "C*m",
        "t_start": -0.5,
        "t_end": 0.5,
        "dt": 0.05,
        "duration": 0.3,
        "t_center": 0.0,
        "envelope_kind": "gaussian_fwhm",
        "modulation_kind": "none",
        "carrier_frequency": 0.1,
        "carrier_frequency_units": "PHz",
        "amplitude": 1.0e8,
        "initial_states": [0],
        "backend": "numpy",
        "storage": "dense",
        "algorithm": "rk4",
        "return_traj": True,
        "sample_stride": 1,
        "nondimensional": False,
        "renorm": False,
        "save": False,
    }
    params.update(overrides)
    return params


def _linmol_case(**overrides):
    params = {
        "basis_type": "linmol",
        "representation": "m_resolved",
        "axes": "xy",
        "V_max": 0,
        "J_max": 1,
        "vibrational_frequency": 0.2,
        "vibrational_frequency_units": "rad/fs",
        "anharmonic_shift": 0.0,
        "anharmonic_shift_units": "rad/fs",
        "rotational_constant": 0.01,
        "rotational_constant_units": "rad/fs",
        "vibration_rotation_coupling": 0.0,
        "vibration_rotation_coupling_units": "rad/fs",
        "dipole_scale": 3.0e-30,
        "dipole_scale_units": "C*m",
        "potential_type": "harmonic",
        "t_start": -0.5,
        "t_end": 0.5,
        "dt": 0.05,
        "duration": 0.3,
        "t_center": 0.0,
        "envelope_kind": "gaussian_fwhm",
        "modulation_kind": "none",
        "carrier_frequency": 0.1,
        "carrier_frequency_units": "PHz",
        "amplitude": 1.0e8,
        "polarization": [1.0, 0.0],
        "initial_states": [0],
        "backend": "numpy",
        "storage": "dense",
        "algorithm": "rk4",
        "return_traj": True,
        "sample_stride": 1,
        "nondimensional": False,
        "renorm": False,
        "save": False,
    }
    params.update(overrides)
    return params


def test_unknown_key_names_the_offending_parameter():
    with pytest.raises(SimulationConfigurationError, match="typo_parameter"):
        validate_simulation_case(_twolevel_case(typo_parameter=1.0))


def test_non_string_key_is_reported_as_unknown():
    params = _twolevel_case()
    params[7] = "invalid"

    with pytest.raises(SimulationConfigurationError, match="7"):
        validate_simulation_case(params)


@pytest.mark.parametrize("key", ["V_max", "potential_type", "representation", "axes"])
def test_twolevel_rejects_model_inapplicable_keys(key):
    with pytest.raises(SimulationConfigurationError, match=key):
        validate_simulation_case(_twolevel_case(**{key: 1}))


def test_scalar_model_rejects_dummy_polarization():
    with pytest.raises(SimulationConfigurationError, match="polarization"):
        validate_simulation_case(_twolevel_case(polarization=[1.0, 0.0]))


def test_rk4_rejects_split_interaction_selector():
    with pytest.raises(SimulationConfigurationError, match="split_interaction"):
        validate_simulation_case(_linmol_case(split_interaction="helicity_projected"))


def test_m_resolved_split_operator_requires_split_interaction_selector():
    with pytest.raises(SimulationConfigurationError, match="split_interaction"):
        validate_simulation_case(_linmol_case(algorithm="split_operator"))


def test_m_resolved_split_operator_accepts_explicit_split_interaction():
    options = validate_simulation_case(
        _linmol_case(
            algorithm="split_operator",
            split_interaction="cartesian",
        )
    )

    assert options.algorithm_name == "split_operator"


def test_scalar_split_operator_rejects_cartesian_selector():
    with pytest.raises(SimulationConfigurationError, match="split_interaction"):
        validate_simulation_case(
            _twolevel_case(
                algorithm="split_operator",
                split_interaction="cartesian",
            )
        )


def test_m_average_split_operator_rejects_cartesian_selector():
    params = _linmol_case(
        representation="m_incoherent_average",
        algorithm="split_operator",
        split_interaction="cartesian",
    )
    params.pop("axes")
    params.pop("polarization")

    with pytest.raises(SimulationConfigurationError, match="split_interaction"):
        validate_simulation_case(params)


@pytest.mark.parametrize("key", ["validate_units", "verbose", "save"])
def test_runner_boolean_controls_reject_non_booleans(key):
    with pytest.raises(SimulationConfigurationError, match=key):
        validate_simulation_case(_twolevel_case(**{key: 1}))


def test_python_parameter_loader_excludes_imported_modules(tmp_path):
    path = tmp_path / "params.py"
    path.write_text(
        "import numpy as np\namplitude = float(np.sqrt(4.0))\nhelper_value = 3\n"
    )

    params = load_params_file(str(path))

    assert params == {"amplitude": 2.0, "helper_value": 3}
    assert "np" not in params


def test_nonfinite_generated_phase_is_rejected():
    with pytest.raises(SimulationConfigurationError, match="phase_rad"):
        validate_simulation_case(_twolevel_case(phase_rad=np.nan))
