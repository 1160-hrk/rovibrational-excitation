"""Regression tests for simulation-level contracts."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from rovibrational_excitation.core.time import TimeGrid
from rovibrational_excitation.core.units import Frequency
from rovibrational_excitation.fields import (
    CartesianField,
    ElectricField,
    ScalarField,
    gaussian,
    gaussian_fwhm,
    lorentzian,
    lorentzian_fwhm,
)
from rovibrational_excitation.io import CheckpointManager
from rovibrational_excitation.simulation.config import load_params_file
from rovibrational_excitation.simulation.runner import (
    _run_one,
    run_all_with_checkpoint,
    run_simulation_case,
)
from rovibrational_excitation.simulation.validation import (
    SimulationConfigurationError,
    validate_simulation_case,
)


def _base_case(**overrides):
    case = {
        "basis_type": "twolevel",
        "energy_gap": 0.2,
        "energy_gap_units": "rad/fs",
        "mu0_Cm": 3.0e-30,
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
        "save": False,
        "backend": "numpy",
        "storage": "dense",
        "algorithm": "rk4",
        "return_traj": True,
        "sample_stride": 1,
        "nondimensional": False,
        "renorm": False,
    }
    case.update(overrides)
    return case


def test_time_grid_includes_exact_endpoints_and_rk_midpoints():
    grid = TimeGrid.from_bounds(-1.0, 1.0, 0.1).field_times_fs

    assert grid.size == 21
    assert grid[0] == -1.0
    assert grid[-1] == 1.0
    np.testing.assert_allclose(np.diff(grid), 0.1)


def test_time_grid_rejects_span_that_solver_would_truncate():
    with pytest.raises(ValueError, match=r"integer multiple of 2 \* dt"):
        TimeGrid.from_bounds(-1.0, 1.0, 0.3)


@pytest.mark.parametrize(
    "model_overrides",
    [
        {},
        {
            "basis_type": "vibladder",
            "V_max": 2,
            "vibrational_frequency": 0.2,
            "vibrational_frequency_units": "rad/fs",
            "anharmonic_shift": 0.0,
            "anharmonic_shift_units": "rad/fs",
            "potential_type": "harmonic",
        },
    ],
)
@pytest.mark.parametrize("nondimensional", [False, True])
def test_scalar_models_propagate_without_polarization(model_overrides, nondimensional):
    params = _base_case(nondimensional=nondimensional)
    if model_overrides:
        params.pop("energy_gap")
        params.pop("energy_gap_units")
        params.update(model_overrides)

    population = _run_one(params)

    expected_size = 3 if model_overrides else 2
    assert population.shape[1] == expected_size
    assert np.all(np.isfinite(population))


def test_final_state_only_uses_final_physical_time_and_state_axis(tmp_path):
    params = _base_case(
        amplitude=0.0,
        return_traj=False,
        save=True,
        outdir=str(tmp_path),
    )

    population = _run_one(params)

    assert population.shape == (1, 2)
    with np.load(tmp_path / "result.npz") as data:
        np.testing.assert_array_equal(data["t_p"], np.array([params["t_end"]]))
        assert data["t_E"][0] == params["t_start"]
        assert data["t_E"][-1] == params["t_end"]


def test_validation_rejects_missing_physical_parameter_before_building():
    params = _base_case()
    del params["mu0_Cm"]

    with pytest.raises(SimulationConfigurationError, match="mu0_Cm"):
        validate_simulation_case(params)


def test_validation_rejects_missing_initial_states():
    params = _base_case()
    del params["initial_states"]

    with pytest.raises(SimulationConfigurationError, match="initial_states"):
        validate_simulation_case(params)


def test_validation_rejects_missing_duration():
    params = _base_case()
    del params["duration"]

    with pytest.raises(SimulationConfigurationError, match="duration"):
        validate_simulation_case(params)


@pytest.mark.parametrize("key", ["envelope_kind", "modulation_kind", "t_center"])
def test_generated_field_requires_explicit_discriminators_and_center(key):
    params = _base_case()
    del params[key]

    with pytest.raises(SimulationConfigurationError, match=key):
        validate_simulation_case(params)


@pytest.mark.parametrize("key", ["carrier_frequency", "carrier_frequency_units"])
def test_generated_field_requires_carrier_frequency_and_unit(key):
    params = _base_case()
    del params[key]

    with pytest.raises(SimulationConfigurationError, match=key):
        validate_simulation_case(params)


def test_generated_field_rejects_removed_ambiguous_carrier_key():
    params = _base_case(carrier_freq=0.1)

    with pytest.raises(SimulationConfigurationError, match="carrier_freq"):
        validate_simulation_case(params)


@pytest.mark.parametrize(
    ("value", "unit"),
    [
        (np.nan, "PHz"),
        (True, "PHz"),
        (0.1, "cycles_per_second"),
    ],
)
def test_generated_field_rejects_invalid_carrier_frequency(value, unit):
    params = _base_case(
        carrier_frequency=value,
        carrier_frequency_units=unit,
    )

    with pytest.raises(
        SimulationConfigurationError,
        match="carrier_frequency/carrier_frequency_units",
    ):
        validate_simulation_case(params)


@pytest.mark.parametrize(
    ("value", "unit"),
    [
        (100.0, "THz"),
        (0.2 * np.pi, "rad/fs"),
        (1.0e14, "Hz"),
        (0.1 / (2.99792458e8 * 1.0e-13), "cm^-1"),
        (0.1 / (2.99792458e8 * 1.0e-13), "wavenumber"),
    ],
)
def test_generated_carrier_frequency_units_preserve_the_same_field(value, unit):
    from rovibrational_excitation.simulation.runner import _generated_sampled_field

    reference_params = _base_case()
    params = _base_case(
        carrier_frequency=value,
        carrier_frequency_units=unit,
    )
    grid = TimeGrid.from_bounds(params["t_start"], params["t_end"], params["dt"])

    reference = _generated_sampled_field(
        reference_params,
        time_grid=grid,
        use_m_average=False,
        expects_cartesian=False,
    )
    actual = _generated_sampled_field(
        params,
        time_grid=grid,
        use_m_average=False,
        expects_cartesian=False,
    )

    np.testing.assert_allclose(
        actual.samples_v_per_m,
        reference.samples_v_per_m,
        rtol=2.0e-15,
        atol=2.0e-7,
    )


@pytest.mark.parametrize(
    ("removed_key", "value"),
    [
        ("envelope_func", gaussian_fwhm),
        ("Sinusoidal_modulation", False),
    ],
)
def test_generated_field_rejects_removed_shape_selectors(removed_key, value):
    params = _base_case(**{removed_key: value})

    with pytest.raises(SimulationConfigurationError, match=removed_key):
        validate_simulation_case(params)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("envelope_kind", "voigt_fwhm"),
        ("modulation_kind", "automatic"),
    ],
)
def test_generated_field_rejects_unknown_discriminators(key, value):
    with pytest.raises(SimulationConfigurationError, match=key):
        validate_simulation_case(_base_case(**{key: value}))


@pytest.mark.parametrize(
    "key",
    [
        "modulation_depth",
        "modulation_delay",
        "modulation_delay_units",
        "modulation_mode",
    ],
)
def test_sinusoidal_modulation_requires_all_non_additive_parameters(key):
    params = _base_case(
        modulation_kind="sinusoidal",
        modulation_depth=0.2,
        modulation_delay=30.0,
        modulation_delay_units="fs",
        modulation_mode="phase",
    )
    del params[key]

    with pytest.raises(SimulationConfigurationError, match=key):
        validate_simulation_case(params)


def test_no_modulation_rejects_sinusoidal_parameters():
    with pytest.raises(SimulationConfigurationError, match="modulation_depth"):
        validate_simulation_case(_base_case(modulation_depth=0.2))


def test_sinusoidal_modulation_rejects_unknown_type():
    params = _base_case(
        modulation_kind="sinusoidal",
        modulation_depth=0.2,
        modulation_delay=30.0,
        modulation_delay_units="fs",
        modulation_mode="frequency",
    )

    with pytest.raises(SimulationConfigurationError, match="modulation_mode"):
        validate_simulation_case(params)


@pytest.mark.parametrize(
    "legacy_key",
    [
        "amplitude_sin_mod",
        "carrier_freq_sin_mod",
        "phase_rad_sin_mod",
        "type_mod_sin_mod",
    ],
)
def test_sinusoidal_modulation_rejects_legacy_keys(legacy_key):
    with pytest.raises(SimulationConfigurationError, match=legacy_key):
        validate_simulation_case(_base_case(**{legacy_key: 0.2}))


def test_sinusoidal_modulation_rejects_invalid_delay_unit():
    params = _base_case(
        modulation_kind="sinusoidal",
        modulation_depth=0.2,
        modulation_delay=30.0,
        modulation_delay_units="not-a-time-unit",
        modulation_mode="phase",
    )

    with pytest.raises(SimulationConfigurationError, match="modulation_delay"):
        validate_simulation_case(params)


@pytest.mark.parametrize("depth", [-0.01, 1.01])
def test_amplitude_modulation_rejects_depth_outside_unit_interval(depth):
    params = _base_case(
        modulation_kind="sinusoidal",
        modulation_depth=depth,
        modulation_delay=30.0,
        modulation_delay_units="fs",
        modulation_mode="amplitude",
    )

    with pytest.raises(SimulationConfigurationError, match=r"0 <= depth <= 1"):
        validate_simulation_case(params)


def test_validation_rejects_unknown_potential_before_model_construction():
    params = _base_case(
        basis_type="vibladder",
        V_max=2,
        vibrational_frequency=0.2,
        vibrational_frequency_units="rad/fs",
        anharmonic_shift=0.0,
        anharmonic_shift_units="rad/fs",
        potential_type="quadratic",
    )

    with pytest.raises(SimulationConfigurationError, match="potential_type"):
        validate_simulation_case(params)


def test_validation_rejects_removed_pulse_duration_alias():
    params = _base_case(pulse_duration=0.3)
    params.pop("duration")

    with pytest.raises(
        SimulationConfigurationError,
        match="pulse_duration was removed",
    ):
        validate_simulation_case(params)


def test_checkpoint_deduplicates_cases_and_ignores_runtime_error(tmp_path):
    manager = CheckpointManager(tmp_path)
    case = {"amplitude": 1.0, "save": True, "outdir": "first"}
    duplicate = {**case, "outdir": "second"}
    failed = {**case, "error": "old failure"}

    manager.save_checkpoint([case, duplicate], [failed], 1, 0.0)
    checkpoint = manager.load_checkpoint()

    assert checkpoint is not None
    assert checkpoint["completed_cases"] == 1
    assert checkpoint["failed_cases"] == 0
    assert manager.failed_cases_file.exists()


def test_summary_keeps_each_result_with_its_original_case(tmp_path):
    params = {
        "description": "summary_mapping",
        "amplitude": [1.0, 2.0],
    }
    successful = np.array([[0.25, 0.75]])

    with (
        patch(
            "rovibrational_excitation.simulation.runner._make_root",
            return_value=Path(tmp_path),
        ),
        patch(
            "rovibrational_excitation.simulation.runner._run_one",
            side_effect=[ValueError("invalid first case"), successful],
        ),
    ):
        results = run_all_with_checkpoint(
            params,
            save=True,
            checkpoint_interval=2,
        )

    assert len(results) == 1
    summary = pd.read_csv(tmp_path / "summary.csv")
    first = summary.loc[summary["amplitude"] == 1.0].iloc[0]
    second = summary.loc[summary["amplitude"] == 2.0].iloc[0]
    assert first["status"] == "failed"
    assert second["status"] == "success"
    assert second["pop_0"] == pytest.approx(0.25)


def test_runner_parameter_template_matches_current_required_contract():
    repository_root = Path(__file__).parents[2]
    params = load_params_file(str(repository_root / "examples" / "params_template.py"))

    options = validate_simulation_case(params)

    assert params["basis_type"] == "linmol"
    assert params["representation"] == "m_resolved"
    assert params["axes"] == "xy"
    assert options.algorithm_name == "rk4"
    assert options.execution.storage.value == "dense"


_GENERATED_FIELD_KEYS = {
    "t_start",
    "t_end",
    "dt",
    "duration",
    "t_center",
    "envelope_kind",
    "modulation_kind",
    "carrier_frequency",
    "carrier_frequency_units",
    "amplitude",
    "polarization",
}


def _without_generated_field_keys(params):
    return {
        key: value for key, value in params.items() if key not in _GENERATED_FIELD_KEYS
    }


@pytest.mark.parametrize(
    ("envelope_kind", "envelope_func"),
    [
        ("gaussian", gaussian),
        ("gaussian_fwhm", gaussian_fwhm),
        ("lorentzian", lorentzian),
        ("lorentzian_fwhm", lorentzian_fwhm),
    ],
)
def test_generated_envelope_kind_preserves_existing_samples(
    envelope_kind, envelope_func
):
    from rovibrational_excitation.simulation.runner import _generated_sampled_field

    params = _base_case(envelope_kind=envelope_kind)
    grid = TimeGrid.from_bounds(params["t_start"], params["t_end"], params["dt"])
    expected = ElectricField.from_time_grid(grid)
    expected.add_dispersed_Efield(
        envelope_func,
        duration=params["duration"],
        t_center=params["t_center"],
        carrier_freq=Frequency(
            params["carrier_frequency"], params["carrier_frequency_units"]
        ).cycles_per_fs,
        amplitude=params["amplitude"],
        polarization=np.array([1.0, 0.0]),
    )

    actual = _generated_sampled_field(
        params,
        time_grid=grid,
        use_m_average=False,
        expects_cartesian=False,
    )

    np.testing.assert_array_equal(
        actual.samples_v_per_m,
        expected.get_scalar_field(),
    )


@pytest.mark.parametrize("modulation_type", ["phase", "amplitude"])
@pytest.mark.parametrize(
    ("phase_parameters", "expected_phase"),
    [({}, 0.0), ({"modulation_phase_rad": 0.4}, 0.4)],
)
def test_sinusoidal_modulation_kind_matches_field_api(
    modulation_type, phase_parameters, expected_phase
):
    from rovibrational_excitation.simulation.runner import _generated_sampled_field

    params = _base_case(
        modulation_kind="sinusoidal",
        modulation_depth=0.2,
        modulation_delay=30.0,
        modulation_delay_units="fs",
        modulation_mode=modulation_type,
        **phase_parameters,
    )
    grid = TimeGrid.from_bounds(params["t_start"], params["t_end"], params["dt"])
    expected = ElectricField.from_time_grid(grid)
    expected.add_dispersed_Efield(
        gaussian_fwhm,
        duration=params["duration"],
        t_center=params["t_center"],
        carrier_freq=Frequency(
            params["carrier_frequency"], params["carrier_frequency_units"]
        ).cycles_per_fs,
        amplitude=params["amplitude"],
        polarization=np.array([1.0, 0.0]),
    )
    expected.apply_sinusoidal_mod(
        center_freq=Frequency(
            params["carrier_frequency"], params["carrier_frequency_units"]
        ).cycles_per_fs,
        modulation_depth=params["modulation_depth"],
        delay_fs=params["modulation_delay"],
        phase_rad=expected_phase,
        mode=params["modulation_mode"],
    )

    actual = _generated_sampled_field(
        params,
        time_grid=grid,
        use_m_average=False,
        expects_cartesian=False,
    )

    np.testing.assert_array_equal(
        actual.samples_v_per_m,
        expected.get_scalar_field(),
    )


def test_sinusoidal_modulation_delay_units_are_physically_equivalent():
    from rovibrational_excitation.simulation.runner import _generated_sampled_field

    fs_params = _base_case(
        modulation_kind="sinusoidal",
        modulation_depth=0.2,
        modulation_delay=30.0,
        modulation_delay_units="fs",
        modulation_phase_rad=0.4,
        modulation_mode="phase",
    )
    ps_params = {
        **fs_params,
        "modulation_delay": 0.03,
        "modulation_delay_units": "ps",
    }
    grid = TimeGrid.from_bounds(
        fs_params["t_start"], fs_params["t_end"], fs_params["dt"]
    )

    field_fs = _generated_sampled_field(
        fs_params,
        time_grid=grid,
        use_m_average=False,
        expects_cartesian=False,
    )
    field_ps = _generated_sampled_field(
        ps_params,
        time_grid=grid,
        use_m_average=False,
        expects_cartesian=False,
    )

    np.testing.assert_allclose(
        field_fs.samples_v_per_m,
        field_ps.samples_v_per_m,
        rtol=2e-15,
        atol=0.0,
    )


def test_external_scalar_field_matches_existing_generated_twolevel_calculation():
    params = _base_case(t_center=0.0)
    expected = _run_one(params)
    grid = TimeGrid.from_bounds(params["t_start"], params["t_end"], params["dt"])
    generated = ElectricField.from_time_grid(grid)
    generated.add_dispersed_Efield(
        gaussian_fwhm,
        duration=params["duration"],
        t_center=params["t_center"],
        carrier_freq=Frequency(
            params["carrier_frequency"], params["carrier_frequency_units"]
        ).cycles_per_fs,
        amplitude=params["amplitude"],
        polarization=np.array([1.0, 0.0]),
    )
    field = ScalarField(grid, generated.get_scalar_field())

    actual = run_simulation_case(
        _without_generated_field_keys(params),
        field=field,
    )

    np.testing.assert_array_equal(actual, expected)


def test_external_cartesian_field_matches_existing_generated_linmol_calculation():
    params = {
        "basis_type": "linmol",
        "V_max": 0,
        "J_max": 1,
        "representation": "m_resolved",
        "axes": "xy",
        "vibrational_frequency": 0.2,
        "vibrational_frequency_units": "rad/fs",
        "anharmonic_shift": 0.0,
        "anharmonic_shift_units": "rad/fs",
        "rotational_constant": 0.001,
        "rotational_constant_units": "rad/fs",
        "vibration_rotation_coupling": 0.0,
        "vibration_rotation_coupling_units": "rad/fs",
        "mu0_Cm": 3.0e-30,
        "potential_type": "harmonic",
        "initial_states": [0],
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
        "polarization": [1.0, 1.0],
        "backend": "numpy",
        "storage": "dense",
        "algorithm": "rk4",
        "return_traj": True,
        "sample_stride": 1,
        "nondimensional": False,
        "renorm": False,
        "save": False,
    }
    expected = _run_one(params)
    grid = TimeGrid.from_bounds(params["t_start"], params["t_end"], params["dt"])
    generated = ElectricField.from_time_grid(grid)
    generated.add_dispersed_Efield(
        gaussian_fwhm,
        duration=params["duration"],
        t_center=params["t_center"],
        carrier_freq=Frequency(
            params["carrier_frequency"], params["carrier_frequency_units"]
        ).cycles_per_fs,
        amplitude=params["amplitude"],
        polarization=np.asarray(params["polarization"]),
    )
    components = generated.get_Efield()
    field = CartesianField(grid, components[:, 0], components[:, 1])

    actual = run_simulation_case(
        _without_generated_field_keys(params),
        field=field,
    )

    np.testing.assert_array_equal(actual, expected)


def test_external_field_rejects_generation_parameters_and_wrong_field_kind():
    grid = TimeGrid.from_bounds(-0.5, 0.5, 0.05)
    scalar = ScalarField(grid, np.ones(grid.field_times_fs.size))
    cartesian = CartesianField(
        grid,
        np.ones(grid.field_times_fs.size),
        np.zeros(grid.field_times_fs.size),
    )
    external_params = _without_generated_field_keys(_base_case())

    with pytest.raises(SimulationConfigurationError, match="amplitude"):
        run_simulation_case({**external_params, "amplitude": 1.0}, field=scalar)
    with pytest.raises(SimulationConfigurationError, match="scalar field"):
        run_simulation_case(external_params, field=cartesian)


def test_generated_cartesian_field_retains_legacy_helicity_inputs():
    from rovibrational_excitation.simulation.runner import _generated_sampled_field

    params = _base_case(
        polarization=np.array([1.0, 1.0j]) / np.sqrt(2.0),
    )
    grid = TimeGrid.from_bounds(params["t_start"], params["t_end"], params["dt"])

    field = _generated_sampled_field(
        params,
        time_grid=grid,
        use_m_average=False,
        expects_cartesian=True,
    )

    np.testing.assert_allclose(
        field.get_pol(),
        np.asarray(params["polarization"]),
        rtol=0.0,
        atol=2.0e-16,
    )
    assert field.get_scalar_field().shape == grid.field_times_fs.shape


def _helicity_runner_case(**overrides):
    params = {
        "basis_type": "linmol",
        "V_max": 1,
        "J_max": 1,
        "representation": "m_resolved",
        "axes": "xy",
        "vibrational_frequency": 0.2,
        "vibrational_frequency_units": "rad/fs",
        "anharmonic_shift": 0.0,
        "anharmonic_shift_units": "rad/fs",
        "rotational_constant": 0.001,
        "rotational_constant_units": "rad/fs",
        "vibration_rotation_coupling": 0.0,
        "vibration_rotation_coupling_units": "rad/fs",
        "mu0_Cm": 3.0e-30,
        "potential_type": "harmonic",
        "initial_states": [0],
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
        "polarization": np.array([1.0, 1.0j]) / np.sqrt(2.0),
        "backend": "numpy",
        "storage": "dense",
        "algorithm": "split_operator",
        "split_interaction": "helicity_projected",
        "return_traj": True,
        "sample_stride": 1,
        "nondimensional": False,
        "renorm": False,
        "validate_units": False,
        "save": False,
    }
    params.update(overrides)
    return params


def test_generated_helicity_runner_retains_existing_decomposition_path():
    population = _run_one(_helicity_runner_case())

    np.testing.assert_allclose(population.sum(axis=1), 1.0, atol=2.0e-14)


def test_external_cartesian_helicity_rejects_missing_decomposition():
    params = _helicity_runner_case()
    grid = TimeGrid.from_bounds(params["t_start"], params["t_end"], params["dt"])
    field = CartesianField(
        grid,
        np.ones(grid.field_times_fs.size),
        np.zeros(grid.field_times_fs.size),
    )

    with pytest.raises(ValueError, match="requires polarization and scalar_field"):
        run_simulation_case(_without_generated_field_keys(params), field=field)
