"""Numerical and rendering-call contracts for visualization helpers."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from rovibrational_excitation.visualization.spectrogram import spectrogram_fast

plot_all_module = import_module("rovibrational_excitation.visualization.plot_all")
field_module = import_module(
    "rovibrational_excitation.visualization.plot_electric_field"
)
vector_module = import_module(
    "rovibrational_excitation.visualization.plot_electric_field_vector"
)
population_module = import_module(
    "rovibrational_excitation.visualization.plot_population"
)


@pytest.fixture(autouse=True)
def _close_figures():
    plt.close("all")
    yield
    plt.close("all")


def test_spectrogram_preserves_window_centers_frequency_and_magnitudes():
    x = np.arange(4, dtype=float)
    y = np.array([1.0, 2.0, 3.0, 4.0])

    x_spec, frequency, spectrum, maxima = spectrogram_fast(
        x,
        y,
        T=2,
        unit_T="index",
        window_type="rectangle",
        return_max_index=True,
        step=1,
    )

    np.testing.assert_array_equal(x_spec, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(frequency, np.array([0.0, 0.5]))
    np.testing.assert_array_equal(
        spectrum,
        np.array([[3.0, 5.0, 7.0], [1.0, 1.0, 1.0]]),
    )
    np.testing.assert_array_equal(maxima, np.array([0, 0, 0]))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"unit_T": "seconds"}, "unit_T must be 'index' or 'x'"),
        ({"T": 5}, "Window size T is larger than input signal"),
        ({"window_type": "gaussian"}, "Unknown window type"),
    ],
)
def test_spectrogram_preserves_explicit_input_errors(overrides, message):
    kwargs = {
        "x": np.arange(4, dtype=float),
        "y": np.ones(4),
        "T": 2,
        "unit_T": "index",
        "window_type": "rectangle",
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=message):
        spectrogram_fast(**kwargs)


def test_plot_all_preserves_explicit_times_labels_and_save_calls(tmp_path, monkeypatch):
    events: list[tuple[str, object, object]] = []
    monkeypatch.setattr(plot_all_module.time, "strftime", lambda _format: "STAMP")
    monkeypatch.setattr(
        plt,
        "show",
        lambda: events.append(("show", None, None)),
    )
    monkeypatch.setattr(
        plt,
        "savefig",
        lambda path, **kwargs: events.append(("save", Path(path), kwargs)),
    )
    optimizer = SimpleNamespace(tlist=np.array([0.0, 0.5, 1.0]), target_idx=1)
    field_data = np.array([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]])
    trajectory = np.array([[1.0, 0.0], [0.5, np.sqrt(0.75)]], dtype=complex)
    trajectory_times = np.array([0.0, 1.0])
    figures_dir = tmp_path / "figures"

    plot_all_module.plot_all(
        basis=object(),
        optimizer_like=optimizer,
        efield=object(),
        psi_traj=trajectory,
        field_data=field_data,
        sample_stride=7,
        trajectory_times_fs=trajectory_times,
        omega_center_cm=None,
        figures_dir=str(figures_dir),
        filename_prefix="contract",
        do_spectrum=False,
        do_spectrogram=False,
    )

    axes = [plt.figure(number).axes[0] for number in plt.get_fignums()]
    assert len(axes) == 2
    np.testing.assert_array_equal(axes[0].lines[0].get_xdata(), optimizer.tlist)
    np.testing.assert_array_equal(axes[0].lines[0].get_ydata(), field_data[:, 0])
    np.testing.assert_array_equal(axes[0].lines[1].get_ydata(), field_data[:, 1])
    assert axes[0].get_xlabel() == "Time [fs]"
    assert axes[0].get_ylabel() == "Electric Field [V/m]"
    np.testing.assert_array_equal(axes[1].lines[0].get_xdata(), trajectory_times)
    np.testing.assert_allclose(axes[1].lines[0].get_ydata(), np.array([0.0, 0.75]))
    assert axes[1].get_ylim() == pytest.approx((0.0, 1.05))
    assert events == [
        (
            "save",
            figures_dir / "contract_field_STAMP.png",
            {"dpi": 300, "bbox_inches": "tight"},
        ),
        ("show", None, None),
        (
            "save",
            figures_dir / "contract_fidelity_STAMP.png",
            {"dpi": 300, "bbox_inches": "tight"},
        ),
        ("show", None, None),
    ]


def test_result_directory_plotters_preserve_files_labels_and_show_save_order(
    tmp_path, monkeypatch
):
    time_axis = np.array([0.0, 0.5, 1.0])
    field = np.array([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]])
    vector = field.astype(complex) + 1j * 9.0
    population = np.array([[1.0, 0.0], [0.6, 0.4], [0.2, 0.8]])
    np.save(tmp_path / "tlist.npy", time_axis)
    np.save(tmp_path / "Efield_real.npy", field)
    np.save(tmp_path / "Efield_vector.npy", vector)
    np.save(tmp_path / "population.npy", population)
    events: list[tuple[str, Path | None]] = []
    monkeypatch.setattr(plt, "show", lambda: events.append(("show", None)))
    monkeypatch.setattr(
        plt,
        "savefig",
        lambda path, **_kwargs: events.append(("save", Path(path))),
    )

    field_module.plot_electric_field(tmp_path)
    field_axis = plt.gca()
    assert field_axis.get_xlabel() == "Time (fs)"
    assert field_axis.get_ylabel() == "Electric Field Amplitude"
    plt.close("all")

    vector_module.plot_electric_vector(tmp_path)
    vector_axis = plt.gca()
    np.testing.assert_array_equal(vector_axis.lines[0].get_ydata(), field[:, 0])
    np.testing.assert_array_equal(vector_axis.lines[1].get_ydata(), field[:, 1])
    plt.close("all")

    population_module.plot_population(tmp_path, state_index=1)
    population_axis = plt.gca()
    assert len(population_axis.lines) == 2
    assert [line.get_label() for line in population_axis.lines] == [
        "State 0",
        "State 1",
    ]
    np.testing.assert_array_equal(
        population_axis.lines[1].get_ydata(), population[:, 1]
    )
    assert events == [
        ("show", None),
        ("save", tmp_path / "electric_field_plot.png"),
        ("show", None),
        ("save", tmp_path / "electric_field_vector_plot.png"),
        ("show", None),
        ("save", tmp_path / "population_plot.png"),
    ]
