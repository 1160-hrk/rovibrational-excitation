# rovibrational-excitation
[![PyPI version](https://img.shields.io/pypi/v/rovibrational-excitation.svg)](https://pypi.org/project/rovibrational-excitation/)
[![Python](https://img.shields.io/pypi/pyversions/rovibrational-excitation.svg)](https://pypi.org/project/rovibrational-excitation/)
[![License](https://img.shields.io/github/license/1160-hrk/rovibrational-excitation.svg)](https://github.com/1160-hrk/rovibrational-excitation/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/rovibrational-excitation.svg)](https://pypi.org/project/rovibrational-excitation/)
[![Coverage](https://img.shields.io/badge/coverage-75%25-yellowgreen.svg)](tests/README.md#現在のテストカバレッジ)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Python package for **time-dependent quantum dynamics** of
linear molecules (rotation × vibration) driven by femtosecond–picosecond
laser pulses.

<div align="center">

| CPU / GPU (CuPy) | Numba-JIT RK4 propagator | Lazy, cached dipole matrices |
|------------------|--------------------------|------------------------------|

</div>

---

## Key features

* **Runge–Kutta 4 (RK-4)** propagators for the Schrödinger and Liouville–von Neumann equations (`complex128`, cache-friendly).
* **Lazy, high-speed construction** of transition-dipole matrices (`rovibrational_excitation.dipole.*`)  
  * rigid-rotor + harmonic / Morse vibration  
  * Numba (CPU) or CuPy (GPU) backend
* **Vector electric-field objects** with Gaussian envelopes, chirp, optional sinusoidal and binned modulation.
* **Batch runner** for pump–probe / parameter sweeps with automatic directory creation, progress-bar and compressed output (`.npz`).
* 100 % pure-Python, **no compiled extension to ship** (Numba compiles at runtime).
* Currently, only linear molecules are supported; that is, only the rotational quantum numbers J and M are taken into account.


---

## Testing & Coverage

The package includes a comprehensive test suite with **75% code coverage** across all modules.

- 🟢 **Basis classes**: 100% coverage (LinMol, TwoLevel, VibLadder)
- 🟢 **Core physics**: 83-98% coverage (Hamiltonian, States, Propagator)
- 🟡 **Electric field**: 53% coverage
- 🔴 **Low-level propagators**: 25-38% coverage (ongoing development)

See [`tests/README.md`](tests/README.md) for detailed coverage reports and test instructions.

```bash
# Run tests
cd tests/ && python -m pytest -v

# Generate coverage report
coverage run -m pytest && coverage report
```

---

## Installation

```bash
# From PyPI  (stable)
pip install rovibrational-excitation          # installs sub-packages as well

# Or from GitHub (main branch, bleeding-edge)
pip install git+https://github.com/1160-hrk/rovibrational-excitation.git
````

> **CuPy (optional)** – for GPU acceleration
>
> ```bash
> pip install cupy-cuda12x     # pick the wheel that matches your CUDA
> ```

---

## Quick start : library API

```python
import numpy as np
import rovibrational_excitation as rve

# --- 1. Basis & dipole matrices ----------------------------------
c_vacuum = 299792458 * 1e2 / 1e15  # cm/fs
debye_unit = 3.33564e-30                       # 1 D → C·m
Omega01_rad_phz = 2349*2*np.pi*c_vacuum
Delta_omega_rad_phz = 25*2*np.pi*c_vacuum
B_rad_phz = 0.39e-3*2*np.pi*c_vacuum
Mu0_Cm = 0.3 * debye_unit                      # 0.3 Debye 相当
Potential_type = "harmonic"  # or "morse"
V_max = 2
J_max = 4

basis = rve.LinMolBasis(
            V_max=V_max,
            J_max=J_max,
            use_M = True,
            omega_rad_phz = Omega01_rad_phz,
            delta_omega_rad_phz = Delta_omega_rad_phz
            )           # |v J M⟩ direct-product

dip   = rve.LinMolDipoleMatrix(
            basis, mu0=Mu0_Cm, potential_type=Potential_type,
            backend="numpy", dense=True)            # CSR on GPU

mu_x  = dip.mu_x            # lazy-built, cached thereafter
mu_y  = dip.mu_y
mu_z  = dip.mu_z

# --- 2. Hamiltonian ----------------------------------------------
H0 = rve.generate_H0_LinMol(
        basis,
        omega_rad_phz       = Omega01_rad_phz,
        delta_omega_rad_phz = Delta_omega_rad_phz,
        B_rad_phz           = B_rad_phz,
)

# --- 3. Electric field -------------------------------------------
t  = np.linspace(-200, 200, 4001)                   # fs
E  = rve.ElectricField(tlist=t)
E.add_dispersed_Efield(
        envelope_func=rve.core.electric_field.gaussian_fwhm,
        duration=50.0,             # FWHM (fs)
        t_center=0.0,
        carrier_freq=2349*2*np.pi*c_vacuum,   # rad/fs
        amplitude=1.0,
        polarization=[1.0, 0.0],   # x-pol.
)

# --- 4. Initial state |v=0,J=0,M=0⟩ ------------------------------
from rovibrational_excitation.core.states import StateVector
psi0 = StateVector(basis)
psi0.set_state((0,0,0), 1.0)
psi0.normalize()

# --- 5. Time propagation (Schrödinger) ---------------------------
psi_t = rve.schrodinger_propagation(
            H0, E, dip,
            psi0.data,
            axes="xy",              # Ex→μx, Ey→μy
            sample_stride=10,
            backend="numpy")        # or "cupy"

population = np.abs(psi_t)**2
print(population.shape)            # (Nt, dim)
```

---

## Quick start : batch runner

1. **Create a parameter file** (`params_CO2.py`)

```python
# description is used in results/<timestamp>_<description>/
description = "CO2_antisymm_stretch"

# --- time axis (fs) ---------------------------------------------
t_start, t_end, dt = -200.0, 200.0, 0.1       # Unit is fs

# --- electric-field scan ----------------------------------------
duration       = [50.0, 80.0]                 # Gaussian FWHM (fs)
polarization   = [[1,0], [1/2**0.5,1j/2**0.5]]
t_center       = [0.0, 100.0]

carrier_freq   = 2349*2*np.pi*1e12*1e-15      # rad/fs
amplitude      = 1.0e9                        # V/m

# --- molecular constants ----------------------------------------
V_max, J_max   = 2, 4
omega_rad_phz  = carrier_freq * 2 * np.pi
mu0_Cm         = 0.3 * 3.33564e-30            # 0.3 D
```

2. **Run**

```bash
python -m rovibrational_excitation.simulation.runner \
       examples/params_CO2.py     -j 4      # 4 processes
```

* Creates `results/YYYY-MM-DD_hh-mm-ss_CO2_antisymm_stretch/…`
* For each case a folder with `result.npz`, `parameters.json`
* Top-level `summary.csv` (final populations etc.)

> Add `--dry-run` to just list cases without running.

---

## Directory layout (after refactor)

```
rovibrational_excitation/
  __init__.py                # public re-export
  core/                      # low-level numerics
    basis.py, propagator.py, ...
  dipole/
    linmol/                  # high-level dipole API
      builder.py, cache.py
    rot/                     # rotational TDM formulae
      jm.py, j.py
    vib/
      harmonic.py, morse.py
  plots/                     # helper scripts (matplotlib)
  simulation/                # batch manager, CLI
```

---

## Development

```bash
git clone https://github.com/1160-hrk/rovibrational-excitation.git
cd rovibrational-excitation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -v
```

Black + Ruff + MyPy configs are in *pyproject.toml*.

---

## License

[MIT](LICENSE)

© 2025 Hiroki Tsusaka. All rights reserved.
