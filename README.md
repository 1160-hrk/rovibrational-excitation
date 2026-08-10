# rovibrational-excitation

<!-- Core Information -->
[![PyPI version](https://img.shields.io/pypi/v/rovibrational-excitation.svg)](https://pypi.org/project/rovibrational-excitation/)
[![Python](https://img.shields.io/pypi/pyversions/rovibrational-excitation.svg)](https://pypi.org/project/rovibrational-excitation/)
[![Downloads](https://img.shields.io/pypi/dm/rovibrational-excitation.svg)](https://pypi.org/project/rovibrational-excitation/)
[![License](https://img.shields.io/github/license/1160-hrk/rovibrational-excitation.svg)](https://github.com/1160-hrk/rovibrational-excitation/blob/main/LICENSE)

<!-- Quality Assurance -->
[![Tests](https://github.com/1160-hrk/rovibrational-excitation/actions/workflows/tests.yml/badge.svg)](https://github.com/1160-hrk/rovibrational-excitation/actions/workflows/tests.yml)
[![Coverage](https://codecov.io/gh/1160-hrk/rovibrational-excitation/branch/main/graph/badge.svg)](https://codecov.io/gh/1160-hrk/rovibrational-excitation)
[![Code Quality](https://github.com/1160-hrk/rovibrational-excitation/actions/workflows/ci.yml/badge.svg)](https://github.com/1160-hrk/rovibrational-excitation/actions/workflows/ci.yml)

Python package for **time-dependent quantum dynamics** of
linear molecules (rotation × vibration) driven by femtosecond–picosecond
laser pulses.

<div align="center">

| CPU / GPU (CuPy) | Numba-JIT RK4 propagator | Lazy, cached dipole matrices |
|------------------|--------------------------|------------------------------|

</div>

---

## Key features

### 🔧 High-Performance Time Evolution Engine
* **Runge–Kutta 4 (RK-4)** propagators for the Schrödinger and Liouville–von Neumann equations (`complex128`, cache-friendly).
* **Split-operator method** with CPU/GPU backends for efficient propagation.

### ⚡ High-Speed Dipole Matrix Construction
* **Lazy, high-speed construction** of transition-dipole matrices (`rovibrational_excitation.dipole.*`)  
  * rigid-rotor + harmonic / Morse vibration  
  * Numba (CPU) or CuPy (GPU) backend
* **Lazy evaluation & caching** for fast computation

### 🌊 Flexible Electric Field Control
* **Vector electric-field objects** with Gaussian envelopes, chirp, optional sinusoidal and binned modulation.
* Gaussian envelope, chirp functionality
* Sinusoidal and binned modulation options
* Vector field support

### 📊 Batch Processing & Analysis
* **Batch runner** for pump–probe / parameter sweeps with automatic directory creation, progress-bar and compressed output (`.npz`).
* Pump-probe experiment simulation
* Parameter sweep capabilities
* Automatic directory creation
* Progress bar display
* Compressed output (`.npz`)

### 🔬 Supported Molecules
* Currently, only **linear molecules** are supported; that is, only the rotational quantum numbers J and M are taken into account.
* Future extension to non-linear molecules is planned.

### 🏗️ Pure Python Implementation
* 100 % pure-Python, **no compiled extension to ship** (Numba compiles at runtime).

---

## Testing & Coverage

The package includes a comprehensive test suite with **63% code coverage** across all modules.

- 🟢 **Basis classes**: 100% coverage (LinMol, TwoLevel, VibLadder)
- 🟡 **Core physics**: 55% overall coverage
  - States: 98% coverage  
  - Propagator: 83% coverage
  - Hamiltonian: 67% coverage
- 🟡 **Electric field**: 53% coverage
- 🟡 **Dipole matrices**: 52-96% coverage (varies by subsystem)
- 🔴 **Low-level propagators**: 25-38% coverage (ongoing development)
- 🟡 **Simulation runner**: 62% coverage

See [`tests/README.md`](tests/README.md) for detailed coverage reports and test instructions.

```bash
# Run tests
cd tests/ && python -m pytest -v

# Generate coverage report
coverage run -m pytest && coverage report
```

---

## Installation

### Stable Release (PyPI)
```bash
# From PyPI  (stable)
pip install rovibrational-excitation          # installs sub-packages as well
```

### Development Version (GitHub)
```bash
# Or from GitHub (main branch, bleeding-edge)
pip install git+https://github.com/1160-hrk/rovibrational-excitation.git
```

### GPU Acceleration (Optional)
> **CuPy (optional)** – for GPU acceleration
>
> ```bash
> pip install cupy-cuda12x     # pick the wheel that matches your CUDA
> ```

---

## Requirements

### Python Environment
- **Python**: 3.10+
- **NumPy**: Array operations & numerical computing
- **SciPy**: Scientific computing library
- **Numba**: JIT compilation (CPU acceleration)

### Optional Dependencies
- **CuPy**: GPU computing (requires CUDA)
- **Matplotlib**: Graph plotting
- **tqdm**: Progress bars

---

## 📚 Documentation

For detailed usage instructions and parameter reference:

| Document | Description | Audience |
|----------|-------------|----------|
| **[docs/PARAMETER_REFERENCE.md](docs/PARAMETER_REFERENCE.md)** | **Complete parameter reference** | All users |
| [docs/SWEEP_SPECIFICATION.md](docs/SWEEP_SPECIFICATION.md) | Parameter sweep specification | Intermediate |
| [docs/README.md](docs/README.md) | Documentation index & quick guides | All users |
| [examples/params_template.py](examples/params_template.py) | Parameter file template | Beginners |

### 🚀 Getting Started

1. **Read the parameter reference**: [docs/PARAMETER_REFERENCE.md](docs/PARAMETER_REFERENCE.md)
2. **Copy the template**: `cp examples/params_template.py my_params.py`
3. **Edit parameters** according to your system
4. **Run simulation**: `python -m rovibrational_excitation.simulation.runner my_params.py`

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
omega_rad_phz   = carrier_freq * 2 * np.pi
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

## Applications

### CO2 Antisymmetric Stretch Vibration Excitation
- **Molecule**: CO2 (linear triatomic molecule)
- **Excitation mode**: Antisymmetric stretch vibration (ν₃ ≈ 2349 cm⁻¹)
- **Laser**: Femtosecond pulse
- **Analysis**: Population transfer between vibrational levels

### Pump-Probe Experiments
- **Pump pulse**: Molecular excitation
- **Probe pulse**: State exploration after time delay
- **Measurements**: Time-resolved spectra, population dynamics

---

## Directory layout

```
rovibrational_excitation/
├── src/rovibrational_excitation/
│   ├── __init__.py          # public re-export
│   ├── core/                # low-level numerics
│   │   ├── basis/           # quantum basis classes
│   │   │   ├── __init__.py
│   │   │   ├── base.py      # abstract base class
│   │   │   ├── linmol.py    # linear molecule basis
│   │   │   ├── twolevel.py  # two-level system
│   │   │   └── viblad.py    # vibrational ladder
│   │   ├── propagator.py    # time evolution
│   │   ├── electric_field.py
│   │   ├── hamiltonian.py   # DEPRECATED
│   │   ├── states.py        # quantum state vectors
│   │   ├── _rk4_schrodinger.py
│   │   ├── _rk4_lvne.py
│   │   └── _splitop_schrodinger.py
│   ├── dipole/              # transition dipole matrices
│   │   ├── linmol/          # linear molecules
│   │   │   ├── builder.py   # matrix construction
│   │   │   └── cache.py     # caching system
│   │   ├── twolevel/        # two-level systems
│   │   ├── viblad/          # vibrational ladder
│   │   ├── rot/             # rotational elements
│   │   │   ├── j.py         # J quantum number
│   │   │   └── jm.py        # J,M quantum numbers
│   │   └── vib/             # vibrational elements
│   │       ├── harmonic.py  # harmonic oscillator
│   │       └── morse.py     # Morse oscillator
│   ├── plots/               # visualization helpers
│   │   ├── plot_electric_field.py
│   │   ├── plot_electric_field_vector.py
│   │   └── plot_population.py
│   └── simulation/          # batch runner & CLI
│       ├── runner.py        # main execution engine
│       ├── manager.py       # execution management
│       └── config.py        # configuration handling
├── tests/                   # unit tests (pytest)
├── validation/              # collected-validation policy redirect
│   └── README.md
├── examples/                # usage examples
└── docs/                    # documentation
```

### Validation and Testing

All scientific correctness checks are collected by pytest. Physics references
live in `tests/physics/`, API and capability guarantees in
`tests/contracts/`, and cross-component workflows in `tests/integration/`.
Reproducible runtime and memory measurements live in `benchmarks/` and
`tests/performance/`.

```bash
pytest -q
pytest -m physics
```

The retained `validation/README.md` is a policy redirect; standalone
print/plot diagnostics are not correctness gates.

---

## Development

```bash
git clone https://github.com/1160-hrk/rovibrational-excitation.git
cd rovibrational-excitation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -v
```

### Development Tools
- **Black**: Code formatter
- **Ruff**: High-speed linter  
- **MyPy**: Static type checking
- **pytest**: Testing framework

Black + Ruff + MyPy configs are in *pyproject.toml*.

---

## Contributing

1. **Issue Reports**: Bug reports & feature requests
2. **Pull Requests**: Code improvements & new features
3. **Documentation**: Usage examples & tutorials

### Development Guidelines
- PEP8-compliant code style
- Type hints required
- Maintain test coverage
- Detailed docstrings

---

## References

1. **Quantum Mechanics**: Griffiths, "Introduction to Quantum Mechanics"
2. **Molecular Spectroscopy**: Herzberg, "Molecular Spectra and Molecular Structure"
3. **Numerical Methods**: Press et al., "Numerical Recipes"

---

## License

[MIT](LICENSE)

© 2025 Hiroki Tsusaka. All rights reserved.

---

## Contact

- **GitHub Issues**: [Repository](https://github.com/1160-hrk/rovibrational-excitation)
- **Email**: Please check the project page

---

*Last updated: January 2025*
