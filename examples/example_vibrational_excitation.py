#!/usr/bin/env python
"""
Vibrational Ladder System Excitation Simulation Example
======================================================

Time evolution simulation for pure vibrational systems (no rotation).
Tests both harmonic and Morse oscillators.

Usage:
    python examples/example_vibrational_excitation.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import matplotlib.pyplot as plt
from rovibrational_excitation.core.basis import VibLadderBasis
from rovibrational_excitation.core.states import StateVector
from rovibrational_excitation.core.electric_field import ElectricField, gaussian
from rovibrational_excitation.core.propagator import schrodinger_propagation
from rovibrational_excitation.dipole.viblad import VibLadderDipoleMatrix


# ============================================================================
# SIMULATION PARAMETERS - EDIT HERE
# ============================================================================
#
# USAGE:
# ------
# Simply modify the parameters below to customize your simulation:
#
# 1. SYSTEM_PARAMS: Control the quantum system properties
#    - V_max: Increase for higher vibrational states (computational cost increases)
#    - omega01: Fundamental vibrational frequency (affects energy spacing)
#    - domega_*: Anharmonicity (0 = harmonic, >0 = anharmonic)
#    - mu0_au: Dipole moment strength (affects transition rates)
#
# 2. FIELD_PARAMS: Control the laser pulse properties
#    - amplitude: Higher values = stronger excitation
#    - duration: Pulse width (shorter = broader spectrum)
#    - carrier_freq_*: Laser frequency tuning
#
# 3. TIME_PARAMS: Control simulation time and resolution
#    - tf: Longer times to see more dynamics
#    - dt: Smaller values for better accuracy (but slower)
#
# ============================================================================

# System parameters
SYSTEM_PARAMS = {
    'V_max': 5,                    # Maximum vibrational quantum number
    'omega01': 2349 * 2 * np.pi *3e10/1e15,                # Fundamental frequency (rad/fs)
    'domega_harmonic': 25 * 2 * np.pi *3e10/1e15,        # Anharmonicity for harmonic oscillator (rad/fs)
    'domega_morse': 25 * 2 * np.pi *3e10/1e15,          # Anharmonicity for Morse oscillator (rad/fs)
    'mu0_au': 1e-30,               # Dipole moment magnitude (a.u.)
}

# Electric field parameters
FIELD_PARAMS = {
    'amplitude': 5e9,             # Electric field amplitude (a.u.)
    'duration': 60.0,              # Pulse duration (fs)
    'polarization': [1, 0],        # Polarization vector [Ex, Ey]
    'axes': "zy",                  # Coupling axes (Ey with μ_z)
    'carrier_freq_factor': 1,    # Carrier frequency factor (multiply with omega01/(2π))
    'use_resonance': True,         # If True, use resonant frequency; if False, use custom frequency
    'custom_carrier_freq': 0.159,  # Custom carrier frequency (rad/fs) - used when use_resonance=False
}

# Time grid parameters
TIME_PARAMS = {
    'ti': 0.0,                     # Start time (fs)
    'tf': 1000.0,                   # End time (fs)
    'dt': 0.01,                     # Time step for electric field (fs)
    'sample_stride': 5,            # Sampling stride for time evolution
}

# Plot parameters
PLOT_PARAMS = {
    'max_states_plot': 6,          # Maximum number of states to plot
    'figsize_main': (10, 8),       # Figure size for main plots
    'figsize_comp': (10, 6),       # Figure size for comparison plot
    'dpi': 300,                    # Resolution for saved plots
}

# ============================================================================


def print_simulation_parameters():
    """Display current simulation parameters"""
    print("=" * 80)
    print("CURRENT SIMULATION PARAMETERS")
    print("=" * 80)
    
    print("\n[SYSTEM PARAMETERS]")
    for key, value in SYSTEM_PARAMS.items():
        print(f"  {key:<20}: {value}")
    
    print("\n[ELECTRIC FIELD PARAMETERS]")
    for key, value in FIELD_PARAMS.items():
        print(f"  {key:<20}: {value}")
    
    print("\n[TIME PARAMETERS]")
    for key, value in TIME_PARAMS.items():
        print(f"  {key:<20}: {value}")
    
    print("\n[PLOT PARAMETERS]")
    for key, value in PLOT_PARAMS.items():
        print(f"  {key:<20}: {value}")
    
    print("=" * 80)


def run_vibrational_excitation_simulation(potential_type="harmonic"):
    """
    Run vibrational excitation simulation
    
    Parameters
    ----------
    potential_type : str
        'harmonic' or 'morse'
    """
    print(f"=== Vibrational Excitation Simulation ({potential_type}) ===")
    
    # Get parameters from global settings
    V_max = SYSTEM_PARAMS['V_max']
    omega01 = SYSTEM_PARAMS['omega01']
    if potential_type == "morse":
        domega = SYSTEM_PARAMS['domega_morse']
    else:
        domega = SYSTEM_PARAMS['domega_harmonic']
    mu0_au = SYSTEM_PARAMS['mu0_au']
    
    # Generate basis and Hamiltonian
    basis = VibLadderBasis(V_max=V_max, omega_rad_phz=omega01, delta_omega_rad_phz=domega)
    H0 = basis.generate_H0()
    print(f"Vibrational energy levels: {np.diag(H0)}")
    
    # Generate dipole matrix
    dipole_matrix = VibLadderDipoleMatrix(
        basis,
        mu0=mu0_au,
        potential_type=potential_type
    )
    
    # Initial state: ground vibrational state |v=0⟩
    state = StateVector(basis)
    state.set_state((0,), 1.0)
    psi0 = state.data
    
    # Time grid and electric field settings
    ti = TIME_PARAMS['ti']
    tf = TIME_PARAMS['tf']
    dt4Efield = TIME_PARAMS['dt']
    time4Efield = np.arange(ti, tf + 2*dt4Efield, dt4Efield)
    
    # Gaussian pulse electric field settings
    duration = FIELD_PARAMS['duration']
    tc = (time4Efield[-1] + time4Efield[0]) / 2
    
    # Set carrier frequency
    if FIELD_PARAMS['use_resonance']:
        carrier_freq = omega01 / (2*np.pi) * FIELD_PARAMS['carrier_freq_factor']  # Resonant with vibrational frequency
    else:
        carrier_freq = FIELD_PARAMS['custom_carrier_freq']  # Custom frequency
    
    amplitude = FIELD_PARAMS['amplitude']
    polarization = np.array(FIELD_PARAMS['polarization'])  # 2-element vector: Ey direction (for vibrational transitions)
    
    Efield = ElectricField(tlist=time4Efield)
    Efield.add_dispersed_Efield(
        envelope_func=gaussian,
        duration=duration,
        t_center=tc,
        carrier_freq=carrier_freq,
        amplitude=amplitude,
        polarization=polarization,
        const_polarisation=False,
    )
    
    # Time evolution calculation
    print("Starting time evolution calculation...")
    sample_stride = TIME_PARAMS['sample_stride']
    time4psi, psi_t = schrodinger_propagation(
        H0=H0,
        Efield=Efield,
        dipole_matrix=dipole_matrix,
        psi0=psi0,
        axes=FIELD_PARAMS['axes'],  # Couple Ey with μ_z
        return_traj=True,
        return_time_psi=True,
        sample_stride=sample_stride
        )
    print("Time evolution calculation completed.")
    
    return time4Efield, Efield, time4psi, psi_t, basis


def plot_results(time4Efield, Efield, time4psi, psi_t, basis, potential_type):
    """Plot simulation results"""
    fig, axes = plt.subplots(2, 1, figsize=PLOT_PARAMS['figsize_main'], sharex=True)
    
    # Electric field time evolution
    Efield_data = Efield.get_Efield()
    axes[0].plot(time4Efield, Efield_data[:, 0], 'b-', linewidth=1.5, label=r"$E_y(t)$")
    axes[0].set_ylabel("Electric Field [a.u.]")
    axes[0].set_title(f"Vibrational Excitation Simulation ({potential_type.capitalize()})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Population dynamics for each vibrational state
    max_states = min(PLOT_PARAMS['max_states_plot'], basis.size())
    colors = plt.cm.viridis(np.linspace(0, 1, max_states))
    for i in range(max_states):  # Plot first N states only
        state = basis.get_state(i)
        v = state[0]
        prob = np.abs(psi_t[:, i])**2
        axes[1].plot(time4psi, prob, color=colors[i], linewidth=2, 
                    label=f"|v={v}⟩")
    
    # Check total probability conservation
    total_prob = np.sum(np.abs(psi_t)**2, axis=1)
    axes[1].plot(time4psi, total_prob, 'k--', alpha=0.8, linewidth=1, 
                label="Total")
    
    axes[1].set_xlabel("Time [fs]")
    axes[1].set_ylabel("Population")
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    
    # Save plot
    filename = f"examples/results/vibrational_excitation_{potential_type}.png"
    plt.savefig(filename, dpi=PLOT_PARAMS['dpi'], bbox_inches='tight')
    print(f"Plot saved to {filename}")
    
    return fig


def analyze_final_populations(psi_t, basis, potential_type):
    """Analyze final state populations"""
    print(f"\n=== Final State Analysis ({potential_type}) ===")
    final_populations = np.abs(psi_t[-1, :])**2
    
    print("Final vibrational state populations:")
    max_states = min(PLOT_PARAMS['max_states_plot'], basis.size())
    for i in range(max_states):
        state = basis.get_state(i)
        v = state[0]
        pop = final_populations[i]
        print(f"  |v={v}⟩: {pop:.4f}")
    
    # Calculate excitation efficiency
    ground_state_pop = final_populations[0]
    excited_pop = 1.0 - ground_state_pop
    print(f"\nGround state population: {ground_state_pop:.4f}")
    print(f"Excited state population: {excited_pop:.4f}")
    
    return final_populations


def main():
    """Main execution function"""
    print("Vibrational Ladder System Excitation Simulation")
    print("=" * 50)
    
    # Display current parameters
    print_simulation_parameters()
    
    # Create results directory
    os.makedirs("examples/results", exist_ok=True)
    
    # Harmonic oscillator simulation
    time4Efield_h, Efield_h, time4psi_h, psi_t_h, basis_h = run_vibrational_excitation_simulation("harmonic")
    plot_results(time4Efield_h, Efield_h, time4psi_h, psi_t_h, basis_h, "harmonic")
    final_pop_h = analyze_final_populations(psi_t_h, basis_h, "harmonic")
    
    # Morse oscillator simulation
    time4Efield_m, Efield_m, time4psi_m, psi_t_m, basis_m = run_vibrational_excitation_simulation("morse")
    plot_results(time4Efield_m, Efield_m, time4psi_m, psi_t_m, basis_m, "morse")
    final_pop_m = analyze_final_populations(psi_t_m, basis_m, "morse")
    
    # Comparison plot
    fig_comp, ax = plt.subplots(figsize=PLOT_PARAMS['figsize_comp'])
    
    max_states = min(PLOT_PARAMS['max_states_plot'], basis_h.size())
    v_states = np.arange(max_states)
    width = 0.35
    
    ax.bar(v_states - width/2, final_pop_h[:len(v_states)], width, 
           label='Harmonic', alpha=0.8, color='blue')
    ax.bar(v_states + width/2, final_pop_m[:len(v_states)], width, 
           label='Morse', alpha=0.8, color='red')
    
    ax.set_xlabel('Vibrational Quantum Number (v)')
    ax.set_ylabel('Final Population')
    ax.set_title('Final Population Comparison: Harmonic vs Morse Oscillator')
    ax.set_xticks(v_states)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("examples/results/vibrational_comparison.png", dpi=PLOT_PARAMS['dpi'], bbox_inches='tight')
    print("\nComparison plot saved to examples/results/vibrational_comparison.png")
    
    plt.show()
    
    print("\nAll calculations completed!")


if __name__ == "__main__":
    main() 