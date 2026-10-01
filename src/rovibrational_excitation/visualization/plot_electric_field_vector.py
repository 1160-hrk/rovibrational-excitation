import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .result_data import ResultPath, load_cartesian_field_plot_data


def plot_electric_vector(result_dir: ResultPath) -> None:
    """Plot both validated Cartesian electric-field components of one result."""
    result_path = Path(result_dir)
    tlist, E_vec = load_cartesian_field_plot_data(result_path)
    print(f"E_vec shape: {E_vec.shape}")

    Ex = np.real(E_vec[:, 0])
    Ey = np.real(E_vec[:, 1])

    plt.figure(figsize=(8, 4))
    plt.plot(tlist, Ex, label="Re(E_x)", color="tab:blue")
    plt.plot(tlist, Ey, label="Re(E_y)", color="tab:orange")
    plt.xlabel("Time (fs)")
    plt.ylabel("Electric Field Amplitude")
    plt.title(f"Real part of Jones vector (Electric field)\n{result_path.name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # 保存
    filename = result_path / "electric_field_vector_plot.png"
    plt.savefig(filename, dpi=300)
    print(f"✅ Saved electric field vector plot to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot Cartesian electric-field components from a versioned result"
    )
    parser.add_argument(
        "result_dir", help="Path to result directory (contains result_current.json)"
    )
    args = parser.parse_args()
    plot_electric_vector(args.result_dir)
