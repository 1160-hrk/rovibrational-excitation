import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from .result_data import ResultPath, load_field_plot_data


def plot_electric_field(result_dir: ResultPath) -> None:
    """Plot the validated scalar or Cartesian electric field of one result."""
    result_path = Path(result_dir)
    tlist, E_real = load_field_plot_data(result_path)
    print(f"E_real shape: {E_real.shape}")

    plt.figure(figsize=(8, 4))
    plt.plot(tlist, E_real)
    # plt.plot(tlist, E_real[1], label='Y polarization')
    plt.xlabel("Time (fs)")
    plt.ylabel("Electric Field Amplitude")
    plt.title(f"Electric Field in {result_path.name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # 保存
    filename = result_path / "electric_field_plot.png"
    plt.savefig(filename, dpi=300)
    print(f"✅ Saved electric field vector plot to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot electric field from a versioned simulation result"
    )
    parser.add_argument(
        "result_dir", help="Path to result directory (contains result_current.json)"
    )
    args = parser.parse_args()
    plot_electric_field(args.result_dir)
