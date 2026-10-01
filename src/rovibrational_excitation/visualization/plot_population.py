import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from .result_data import ResultPath, load_population_plot_data


def plot_population(result_dir: ResultPath, state_index: int = 0) -> None:
    """Plot all populations from one validated versioned result."""
    result_path = Path(result_dir)
    tlist, population = load_population_plot_data(result_path)

    plt.figure(figsize=(8, 4))
    for i in range(population.shape[1]):
        plt.plot(tlist, population[:, i], label=f"State {i}")

    plt.xlabel("Time (fs)")
    plt.ylabel("Population")
    plt.title(f"Population dynamics in {result_path.name}")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    # 保存
    filename = result_path / "population_plot.png"
    plt.savefig(filename, dpi=300)
    print(f"✅ Saved population plot to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot populations from a versioned simulation result"
    )
    parser.add_argument(
        "result_dir", help="Path to result directory (contains result_current.json)"
    )
    args = parser.parse_args()
    plot_population(args.result_dir)
