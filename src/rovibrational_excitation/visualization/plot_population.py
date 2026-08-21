# plot_population.py
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_population(result_dir, state_index=0):
    tlist_path = os.path.join(result_dir, "tlist.npy")
    pop_path = os.path.join(result_dir, "population.npy")

    if not os.path.exists(tlist_path) or not os.path.exists(pop_path):
        print(f"Missing data in: {result_dir}")
        return

    tlist = np.load(tlist_path)
    population = np.load(pop_path)

    plt.figure(figsize=(8, 4))
    for i in range(population.shape[1]):
        plt.plot(tlist, population[:, i], label=f"State {i}")

    plt.xlabel("Time (fs)")
    plt.ylabel("Population")
    plt.title(f"Population dynamics in {os.path.basename(result_dir)}")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    # 保存
    filename = os.path.join(result_dir, "population_plot.png")
    plt.savefig(filename, dpi=300)
    print(f"✅ Saved population plot to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot population from simulation result folder"
    )
    parser.add_argument(
        "result_dir", help="Path to result directory (contains population.npy)"
    )
    args = parser.parse_args()
    plot_population(args.result_dir)
