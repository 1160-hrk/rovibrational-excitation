# plot_electric_field.py
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_electric_field(result_dir):
    ereal_path = os.path.join(result_dir, "Efield_real.npy")
    tlist_path = os.path.join(result_dir, "tlist.npy")

    if not os.path.exists(ereal_path) or not os.path.exists(tlist_path):
        print(f"Missing electric field data in: {result_dir}")
        return

    E_real = np.load(ereal_path)  # shape = [2, T]
    print(f"E_real shape: {E_real.shape}")
    tlist = np.load(tlist_path)

    plt.figure(figsize=(8, 4))
    plt.plot(tlist, E_real)
    # plt.plot(tlist, E_real[1], label='Y polarization')
    plt.xlabel("Time (fs)")
    plt.ylabel("Electric Field Amplitude")
    plt.title(f"Electric Field in {os.path.basename(result_dir)}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # 保存
    filename = os.path.join(result_dir, "electric_field_plot.png")
    plt.savefig(filename, dpi=300)
    print(f"✅ Saved electric field vector plot to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot electric field from simulation result folder"
    )
    parser.add_argument(
        "result_dir", help="Path to result directory (contains Efield_real.npy)"
    )
    args = parser.parse_args()
    plot_electric_field(args.result_dir)

    # Test with a sample result directory
    # test_result_dir = "../results/2025-04-10_02-51-24_CO2_antisymm_stretch/gauss_width_50.0/pol_[1, 0]/delay_100.0"
    # plot_electric_field(test_result_dir)
