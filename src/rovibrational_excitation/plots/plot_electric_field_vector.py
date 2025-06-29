# plot_electric_vector.py
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_electric_vector(result_dir):
    evec_path = os.path.join(result_dir, "Efield_vector.npy")
    tlist_path = os.path.join(result_dir, "tlist.npy")

    if not os.path.exists(evec_path) or not os.path.exists(tlist_path):
        print(f"Missing vector field data in: {result_dir}")
        return

    E_vec = np.load(evec_path)  # shape = [2, T]
    tlist = np.load(tlist_path)

    E_vec = np.squeeze(E_vec)
    print(f"E_vec shape: {E_vec.shape}")

    Ex = np.real(E_vec[:, 0])
    Ey = np.real(E_vec[:, 1])

    plt.figure(figsize=(8, 4))
    plt.plot(tlist, Ex, label="Re(E_x)", color="tab:blue")
    plt.plot(tlist, Ey, label="Re(E_y)", color="tab:orange")
    plt.xlabel("Time (fs)")
    plt.ylabel("Electric Field Amplitude")
    plt.title(
        f"Real part of Jones vector (Electric field)\n{os.path.basename(result_dir)}"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # 保存
    filename = os.path.join(result_dir, "electric_field_vector_plot.png")
    plt.savefig(filename, dpi=300)
    print(f"✅ Saved electric field vector plot to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot complex electric vector field (real part) from result folder"
    )
    parser.add_argument(
        "result_dir", help="Path to result directory (contains Efield_vector.npy)"
    )
    args = parser.parse_args()
    plot_electric_vector(args.result_dir)

    # Test with a sample result directory
    # test_result_dir = "../results/2025-04-10_02-51-24_CO2_antisymm_stretch/gauss_width_50.0/pol_[1, 0]/delay_100.0"
    # plot_electric_vector(test_result_dir)
