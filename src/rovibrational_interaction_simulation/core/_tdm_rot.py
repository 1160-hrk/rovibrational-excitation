import numpy as np

def tdm_jm_x(quanta1, quanta2):
    delta = quanta2 - quanta1
    J1 = quanta1[0]
    M1 = quanta1[1]
    J2 = quanta2[0]
    M2 = quanta2[1]
    if delta[0] == 1:
        if delta[1] in [1, -1]:
            tdm = -delta[1] * np.sqrt(
                (J2 + delta[1] * M2 + 1) * (J2 + delta[1] * M2 + 2)
                / ((2 * J2 + 1) * (2 * J2 + 3))
                ) / 2
        else:
            return 0.0
    elif delta[0] == -1:
        if delta[1] in [1, -1]:
            tdm = delta[1] * np.sqrt(
                (J1 - delta[1] * M1 + 1) * (J1 - delta[1] * M1 + 2)
                / ((2 * J1 + 1) * (2 * J1 + 3))
                ) / 2
        else:
            return 0.0
    else:
        return 0.0  # ΔJ ≠ ±1 → 禁制
    return tdm

def tdm_jm_y(quanta1, quanta2):
    delta = quanta2 - quanta1
    J1 = quanta1[0]
    M1 = quanta1[1]
    J2 = quanta2[0]
    M2 = quanta2[1]
    if delta[0] == 1:
        if delta[1] in [1, -1]:
            tdm = -1j * np.sqrt(
                (J2 - delta[1] * M2 + 1) * (J2 - delta[1] * M2 + 2)
                / ((2 * J2 + 1) * (2 * J2 + 3))
                ) / 2
        else:
            return 0.0
    elif delta[0] == -1:
        if delta[1] in [1, -1]:
            tdm = 1j * np.sqrt(
                (J1 + delta[1] * M1 + 1) * (J1 + delta[1] * M1 + 2)
                / ((2 * J1 + 1) * (2 * J1 + 3))
                ) / 2
        else:
            return 0.0
    else:
        return 0.0
    return tdm

def tdm_jm_z(quanta1, quanta2):
    delta = quanta2 - quanta1
    J1 = quanta1[0]
    M1 = quanta1[1]
    J2 = quanta2[0]
    M2 = quanta2[1]
    if delta[0] == 1:
        if delta[1] == 0:
            tdm = np.sqrt(
                (J2 + 1 - M2) * (J2 + 1 + M2)
                / ((2 * J2 + 1) * (2 * J2 + 3))
                )
        else:
            return 0.0
    elif delta[0] == -1:
        if delta[1] == 0:
            tdm = np.sqrt(
                (J1 + 1 - M1) * (J1 + 1 + M1)
                / ((2 * J1 + 1) * (2 * J1 + 3))
                )
        else:
            return 0.0
    else:
        return 0.0
    return tdm

tdm_jm_dict = {
    'x': tdm_jm_x,
    'y': tdm_jm_y,
    'z': tdm_jm_z,
}

def tdm_j(j1, j2):
    delta = j2 - j1
    if delta == 1:
        tdm *= np.sqrt(j1 / (2 * j1 + 1)) / 2
    elif delta == -1:
        tdm *= np.sqrt((j1 + 1) / (2 * j1 + 1)) / 2
    else:
        return 0.0  # ΔJ ≠ ±1 → 禁制
    return tdm