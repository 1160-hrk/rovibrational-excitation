import numpy as np

global N
N = 200


def omega01_domega_to_N(omega01, domega):
    global N
    if domega != 0:
        N = (omega01 + domega) / domega - 1 / 2
    return None


def tdm_vib_morse(v1, v2):
    global N
    tdm0 = 2 / (2 * N - 1) * np.sqrt((N - 1) * N / (2 * N))
    if v1 > v2:
        Nu, Nl = v1, v2
    elif v1 < v2:
        Nu, Nl = v2, v1
    else:
        return 0
    array_for_gamma_fun = np.arange(-Nu + 1, -Nl + 1) + 2 * N
    array_for_factorial = np.arange(Nl + 1, Nu + 1)
    tdm = (
        2
        * (-1) ** (Nu - Nl + 1)
        / ((Nu - Nl) * (2 * N - Nl - Nu))
        * np.sqrt(
            (N - Nl)
            * (N - Nu)
            * np.prod(array_for_factorial)
            / (np.prod(array_for_gamma_fun))
        )
        / tdm0
    )
    return tdm
