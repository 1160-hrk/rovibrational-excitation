import numpy as np


def tdm_vib_harm(v1, v2):
    if v1 == v2 + 1:
        return np.sqrt(v1)
    elif v1 == v2 - 1:
        return np.sqrt(v2)
    else:
        return 0.0
