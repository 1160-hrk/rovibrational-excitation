import math

import numpy as np


def omega01_domega_to_N(omega01: float, domega: float) -> float:
    """Return the Morse level parameter derived from spectroscopic constants."""
    if domega == 0:
        raise ValueError("delta_omega must be non-zero when potential_type='morse'")
    return (omega01 + domega) / domega - 1 / 2


def tdm_vib_morse(v1, v2, level_parameter):
    """Return a Morse vibrational transition element without global state."""
    tdm0 = (
        2
        / (2 * level_parameter - 1)
        * np.sqrt((level_parameter - 1) * level_parameter / (2 * level_parameter))
    )
    if v1 > v2:
        Nu, Nl = v1, v2
    elif v1 < v2:
        Nu, Nl = v2, v1
    else:
        return 0
    array_for_gamma_fun = np.arange(-Nu + 1, -Nl + 1) + 2 * level_parameter
    array_for_factorial = np.arange(Nl + 1, Nu + 1)
    tdm = (
        2
        * (-1) ** (Nu - Nl + 1)
        / ((Nu - Nl) * (2 * level_parameter - Nl - Nu))
        * np.sqrt(
            (level_parameter - Nl)
            * (level_parameter - Nu)
            * np.prod(array_for_factorial)
            / (np.prod(array_for_gamma_fun))
        )
        / tdm0
    )
    return tdm


def validate_morse_v_max(v_max: int, level_parameter: float) -> None:
    """Reject a vibrational basis extending beyond the Morse bound levels."""
    max_allowed = math.floor(level_parameter) - 1
    if v_max > max_allowed:
        raise ValueError(
            f"V_max={v_max} exceeds the Morse limit {max_allowed} "
            f"derived from N={level_parameter:g}"
        )
