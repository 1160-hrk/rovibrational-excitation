import numpy as np
from typing import Tuple, Union, Optional

def spectrogram_fast(
    x: np.ndarray,
    y: np.ndarray,
    T: Union[int, float],
    unit_T: str = 'index',
    window_type: str = 'hamming',
    return_max_index: bool = False,
    step: int = 1,
    N_pad: Optional[int] = None,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Fast version of spectrogram using precomputed FFT frequency and preallocated arrays.

    This function computes the spectrogram of a signal using a sliding window approach.
    It applies a window function to each segment of the signal and performs a Fast Fourier Transform (FFT)
    on the windowed segment. Only the non-negative frequency components are kept. The output is a 2D array
    where each column represents the FFT magnitude spectrum at a specific time point.

    Parameters:
    x : np.ndarray
        Time axis (1D array).
    y : np.ndarray
        Signal to analyze (same shape as x).
    T : int or float
        Window size. If unit_T is 'index', interpreted as number of samples.
        If unit_T is 'x', interpreted as range in units of x.
    unit_T : str, default='index'
        Unit of T. Either 'index' or 'x'.
    window_type : str, default='hamming'
        Window function to apply. Options: 'rectangle', 'triangle', 'hamming', 'han', 'blackman'.
    return_max_index : bool, default=False
        Whether to return the frequency index with the maximum amplitude for each window.
    step : int, default=1
        Step size for sliding the window.

    Returns:
    Tuple containing:
    - x_spec : np.ndarray
        Center positions of each window on the x-axis.
    - freq : np.ndarray
        Array of positive frequency components.
    - spec : np.ndarray
        2D spectrogram array of shape (len(freq), number of windows).
    - max_indices : np.ndarray, optional
        Indices of the frequency component with the maximum amplitude per window.
        Only returned if return_max_index is True.
    """
    dx = x[1] - x[0]
    N = len(x)

    if unit_T == 'x':
        T_index = int(np.round(T / dx))
    elif unit_T == 'index':
        T_index = int(T)
    else:
        raise ValueError("unit_T must be 'index' or 'x'")

    if T_index > N:
        raise ValueError("Window size T is larger than input signal.")

    if window_type == 'rectangle':
        window = np.ones(T_index)
    elif window_type == 'triangle':
        window = 1 - np.abs((np.arange(T_index) - T_index / 2) / (T_index / 2))
    elif window_type == 'hamming':
        window = np.hamming(T_index)
    elif window_type == 'han':
        window = np.hanning(T_index)
    elif window_type == 'blackman':
        window = np.blackman(T_index)
    else:
        raise ValueError("Unknown window type")

    # Use rFFT with zero-padding length N_pad if provided
    n_rfft = int(T_index if N_pad is None else max(int(N_pad), int(T_index)))
    freq = np.fft.rfftfreq(n_rfft, d=dx)
    num_freq = len(freq)

    num_windows = (N - T_index) // step + 1
    spec = np.empty((num_freq, num_windows), dtype=np.float64)
    max_indices = np.empty(num_windows, dtype=int) if return_max_index else None
    x_spec = np.empty(num_windows, dtype=np.float64)

    for i in range(num_windows):
        start = i * step
        end = start + T_index
        segment = y[start:end] * window
        # rFFT with zero-padding to n_rfft
        fft_segment = np.fft.rfft(segment, n=n_rfft)
        abs_fft = np.abs(fft_segment)
        spec[:, i] = abs_fft
        if return_max_index and max_indices is not None:
            max_indices[i] = np.argmax(abs_fft)
        x_spec[i] = x[start + T_index // 2]

    if return_max_index:
        if max_indices is None:
            max_indices = np.array([], dtype=int)
        return x_spec, freq, spec, max_indices
    else:
        return x_spec, freq, spec

