import numpy as np
from scipy.signal import convolve


def parallel_filter(n, period):
    """Analytic filter for parallel Filtered Backprojection.

    Args:
        n (np.ndarray): Discrete axis over which the filter impulse
            response will be computed.
        period (int, float): Sampling period of the discrete axis.

    Returns:
        filter (np.ndarray): Discrete filter impulse response over the
            axis n. Has same shape as n.
    """
    assert isinstance(n, np.ndarray), "n has wrong type. Should be numpy array"
    assert isinstance(period, (int, float)), (
        "period has wrong type. Should be int or float."
    )

    filter = np.zeros_like(n, dtype=float)

    filter[n % 2 == 1] = -(1 / (n[n % 2 == 1] * np.pi * period) ** 2)
    filter[n == 0] = 1 / (4 * period**2)

    return filter


def equiangular_filter(n, period):
    """Analytic filter for equiangular Filtered Backprojection.

    Args:
        n (np.ndarray): Discrete axis over which the filter impulse
            response will be computed.
        period (int, float): Sampling period of the discrete axis.

    Returns:
        filter (np.ndarray): Discrete filter impulse response over the
            axis n. Has same shape as n.
    """
    assert isinstance(n, np.ndarray), "n has wrong type. Should be numpy array"
    assert isinstance(period, (int, float)), (
        "period has wrong type. Should be int or float."
    )

    filter = np.zeros_like(n, dtype=float)

    filter[n % 2 == 1] = (
        period / (np.pi * period * np.sin(n[n % 2 == 1] * period))
    ) ** 2
    filter[n == 0] = 1 / (8 * period**2)

    return filter


def equidistant_filter(n, period):
    """Analytic filter for equidistant Filtered Backprojection.

    Args:
        n (np.ndarray): Discrete axis over which the filter impulse
            response will be computed.
        period (int, float): Sampling period of the discrete axis.

    Returns:
        filter (np.ndarray): Discrete filter impulse response over the
            axis n. Has same shape as n.
    """
    assert isinstance(n, np.ndarray), "n has wrong type. Should be numpy array"
    assert isinstance(period, (int, float)), (
        "period has wrong type. Should be int or float."
    )

    filter = np.zeros_like(n, dtype=float)

    filter[n % 2 == 1] = -(1 / 2 * (np.pi * period * (n[n % 2 == 1])) ** 2)
    filter[n == 0] = 1 / (8 * period**2)

    return filter


def build_freq_filter(N, filter_type="ramp", cutoff=None, period=None, num_terms=100):
    """Builds a filter in the frequency domain.

    Parameters
    ----------
        N (int): Length of filter.
        filter_type (str, optional): Type of the filter to
            generate. Default value is "ramp".
        cutoff (float, optional): Filter cutoff. Default value is 1.
        period (float, optional): Sampling period. Default value is None.
        num_terms (int, optional): Number of terms to consider for
            the computation of Laplacian of Sinc filter (LoSinc).
            Default value is 100.

    Returns
    -------
        (np.ndarray): Filter.
    """
    # Frequency vector
    freqs = np.fft.fftfreq(N).reshape(-1, 1)

    # Ramp base
    filt = np.abs(freqs)

    if filter_type == "shepp-logan":
        filt *= np.sinc(freqs / cutoff)
    elif filter_type == "cosine":
        filt *= np.cos(np.pi * freqs / (2 * cutoff))
    elif filter_type == "hamming":
        filt *= 0.54 + 0.46 * np.cos(np.pi * freqs / cutoff)
    elif filter_type == "hann":
        filt *= 0.5 * (1 + np.cos(np.pi * freqs / cutoff))
    elif filter_type != "ramp":
        raise ValueError(f"Unknown filter: {filter_type}")

    # Zero out frequencies beyond cutoff
    if cutoff is not None:
        filt[np.abs(freqs) > cutoff] = 0

    return filt.flatten()


def filter_multiply(sinogram, filter_kernel, period=1.0):
    """
    Filter sinogram projections using multiplication by the
    frequency response filter.

    Parameters
    ----------
    sinogram (np.ndarray): The sinogram array of shape
        (N_views, N_det).
    filter_kernel (np.ndarray): The 1D frequency-domain
        filter of shape (N_det,).
    period (float): Sampling period (e.g., detector spacing).
        Default is 1.0.

    Returns
    -------
    filtered (np.ndarray): The filtered sinogram of shape
        (N_views, N_det).
    """
    if sinogram.ndim != 2:
        raise ValueError("sinogram must be a 2D array (N_views, N_det).")
    if filter_kernel.ndim != 1:
        raise ValueError("filter_kernel must be a 1D array (N_det,).")

    N_views, N_det = sinogram.shape
    if filter_kernel.shape[0] != N_det:
        raise ValueError(
            "filter_kernel length must match the number of detectors (sinogram.shape[1])."
        )

    # FFT of sinogram along detector axis
    sinogram_fft = np.fft.fft(sinogram, axis=1)

    # Multiply by the filter in frequency domain
    filtered_fft = sinogram_fft * filter_kernel  # broadcasted multiplication

    # Inverse FFT to get filtered sinogram in time domain and scale by period
    filtered = np.fft.ifft(filtered_fft, axis=1).real
    # filtered *= period

    return filtered


def filter_convolve(sinogram, filter_impulse, period=1.0):
    """
    Filter sinogram projections using convolution by the
    impulse response of a filter.

    Parameters
    ----------
    sinogram (np.ndarray): The sinogram array of shape
        (N_views, N_det).
    filter_impulse (np.ndarray): The 1D impulse response of the
        filter of shape (N_det,).
    period (float): Sampling period (e.g., detector spacing).
        Default is 1.0.

    Returns
    -------
    filtered (np.ndarray): The filtered sinogram of shape
        (N_views, N_det).
    """
    if sinogram.ndim != 2:
        raise ValueError("sinogram must be a 2D array (N_views, N_det).")
    if filter_impulse.ndim != 1:
        raise ValueError("filter_impulse must be a 1D array (N_det,).")

    # Normalize filter to preserve scaling (optional)
    filter_impulse = filter_impulse / np.sum(filter_impulse)

    # Multiply by the filter in frequency domain
    filtered = convolve(sinogram, filter_impulse.reshape(1, -1), mode="same")

    # Scale by sampling period
    filtered *= period

    return filtered


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    N_det = 512
    cutoff = 1

    freq = np.fft.fftshift(np.fft.fftfreq(N_det))

    filter_types = ("ramp", "shepp-logan", "cosine", "hamming", "hann")

    fig, axs = plt.subplots(2, 3)
    for ax, filter_type in zip(axs.flat, filter_types):
        filter = build_freq_filter(N_det, filter_type, cutoff=cutoff)
        ax.plot(freq, np.fft.fftshift(filter))
        ax.set_title(f"Filter {filter_type} with cutoff = {cutoff}")
        ax.set_xlabel("f[Hz]")
        ax.set_ylabel("h(f)")
        ax.grid()
    plt.tight_layout()
    plt.show()
