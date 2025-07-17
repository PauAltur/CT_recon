import numpy as np


def build_filter(N, filter_type="ramp", cutoff=1.0):
    """Builds a filter.

    Parameters
    ----------
        N (int): Length of filter.
        filter_type (str, optional): Type of the filter to
            generate. Default value is "ramp".
        cutoff (float): Filter cutoff. Default value is 1.

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
    filt[np.abs(freqs) > cutoff] = 0

    return filt.flatten()


def filter_projections_freq_kernel(sinogram, filter_kernel, delta_beta=1.0):
    """Filter projections using a frequency domain filter kernel.

    Parameters
    ----------
        sinogram (np.ndarray): Array of projections of shape
            (N_views, N_det).
        filter_kernel (np.ndarray): Filter kernel of shape (N_det,).
        delta_beta (float, optional): Angular step size in radians.
            Default is 1.0.

    Returns
    -------
        (np.ndarray): Array of filtered projections of shape
            (N_views, N_det).
    """
    # Ensure filter_kernel is a 1D array
    if filter_kernel.ndim != 1:
        raise ValueError("filter_kernel must be a 1D array.")
    if sinogram.ndim != 2:
        raise ValueError("sinogram must be a 2D array.")

    # Pad sinogram and filter_kernel to the same length
    N = sinogram.shape[1]
    pad = N
    sinogram_padded = np.pad(sinogram, ((0, 0), (0, pad)), mode="constant")
    filter_kernel_padded = np.pad(filter_kernel, (0, pad), mode="constant")

    # Perform FFT, multiply, and IFT
    sinogram_fft = np.fft.fft(sinogram_padded, axis=1)
    filter_fft = np.fft.fft(filter_kernel_padded)

    filtered_fft = sinogram_fft * filter_fft
    filtered = np.fft.ifft(filtered_fft, axis=1).real

    filtered = filtered[:, :N]  # Crop to original size

    # Scale by delta_beta
    return filtered * delta_beta


def filter_projections(sinogram, filter, factor=1, smooth=False):
    """Filter projections.

    Parameters
    ----------
        sinogram (np.ndarray): Array of projections of shape
            (N_views, N_det)
        filter (callable): Filter instance that can generate a vector
            to be convolved with the sinogram.
        factor (int, optional): Factor by which the filtered projections
            are multiplied after the IFT.
        smooth (bool, optional): Whether the Filtered projections
            should be smoothed with a Hamming window before the IFT.
            Default is False.

    Returns
    -------
        filtered_sinogram (np.ndarray): Array of filtered
            projections of shape (n_projections, n_detector_pixels)
    """
    # generate filter response
    _, n_det = sinogram.shape
    n_fft = 2 * n_det - 1
    n = np.arange(-n_det // 2, n_det // 2)

    h_n = filter(n)

    # transform to FFT
    H = np.fft.rfft(h_n, n_fft)
    S = np.fft.rfft(sinogram, n_fft, axis=1)

    # optional smoothing
    if smooth:
        window_time = np.hamming(n_fft)
        W = np.fft.rfft(window_time)
        filtered_FT = S * H * W
    else:
        filtered_FT = S * H

    # IFT
    filtered_sinogram = factor * np.fft.irfft(filtered_FT, n_fft, axis=1)

    # trim to "same"
    start = (n_det - 1) // 2
    end = start + n_det
    filtered_sinogram = filtered_sinogram[:, start:end]

    return filtered_sinogram


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    N_det = 512
    cutoff = 1

    freq = np.fft.fftshift(np.fft.fftfreq(N_det))

    filter_types = ("ramp", "shepp-logan", "cosine", "hamming", "hann")

    fig, axs = plt.subplots(2, 3)
    for ax, filter_type in zip(axs.flat, filter_types):
        filter = build_filter(N_det, filter_type, cutoff=cutoff)
        ax.plot(freq, np.fft.fftshift(filter))
        ax.set_title(f"Filter {filter_type} with cutoff = {cutoff}")
        ax.set_xlabel("f[Hz]")
        ax.set_ylabel("h(f)")
        ax.grid()
    plt.tight_layout()
    plt.show()
