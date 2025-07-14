import numpy as np
from scipy import interpolate


class DiscreteWindowedRampFilter():
    """Discrete windowed ramp filter for filtered backprojection.

    This filter is used in computed tomography to reconstruct images
    from their sinograms. It is defined for both integer and array inputs.
    Its name comes from the shape of its frequency response.

    Attributes:
    ----------
        tau (float): Parameter for the filter, must be a positive numeric value.
    """
    def __init__(self, tau):
        """Initialize the discrete windowed ramp filter with a given tau.
        
        Parameters:
        ----------
            tau (float): Parameter for the filter.
        """
        if not isinstance(tau, (int, float)) or tau <= 0:
            raise ValueError("tau must be a positive numeric value.")
        self.tau = tau

    def __call__(self, n):
        """Evaluate the discrete windowed ramp filter at n.
        
        Parameters:
        ----------
            n (int, np.ndarray): Number of points in the filter.
        
        Returns:
        -------
            (int, np.ndarray): Discrete windowed ramp filter evaluated at n.
        """
        if isinstance(n, int):
            return self._compute_integer_filter(n)
        elif isinstance(n, np.ndarray):
            return self._compute_array_filter(n)
        
    def _compute_integer_filter(self, n):
        """Compute the discrete windowed ramp filter for an integer n.
        
        Parameters:
        ----------
            n (int): Number of points in the filter.
        
        Returns:
        -------
            (int): Discrete windowed ramp filter evaluated at n.
        """
        if n == 0:
            return 1 / (4 * self.tau**2)
        elif n % 2 == 0:
            return 0
        elif n % 2 == 1:
            return -1 / (n * np.pi * self.tau)**2
        
    def _compute_array_filter(self, n):
        """Compute the discrete windowed ramp filter for an array n.
        
        Parameters:
        ----------
            n (np.ndarray): Number of points in the filter.
        
        Returns:
        -------
            (np.ndarray): Discrete windowed ramp filter evaluated at n.
        """
        filter = np.zeros_like(n, dtype=float)
        filter[n % 2 == 0] = 0
        filter[n % 2 == 1] = -1 / (n[n % 2 == 1] * np.pi * self.tau)**2
        filter[n == 0] = 1 / (4 * self.tau**2)
        return filter


class DiscreteFanBeamFilter():
    """Discrete fan beam filter for filtered backprojection.

    This filter is used in computed tomography to reconstruct images
    from their sinograms acquired in a fan beam geometry.
    It is defined for both integer and array inputs.

    Attributes:
    ----------
        alpha (float): Parameter for the filter, must be a positive numeric value.
    """

    def __init__(self, alpha):
        """Initialize the discrete fan beam filter with a given alpha.
        
        Parameters:
        ----------
            alpha (float): Angle in radians between contiguous rays.
        """
        if not isinstance(alpha, (int, float)) or alpha <= 0:
            raise ValueError("alpha must be a positive numeric value.")
        self.alpha = alpha

    def __call__(self, n):
        """Evaluate the discrete fan beam filter at n.
        
        Parameters:
        ----------
            n (int, np.ndarray): Number of points in the filter.
        
        Returns:
        -------
            (int, np.ndarray): Discrete fan beam filter evaluated at n.
        """
        if isinstance(n, int):
            return self._compute_integer_filter(n)
        elif isinstance(n, np.ndarray):
            return self._compute_array_filter(n)
        
    def _compute_integer_filter(self, n):
        """Compute the discrete fan beam filter for an integer n.
        
        Parameters:
        ----------
            n (int): Number of points in the filter.
        
        Returns:
        -------
            (int): Discrete fan beam filter evaluated at n.
        """
        if n == 0:
            return 1 / (8 * self.alpha**2)
        elif n % 2 == 0:
            return 0
        elif n % 2 == 1:
            return (self.alpha / (np.pi * self.alpha * np.sin(n*self.alpha)))**2
        
    def _compute_array_filter(self, n):
        """Compute the discrete fan beam filter for an array n.
        
        Parameters:
        ----------
            n (np.ndarray): Number of points in the filter.
        
        Returns:
        -------
            (np.ndarray): Discrete fan beam filter evaluated at n.
        """
        filter = np.zeros_like(n, dtype=float)
        filter[n % 2 == 0] = 0
        filter[n % 2 == 1] = (self.alpha / (np.pi * self.alpha * np.sin(n[n % 2 == 1]*self.alpha)))**2
        filter[n == 0] = 1 / (8 * self.alpha**2)
        return filter
    

def build_filter(N, filter_type="ramp", cutoff=1.0):
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
    N = sinogram.shape[1]
    pad = N
    sinogram_padded = np.pad(sinogram, ((0,0), (0,pad)), mode="constant")
    filter_kernel_padded = np.pad(filter_kernel, (0, pad), mode="constant")

    sinogram_fft = np.fft.fft(sinogram_padded, axis=1)
    filter_fft = np.fft.fft(filter_kernel_padded)

    filtered_fft = sinogram_fft * filter_fft
    filtered = np.fft.ifft(filtered_fft, axis=1).real

    filtered = filtered[:, :N]  # Crop to original size
    return filtered * delta_beta


def filter_projections(sinogram, filter, factor=1, smooth=False):
    """Filter projections.

    Parameters
    ----------
        sinogram (np.ndarray): Array of projections of shape 
            (n_projections, n_detector_pixels)
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
    n_fft = 2*n_det - 1
    n = np.arange(-n_det // 2, n_det // 2)

    h_n = filter(n)

    # transform to FFT
    H = np.fft.rfft(h_n, n_fft)
    S = np.fft.rfft(sinogram, n_fft, axis=1)

    # optional smoothing
    if smooth:
        window_time = np.hamming(n_fft)
        W = np.fft.rfft(window_time)
        filtered_FT = S*H*W
    else:
        filtered_FT = S*H
    
    # IFT
    filtered_sinogram = factor * np.fft.irfft(filtered_FT, n_fft, axis=1)

    # trim to "same"
    start = (n_det - 1) // 2
    end = start + n_det
    filtered_sinogram = filtered_sinogram[:, start:end]

    return filtered_sinogram


def interpolate_projections(t, sinogram, factor):
    """Interpolates sinogram by a factor to make backprojection more accurate.
    
    Parameters
    ----------
        t (np.ndarray): X array of projections with shape (n_detection_elements,).
        sinogram (np.ndarray): Array of projections with shape (n_projections,
            n_detection_elements).
        factor (int): Factor by which the interpolation augments the number of
            sample points.

    Returns
    -------
        (np.ndarray): Array of interpolated projections with shape (n_projections,
            n_detection_elements * factor).
    """
    # estimate interpolation function
    f = interpolate.interp1d(t, sinogram, axis=1)

    # perform interpolation
    t_new = np.linspace(t[0], t[-1], num=len(t)*factor)

    return t_new, f(t_new)

 
def parallel_backproject(Q_theta, N, t, tau):
    """Backproject parallel filtered projections to reconstruct image.

    Parameters
    ----------
        Q_theta (np.ndarray): Array of projections with shape
            (n_projections, n_detector_elements).
        N (int): Number of pixels in the image. Assumption of square
            image.
        t (np.ndarray): Projection axis with shape (n_detection_elements,).
        tau (float): Period of detector elements (cm).

    Returns
    -------
        (np.ndarray): Reconstructed image with shape (n_pixels, n_pixels). 
    """
    # magnitude setup
    K, Nt = Q_theta.shape
    theta = np.deg2rad(np.linspace(0, 180, K))

    # vectorized computation of t for each (x,y) and theta
    x, y = np.mgrid[-N//2:N//2, -N//2:N//2] * tau
    cos_theta = np.cos(theta)[:, None, None]
    sin_theta = np.sin(theta)[:, None, None]
    t_corresp = x[None, ...] * cos_theta + y[None, ...] * sin_theta

    # find nearest detector bin index for each t in one shot
    idx = np.searchsorted(t, t_corresp, side="left")
    idx = np.clip(idx, 0, Nt - 1)

    # index the elements corresponding to each point
    angle_idx = np.arange(K)[:, None, None]
    backproj = Q_theta[angle_idx, idx]

    return (np.pi / K) * backproj.sum(axis=0)


def angle_factor(R_beta, D, beta):
    """Adjust each ray in the projection by its corresponding angle factor.

    Parameters
    ----------
        R_beta (np.ndarray): Fan beam projection with shape (N_det,).
        beta (np.ndarray): Array of angles in radians corresponding to
            each ray in the projection (N_det,).

    Returns
    -------
        (np.ndarray): Adjusted fan beam projection with shape (N_det,).
    """
    # convert beta to radians
    return R_beta * D * np.cos(beta)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from geometry import shepp_logan
    from project import acquire_projections

    N = 128
    n_proj = 101
    tau=[100, 10, 1, 0.1, 0.01]
    
    fig, axs = plt.subplots(3,3)

    phantom = shepp_logan(N)
    sinogram = acquire_projections(phantom, n_projections=n_proj)

    axs[0,0].imshow(phantom, cmap="gray")
    axs[0,0].set_title("Shepp-Logan phantom")

    axs[0,1].imshow(sinogram, cmap="gray")
    axs[0,1].set_title("Sinogram of Shepp-Logan phantom")
    
    for ax, tau_i in zip(axs.flat[2:], tau):
        filtered_sinogram = filter_projections(sinogram, tau=tau_i, smooth=True)  
        ax.imshow(filtered_sinogram, cmap="gray")
        ax.set_title(f"Filtered sinogram of Shepp-Logan phantom with $\\tau={tau_i}$")

    plt.show()
    