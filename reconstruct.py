import numpy as np
from scipy import signal, interpolate


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
    

def filter_projections(sinogram, tau, smooth=False):
    """Filter projections with a Windowed Ramp Filter

    Parameters
    ----------
        sinogram (np.ndarray): Array of projections of shape 
            (n_projections, n_detector_pixels)
        tau (float): Period for the windowed ramp filter. 
            Inverse of the Nyquist frequency.
        smooth (bool, optional): Whether the Filtered projections
            should be smoothed with a Hamming window before the IFT.
            Default is False.

    Returns
    -------
        filtered_sinogram (np.ndarray): Array of filtered
            projections of shape (n_projections, n_detector_pixels)
    """
    # generate filter response
    n_proj, n_det = sinogram.shape
    n_fft = 2*n_det - 1
    t = np.arange(-n_det // 2, n_det // 2)

    filter = DiscreteWindowedRampFilter(tau)
    h_t = filter(t)

    # transform to FFT
    H = np.fft.rfft(h_t, n_fft)
    S = np.fft.rfft(sinogram, n_fft, axis=1)

    # optional smoothing
    if smooth:
        window_time = np.hamming(n_fft)
        W = np.fft.rfft(window_time)
        filtered_FT = S*H*W
    else:
        filtered_FT = S*H
    
    # IFT
    filtered_sinogram = np.fft.irfft(filtered_FT, n_fft, axis=1)

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

def backproject(Q_theta, N, t, tau):
    """Backproject filtered projections to reconstruct image.

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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from shepp_logan import shepp_logan
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
    