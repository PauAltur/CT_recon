import numpy as np


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
    

def filtered_backprojection(sinogram):
    """Reconstruct an image from its sinogram using filtered backprojection.
    
    Parameters:
    ----------
        sinogram (np.ndarray): Input sinogram to be reconstructed.
    
    Returns:
    -------
        np.ndarray: Reconstructed image from the sinogram.
    """

    pass


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example usage of the DiscreteWindowedRampFilter
    tau = 10
    filter = DiscreteWindowedRampFilter(tau)
    
    n = np.arange(-10, 11)  # Example range for n
    filtered_values = filter(n)
    
    plt.plot(n, filtered_values)
    plt.title("Discrete Windowed Ramp Filter")
    plt.xlabel("n")
    plt.ylabel("Filter Value")
    plt.grid()
    plt.show()
