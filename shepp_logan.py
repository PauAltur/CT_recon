import numpy as np


def shepp_logan(N):    
    """ Generate the Shepp-Logan phantom image.

    Parameters:
    ----------
        N (int): Size of the output image (N x N).

    Returns:
    -------
        np.ndarray: Shepp-Logan phantom image.
    """
    if N <= 0 or not isinstance(N, int):
        raise ValueError("N must be a positive integer.")
    
    # Create a grid of coordinates
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    
    # Initialize the phantom image
    phantom = np.zeros((N, N))
    
    # Define the ellipses parameters
    ELLIPSES = [
        (0, 0, 0.69, 0.92, 0, 256),
        (0, 0.0184, 0.6624, 0.874, 0, 75),
        (0.22, 0, 0.11, 0.31, 18, 25),
        (-0.22, 0, 0.16, 0.41, -18, 25),
        (0, -0.35, 0.21, 0.25, 0, 125), 
        (0, -0.1, 0.046, 0.046, 0, 150),
        (0, 0.1, 0.046, 0.046, 0, 150),
        (-0.08, 0.605, 0.046, 0.023, 0, 135),
        (0, 0.605, 0.023, 0.023, 0, 165),
        (0.06, 0.605, 0.023, 0.046, 0, 175)
    ]

    # Draw the ellipses
    for (x0, y0, a, b, theta, value) in ELLIPSES:
        # Calculate the ellipse equation
        theta = np.radians(theta)
        ellipse = ((X - x0) * np.cos(theta) + (Y - y0) * np.sin(theta))**2 / a**2 + \
                  ((X - x0) * np.sin(theta) - (Y - y0) * np.cos(theta))**2 / b**2
        
        # Set the value in the phantom image where the ellipse condition is met
        phantom[ellipse <= 1] = value

    # Normalize the phantom image to the range [0, 1]
    phantom = (phantom - phantom.min()) / (phantom.max() - phantom.min())
    
    return phantom


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Generate the Shepp-Logan phantom
    N = 256
    phantom_image = shepp_logan(N)

    # Display the phantom image
    plt.imshow(phantom_image, cmap='gray', extent=(0, 1, 0, 1))
    plt.title('Shepp-Logan Phantom')
    plt.axis('off')
    plt.show()
