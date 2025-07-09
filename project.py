import numpy as np
from rotate import rotate


def parallel_projection(image, axis="y"):
    """Project an image along a specified axis.
    
    Parameters:
    ----------
        image (np.ndarray): Input image to be projected.
        axis (str): Axis along which to project the image. 
                    Options are "y" for vertical projection
                    and "x" for horizontal projection.
    
    Returns:
    -------
        np.ndarray: Projected image along the specified axis.
    """
    return np.sum(image, axis=0) if axis == "y" else np.sum(image, axis=1)


def acquire_projections(image, max_angle=180, n_projections=180, axis="y"):
    """Acquire n projections of an image from 0 to max_angle degrees.
    
    Parameters:
    ----------
        image (np.ndarray): Input image to be projected.
        max_angle (int): Maximum angle for projections in degrees.
        n_projections (int): Number of projections to acquire.
        axis (str): Axis along which to project the image. 
            Options are "y" for vertical projection and "x"
            for horizontal projection.
    
    Returns:
    -------
        np.ndarray: Sinogram of the image, containing n_projections of
            size n_det_elements, where n_det_elements is the number of
            detector elements along the specified axis.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a numpy array.")
    if not isinstance(max_angle, (int, float)) or max_angle <= 0:
        raise ValueError("max_angle must be a positive numeric value.")
    if not isinstance(n_projections, int) or n_projections <= 0:
        raise ValueError("n_projections must be a positive integer.")

    angles = np.linspace(0, max_angle, n_projections)
    n_det_elements= image.shape[0] if axis == "y" else image.shape[1]
    projections = np.empty((n_projections, n_det_elements), dtype=image.dtype)

    for i, angle in enumerate(angles):
        rotated_image = rotate(image, angle)
        projections[i, :] = parallel_projection(rotated_image, axis=axis)

    return projections


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from shepp_logan import shepp_logan

    # Generate the Shepp-Logan phantom
    N = 1024
    phantom = shepp_logan(N)

    # Project the phantom along the y-axis
    projections = acquire_projections(phantom, n_projections=1024)

    # Plot the original phantom and its projection
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Shepp-Logan Phantom")
    plt.imshow(phantom, cmap='gray')
    
    plt.subplot(1, 2, 2)
    plt.title("Projection along Y-axis")
    plt.imshow(projections.T, cmap='gray')
    
    plt.tight_layout()
    plt.show()