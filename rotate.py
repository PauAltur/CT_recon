import numpy as np


def rotate(image, angle):
    """Rotate an image by a given angle.

    Parameters:
    ----------
        image (np.ndarray): Input image to be rotated.
        angle (float): Angle in degrees to rotate the image.

    Returns:
    -------
        np.ndarray: Rotated image.
    """

    if not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a numpy array.")
    
    if not isinstance(angle, (int, float)):
        raise ValueError("Angle must be a numeric value.")

    # Convert angle to radians
    angle_rad = np.radians(angle)

    # Get the dimensions of the image
    h, w = image.shape[:2]

    # Calculate the center of the image
    center = (w / 2, h / 2)

    # Create the rotation matrix
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                 [np.sin(angle_rad), np.cos(angle_rad)]])

    # Create an output image with the same shape as the input
    rotated_image = np.zeros_like(image)

    # Create meshgrid of (x, y) coordinates
    y_indices, x_indices = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    coords = np.stack([x_indices.ravel(), y_indices.ravel()], axis=1)

    # Translate coordinates to center
    coords_centered = coords - np.array(center)

    # Inverse rotation matrix for backward mapping
    inv_rotation_matrix = np.linalg.inv(rotation_matrix)

    # Apply inverse rotation
    rotated_coords = coords_centered @ inv_rotation_matrix.T

    # Translate back to original position
    rotated_coords += np.array(center)

    # Round and convert to integer indices
    x_rot = np.round(rotated_coords[:, 0]).astype(int)
    y_rot = np.round(rotated_coords[:, 1]).astype(int)

    # Mask for valid coordinates
    mask = (
        (x_rot >= 0) & (x_rot < w) &
        (y_rot >= 0) & (y_rot < h)
    )

    # Assign pixel values using valid indices
    rotated_image[y_indices.ravel()[mask], x_indices.ravel()[mask]] = image[y_rot[mask], x_rot[mask]]

    return rotated_image


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from shepp_logan import shepp_logan

    # Example usage
    N = 1024
    phantom = shepp_logan(N)  # Placeholder for the phantom image

    for i, angle in enumerate(np.linspace(0,360, 16)):
        print("Rotating image by {:.1f} degrees...".format(angle))
        # Rotate the phantom image
        rotated_phantom = rotate(phantom, angle)

        # Display the original and rotated images
        plt.subplot(4, 4, int(i+1))
        plt.title("Rotation: {:.1f}º".format(angle))
        plt.imshow(rotated_phantom, cmap='gray')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
