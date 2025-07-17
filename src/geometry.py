import numpy as np


def shepp_logan(N):
    """Generate the Shepp-Logan phantom image.

    Parameters:
    ----------
        N (int): Size of the output image (N x N).

    Returns:
    -------
        (np.ndarray): Shepp-Logan phantom image.
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
        (0.06, 0.605, 0.023, 0.046, 0, 175),
    ]

    # Draw the ellipses
    for x0, y0, a, b, theta, value in ELLIPSES:
        # Calculate the ellipse equation
        theta = np.radians(theta)
        ellipse = ((X - x0) * np.cos(theta) + (Y - y0) * np.sin(theta)) ** 2 / a**2 + (
            (X - x0) * np.sin(theta) - (Y - y0) * np.cos(theta)
        ) ** 2 / b**2

        # Set the value in the phantom image where the ellipse condition is met
        phantom[ellipse <= 1] = value

    # Normalize the phantom image to the range [0, 1]
    phantom = (phantom - phantom.min()) / (phantom.max() - phantom.min())

    return phantom


def setup_geometry(D_so, D_sd, R_obj, N_det, N_views):
    """
    Set up the geometry for the CT reconstruction.

    Parameters
    ----------
        D_so (float): Source-object distance.
        D_sd (float): Source-detector distance.
        R_obj (float): Radius of the object.
        N_det (int): Number of detector bins.

    Returns
    -------
        S (np.ndarray): Source positions in lab frame.
        D_theta (np.ndarray): Detector line positions in lab frame.
        D (np.ndarray): Positions of all detector bins at every view in lab frame.
        theta (np.ndarray): View angles.
        beta (np.ndarray): Detector angles wrt central ray.
    """
    fan_angle = 2 * np.arcsin(R_obj / D_so)  # full fan angle covering object
    delta_beta = fan_angle / N_det  # equi‑angular bin spacing
    beta = (
        np.arange(N_det) - (N_det - 1) / 2
    ) * delta_beta  # detector angles wrt central ray
    theta = np.linspace(0, 2 * np.pi, N_views, endpoint=False)  # view angles
    C = (0, 0)  # centre of rotation in lab frame

    # Unit vectors defining source frame at every view
    e_r = np.stack(
        [np.cos(theta), np.sin(theta)], axis=1
    )  # (N_views, 2) radial unit vectors from C to S
    e_t = np.stack(
        [-np.sin(theta), np.cos(theta)], axis=1
    )  # (N_views, 2) tangential unit vectors (right‑hand)

    # Source position in lab frame for every view
    S = D_so * e_r  # (N_views, 2) source positions in lab frame

    # Position of detector line at every view in lab frame
    D_theta = (
        C - (D_sd - D_so) * e_r
    )  # (N_views, 2) detector line positions in lab frame

    # Radial unit vectors in detector frame
    tan_beta = np.tan(beta).reshape(
        1, -1, 1
    )  # (1, N_det, 1) radial unit vectors in detector frame

    # Position of all detector bins at every view in lab frame
    D = D_theta[:, None, :] + D_sd * tan_beta * e_t[:, None, :]  # (N_views, N_det, 2)

    return S, D, theta, beta, fan_angle, delta_beta


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
    rotation_matrix = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )

    # Create an output image with the same shape as the input
    rotated_image = np.zeros_like(image)

    # Create meshgrid of (x, y) coordinates
    y_indices, x_indices = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
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
    mask = (x_rot >= 0) & (x_rot < w) & (y_rot >= 0) & (y_rot < h)

    # Assign pixel values using valid indices
    rotated_image[y_indices.ravel()[mask], x_indices.ravel()[mask]] = image[
        y_rot[mask], x_rot[mask]
    ]

    return rotated_image


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Generate the Shepp-Logan phantom
    N = 256
    phantom_image = shepp_logan(N)

    # Display the phantom image
    plt.imshow(phantom_image, cmap="gray", extent=(0, 1, 0, 1))
    plt.title("Shepp-Logan Phantom")
    plt.axis("off")
    plt.show()
