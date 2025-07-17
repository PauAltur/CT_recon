import numpy as np
from numba import njit, prange
from src.geometry import rotate


def parallel_projection(image, axis):
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


def acquire_parallel_projections_loop(
    image, max_angle=180, n_projections=100, axis="x"
):
    """Acquire n parallel projections of an image from 0 to max_angle degrees
    in a python loop.

    This function computes a sinogram (parallel-beam projections) of `image`
    by rotating the image and summing along the specified axis. This is usually
    slower than the vectorised version, but it is easier to understand and uses
    less memory.

    Parameters:
    ----------
        image (np.ndarray): Input image to be projected.
        max_angle (int, optional): Maximum angle for projections in degrees.
            Default is 180 degrees.
        n_projections (int, optional): Number of projections to acquire.
            Default is 100.
        axis (str, optional): Axis along which to project the image. Options
            are "y" for vertical projection and "x" for horizontal projection.
            Default is "x".

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
    n_det_elements = image.shape[0] if axis == "y" else image.shape[1]
    projections = np.empty((n_projections, n_det_elements), dtype=image.dtype)

    for i, angle in enumerate(angles):
        rotated_image = rotate(image, angle)
        projections[i, :] = parallel_projection(rotated_image, axis=axis)

    return projections


def acquire_parallel_projections_vectorised(
    image, max_angle: float = 180, n_projections: int = 100, axis: str = "x"
) -> np.ndarray:
    """
    Compute a sinogram (parallel‑beam projections) of `image` without any
    explicit Python loop over the angles.

    Parameters
    ----------
    image (np.ndarray): A 2‑D greyscale image.
    max_angle (float, optional): Upper bound (degrees) of the fan of projections.
        Default is 180 degrees.
    n_projections (int, optional): Number of projections to acquire.
        Default is 100.
    axis (str): Detector orientation (“y” → vertical detector columns, a Radon
        transform along the y–axis; “x” → horizontal detector rows). Default is "x".

    Returns
    -------
    projections (np.ndarray): Sinogram in which each row is the line‑integral
        at a given angle. The shape is (n_projections, n_detectors).
    """
    if image.ndim != 2:
        raise ValueError("Only 2‑D greyscale input supported.")

    # Prepare the static pixel coordinate grid (centred at 0,0)
    h, w = image.shape
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")  # (H, W)
    coords = np.stack((x.ravel(), y.ravel()), axis=1).astype(
        float
    )  # (Npix, 2) where Npix = H*W

    centre = np.array(
        [
            (w - 1) / 2.0,  # note: (w‑1, h‑1) centres exactly
            (h - 1) / 2.0,
        ]
    )  # (2,)

    coords -= centre  # shift to 0,0

    # Build the whole stack of 2×2 rotation matrices at once
    angles = np.deg2rad(
        np.linspace(0.0, max_angle, n_projections, endpoint=False)
    )  # (P,)
    c, s = np.cos(angles), np.sin(angles)  # (P,)

    rotmats = np.stack(
        (np.stack((c, -s), axis=-1), np.stack((s, c), axis=-1)), axis=-2
    )  # (P, 2, 2)

    # Rotate *all* pixel coordinates for *all* angles in one call
    rotated_coords = coords @ rotmats.transpose(0, 2, 1)  # (P, Npix, 2)
    rotated_coords += centre  # undo shift

    # Map floating‑point coords → integer pixel indices and reject points
    # that fall outside the FOV
    x_rot = np.rint(rotated_coords[..., 0]).astype(np.int32)  # (P, Npix)
    y_rot = np.rint(rotated_coords[..., 1]).astype(np.int32)  # (P, Npix)

    valid = (0 <= x_rot) & (x_rot < w) & (0 <= y_rot) & (y_rot < h)  # (P, Npix)

    # Gather intensity values for every (angle, pixel) pair in one advanced‑indexing operation
    img_flat = image.ravel()  # (Npix,)
    flat_idx = y_rot * w + x_rot  # (P, Npix)
    pixel_vals = np.where(valid, img_flat.take(flat_idx, mode="clip"), 0)  # (P, Npix)

    # Reduce (sum) along the chosen detector direction
    # Reshape to (P, H, W) so the reduction axis is contiguous.
    cube = pixel_vals.reshape(n_projections, h, w)  # (P, H, W)
    projections = (
        cube.sum(axis=2) if axis == "x" else cube.sum(axis=1)
    )  # (P, n_detectors)

    return projections


def acquire_parallel_projections(
    image, max_angle=180, n_projections=100, axis="x", vectorised=True
):
    """Acquire n parallel projections of an image from 0 to max_angle degrees.

    This function computes a sinogram (parallel-beam projections) of `image`
    by rotating the image and summing along the specified axis. It can use
    either a vectorised or a loop-based approach.

    Parameters:
    ----------
        image (np.ndarray): Input image to be projected.
        max_angle (int): Maximum angle for projections in degrees.
        n_projections (int): Number of projections to acquire.
        axis (str): Axis along which to project the image.
            Options are "y" for vertical projection and "x"
            for horizontal projection.
        vectorised (bool): If True, use vectorised computation; otherwise, use loop-based.

    Returns:
    -------
        np.ndarray: Sinogram of the image, containing n_projections of
            size n_det_elements, where n_det_elements is the number of
            detector elements along the specified axis.
    """
    if vectorised:
        return acquire_parallel_projections_vectorised(
            image, max_angle, n_projections, axis
        )
    else:
        return acquire_parallel_projections_loop(image, max_angle, n_projections, axis)


@njit  # ← compiles and accelerates the function
def siddon_geom_numba(image, Sg, Dg):
    """Siddon ray‑tracing for 2‑D images **with geometric (y‑up)
    input**.

    Parameters
    ----------
        image (np.ndarray): 2‑D float32 array. Voxel size = 1 pixel.
        Sg (np.ndarray): Coordinates of the source point in geometric coords.
            Shape is (2,).
        Dg (np.ndarray): Coordinates of the detector point in geometric coords.
            Shape is (2,).

    Returns
    -------
        vals (np.ndarray): Line integral  ∑ f · Δl  along the segment Sg → Dg
    """
    nx, ny = image.shape

    # Convert geometric to image coords
    Sx = Sg[0] + nx / 2.0
    Sy = ny / 2.0 - Sg[1]
    Dx = Dg[0] + nx / 2.0
    Dy = ny / 2.0 - Dg[1]

    delta_x = Dx - Sx
    delta_y = Dy - Sy

    # Handle vertical and horizontal rays
    if delta_x != 0:
        tx = (np.arange(nx + 1) - Sx) / delta_x
    else:
        tx = np.array([-1e10, 1e10])  # Numba doesn't like inf

    if delta_y != 0:
        ty = (np.arange(ny + 1) - Sy) / delta_y
    else:
        ty = np.array([-1e10, 1e10])

    # Concatenate, clip, sort, and make unique
    t = np.concatenate((tx, ty))
    t = np.clip(t, 0.0, 1.0)
    t.sort()
    t_unique = []
    prev = -1.0
    for i in range(len(t)):
        if abs(t[i] - prev) > 1e-8:
            t_unique.append(t[i])
            prev = t[i]
    if t_unique[0] != 0.0:
        t_unique.insert(0, 0.0)
    if t_unique[-1] != 1.0:
        t_unique.append(1.0)
    t = np.array(t_unique)

    # Midpoints and segment lengths
    vals = 0.0
    for i in range(len(t) - 1):
        xm = Sx + delta_x * (t[i] + t[i + 1]) / 2.0
        ym = Sy + delta_y * (t[i] + t[i + 1]) / 2.0
        xi = int(xm)
        yi = int(ym)

        if 0 <= xi < nx and 0 <= yi < ny:
            seg_len = np.hypot(delta_x * (t[i + 1] - t[i]), delta_y * (t[i + 1] - t[i]))
            vals += image[xi, yi] * seg_len

    return vals


@njit(parallel=True)
def acquire_fanbeam_projections(image, S_arr, D_arr):
    """
    Compute the sinogram of a 2D image using Siddon's ray-tracing algorithm.

    Parameters
    ----------
        image (np.ndarray): 2D float32 array. Voxel size = 1 pixel.
        S_arr (np.ndarray): 2D array of shape (N_views, 2) containing
            source points in geometric coordinates.
        D_arr (np.ndarray): 3D array of shape (N_views, N_det, 2) containing
            detector bins in geometric coordinates.

    Returns
    -------
        sinogram (np.ndarray): 2D array of shape (N_views, N_det) containing
            the sinogram.
    """
    N_views, N_det, _ = D_arr.shape
    sinogram = np.zeros((N_views, N_det), dtype=np.float32)

    for v in prange(N_views):
        Sg = S_arr[v]
        for i in range(N_det):
            Dg = D_arr[v, i]
            sinogram[v, i] = siddon_geom_numba(image, Sg, Dg)

    return sinogram


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from geometry import shepp_logan

    # Generate the Shepp-Logan phantom
    N = 1024
    phantom = shepp_logan(N)

    # Project the phantom along the y-axis
    projections = acquire_parallel_projections(phantom, n_projections=1024)

    # Plot the original phantom and its projection
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Shepp-Logan Phantom")
    plt.imshow(phantom, cmap="gray")

    plt.subplot(1, 2, 2)
    plt.title("Projection along Y-axis")
    plt.imshow(projections.T, cmap="gray")

    plt.tight_layout()
    plt.show()
