import numpy as np
from numba import njit, prange


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
            vals += image[yi, xi] * seg_len

    return vals


@njit(parallel=True)
def acquire_fanbeam_projections(image, S_arr, D_arr):
    """
    Compute the sinogram of a 2D image with a fan beam using
    Siddon's ray-tracing algorithm.

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


@njit(parallel=True)
def acquire_parallel_projections(image, S_arr, D_arr):
    """
    Compute the sinogram of a 2D image with a parallel beam using
    Siddon's ray-tracing algorithm.

    Parameters
    ----------
        image (np.ndarray): 2D float32 array. Voxel size = 1 pixel.
        S_arr (np.ndarray): 3D array of shape (N_views, N_det, 2) containing
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
        for i in range(N_det):
            Sg = S_arr[v, i]
            Dg = D_arr[v, i]
            sinogram[v, i] = siddon_geom_numba(image, Sg, Dg)

    return sinogram


def acquire_projections(image, S_arr, D_arr, mode="parallel"):
    """
    Dispatch function to compute sinogram for either parallel or fan-beam geometry.

    Parameters
    ----------
        image : np.ndarray
            2D float32 image array.
        S_arr : np.ndarray
            Source coordinates. Shape:
                - (N_views, N_det, 2) for 'parallel'
                - (N_views, 2) for 'fanbeam'
        D_arr : np.ndarray
            Detector coordinates. Shape: (N_views, N_det, 2)
        mode : str
            Either 'parallel' or 'fanbeam'.

    Returns
    -------
        sinogram : np.ndarray
            2D array of shape (N_views, N_det)
    """
    if mode == "parallel":
        return acquire_parallel_projections(image, S_arr, D_arr)
    elif mode == "fanbeam":
        return acquire_fanbeam_projections(image, S_arr, D_arr)
    else:
        raise ValueError(f"Unknown projection mode: {mode}")


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
