import numpy as np
from scipy import interpolate
from src.filter import (
    parallel_filter,
    build_freq_filter,
    filter_multiply,
    filter_convolve,
)


def interpolate_projections(t, sinogram, f_interp):
    """Interpolates sinogram by a factor to make backprojection more accurate.

    Parameters
    ----------
        t (np.ndarray): X array of projections with shape (N_det,).
        sinogram (np.ndarray): Array of projections with shape (N_views, N_det).
        factor (int): Factor by which the interpolation augments the number of
            sample points.

    Returns
    -------
        t_interp (np.ndarray): Interpolated X array with shape (N_det * factor)
        sinogram_interp (np.ndarray): Array of interpolated projections with
            shape (N_views, N_det * factor).
    """
    # estimate interpolation function
    f = interpolate.interp1d(t, sinogram, axis=1)

    # perform interpolation
    t_interp = np.linspace(t[0], t[-1], num=len(t) * f_interp)
    sinogram_interp = f(t_interp)

    return t_interp, sinogram_interp


def parallel_reconstruction(
    sinogram,
    theta,
    N_pixels,
    period,
    f_interp=1,
    interpolation="linear",
    normalize=True,
):
    """
    Reconstruct image from sinogram acquired with parallel projections.

    Parameters
    ----------
        sinogram (np.ndarray): (N_views, N_det) array of projections.
        theta (np.ndarray): (N_views,) array of projection angles.
        N_pixels (int): Number of pixels per side of the reconstructed image.
        period (float): Detector element spacing.
        f_interp (int, optional): Interpolation factor for detector bins. The
            default is 1 which equals no interpolation.
        interpolation (str, optional): Interpolation method: "linear" or "nearest"
            (default "linear").
        normalize (bool, optional): Whether the reconstruction values should be
            normalized to the 0-1 range. Default is True.

    Returns
    -------
        recon (np.ndarray): Reconstructed image (N_pixels, N_pixels).
    """
    N_views, N_det = sinogram.shape
    n = np.arange(-N_det // 2, N_det // 2)
    t = n * period

    # Filter projections
    filter = parallel_filter(n, period)
    sinogram_filt = filter_convolve(sinogram, filter, period)

    # Preinterpolate projections if factor > 1
    if f_interp > 1:
        t_interp, sinogram_interp = interpolate_projections(t, sinogram_filt, f_interp)
    else:
        t_interp, sinogram_interp = t, sinogram_filt

    # Create image grid (x,y) centered at 0 TODO: move to function
    size_physical = N_det * period
    y, x = np.mgrid[
        -size_physical / 2 : size_physical / 2 : N_pixels * 1j,
        -size_physical / 2 : size_physical / 2 : N_pixels * 1j,
    ]
    y = -y

    cos_theta = np.cos(theta)[:, None, None]  # (N_views,1,1)
    sin_theta = np.sin(theta)[:, None, None]  # (N_views,1,1)
    t_corresp = (
        -x[None, :, :] * sin_theta + y[None, :, :] * cos_theta
    )  # (N_views, N_pixels, N_pixels)

    view_idx = np.arange(N_views)[:, None, None]

    if interpolation == "nearest":
        # TODO: Move to function
        # Nearest neighbor interpolation
        idx = np.searchsorted(t_interp, t_corresp, side="left")
        idx = np.clip(idx, 0, len(t_interp) - 1)
        backproj = sinogram_interp[view_idx, idx]

    elif interpolation == "linear":
        # TODO: Move to function
        # Find indices bounding t_corresp
        idx_right = np.searchsorted(t_interp, t_corresp, side="left")
        idx_right = np.clip(idx_right, 1, len(t_interp) - 1)
        idx_left = idx_right - 1

        t_left = t_interp[idx_left]
        t_right = t_interp[idx_right]

        # Weights for linear interpolation
        weight_right = (t_corresp - t_left) / (t_right - t_left + 1e-12)
        weight_left = 1.0 - weight_right

        vals_left = sinogram_interp[view_idx, idx_left]
        vals_right = sinogram_interp[view_idx, idx_right]

        backproj = weight_left * vals_left + weight_right * vals_right

    else:
        raise ValueError("Interpolation must be 'linear' or 'nearest'")

    recon = (np.pi / N_views) * backproj.sum(axis=0)

    # Normalize the recon
    if normalize is True:
        recon = (recon - recon.min()) / (recon.max() - recon.min())

    return recon


def compute_l_gamma_coords(N_pixels, D_so, theta):
    """Compute the source frame L and gamma coordinates for each pixel in each view
    of an equiangular fan beam acquisition.

    Parameters
    ----------
        N_pixels (int): Number of pixels per side in image. Assumes square image.
        D_so (int): Source object distance in pixels.
        theta (np.ndarray): Angle of central ray with respect to origin for each
            view. Shape (N_views,).

    Returns
    -------
        L (np.ndarray): L coordinate for each pixel in each projection view. Shape
            (N_pixels, N_pixels, N_views).
        gamma (np.ndarray): Gamma coordinate for each pixel in each projection view.
            Shape (N_pixels, N_pixels, N_views).
    """
    # Generate cartesian coords and transform to polar
    y, x = np.mgrid[
        -N_pixels // 2 : N_pixels // 2, -N_pixels // 2 : N_pixels // 2
    ]  # (N_pixels, N_pixels)
    y = -y

    r = np.sqrt(x**2 + y**2)  # (N_pixels, N_pixels)
    phi = np.arctan2(y, x)  # (N_pixels, N_pixels)

    # Compute source frame coordinates
    theta_min90 = (
        theta - np.pi / 2
    )  # adjust for different frame of reference wrt view angle
    L = np.sqrt(
        (
            D_so
            + r[..., None]
            * np.sin(theta_min90[np.newaxis, np.newaxis, :] - phi[..., None])
        )
        ** 2
        + (
            r[..., None]
            * np.cos(theta_min90[np.newaxis, np.newaxis, :] - phi[..., None])
        )
        ** 2
    )  # (N_pixels, N_pixels, N_views)
    gamma = np.arctan2(
        -(
            r[..., None]
            * np.cos(theta_min90[np.newaxis, np.newaxis, :] - phi[..., None])
        ),
        (
            D_so
            + r[..., None]
            * np.sin(theta_min90[np.newaxis, np.newaxis, :] - phi[..., None])
        ),
    )  # (N_pixels, N_pixels, N_views)

    return L, gamma


def compute_equiangular_ray_indices(gamma, fan_angle, delta_beta, f_interp, N_det):
    """Compute ray indices from the angular coordinates of equiangular projections.

    This function takes the angular coordinates of each pixel in the image
    for each acquired view (gamma), and computes a ray index that will
    later be used to select the relevant rays to backproject each pixel.

    Parameters
    ----------
        gamma (np.ndarray): Gamma coordinate for each pixel in each projection
            view. Shape (N_pixels, N_pixels, N_views).
        fan_angle (float): Angle of the X-ray beam.
        delta_beta (float): Angle spacing between detector bins.
        f_interp (int): Interpolation factor of the filtered projections.
        N_det (int): Number of detector elements/bins.

    Returns
    -------
        k0 (np.ndarray): Floor ray indices for each pixel in each view.
            Shape (N_views, N_pixels^2).
        k1 (np.ndarray): Ceil ray indices for each pixel in each view.
            Shape (N_views, N_pixels^2).
        w (np.ndarray): Linear interpolation weights for each pixels and
            view.
    """
    N_views = gamma.shape[-1]
    beta0 = -fan_angle / 2
    if f_interp is not None:
        beta_spacing_ang = delta_beta / f_interp
    else:
        beta_spacing_ang = delta_beta

    # Fractional index
    k = (gamma - beta0) / beta_spacing_ang

    # Floor and ceil indices
    k0 = np.floor(k).astype(int)
    k1 = k0 + 1

    # Weights
    w = k - k0  # Linear interpolation weight

    # Clip to valid range
    k0 = np.clip(k0, 0, N_det - 1)
    k1 = np.clip(k1, 0, N_det - 1)

    # Reshape for gather
    k0 = k0.transpose(2, 0, 1).reshape(N_views, -1)
    k1 = k1.transpose(2, 0, 1).reshape(N_views, -1)
    w = w.transpose(2, 0, 1).reshape(N_views, -1)

    return k0, k1, w


def equiangular_reconstruction(
    sinogram,
    beta,
    theta,
    N_pixels,
    D_so,
    filter_type,
    cutoff,
    period,
    fan_angle,
    f_interp=1,
    mode="nearest",
    normalize=True,
):
    """Compute backprojection of fan beam acquisition with equiangular detector bins.

    Parameters
    ----------
        sinogram (np.ndarray): Sinogram of acquisitions. Shape (N_views, N_det).
        beta (np.ndarray): Array of bin angles wrt central ray. Shape (N_det,).
        theta (np.ndarray): Array of view angles with shape (N_views,).
        N_pixels (int): Number of pixels per image side. Assumes square image.
        D_so (int): Source object distance in pixels.
        filter (str): Filter type to convolve with the sinogram.
        cutoff (float): Cutoff for windowed filters, between 0 and 1.
        period (float): Discretization period of detector.
        fan_angle (float): Angle of the X-ray beam.
        delta_beta (float): Angle spacing of detector bins.
        f_interp (int, optional): Factor by which the detector bins will be interpolated.
            Default is 1 (i.e. no interpolation).
        mode (str, optional): Interpolation mode. Can be either "nearest" or "linear".
            Default is "nearest".
        normalize (bool, optional): Whether the reconstruction values should be normalized
            to the 0-1 range. Default is True.

    Returns
    -------
        recon (np.ndarray): Reconstruction array with shape (N_pixels, N_pixels).
    """
    # Dimension setup
    Nx = Ny = N_pixels
    N_views = theta.shape[0]
    N_det = sinogram.shape[-1]

    # Distance correction
    sinogram_corr = sinogram * D_so * np.cos(beta[None, :])

    # Filter the projections
    filter = build_freq_filter(N_det, filter_type, cutoff)
    sinogram_filt = filter_multiply(sinogram_corr, filter, period)

    # Compute source frame coordinates
    L, gamma = compute_l_gamma_coords(N_pixels, D_so, theta)

    # Compute ordinal coords for detector bins
    k0, k1, w = compute_equiangular_ray_indices(
        gamma, fan_angle, period, f_interp, N_det
    )

    # Perform interpolation either through NN or linear
    if mode == "nearest":
        sinogram_interp = np.take_along_axis(
            sinogram_filt, k0, axis=1
        )  # (N_views, N_pixels^2)
    elif mode == "linear":
        # Gather values from Q at both floor and ceil indices
        sinogram_filt_0 = np.take_along_axis(
            sinogram_filt, k0, axis=1
        )  # (N_views, N_pixels^2)
        sinogram_filt_1 = np.take_along_axis(
            sinogram_filt, k1, axis=1
        )  # (N_views, N_pixels^2)

        # Interpolate linearly
        sinogram_interp = (1 - w) * sinogram_filt_0 + w * sinogram_filt_1

    # Compute backprojection
    inv_L2 = 1.0 / (L**2)  # (N_pixels, N_pixels, N_views)
    inv_L2 = inv_L2.transpose(2, 0, 1).reshape(N_views, -1)  # (N_views, N_pixels^2)
    cos_gamma = (
        np.cos(gamma).transpose(2, 0, 1).reshape(N_views, -1)
    )  # (N_views, N_pixels^2)
    sinogram_weighted = sinogram_interp * inv_L2 * cos_gamma  # (N_views, N_pixels^2)

    recon_flat = (2 * np.pi / N_views) * np.sum(
        sinogram_weighted, axis=0
    )  # (N_pixels^2,)
    recon = recon_flat.reshape(Nx, Ny)  # (N_pixels, N_pixels)

    # Normalize the recon
    if normalize is True:
        recon = (recon - recon.min()) / (recon.max() - recon.min())

    return recon


def compute_u_sprime_coords(N_pixels, D_so, theta):
    """Compute the source frame U and sprime coordinates for each pixel in each view
    of an equidistant fan beam acquisition.

    Parameters
    ----------
        N_pixels (int): Number of pixels per side in image. Assumes square image.
        D_so (int): Source object distance in pixels.
        theta (np.ndarray): Angle of central ray with respect to origin for each
            view. Shape (N_views,).

    Returns
    -------
        U (np.ndarray): U ratio for each pixel in each projection view. Shape
            (N_pixels, N_pixels, N_views).
        sprime (np.ndarray): S prime coordinate for each pixel in each projection view.
            Shape (N_pixels, N_pixels, N_views).
    """
    # Generate cartesian coords and transform to polar
    y, x = np.mgrid[
        -N_pixels // 2 : N_pixels // 2, -N_pixels // 2 : N_pixels // 2
    ]  # (N_pixels, N_pixels)
    y = -y

    r = np.sqrt(x**2 + y**2)  # (N_pixels, N_pixels)
    phi = np.arctan2(y, x)  # (N_pixels, N_pixels)

    # Compute U and sprime
    theta_min90 = (
        theta - np.pi / 2
    )  # adjust for different frame of reference wrt view angle
    U = (
        D_so + r[..., None] * np.sin(theta_min90[None, None, :] - phi[..., None])
    ) / D_so
    sprime = (
        D_so
        * (r[..., None] * np.cos(theta_min90[None, None, :] - phi[..., None]))
        / (D_so + r[..., None] * np.sin(theta_min90[None, None, :] - phi[..., None]))
    )

    return U, sprime


def equidistant_reconstruction(
    sinogram,
    det_coords,
    theta,
    N_pixels,
    D_so,
    filter_type,
    cutoff,
    period,
    mode="linear",
    normalize=True,
):
    """Compute backprojection of fan beam acquisition with equidistant detector bins.

    Args:
        sinogram (np.ndarray): Sinogram of acquisitions. Shape (N_views, N_det).
        det_coords (np.ndarray): Array of bin distances wrt central bin. Shape (N_det,).
        theta (np.ndarray): Array of view angles with shape (N_views,).
        N_pixels (int): Number of pixels per image side. Assumes square image.
        D_so (int): Source object distance in pixels.
                filter (str): Filter type to convolve with the sinogram.
        cutoff (float): Cutoff for windowed filters, between 0 and 1.
        period (float): Discretization period of detector.
        mode (str, optional): Interpolation mode. Can be either "nearest" or "linear".
            Default is "nearest".
        normalize (bool, optional): Whether the reconstruction values should be normalized
            to the 0-1 range. Default is True.
    """
    # Dimension setup
    Nx = Ny = N_pixels
    N_views = theta.shape[0]
    N_det = sinogram.shape[-1]

    # Distance correction
    sinogram_corr = sinogram * (D_so / np.sqrt(D_so ** +(det_coords[None, :] ** 2)))

    # High-pass filtering to counter blurring
    filter = build_freq_filter(N_det, filter_type, cutoff, period)
    sinogram_filt = filter_multiply(sinogram_corr, filter, period)

    # Compute source frame coordinates
    U, sprime = compute_u_sprime_coords(N_pixels, D_so, theta)

    pass


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from geometry import shepp_logan
    from project import acquire_projections
    from filter import filter_projections

    N_pixels = 128
    n_proj = 101
    tau = [100, 10, 1, 0.1, 0.01]

    fig, axs = plt.subplots(3, 3)

    phantom = shepp_logan(N_pixels)
    sinogram = acquire_projections(phantom, n_projections=n_proj)

    axs[0, 0].imshow(phantom, cmap="gray")
    axs[0, 0].set_title("Shepp-Logan phantom")

    axs[0, 1].imshow(sinogram, cmap="gray")
    axs[0, 1].set_title("Sinogram of Shepp-Logan phantom")

    for ax, tau_i in zip(axs.flat[2:], tau):
        filtered_sinogram = filter_projections(sinogram, tau=tau_i, smooth=True)
        ax.imshow(filtered_sinogram, cmap="gray")
        ax.set_title(f"Filtered sinogram of Shepp-Logan phantom with $\\tau={tau_i}$")

    plt.show()
