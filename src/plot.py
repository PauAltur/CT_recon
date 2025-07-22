import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon


def animate_S_D(S, D, image):
    """
    Animate the evolution over N_views of point S and detector array D,
    shading polygon between S[i] and edge points of D[i],
    and showing an image at the center.

    Parameters:
    -----------
    S : np.ndarray, shape (N_views, 2)
        Point positions per view.

    D : np.ndarray, shape (N_views, N_det, 2)
        Detector element positions per view.

    image : np.ndarray
        Image array to show in the background.
    """

    N_views = S.shape[0]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")

    # Show image centered at origin
    H, W = image.shape[:2]
    extent = [-W / 2, W / 2, -H / 2, H / 2]

    ax.imshow(image, extent=extent, cmap="gray")

    # Plot handles: point S and detector elements D as scatter plots
    scat_S = ax.scatter([], [], c="red", s=80, label="Source S")
    scat_D = ax.scatter([], [], c="blue", s=30, label="Detector D")

    # Polygon patch to shade the area between S and detector edges
    poly_patch = Polygon([[0, 0]], closed=True, color="orange", alpha=0.3)
    ax.add_patch(poly_patch)

    ax.legend()

    # Compute combined bounds for S and D over all frames
    all_points = np.vstack([S.reshape(-1, 2), D.reshape(-1, 2)])

    x_min = min(all_points[:, 0].min(), extent[0])
    x_max = max(all_points[:, 0].max(), extent[1])
    y_min = min(all_points[:, 1].min(), extent[2])
    y_max = max(all_points[:, 1].max(), extent[3])

    # Add a small margin (say 5%)
    x_margin = 0.05 * (x_max - x_min)
    y_margin = 0.05 * (y_max - y_min)

    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_title("Animation of S and D with shaded polygon")

    def update(frame):
        # Update point S
        scat_S.set_offsets(S[frame].reshape(1, 2))

        # Update detector points
        scat_D.set_offsets(D[frame])

        # Polygon vertices:
        # Connect S to the first detector point, then all detector points, then back to S
        polygon_points = np.vstack([S[frame], D[frame], S[frame]])
        poly_patch.set_xy(polygon_points)
        return scat_S, scat_D, poly_patch

    anim = FuncAnimation(fig, update, frames=N_views, interval=100, blit=True)

    plt.show()

    return anim
