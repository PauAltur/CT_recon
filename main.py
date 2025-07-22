import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
from src.project import acquire_projections
from src.geometry import shepp_logan, delta_phantom, setup_geometry
from src.reconstruct import equiangular_reconstruction, parallel_reconstruction


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    plt.ion()  # Turn on interactive mode

    # ---------------------------
    # Set up geometry parameters
    # ---------------------------
    geom_tuple = setup_geometry(
        cfg["geometry"]["D_so"],
        cfg["geometry"]["D_sd"],
        cfg["geometry"]["R_obj"],
        cfg["projection"]["N_det"],
        cfg["projection"]["N_views"],
        cfg["projection"]["view_range"],
        cfg["projection"]["type"],
    )
    if cfg["projection"]["type"] == "parallel":
        S, D, theta, det_pitch = geom_tuple
    elif cfg["projection"]["type"] == "equiangular":
        S, D, theta, beta, fan_angle, delta_beta = geom_tuple
    elif cfg["projection"]["type"] == "equidistant":
        raise NotImplementedError("Equidistant geometry setup not implemented yet")

    # --------------------------------------------------------
    # Generate phantom
    # --------------------------------------------------------
    if cfg["geometry"]["phantom"]["name"] == "shepp-logan":
        f = shepp_logan(cfg["geometry"]["N_pixels"])

    elif cfg["geometry"]["phantom"]["name"] == "delta":
        f = delta_phantom(
            cfg["geometry"]["N_pixels"], **cfg["geometry"]["phantom"]["args"]
        )
    plt.figure()
    plt.imshow(f, cmap="gray", origin="lower")
    plt.title("Shepp-Logan phantom")
    plt.axis("off")
    plt.draw()
    plt.pause(0.001)
    # time.sleep(0.5)

    # --------------------------------------------------------
    # Acquire projections of the phantom
    # --------------------------------------------------------
    P = acquire_projections(f, S, D, mode=cfg["projection"]["type"])
    plt.figure()
    plt.imshow(P, cmap="gray")
    plt.title("Sinogram of phantom")
    plt.axis("off")
    plt.draw()
    plt.pause(0.001)
    # plt.show(block=False)
    # time.sleep(0.5)

    # --------------------------------------------------------------
    # Reconstruct the Shepp-Logan phantom from projections
    # --------------------------------------------------------------
    if cfg["projection"]["type"] == "parallel":
        recon = parallel_reconstruction(
            P,
            theta,
            cfg["geometry"]["N_pixels"],
            cfg["filter"]["type"],
            cfg["filter"]["cutoff"],
            det_pitch,
            cfg["interpolation"]["factor"],
            cfg["interpolation"]["type"],
        )
    elif cfg["projection"]["type"] == "equiangular":
        recon = equiangular_reconstruction(
            P,
            beta,
            theta,
            cfg["geometry"]["N_pixels"],
            cfg["geometry"]["D_so"],
            cfg["filter"]["type"],
            cfg["filter"]["cutoff"],
            delta_beta,
            fan_angle,
            f_interp=cfg["interpolation"]["factor"],
            mode=cfg["interpolation"]["type"],
        )
    elif cfg["projection"]["type"] == "equidistant":
        raise NotImplementedError(
            "Reconstruction of equidistant fan beam projections is not implemented"
        )
    else:
        raise ValueError(
            "Projection type not recognized. Should be parallel, equiangular or equidistant."
        )

    plt.figure()
    plt.imshow(recon, cmap="gray", origin="lower")
    plt.title(f"Reconstructed image {cfg['projection']['type']} projections")
    plt.axis("off")
    plt.draw()
    plt.pause(0.001)

    input("Press Enter to close all plots...")
    plt.ioff()
    plt.close("all")


if __name__ == "__main__":
    main()
