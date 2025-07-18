import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
from src.project import acquire_projections
from src.geometry import shepp_logan, setup_geometry
from src.reconstruct import equiangular_reconstruction, parallel_reconstruction


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    # -----------------------------------
    # Set up geometry parameters
    # -----------------------------------
    if cfg["projection"] == "parallel":
        S, D, theta, det_pitch = setup_geometry(
            cfg["geometry"]["D_so"],
            cfg["geometry"]["D_sd"],
            cfg["geometry"]["R_obj"],
            cfg["projection"]["N_det"],
            cfg["projection"]["N_views"],
            cfg["projection"],
            cfg["projection"]["view_range"],
        )
    elif cfg["projection"] in ["equiangular", "equidistant"]:
        S, D, theta, beta, fan_angle, delta_beta = setup_geometry(
            cfg["geometry"]["D_so"],
            cfg["geometry"]["D_sd"],
            cfg["geometry"]["R_obj"],
            cfg["projection"]["N_det"],
            cfg["projection"]["N_views"],
            cfg["projection"],
            cfg["projection"]["view_range"],
        )

    # --------------------------------------------------------
    # Generate Shepp-Logan phantom
    # --------------------------------------------------------
    f = shepp_logan(cfg["geometry"]["N_pixels"])
    # plt.figure(1)
    # plt.imshow(f, cmap="gray")
    # plt.title("Shepp-Logan phantom")
    # plt.axis("off")
    # plt.show()

    # --------------------------------------------------------
    # Acquire projections of the phantom
    # --------------------------------------------------------
    P = acquire_projections(f, S, D, mode=cfg["projection"])
    plt.figure()
    plt.imshow(P, cmap="gray")
    plt.title("Sinogram of Shepp-Logan phantom")
    plt.axis("off")
    plt.show()

    # --------------------------------------------------------------
    # Reconstruct the Shepp-Logan phantom from projections
    # --------------------------------------------------------------

    # Step 4: Backproject the filtered projections
    if cfg["projection"] == "linear":
        recon = parallel_reconstruction(
            P,
            theta,
            cfg["geometry"]["N_pixels"],
            cfg["filter"]["type"],
            cfg["filter"]["cutoff"],
            det_pitch,
            cfg["interpolation"]["factor"],
            cfg["projection"]["view_range"],
        )
    elif cfg["projection"] == "equiangular":
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
            delta_beta,
            f_interp=cfg["interpolation"]["factor"],
            mode=cfg["interpolation"]["type"],
        )
    plt.figure()
    plt.imshow(recon, cmap="gray", origin="lower")
    plt.title("Reconstructed image from fan beam projections")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
