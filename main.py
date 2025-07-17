import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
from src.project import acquire_fanbeam_projections
from src.geometry import shepp_logan, setup_geometry
from src.reconstruct import  interpolate_projections, distance_correction, equiangular_backproject
from src.filter import build_filter, filter_projections_freq_kernel


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    # -----------------------------------
    # Set up geometry parameters
    # -----------------------------------
    S, D, theta, beta, fan_angle, delta_beta = setup_geometry(
        cfg["geometry"]["D_so"], 
        cfg["geometry"]["D_sd"],
        cfg["geometry"]["R_obj"], 
        cfg["geometry"]["N_det"], 
        cfg["geometry"]["N_views"]
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
    # Acquire fan beam projections of the phantom
    # --------------------------------------------------------
    P = acquire_fanbeam_projections(f, S, D)
    plt.figure(2)
    plt.imshow(P, cmap="gray")
    plt.title("Sinogram of Shepp-Logan phantom")
    plt.axis("off")
    plt.show()

    # --------------------------------------------------------------
    # Reconstruct the Shepp-Logan phantom from fan beam projections
    # --------------------------------------------------------------

    # Step 1: Factor the sinogram by the source-object distance
    P = distance_correction(P, cfg["geometry"]["D_so"], beta)
    # plt.figure(3)
    # plt.imshow(P, cmap="gray")
    # plt.title("Sinogram of Shepp-Logan phantom (distance corrected)")
    # plt.axis("off")
    # plt.show()

    # Step 2: Filter the projections using a discrete fan beam filter
    g = build_filter(cfg["geometry"]["N_det"], filter_type=cfg["filter"]["type"], cutoff=cfg["filter"]["cutoff"])
    Q = filter_projections_freq_kernel(P, g, delta_beta=delta_beta)
    plt.figure(4)
    plt.imshow(Q, cmap="gray")
    plt.title("Filtered sinogram of Shepp-Logan phantom")
    plt.axis("off")
    plt.show()

    # Step 3: Interpolate projections
    if cfg["geometry"]["f_interp"] is not None:
        _, Q = interpolate_projections(beta, Q, f_interp=cfg["geometry"]["f_interp"])        
    # plt.figure(5)
    # plt.imshow(Q_interp, cmap="gray")
    # plt.title("Interpolated filtered sinogram of Shepp-Logan phantom")
    # plt.axis("off")
    # plt.show()

    # Step 4: Backproject the filtered projections
    recon = equiangular_backproject(
        Q,
        cfg["geometry"]["N_pixels"],
        cfg["geometry"]["D_so"],
        theta,
        fan_angle,
        delta_beta,
        f_interp=cfg["geometry"]["f_interp"],
        mode=cfg["backprojection"]["mode"]
    )
    plt.figure(5)
    plt.imshow(recon, cmap="gray", origin="lower")
    plt.title("Reconstructed image from fan beam projections")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()