import os
import numpy as np


def generate_txt_images(redshift, B_1, K_1, Gamma,
                        LOS_coeff, HOAngle_deg,
                        n_along, n_across, lg_pixsize_min_mas, lg_pixsize_max_mas,
                        flare_params, ts_obs_days,
                        exec_dir, parallels_run_file, calculon):
    """
    :param redshift:
    :param B_1:
    :param K_1:
    :param Gamma:
    :param LOS_coeff:
        1 for superluminal speed max or 0.5 for mode.
    :param HOAngle_deg:
         The median full opening angle is 20 deg, tan(alpha) = tan(alpha_app) * sin(theta).
    :param n_along:
    :param n_across:
    :param lg_pixsize_min_mas:
    :param lg_pixsize_max_mas:
    :param flare_params:
        frac.amp.N, frac.amp.B, t_start[days], width[pc]
    :param ts_obs_days:
    :param exec_dir:
    :param parallels_run_file:
    """

    los_angle_deg = np.round(np.rad2deg(np.arcsin(LOS_coeff/Gamma)), 2)
    cone_half_angle_deg = np.round(np.rad2deg(np.arctan(np.tan(np.deg2rad(HOAngle_deg)) * np.sin(np.deg2rad(los_angle_deg)))), 2)

    # Construct params file
    with open(f"{parallels_run_file}", "w") as fo:
        for t_obs_days in ts_obs_days:
            fo.write("{} {} {} {} {} {} {} {} {} {} {:.1f} ".format(redshift, los_angle_deg, cone_half_angle_deg,
                                                                    B_1, K_1, Gamma, n_along, n_across,
                                                                    lg_pixsize_min_mas, lg_pixsize_max_mas,
                                                                    t_obs_days))
            # Write flare parameters
            for flare_param in flare_params[:-1]:
                fo.write("{} ".format(flare_param))
            fo.write("{}".format(flare_params[-1]))
            fo.write("\n")

    os.chdir(exec_dir)
    n_jobs = 4
    if calculon:
        n_jobs = 40
    os.system("parallel --files --results t_obs_{11}" + f" --joblog log --jobs {n_jobs} -a {parallels_run_file} -n 1 -m --colsep ' ' \"./bk_transfer\"")


if __name__ == "__main__":
    redshift = 1.0
    B_1 = 1.0
    K_1 = 500.
    Gamma = 10.
    LOS_coeff = 0.5
    HOAngle_deg = 15.
    n_along = 400
    n_across = 80
    lg_pixsize_min_mas = -2.5
    lg_pixsize_max_mas = -0.5
    flare_params = [30.0, 0.0, 0.0, 0.2]
    ts_obs_days = np.linspace(-400.0, 8*360, 20)
    calculon = True
    if not calculon:
        exec_dir = "/home/ilya/github/bk_transfer/Release"
        parallels_run_file = "/home/ilya/github/bk_transfer/parallels_run.txt"
    else:
        exec_dir = "/home/ilya/github/flares/bk_transfer/Release"
        parallels_run_file = "/home/ilya/github/flares/bk_transfer/parallels_run.txt"

    generate_txt_images(redshift, B_1, K_1, Gamma,
                        LOS_coeff, HOAngle_deg,
                        n_along, n_across, lg_pixsize_min_mas, lg_pixsize_max_mas,
                        flare_params, ts_obs_days,
                        exec_dir, parallels_run_file, calculon)