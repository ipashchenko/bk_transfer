import os
import sys

import numpy as np


def generate_txt_images(redshift=None, B_1=None, m_b=None, K_1=None, Gamma=None,
                        LOS_coeff=None, HOAngle_deg=None,
                        n_along=None, n_across=None, lg_pixsize_min_mas=None, lg_pixsize_max_mas=None,
                        flare_params=None, ts_obs_days=None,
                        exec_dir=None, parallels_run_file=None, n_jobs=None):
    """
    :param LOS_coeff:
        1 for superluminal speed max or 0.5 for mode.
    :param HOAngle_deg:
         The median full opening angle is 20 deg, tan(alpha) = tan(alpha_app) * sin(theta).
    :param flare_params:
        Iterable of (frac.amp.N, frac.amp.B, t_start[days], width[pc])
    """

    los_angle_deg = np.round(np.rad2deg(np.arcsin(LOS_coeff/Gamma)), 2)
    cone_half_angle_deg = np.round(np.rad2deg(np.arctan(np.tan(np.deg2rad(HOAngle_deg)) * np.sin(np.deg2rad(los_angle_deg)))), 2)

    # Construct params file
    with open(f"{parallels_run_file}", "w") as fo:
        for t_obs_days in ts_obs_days:
            fo.write("{} {} {} {} {} {} {} {} {} {} {} {:.1f}".format(redshift, los_angle_deg, cone_half_angle_deg,
                                                                   B_1, m_b, K_1, Gamma, n_along, n_across,
                                                                   lg_pixsize_min_mas, lg_pixsize_max_mas,
                                                                   t_obs_days))
            # Write flare parameters
            for single_flare_params in flare_params:
                for flare_param in single_flare_params:
                    fo.write(" {}".format(flare_param))
            fo.write("\n")

    os.chdir(exec_dir)
    os.system("parallel --files --results t_obs_{12}" + f" --joblog log --jobs {n_jobs} -a {parallels_run_file} -n 1 -m --colsep ' ' \"./bk_transfer\"")
