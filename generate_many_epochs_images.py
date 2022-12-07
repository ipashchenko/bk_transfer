import os
import numpy as np

redshift = 1.0

# B_1 = 0.85
B_1 = 1.0
K_1 = 500.
# Gamma = 0.75/theta
# Gamma = 8.6
# Peak in simulations
Gamma = 10
# 1 for superluminal speed max or 0.5 for mode
los_angle_deg = np.round(np.rad2deg(np.arcsin(0.5/Gamma)), 2)
# The median full opening angle is 20 deg, tan(alpha) = tan(alpha_app) * sin(theta)
cone_half_angle_deg = np.round(np.rad2deg(np.arctan(np.tan(np.deg2rad(15.)) * np.sin(np.deg2rad(los_angle_deg)))), 2)

# n_along = 400
n_along = 2000
# n_across = 80
n_across = 1400
# lg_pixsize_min_mas = -2.5
lg_pixsize_min_mas = -1.0
# lg_pixsize_max_mas = -0.5
lg_pixsize_max_mas = -1.0
ts_obs_days = np.linspace(0.0, 8*360, 20)
ts_obs_days = [0.0]
# w/o LTTD
# ts_obs_days = np.linspace(0, 40*360, 60)
flare_params = [2.0, 0.0, 0.0, 0.2]
# WISE stuff
# ts_obs_days = np.linspace(0.0, 20*360, 241)
# frac.amp.N, frac.amp.B, t_start[days], width[pc]
# flare_params = [1.0, 0.0, 300.0, 0.2]
# WISE stuff
# flare_params = [0.33, 0.0, 300.0, 0.2]
                # 1.0, 0.0, 300.0, 0.2]
                # 3.0, 0.0, 500.0, 0.4]

# Construct params file
with open("parallels_run.txt", "w") as fo:
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

exec_dir = "/home/ilya/github/bk_transfer/Release"
os.chdir(exec_dir)
os.system("parallel --files --results t_obs_{11} --joblog log --jobs 4"
          " -a /home/ilya/github/bk_transfer/parallels_run.txt -n 1 -m --colsep ' ' \"./bk_transfer\"")
