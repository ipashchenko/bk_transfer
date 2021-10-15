import os
import numpy as np

redshift = 0.5
los_angle_deg = 5
cone_half_angle_deg = 1.5
B_1 = 1.0
K_1 = 1000.
Gamma = 10.
n_along = 1024
n_across = 256
lg_pixsize_min_mas = -2.0
lg_pixsize_max_mas = -1.0
data_dir = "/home/ilya/data/rfc"
source_template = "J0102+5824"
ts_obs_month = np.loadtxt(os.path.join(data_dir, "{}_times.txt".format(source_template)))
n_first = None
# frac.amp, t_start[month], width[pc]
flare_params = [5.0, 10.0, 2.0,
                10.0, 30.0, 2.5]

# Construct params file
with open("parallels_run.txt".format(source_template), "w") as fo:
    for t_obs_month in ts_obs_month[:n_first]:
        fo.write("{} {} {} {} {} {} {} {} {} {} {} ".format(redshift, los_angle_deg, cone_half_angle_deg,
                                                            B_1, K_1, Gamma, n_along, n_across,
                                                            lg_pixsize_min_mas, lg_pixsize_max_mas,
                                                            t_obs_month))
        # Write flare parameters
        for flare_param in flare_params[:-1]:
            fo.write("{}, ".format(flare_param))
        fo.write("{}".format(flare_params[-1]))
        fo.write("\n")


exec_dir = "/home/ilya/github/bk_transfer/Release"
os.chdir(exec_dir)

os.system("parallel --files --results t_obs_{11} --joblog log --jobs 4"
          " -a /home/ilya/github/bk_transfer/parallels_run.txt -n 1 -m --colsep ' ' \"./bk_transfer\"")
