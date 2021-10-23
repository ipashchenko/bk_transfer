import os
import numpy as np

redshift = 1.0
los_angle_deg = 5
cone_half_angle_deg = 1.5
B_1 = 1.0
K_1 = 500.
Gamma = 10.
n_along = 400
n_across = 80
lg_pixsize_min_mas = -2.0
lg_pixsize_max_mas = -0.0
data_dir = "/home/ilya/data/rfc"
source_template = "J0102+5824"
# ts_obs_days = np.loadtxt(os.path.join(data_dir, "{}_times.txt".format(source_template)))
# 5 years once per 3 months
ts_obs_days = np.linspace(300, 10*360, 50)
n_first = None
# frac.amp.N, frac.amp.B, t_start[days], width[pc]
flare_params = [5.0, 0.0, 0.0, 2.]
                # 10.0, 100.0, 2.,
                # 10.0, 200.0, 2.]

# Construct params file
with open("parallels_run.txt".format(source_template), "w") as fo:
    for t_obs_days in ts_obs_days[:n_first]:
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
os.system("parallel --files --results t_obs_{11} --joblog log --jobs 6"
          " -a /home/ilya/github/bk_transfer/parallels_run.txt -n 1 -m --colsep ' ' \"./bk_transfer\"")
