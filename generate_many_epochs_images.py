import os
import numpy as np

z = 0.5
los_anlge_deg = 5
cone_half_angle = 1.5
B_1 = 1.0
K_1 = 1000.
Gamma = 10.
n_along = 1024
n_across = 256
lg_pixsize_min = -2.0
lg_pixsize_max = -1.0
data_dir = "/home/ilya/data/rfc"
source_template = "J0102+5824"
t_obs_month = np.loadtxt(os.path.join(data_dir, "{}_times.txt".format(source_template)))
flare_params = [5.0, 10.0, 2.0, 10.0, 30.0, 2.5]
# Construct params file
params = list()
for t_obs in t_obs_month:
    params.append([z, los_anlge_deg, cone_half_angle, B_1, K_1, Gamma, n_along, n_across, lg_pixsize_min,
                   lg_pixsize_max, t_obs, flare_params])
params = np.atleast_2d(params)
np.savetxt("/home/ilya/github/bk_transfer/params_flaring_jet.txt", params)


exec_dir = "/home/ilya/github/bk_transfer/Release"
os.chdir(exec_dir)

os.system("parallel --files --results t_obs_{11} --joblog log --jobs 1"
          " -a /home/ilya/github/bk_transfer/params_flaring_jet.txt -n 1 -m --colsep ' ' \"./bk_transfer\"")
