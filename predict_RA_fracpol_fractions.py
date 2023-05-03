import os
import glob
import numpy as np
from astropy import constants as const
from astropy import units as u
from jet_image import JetImage
import sys
sys.path.insert(0, '/home/ilya/github/ve/vlbi_errors')
from utils import get_uv_correlations
import matplotlib
label_size = 20
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams['ytick.labelsize'] = label_size
matplotlib.rcParams['axes.titlesize'] = label_size
matplotlib.rcParams['axes.labelsize'] = label_size
matplotlib.rcParams['font.size'] = label_size
matplotlib.rcParams['legend.fontsize'] = label_size
matplotlib.rcParams["contour.negative_linestyle"] = 'dotted'
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import LogFormatter


def resize_images(basename, files_dir):
    cwd = os.getcwd()
    os.chdir(files_dir)

    def run_for_files(files):
        for fn in files:
            print(f"Resizing file {fn}")
            os.system(f"convert -resize 15% {fn} small_{fn}")

    files = glob.glob(f"{basename}_*.png")
    # Or use getctime for creation time
    files.sort(key=os.path.getmtime)
    run_for_files(files)
    os.chdir(cwd)


def create_movie(basename, files_dir):
    cwd = os.getcwd()
    os.chdir(files_dir)
    os.system(f"convert -delay 50 -loop 0 `ls -tr small_{basename}_*.png` {basename}.gif")
    os.chdir(cwd)


z = 0.3
n_along = 1200
n_across = 100
lg_pixsize_min_mas = -4.0
lg_pixsize_max_mas = -1.0
rot_angle_deg = 0.0
freq_ghz = 15.4
freq_names = {15.4: "u"}
jetpol_run_directory = "/home/ilya/fs/sshfs/calculon/github/flares/bk_transfer/Release"
save_dir = "/home/ilya/Documents/polarization_flares"


ts_obs_days = np.linspace(-400.0, 9*360, 88)

# baseline_length_ed = 5.675
baseline_length_ed_max = 10
lambda_obs = const.c/(freq_ghz*u.GHz)
baseline_length_max = (baseline_length_ed_max*const.R_earth/lambda_obs).to(u.Unit('')).value

ED = (const.R_earth/lambda_obs).to(u.Unit('')).value

x = np.linspace(-baseline_length_max, baseline_length_max, 100)
y = np.linspace(-baseline_length_max, baseline_length_max, 100)
uv = np.dstack(np.meshgrid(x, y)).reshape(-1, 2)
uv_x, uv_y = np.meshgrid(x, y)


# epoch = -232.6
for epoch in ts_obs_days:

    jm_i = JetImage(z=z, n_along=n_along, n_across=n_across,
                    lg_pixel_size_mas_min=lg_pixsize_min_mas, lg_pixel_size_mas_max=lg_pixsize_max_mas,
                    jet_side=True, rot=np.deg2rad(rot_angle_deg))
    jm_q = JetImage(z=z, n_along=n_along, n_across=n_across,
                    lg_pixel_size_mas_min=lg_pixsize_min_mas, lg_pixel_size_mas_max=lg_pixsize_max_mas,
                    jet_side=True, rot=np.deg2rad(rot_angle_deg))
    jm_u = JetImage(z=z, n_along=n_along, n_across=n_across,
                    lg_pixel_size_mas_min=lg_pixsize_min_mas, lg_pixel_size_mas_max=lg_pixsize_max_mas,
                    jet_side=True, rot=np.deg2rad(rot_angle_deg))

    image_file = "{}/jet_image_{}_{}_{:.1f}.txt".format(jetpol_run_directory, "i", freq_names[freq_ghz], epoch)
    image = np.loadtxt(image_file)
    print(f"Flux = {np.sum(image)}")
    flux = np.sum(image)

    jm_i.load_image_stokes("I", "{}/jet_image_{}_{}_{:.1f}.txt".format(jetpol_run_directory, "i", freq_names[freq_ghz], epoch), scale=1.0)
    jm_q.load_image_stokes("Q", "{}/jet_image_{}_{}_{:.1f}.txt".format(jetpol_run_directory, "q", freq_names[freq_ghz], epoch), scale=1.0)
    jm_u.load_image_stokes("U", "{}/jet_image_{}_{}_{:.1f}.txt".format(jetpol_run_directory, "u", freq_names[freq_ghz], epoch), scale=1.0)
    correlations_dict = get_uv_correlations(uv, [jm_i, jm_q, jm_u])
    Q = 0.5*(correlations_dict["RL"] + correlations_dict["LR"])
    U = (correlations_dict["RL"] - correlations_dict["LR"])/(2j)
    I = 0.5*(correlations_dict["RR"] + correlations_dict["LL"])
    # Vis. domain polarization
    P = Q + 1j*U
    frac_1 = P/I
    frac_2 = 2*correlations_dict["RL"]/(correlations_dict["RR"] + correlations_dict["LL"])

    # print(correlations_dict)
    #
    # print("U = ", U)
    # print("Q = ", Q)
    # print("P = ", P)
    # print("I = ", np.abs(I))
    # print("frac_1 = ", frac_1)
    # print("frac_2 = ", frac_2)
    # print("|frac| = ", np.abs(frac_1))


    fig, axes = plt.subplots(1, 1, figsize=(8, 8))
    im = axes.pcolor(uv_x/ED, uv_y/ED, np.abs(frac_1.reshape(100, 100)), cmap="gist_ncar", norm=matplotlib.colors.LogNorm())
    divider = make_axes_locatable(axes)
    formatter = LogFormatter(10, labelOnlyBase=False)

    cax = divider.append_axes("right", size="5%", pad=0.00)
    cb = fig.colorbar(im, cax=cax, format=formatter)
    cb.set_label(r"Frac. pol.")
    axes.set_xlabel(r"$u$, ED")
    axes.set_ylabel(r"$v$, ED")
    axes.set_aspect("equal")
    axes.set_yticks([-10, -5, 0, 5, 10])
    axes.set_xticks([10, 5, 0, -5, -10])
    # axes.set_ylim([-250, 250])
    # axes.set_xlim([-250, 250])
    axes.invert_xaxis()
    title = axes.set_title("{:05.1f} months, flux = {:.1f} Jy".format((1+z)*epoch/30, flux), fontsize='large')
    plt.savefig(os.path.join(save_dir, "frac_pol_amp_{:.1f}.png".format(epoch)), bbox_inches="tight", dpi=600)
    # plt.show()
    plt.close()


    fig, axes = plt.subplots(1, 1, figsize=(8, 8))
    im = axes.pcolor(uv_x/ED, uv_y/ED, np.angle(frac_1.reshape(100, 100)), vmin=-np.pi, vmax=np.pi, cmap="bwr")
    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size="5%", pad=0.00)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"Pol. Angle, rad")
    axes.set_xlabel(r"$u$, ED")
    axes.set_ylabel(r"$v$, ED")
    axes.set_aspect("equal")
    axes.set_yticks([-10, -5, 0, 5, 10])
    axes.set_xticks([10, 5, 0, -5, -10])
    # axes.set_ylim([-250, 250])
    # axes.set_xlim([-250, 250])
    axes.invert_xaxis()
    title = axes.set_title("{:05.1f} months, flux = {:.1f} Jy".format((1+z)*epoch/30, flux), fontsize='large')
    plt.savefig(os.path.join(save_dir, "frac_pol_angle_{:.1f}.png".format(epoch)), bbox_inches="tight", dpi=300)
    # plt.show()
    plt.close()


resize_images("frac_pol_amp", save_dir)
resize_images("frac_pol_angle", save_dir)
create_movie("frac_pol_amp", save_dir)
create_movie("frac_pol_angle", save_dir)
