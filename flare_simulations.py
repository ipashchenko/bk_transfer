import os
import glob
import sys

import numpy as np
from generate_many_epochs_images import generate_txt_images
from process_raw_image import process_raw_images
from generate_and_model_many_epochs_uvdata import make_and_model_visibilities


def clear_txt_images(files_dir):
    files = glob.glob(os.path.join(files_dir, "jet_image_[i,tau]*_[X, S]_*.txt"))
    for fn in files:
        try:
            os.unlink(fn)
        except FileNotFoundError:
            pass


def clear_tobs(files_dir):
    files = glob.glob(os.path.join(files_dir, "t_obs_*"))
    for fn in files:
        try:
            os.unlink(fn)
        except FileNotFoundError:
            pass


def clear_fits(files_dir):
    files = glob.glob(os.path.join(files_dir, "template_[X, S]_*.uvf"))
    for fn in files:
        try:
            os.unlink(fn)
        except FileNotFoundError:
            pass
    files = glob.glob(os.path.join(files_dir, "model_cc_i_[X, S]_*.fits"))
    for fn in files:
        try:
            os.unlink(fn)
        except FileNotFoundError:
            pass


def create_movie_raw(basename, files_dir):
    cwd = os.getcwd()
    os.chdir(files_dir)
    os.system(f"convert -delay 10 -loop 0 `ls -tr {basename}_raw_nupixel_*.png` {basename}_raw.gif")
    os.chdir(cwd)


def create_movie_clean(basename, files_dir, only_band):
    cwd = os.getcwd()
    os.chdir(files_dir)
    if only_band is None or only_band != "S":
        os.system(f"convert -delay 10 -loop 0 `ls -tr {basename}_observed_i_X_*.png` {basename}_CLEAN_X.gif")
    if only_band is None or only_band != "X":
        os.system(f"convert -delay 10 -loop 0 `ls -tr {basename}_observed_i_S_*.png` {basename}_CLEAN_S.gif")
    os.chdir(cwd)


def clear_pics(basename, files_dir):
    files = glob.glob(os.path.join(files_dir, f"{basename}_raw_nupixel_*.png"))
    for fn in files:
        try:
            os.unlink(fn)
        except FileNotFoundError:
            pass

    files = glob.glob(os.path.join(files_dir, f"{basename}_observed_i_[X, S]_*.png"))
    for fn in files:
        try:
            os.unlink(fn)
        except FileNotFoundError:
            pass


redo = [False, True, True]
calculon = False
basename = "test"
only_band = None
redshift = 0.8
B_1 = 2.0
K_1 = 5000.
# TODO: Changing this => edit main.cpp! ################################################################################
b = 1.25
n = 2.0
s = 2.
gamma_min = 10.
########################################################################################################################
# TODO: Changing this => edit utils.h! #################################################################################
gamma_max = 1E+04
########################################################################################################################

Gamma = 10.
LOS_coeff = 0.5
HOAngle_deg = 15.

los_angle_deg = np.round(np.rad2deg(np.arcsin(LOS_coeff/Gamma)), 2)
cone_half_angle_deg = np.round(np.rad2deg(np.arctan(np.tan(np.deg2rad(HOAngle_deg)) * np.sin(np.deg2rad(los_angle_deg)))), 2)
print(f"LOS(deg) = {los_angle_deg}")
print(f"Cone HA (deg) = {cone_half_angle_deg}")

# sys.exit(0)

n_along = 1000
n_across = 200
lg_pixsize_min_mas = -3.0
lg_pixsize_max_mas = -1.0
match_resolution = False
flare_params = [10.0, 0.0, 0.0, 0.3]
# TODO: Changing this => edit NField.cpp! ##############################################################################
flare_shape = 10.0
########################################################################################################################

ts_obs_days = np.linspace(-400.0, 8*360, 4)
# ts_obs_days = np.array([0.0])
noise_scale_factor = 1.0
mapsizes_dict = {2.3: (2048, 0.05,), 8.6: (2048, 0.05,)}
plot_raw = True
plot_clean = True
only_plot_raw = False
extract_extended = True
use_scipy_for_extract_extended = False
beam_fractions = (1.0,)
two_stage = False
n_components = 4

if not calculon:
    exec_dir = "/home/ilya/github/bk_transfer/Release"
    parallels_run_file = "/home/ilya/github/bk_transfer/parallels_run.txt"
    save_dir = "/home/ilya/github/bk_transfer/pics/flares"
    path_to_script = "/home/ilya/github/bk_transfer/scripts/script_clean_rms"
else:
    exec_dir = "/home/ilya/github/flares/bk_transfer/Release"
    parallels_run_file = "/home/ilya/github/flares/bk_transfer/parallels_run.txt"
    save_dir = "/home/ilya/github/flares/bk_transfer/pics/flares"
    path_to_script = "/home/ilya/github/flares/bk_transfer/scripts/script_clean_rms"


if redo[0]:
    clear_txt_images(exec_dir)
    clear_tobs(exec_dir)
clear_fits(save_dir)
clear_pics(basename, save_dir)

if redo[0]:
    generate_txt_images(redshift, B_1, K_1, Gamma,
                        LOS_coeff, HOAngle_deg,
                        n_along, n_across, lg_pixsize_min_mas, lg_pixsize_max_mas,
                        flare_params, ts_obs_days,
                        exec_dir, parallels_run_file, calculon)

if redo[1] and only_band is None:
    process_raw_images(basename, exec_dir, save_dir, redshift, plot_raw, match_resolution,
                       n_along, n_across, lg_pixsize_min_mas, lg_pixsize_max_mas,
                       ts_obs_days, flare_params, flare_shape,
                       Gamma, LOS_coeff, b, B_1, n, gamma_min, gamma_max, s)
    create_movie_raw(basename, save_dir)

if redo[2]:
    make_and_model_visibilities(basename, only_band, redshift, lg_pixsize_min_mas, lg_pixsize_max_mas, n_along, n_across, match_resolution,
                                ts_obs_days,
                                noise_scale_factor, mapsizes_dict,
                                plot_clean, only_plot_raw,
                                extract_extended, use_scipy_for_extract_extended, beam_fractions, two_stage,
                                n_components,
                                save_dir, exec_dir, path_to_script)
    create_movie_clean(basename, save_dir, only_band)

clear_tobs(exec_dir)
clear_fits(save_dir)
clear_pics(basename, save_dir)
