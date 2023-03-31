import os
import glob
import sys
import astropy.io.fits as pf
from astropy.time import Time
import numpy as np
from generate_many_epochs_images import generate_txt_images
from process_raw_image import process_raw_images
from generate_and_model_many_epochs_uvdata import make_and_model_visibilities


# Find sources
table1_file = "/home/ilya/data/rfc/J_MNRAS_485_1822_table1.fits"
table2_file = "/home/ilya/data/rfc/J_MNRAS_485_1822_table2.fits"

tab1 = pf.getdata(table1_file)
tab2 = pf.getdata(table2_file)
sources = tab1["J2000"]
source_epochs = dict()
all_cadences = list()
for source in sources:
    source_idx = tab2["J2000"] == source
    source_tab2 = tab2[source_idx]
    epochs = source_tab2["Obs.date"]
    times = Time(epochs)
    times = Time(sorted(times, key=lambda x: x.jd))
    # Times relative to the first time (thus, first time will be always zero)
    dtimes = times - times[0]
    vdtimes = dtimes.value
    source_epochs[str(source)] = vdtimes


def clear_txt_images(files_dir):
    files = glob.glob(os.path.join(files_dir, "jet_image_[i,tau]*_[X, S]_*.txt"))
    for fn in files:
        try:
            print(f"Removing {fn}...")
            os.unlink(fn)
        except FileNotFoundError:
            print(f"No file {fn} to remove...")
            pass


def clear_tobs(files_dir):
    files = glob.glob(os.path.join(files_dir, "t_obs_*"))
    for fn in files:
        try:
            print(f"Removing {fn}...")
            os.unlink(fn)
        except FileNotFoundError:
            print(f"No file {fn} to remove...")
            pass


def clear_fits(files_dir):
    files = glob.glob(os.path.join(files_dir, "template_[X, S]_*.uvf"))
    for fn in files:
        try:
            print(f"Removing {fn}...")
            os.unlink(fn)
        except FileNotFoundError:
            print(f"No file {fn} to remove...")
            pass
    files = glob.glob(os.path.join(files_dir, "model_cc_i_[X, S]_*.fits"))
    for fn in files:
        try:
            print(f"Removing {fn}...")
            os.unlink(fn)
        except FileNotFoundError:
            print(f"No file {fn} to remove...")
            pass


def create_movie_raw(basename, files_dir):
    cwd = os.getcwd()
    os.chdir(files_dir)
    os.system(f"convert -delay 10 -loop 0 `ls -tr {basename}_raw_nupixel_*.png` {basename}_raw.gif")
    os.chdir(cwd)


def create_movie_clean(basename, files_dir):
    cwd = os.getcwd()
    os.chdir(files_dir)
    os.system(f"convert -delay 50 -loop 0 `ls -tr {basename}_observed_pol_u_*.png` {basename}_CLEAN_POL.gif")
    os.chdir(cwd)


def clear_pics(basename, files_dir):
    files = glob.glob(os.path.join(files_dir, f"{basename}_raw_nupixel_*.png"))
    for fn in files:
        try:
            os.unlink(fn)
        except FileNotFoundError:
            pass

    files = glob.glob(os.path.join(files_dir, f"{basename}_observed_pol_[X, S, u]_*.png"))
    for fn in files:
        try:
            os.unlink(fn)
        except FileNotFoundError:
            pass


redo = [True, True, True]
calculon = True
basename = "pol"
only_band = None
redshift = 0.3
K_1 = 5000.
# TODO: Changing this => edit main.cpp! ################################################################################
n = 2.0
s = 2.
gamma_min = 10.
########################################################################################################################
# TODO: Changing this => edit utils.h! #################################################################################
gamma_max = 1E+04
########################################################################################################################


# sys.exit(0)

n_along = 300
n_across = 100
lg_pixsize_min_mas = -2.0
lg_pixsize_max_mas = -0.5
match_resolution = False
# TODO: Changing this => edit NField.cpp! ##############################################################################
flare_shape = 10.0
########################################################################################################################
# local_rfc_dir = "/home/ilya/data/rfc"
# source = "J2258-2758"
# source = "J0006-0623"
# times_file = os.path.join(local_rfc_dir, f"{source}_times.txt")
# ts_obs_days = np.loadtxt(times_file)/(1+redshift)

noise_scale_factor = 1.0
# mapsizes_dict = {2.3: (2048, 0.05,), 8.6: (2048, 0.05,)}
mapsizes_dict = {15.4: (1024, 0.1,)}
plot_raw = True
plot_clean = True
only_plot_raw = False
extract_extended = False
use_scipy_for_extract_extended = False
beam_fractions = (1.0,)
two_stage = False
n_components = 4

if not calculon:
    exec_dir = "/home/ilya/github/bk_transfer/Release"
    parallels_run_file = "/home/ilya/github/bk_transfer/parallels_run.txt"
    save_dir = "/home/ilya/github/bk_transfer/pics/flares"
    # path_to_script = "/home/ilya/github/bk_transfer/scripts/script_clean_rms"
    path_to_script = "/home/ilya/github/bk_transfer/scripts/final_clean_nw"
else:
    exec_dir = "/home/ilya/github/flares/bk_transfer/Release"
    parallels_run_file = "/home/ilya/github/flares/bk_transfer/parallels_run.txt"
    save_dir = "/home/ilya/github/flares/bk_transfer/pics/flares"
    # path_to_script = "/home/ilya/github/flares/bk_transfer/scripts/script_clean_rms"
    path_to_script = "/home/ilya/github/bk_transfer/scripts/final_clean_nw"


# n_sources = 1
# flare_params = [(0.0, 0.0, 0.0, 0.1)]
# ts_obs_days = np.array([0.0])

n_sources = 1
idxs = np.random.choice(np.arange(len(sources), dtype=int), size=n_sources, replace=False)
for i in range(n_sources):
    # B_1 = 2.0
    # b = 1.25
    B_1 = 0.3
    b = 1.0
    Gamma = 10.
    LOS_coeff = 0.3
    HOAngle_deg = 15.

    los_angle_deg = np.round(np.rad2deg(np.arcsin(LOS_coeff/Gamma)), 2)
    cone_half_angle_deg = np.round(np.rad2deg(np.arctan(np.tan(np.deg2rad(HOAngle_deg)) * np.sin(np.deg2rad(los_angle_deg)))), 2)
    print(f"LOS(deg) = {los_angle_deg}")
    print(f"Cone HA (deg) = {cone_half_angle_deg}")

    # Fixed times
    ts_obs_days = np.linspace(-400.0, 9*360, 88)
    # # From real sources times
    # # This will be multiplied on (1+z) to bring to the observer z = 0.
    # ts_obs_days = source_epochs[sources[idxs[i]]]/(1+redshift)
    # # Shift to sample flares right
    # ts_obs_days -= 400

    flare_params = list()

    # # First flare
    # t_start_years = np.random.uniform(-1, 0., size=1)[0]
    # t_start_days = t_start_years*12*30
    # amp_N = np.random.uniform(5, 7, size=1)[0]
    # amp_B = 0.0
    # width_pc = np.random.uniform(0.1, 0.2, size=1)[0]
    # flare_params.append((amp_N, amp_B, t_start_days, width_pc))
    #
    # # Maximal number of flares
    # for i_fl in range(10):
    #     # Waiting time 3 yrs
    #     dt_yrs = 0.0
    #     while dt_yrs < 2.0:
    #         dt_yrs = np.random.exponential(3.0)
    #     t_start_years += dt_yrs
    #     t_start_days = t_start_years*12*30
    #     amp_N = np.random.uniform(4, 7, size=1)[0]
    #     amp_B = 0.0
    #     width_pc = np.random.uniform(0.1, 0.2, size=1)[0]
    #     flare_params.append((amp_N, amp_B, t_start_days, width_pc))
    #
    #     if t_start_days > ts_obs_days[-1]:
    #         break
    #
    # np.savetxt(os.path.join(save_dir, f"flares_param_source_{i}.txt"), np.atleast_2d(flare_params))
    flare_params = np.loadtxt(os.path.join(save_dir, f"flares_param_source_{i}.txt"))

    if redo[0]:
        clear_txt_images(exec_dir)
        clear_tobs(exec_dir)
    clear_fits(save_dir)
    clear_pics(basename + f"_{i-1}", save_dir)

    source_basename = basename + f"_{i}"


    if redo[0]:
        print("==========================================")
        print(f"Generating txt model images for {source_basename}")
        print("==========================================")
        generate_txt_images(redshift, B_1, b, K_1, Gamma,
                            LOS_coeff, HOAngle_deg,
                            n_along, n_across, lg_pixsize_min_mas, lg_pixsize_max_mas,
                            flare_params, ts_obs_days,
                            exec_dir, parallels_run_file, calculon)

    # if redo[1] and only_band is None:
    #     print("==========================================")
    #     print(f"Processing raw image for {source_basename}")
    #     print("==========================================")
    #     process_raw_images(source_basename, exec_dir, save_dir, redshift, plot_raw, match_resolution,
    #                        n_along, n_across, lg_pixsize_min_mas, lg_pixsize_max_mas,
    #                        ts_obs_days, flare_params, flare_shape,
    #                        Gamma, LOS_coeff, b, B_1, n, gamma_min, gamma_max, s)
    #     create_movie_raw(source_basename, save_dir)

    if redo[2]:
        print("==========================================")
        print(f"Generating and modelling visibilities for {source_basename}")
        print("==========================================")
        clear_pics(source_basename, save_dir)
        make_and_model_visibilities(source_basename, only_band, redshift, lg_pixsize_min_mas, lg_pixsize_max_mas, n_along, n_across, match_resolution,
                                    ts_obs_days,
                                    noise_scale_factor, mapsizes_dict,
                                    plot_clean, only_plot_raw,
                                    extract_extended, use_scipy_for_extract_extended, beam_fractions, two_stage,
                                    n_components,
                                    save_dir, exec_dir, path_to_script)
    create_movie_clean(source_basename, save_dir)

    clear_tobs(exec_dir)
    clear_fits(save_dir)
    # clear_pics(basename, save_dir)
