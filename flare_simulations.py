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
source_times_dict = dict()
source_epochs_dict = dict()
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
    source_times_dict[str(source)] = vdtimes
    source_epochs_dict[str(source)] = epochs


# sys.exit(0)

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


redo = [True, True, True]
calculon = True
# basename = "comps_decreaseB_same_flare_params"
basename = "dump"
only_band = None
match_resolution = False
dump_visibilities_for_registration_testing = True
redshift = 0.8
K_1 = 5000.

freq_names = {2.3: "S", 8.6: "X"}
freqs_ghz = tuple(freq_names.keys())
n_along = 600
n_across = 100
lg_pixsize_min_mas = -3.0
lg_pixsize_max_mas = -0.5

if match_resolution:
    lg_pixsize_min = {min(freqs_ghz): lg_pixsize_min_mas, max(freqs_ghz): lg_pixsize_min_mas-np.log10(max(freqs_ghz)/min(freqs_ghz))}
    lg_pixsize_max = {min(freqs_ghz): lg_pixsize_max_mas, max(freqs_ghz): lg_pixsize_max_mas-np.log10(max(freqs_ghz)/min(freqs_ghz))}
else:
    lg_pixsize_min = {min(freqs_ghz): lg_pixsize_min_mas, max(freqs_ghz): lg_pixsize_min_mas}
    lg_pixsize_max = {min(freqs_ghz): lg_pixsize_max_mas, max(freqs_ghz): lg_pixsize_max_mas}

# Some template UVFITS with full polarization. Its uv-coverage and noise will be used while creating fake data
# Originally used template
# template_uvfits = {2.3: "/home/ilya/data/rfc/J0102+5824/J0102+5824_S_2017_10_21_pus_vis.fits",
#                    8.6: "/home/ilya/data/rfc/J0102+5824/J0102+5824_X_2017_10_21_pus_vis.fits"}
# These have smalllest beam
# template_uvfits = {2.3: "/home/ilya/data/rfc/J0102+5824/J0102+5824_S_2009_04_21_pus_vis.fits",
#                    8.6: "/home/ilya/data/rfc/J0102+5824/J0102+5824_X_2009_04_21_pus_vis.fits"}

# Sokolovsky's
template_uvfits = {2.3: "/home/ilya/data/rfc/J2203+3145/J2203+3145_S_2007_04_30_sok_vis.fits",
                   8.6: "/home/ilya/data/rfc/J2203+3145/J2203+3145_X_2007_04_30_sok_vis.fits"}

# TODO: Changing this => edit main.cpp! ################################################################################
n = 2.0
s = 2.
gamma_min = 10.
########################################################################################################################
# TODO: Changing this => edit utils.h! #################################################################################
gamma_max = 1E+04
########################################################################################################################
# TODO: Changing this => edit NField.cpp! ##############################################################################
flare_shape = 10.0
########################################################################################################################

noise_scale_factor = 1.0
mapsizes_dict = {2.3: (2048, 0.05,), 8.6: (2048, 0.05,)}
plot_raw = True
plot_clean = True
only_plot_raw = False
extract_extended = True
use_elliptical = False
use_flare_param_files_from_other_run = False
use_scipy_for_extract_extended = False
if use_scipy_for_extract_extended and use_elliptical:
    raise Exception("Currently, Scipy can't be used to fit elliptical Gaussians.")
beam_fractions = np.round(np.linspace(0.5, 1.5, 11), 2)
two_stage = False
n_components = 4


if not calculon:
    exec_dir = "/home/ilya/github/bk_transfer/Release"
    parallels_run_file = "/home/ilya/github/bk_transfer/parallels_run.txt"
    save_dir = "/home/ilya/github/bk_transfer/pics/flares"
    flare_param_files_dir = "/home/ilya/github/bk_transfer/pics/flares"
    path_to_script = "/home/ilya/github/bk_transfer/scripts/script_clean_rms"
    n_jobs = 4
else:
    exec_dir = "/home/ilya/github/flares/bk_transfer/Release"
    parallels_run_file = "/home/ilya/github/flares/bk_transfer/parallels_run.txt"
    save_dir = "/home/ilya/github/flares/bk_transfer/pics/flares/dump"
    flare_param_files_dir = "/home/ilya/github/flares/bk_transfer/pics/flares/survey/comps"
    path_to_script = "/home/ilya/github/flares/bk_transfer/scripts/script_clean_rms"
    n_jobs = 44


save_dir = os.path.join(save_dir, basename)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

dump_visibilities_directory = os.path.join(save_dir, "shifted_uvfits")
if not os.path.exists(dump_visibilities_directory):
    os.mkdir(dump_visibilities_directory)

# n_sources = 1
# flare_params = [(0.0, 0.0, 0.0, 0.1)]
# ts_obs_days = np.array([0.0])

# n_sources = 40
n_sources = 1
idxs = np.random.choice(np.arange(len(sources), dtype=int), size=n_sources, replace=False)
for i in range(n_sources):
    # First set
    # B_1 = 1.5
    # b = 1.25
    # Second set
    B_1 = 0.7
    b = 1.0
    Gamma = 21.
    LOS_coeff = 0.5
    HOAngle_deg = 20.

    los_angle_deg = np.round(np.rad2deg(np.arcsin(LOS_coeff/Gamma)), 2)
    cone_half_angle_deg = np.round(np.rad2deg(np.arctan(np.tan(np.deg2rad(HOAngle_deg)) * np.sin(np.deg2rad(los_angle_deg)))), 2)
    print(f"LOS(deg) = {los_angle_deg}")
    print(f"Cone HA (deg) = {cone_half_angle_deg}")

    # Fixed times
    # This will be multiplied on (1+z) to bring to the observer z = 0.
    ts_obs_days = np.linspace(-400.0, 10*360, 44)/(1+redshift)
    # ts_obs_days = np.array([0.0])

    # # From real sources times
    # # This will be multiplied on (1+z) to bring to the observer z = 0.
    # ts_obs_days = source_epochs[sources[idxs[i]]]/(1+redshift)
    # # Shift to sample flares right
    # ts_obs_days -= 400

    if not use_flare_param_files_from_other_run:
        flare_params = list()

        # First flare
        t_start_years = np.random.uniform(-1, 1., size=1)[0]
        # t_start_years = 0.
        t_start_days = t_start_years*12*30
        # FIXME:
        amp_N = np.random.uniform(3, 10, size=1)[0]
        # amp_N = 5.
        # amp_N = 0.0
        # Only N flare
        amp_B = 0.0
        # Equipartition flare
        # amp_B = np.sqrt(amp_N)
        # Increasing N, decreasing B flare
        # amp_B = -0.5
        # amp_B = 1./np.sqrt(1. + amp_N) - 1.
        width_pc = np.random.uniform(0.1, 0.2, size=1)[0]
        # width_pc = 0.15
        flare_params.append((amp_N, amp_B, t_start_days, width_pc))

        # Maximal number of flares
        for i_fl in range(10):
            # Waiting time 2 yrs
            dt_yrs = 0.0
            while dt_yrs < 2.0:
                dt_yrs = np.random.exponential(2.0)
            t_start_years += dt_yrs
            t_start_days = t_start_years*12*30
            # FIXME:
            # amp_N = 0.0
            amp_N = np.random.uniform(3, 10, size=1)[0]
            # Only N flare
            amp_B = 0.0
            # Equipartition flare
            # amp_B = np.sqrt(amp_N)
            # Increasing N, decreasing B flare
            # amp_B = -0.5
            # amp_B = 1./np.sqrt(1. + amp_N) - 1.
            width_pc = np.random.uniform(0.1, 0.2, size=1)[0]
            flare_params.append((amp_N, amp_B, t_start_days, width_pc))

            if t_start_days > ts_obs_days[-1]:
                break

        np.savetxt(os.path.join(save_dir, f"flares_param_source_{i}.txt"), np.atleast_2d(flare_params))
    else:
        flare_params = np.loadtxt(os.path.join(flare_param_files_dir, f"flares_param_source_{i}.txt"))
        # Convert 2D array to list of lists
        flare_params = flare_params.tolist()
        # Save as it was generated independently
        np.savetxt(os.path.join(save_dir, f"flares_param_source_{i}.txt"), np.atleast_2d(flare_params))

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
                            exec_dir, parallels_run_file, n_jobs)

    if redo[1] and only_band is None:
        print("==========================================")
        print(f"Processing raw image for {source_basename}")
        print("==========================================")
        # try:
        process_raw_images(basename=source_basename, txt_dir=exec_dir, save_dir=save_dir,
                           z=redshift, plot=plot_raw,
                           n_along=n_along, n_across=n_across,
                           lg_pixsize_min_mas=lg_pixsize_min, lg_pixsize_max_mas=lg_pixsize_max,
                           ts_obs_days=ts_obs_days, flare_params=flare_params, flare_shape=flare_shape,
                           Gamma=Gamma, LOS_coeff=LOS_coeff, b=b, B_1=B_1, n=n,
                           gamma_min=gamma_min, gamma_max=gamma_max, s=s)
        # If for some epoch something went wrong
        # except:
        #     clear_tobs(exec_dir)
        #     clear_fits(save_dir)
        #     clear_pics(basename, save_dir)
        #     continue
        # create_movie_raw(source_basename, save_dir)

    if redo[2]:
        print("==========================================")
        print(f"Generating and modelling visibilities for {source_basename}")
        print("==========================================")
        # try:
        make_and_model_visibilities(basename=source_basename, only_band=only_band, z=redshift, freqs_ghz=freqs_ghz,
                                    freq_names=freq_names,
                                    lg_pixsize_min_mas=lg_pixsize_min, lg_pixsize_max_mas=lg_pixsize_max,
                                    n_along=n_along, n_across=n_across,
                                    ts_obs_days=ts_obs_days, template_uvfits=template_uvfits,
                                    noise_scale_factor=noise_scale_factor, mapsizes_dict=mapsizes_dict,
                                    plot_clean=plot_clean, only_plot_raw=only_plot_raw,
                                    extract_extended=extract_extended, use_scipy=use_scipy_for_extract_extended,
                                    use_elliptical=use_elliptical, beam_fractions=beam_fractions, two_stage=two_stage,
                                    n_components=n_components,
                                    save_dir=save_dir, jetpol_run_directory=exec_dir, path_to_script=path_to_script,
                                    n_jobs=n_jobs,
                                    dump_visibilities_for_registration_testing=dump_visibilities_for_registration_testing,
                                    dump_visibilities_directory=dump_visibilities_directory)
        # If for some epoch something went wrong
        # except:
        #     # clear_tobs(exec_dir)
        #     clear_fits(save_dir)
        #     clear_pics(basename, save_dir)
        #     continue

        create_movie_clean(source_basename, save_dir, only_band)

    # clear_tobs(exec_dir)
    clear_fits(save_dir)
    # clear_pics(basename, save_dir)
