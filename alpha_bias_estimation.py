# This corrects bias of IPOL maps at each frequency given CC models of the artificial data
import os
from pathlib import Path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import numpy as np
import matplotlib
matplotlib.use('Agg')
# from astropy.stats import mad_std
# from vlbi_utils import create_mask, filter_CC, convert_difmap_model_file_to_CCFITS, CCFITS_to_difmap
import sys
sys.path.insert(0, '/home/ilya/github/ve/vlbi_errors')
from uv_data import UVData
from spydiff import clean_difmap
from from_fits import create_clean_image_from_fits_file, create_image_from_fits_file, create_model_from_fits_file


# Use final CLEAN iterations with UVTAPER?
use_uvtaper = False

n_mc = 30

# jet_model = "bk"
jet_model = "2ridges"
# jet_model = "3ridges"
# jet_model = "kh"
if jet_model not in ("bk", "2ridges", "3ridges", "kh"):
    raise Exception

# data_origin = "mojave"
data_origin = "bk145"
# data_origin = "vsop"
# data_origin = "vlba"
if data_origin not in ("mojave", "bk145", "vsop", "vlba"):
    raise Exception

if data_origin in ("mojave", "bk145", "vlba"):
    stokes = "i"
if data_origin == "vsop":
    stokes = "ll"

# FIXME: Using 0.2 mas beam size
# Saving intermediate files
if data_origin == "mojave":
    save_dir = os.path.join("/home/ilya/data/alpha/results/final_run/MOJAVE/orig_beam", jet_model)
elif data_origin == "bk145":
    save_dir = os.path.join("/home/ilya/data/alpha/results/final_run/BK145/orig_beam", jet_model)
elif data_origin == "vsop":
    save_dir = os.path.join("/home/ilya/data/alpha/results/final_run/VSOP", jet_model)
elif data_origin == "vlba":
    save_dir = os.path.join("/home/ilya/data/alpha/results/VLBA", jet_model)
else:
    raise Exception("data_origin must be vsop, mojave of bk145!")
Path(save_dir).mkdir(parents=True, exist_ok=True)


# Observed frequencies of simulations
if data_origin == "mojave" or data_origin == "bk145":
    freqs_obs_ghz = [8.1, 15.4]
    # freqs_obs_ghz = [8.1, 8.4, 12.1, 15.4]
elif data_origin == "vsop":
    freqs_obs_ghz = [1.6, 4.8]
elif data_origin == "vlba":
    freqs_obs_ghz = [24, 43]

# Multiplicative factor for noise added to model visibilities.
noise_scale_factor = 1.0

# Common size of the map and pixel size (mas)
common_mapsize = (1024, 0.1)


path_to_script = "/home/ilya/github/bk_transfer/scripts/final_clean_nw"
# path_to_script = "/home/ilya/github/bk_transfer/scripts/final_clean_nw_uvclip"
# path_to_script = "/home/ilya/github/bk_transfer/scripts/final_clean_nw_same_as_rms"
# path_to_script = "/home/ilya/github/bk_transfer/scripts/script_clean_rms"
path_to_scripts = {15.4: "/home/ilya/github/bk_transfer/scripts/script_clean_rms_u",
                   8.1: "/home/ilya/github/bk_transfer/scripts/script_clean_rms_x"}
if data_origin == "vsop":
    path_to_script = "/home/ilya/github/bk_transfer/scripts/final_clean_vsop"
    path_to_scripts = {1.6: "/home/ilya/github/bk_transfer/scripts/final_clean_vsop_l",
                       4.8: "/home/ilya/github/bk_transfer/scripts/final_clean_vsop_c"}

if data_origin == "bk145":
    # Lesha's data
    need_downscale_uv = True
    template_uvfits_dict = {15.4: "/home/ilya/data/alpha/BK145/1228+126.U.2009_05_23C_ta60.uvf_cal",
                            8.1: "/home/ilya/data/alpha/BK145/1228+126.X.2009_05_23_ta60.uvf_cal"}
    # Low freq
    # FIXME: Create U-template beam
    template_x_ccimage = create_clean_image_from_fits_file("/home/ilya/data/alpha/BK145/X_template_beam.fits")
    common_beam = template_x_ccimage.beam

    # FIXME: Using 0.2 mas beam size
    # common_beam = (1.6, 1.6, 0)
    # common_beam = (0.2, 0.2, 0)

elif data_origin == "mojave":
    # MOJAVE data
    need_downscale_uv = False
    template_uvfits_dict = {15.4: "/home/ilya/data/alpha/MOJAVE/1228+126.x.2006_06_15.uvf",
                            8.1: "/home/ilya/data/alpha/MOJAVE/1228+126.x.2006_06_15.uvf",
                            8.4: "/home/ilya/data/alpha/MOJAVE/1228+126.y.2006_06_15.uvf",
                            12.1: "/home/ilya/data/alpha/MOJAVE/1228+126.j.2006_06_15.uvf"}
    template_ccimage = {8.1: "/home/ilya/data/alpha/MOJAVE/template_cc_i_8.1.fits",
                        15.4: "/home/ilya/data/alpha/MOJAVE/template_cc_i_15.4.fits"}
    template_ccimage_low = create_clean_image_from_fits_file(template_ccimage[8.1])
    template_ccimage_high = create_clean_image_from_fits_file(template_ccimage[15.4])
    common_beam = template_ccimage_low.beam

elif data_origin == "vsop":
    # VSOP data
    need_downscale_uv = True
    template_uvfits_dict = {4.8: "/home/ilya/data/alpha/VSOP/m87.vsop-c.w040a5.split_12s",
                            1.6: "/home/ilya/data/alpha/VSOP/m87.vsop-l.w022a7.split_12s"}
    template_ccimages = {1.6: create_clean_image_from_fits_file("/home/ilya/data/alpha/VSOP/template_l_beam.fits"),
                         4.8: create_clean_image_from_fits_file("/home/ilya/data/alpha/VSOP/template_c_beam.fits")}
    # common_beam = template_ccimages[1.6].beam
    common_beam = (1.0, 1.0, 0)

elif data_origin == "vlba":
    common_mapsize = (4096, 0.025)
    template_uvfits_dict = {24: "/home/ilya/data/alpha/VLBA/m87.k.2018_04_28.fin_uvf_cal",
                            43: "/home/ilya/data/alpha/VLBA/m87.q.2018_04_28.fin_uvf_cal"}
    boxes = {24: "/home/ilya/data/alpha/VLBA/m87.k.2018_04_28.fin_win",
             43: "/home/ilya/data/alpha/VLBA/m87.q.2018_04_28.fin_wins"}
    common_beam = (0.5, 0.5, 0)

npixels_beam_common = np.pi*common_beam[0]*common_beam[1]/(4*np.log(2)*common_mapsize[1]**2)
##############################################
# No need to change anything below this line #
##############################################
# Plot only jet emission and do not plot counter-jet?
jet_only = False

freq_high = max(freqs_obs_ghz)
freq_low = min(freqs_obs_ghz)

for freq in freqs_obs_ghz:

    uvdata = UVData(template_uvfits_dict[freq])
    noise = uvdata.noise(average_freq=False, use_V=False)
    uvdata.zero_data()
    # If one needs to decrease the noise this is the way to do it
    for baseline, baseline_noise_std in noise.items():
        noise.update({baseline: noise_scale_factor*baseline_noise_std})

    ccmodel_file = "model_{}_filtered_convolved.fits".format(freq)
    # ccmodel_file_filtered = "model_cc_i_{}_ccfiltered.fits".format(freq)
    # Filter CC-components
    # image = create_image_from_fits_file(os.path.join(save_dir, ccmodel_file)).image
    # std = mad_std(image)
    # cc_mask = create_mask(image, 3*std)
    # filter_CC(os.path.join(save_dir, ccmodel_file), cc_mask, os.path.join(save_dir, ccmodel_file_filtered), "filtered_cc_{}.png".format(freq))
    # convert_difmap_model_file_to_CCFITS()

    ccmodel = create_model_from_fits_file(os.path.join(save_dir, ccmodel_file))

    uvdata.substitute([ccmodel])
    uvdata.noise_add(noise)
    uvdata.save(os.path.join(save_dir, "template_cc_{}.uvf".format(freq)), rewrite=True, downscale_by_freq=need_downscale_uv)

    for i in range(n_mc):

        print("Creating realization ", i+1)
        uvdata = UVData(template_uvfits_dict[freq])
        uvdata.substitute([ccmodel])
        uvdata.noise_add(noise)
        uvdata.save(os.path.join(save_dir, "template_cc_{}_mc_{}.uvf".format(freq, i+1)), rewrite=True, downscale_by_freq=need_downscale_uv)

    outfname = "model_cc_cc_i_{}.fits".format(freq)

    if os.path.exists(outfname):
        os.unlink(outfname)

    if use_uvtaper:
        path_to_script = path_to_scripts[freq]

    if data_origin == "vlba":
        txt_box = boxes[freq]
    else:
        txt_box = None

    # Just create UVFITS and CLEAN it manually
    if data_origin == "vsop":
        continue

    clean_difmap(fname="template_cc_{}.uvf".format(freq), path=save_dir,
                 outfname=outfname, outpath=save_dir, stokes=stokes,
                 mapsize_clean=common_mapsize, path_to_script=path_to_script,
                 show_difmap_output=False, text_box=txt_box,
                 beam_restore=common_beam, dmap="dmap_cc_{}.fits".format(freq))
    # dfm_model=os.path.join(save_dir, "model_cc_i_{}.mdl".format(freq)))
    for i in range(n_mc):
        print("CLEANing realization ", i+1)
        outfname = "model_cc_cc_i_{}_mc_{}.fits".format(freq, i+1)
        clean_difmap(fname="template_cc_{}_mc_{}.uvf".format(freq, i+1), path=save_dir,
                     outfname=outfname, outpath=save_dir, stokes=stokes,
                     mapsize_clean=common_mapsize, path_to_script=path_to_script,
                     show_difmap_output=False, text_box=txt_box,
                     beam_restore=common_beam, dmap="dmap_cc_{}_mc_{}.fits".format(freq, i+1))

if data_origin == "vsop":
    sys.exit(0)

# Create image of alpha made from true jet models convolved with beam
itrue_convolved_low = create_image_from_fits_file("{}/convolved_true_jet_model_i_rotated_{}.fits".format(save_dir, freq_low)).image
itrue_convolved_high = create_image_from_fits_file("{}/convolved_true_jet_model_i_rotated_{}.fits".format(save_dir, freq_high)).image
true_convolved_spix_array = np.log(itrue_convolved_low/itrue_convolved_high)/np.log(freq_low/freq_high)

cc_itrue_convolved_low = create_image_from_fits_file(os.path.join(save_dir, "model_{}_filtered_convolved.fits".format(freq_low))).image
cc_itrue_convolved_high = create_image_from_fits_file(os.path.join(save_dir, "model_{}_filtered_convolved.fits".format(freq_high))).image
cc_true_convolved_spix_array = np.log(cc_itrue_convolved_low/cc_itrue_convolved_high)/np.log(freq_low/freq_high)


images = dict()
for freq in freqs_obs_ghz:
    images[freq] = list()
# Read all MC data
for i in range(n_mc):
    for freq in freqs_obs_ghz:
        images[freq].append(create_clean_image_from_fits_file(os.path.join(save_dir, "model_cc_cc_i_{}_mc_{}.fits".format(freq, i+1))).image)

image_low = np.sum(images[freq_low], axis=0)/n_mc
image_high = np.sum(images[freq_high], axis=0)/n_mc
spix_arrays = [np.log(images[freq_low][i]/images[freq_high][i])/np.log(freq_low/freq_high) for i in range(n_mc)]
std_spix = np.std(spix_arrays, axis=0)
mean_spix = np.mean(spix_arrays, axis=0)
spix_array = np.log(image_low/image_high)/np.log(freq_low/freq_high)


cc_bias_ipol_low = image_low - cc_itrue_convolved_low
cc_bias_ipol_high = image_high - cc_itrue_convolved_high
cc_bias_alpha = mean_spix - cc_true_convolved_spix_array

np.savetxt(os.path.join(save_dir, "cc_ipol_bias_low.txt"), cc_bias_ipol_low)
np.savetxt(os.path.join(save_dir, "cc_ipol_bias_high.txt"), cc_bias_ipol_high)
np.savetxt(os.path.join(save_dir, "cc_alpha_bias.txt"), cc_bias_alpha)
