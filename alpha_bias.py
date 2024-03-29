import os
from pathlib import Path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import numpy as np
from scipy.ndimage import rotate
import matplotlib
matplotlib.use('Agg')
import sys
from jet_image import JetImage, TwinJetImage
from vlbi_utils import (find_image_std, find_bbox, convert_difmap_model_file_to_CCFITS, rotate_difmap_model,
                        convert_blc_trc)
sys.path.insert(0, '/home/ilya/github/ve/vlbi_errors')
from uv_data import UVData
from spydiff import clean_difmap
from from_fits import create_clean_image_from_fits_file, create_image_from_fits_file
from image_ops import spix_map
from image import plot as iplot


# If `True` then re-substitute model data and re-image in Difmap
only_make_pics = False
# If ``only_make_pics = True``, do we need to re-CLEAN? Note that when new beam is used then we need to go from scratch!
re_clean = False

# If True, then just CLEAN artificial data and do not bother with true model convolution.
only_clean = True
# If True then assume that true model is ready
done_true_convolved = True

# Make model low frequency image from high frequency one using `alpha_true`
artificial_alpha = False
# S ~ nu^{+alpha}
alpha_true = -0.5

# Use final CLEAN iterations with UVTAPER?
use_uvtaper = False

n_mc = 1

# jet_model = "bk"
jet_model = "2ridges"
# jet_model = "3ridges"
# jet_model = "kh"
if jet_model not in ("bk", "2ridges", "3ridges", "kh"):
    raise Exception
# data_origin = "mojave"
# data_origin = "bk145"
data_origin = "vsop"
# data_origin = "vlba"
if data_origin not in ("mojave", "bk145", "vsop", "vlba"):
    raise Exception

if data_origin in ("mojave", "bk145", "vlba"):
    stokes = "i"
if data_origin == "vsop":
    stokes = "ll"

# Saving intermediate files
if data_origin == "mojave":
    save_dir = os.path.join("/home/ilya/data/alpha/results/MOJAVE", jet_model)
elif data_origin == "bk145":
    save_dir = os.path.join("/home/ilya/data/alpha/results/BK145/orig_beam", jet_model)
elif data_origin == "vsop":
    save_dir = os.path.join("/home/ilya/data/alpha/results/VSOP", jet_model)
elif data_origin == "vlba":
    save_dir = os.path.join("/home/ilya/data/alpha/results/VLBA", jet_model)
else:
    raise Exception("data_origin must be vsop, mojave of bk145!")
Path(save_dir).mkdir(parents=True, exist_ok=True)


# Observed frequencies of simulations
if data_origin == "mojave" or data_origin == "bk145":
    freqs_obs_ghz = [8.1, 15.4]
elif data_origin == "vsop":
    freqs_obs_ghz = [1.6, 4.8]
elif data_origin == "vlba":
    freqs_obs_ghz = [24, 43]

# -107 for M87
# rot_angle_deg = -107.0
rot_angle_deg = 0.0

# Scale model image to obtain ~ 3 Jy
scale = 1.0

# Multiplicative factor for noise added to model visibilities.
noise_scale_factor = 0.01

# Common size of the map and pixel size (mas)
common_mapsize = (1024, 0.1)


# C++ code run parameters
# jetpol_files_directory = "/home/ilya/github/bk_transfer/Release"
jetpol_files_directory = "/home/ilya/github/imager_dev"
z = 0.00436
# n_along = 1400
# n_across = 500
# FIXME: Debug FT of imager_dev
n_along = 512
n_across = 1024
# FIXME: For 24 & 43 GHz VLBA data we need smaller pixel sizes?
# lg_pixel_size_mas_min = -1.5
# lg_pixel_size_mas_max = -1.5
# FIXME: Debug FT of imager_dev
lg_pixel_size_mas_min = -1
lg_pixel_size_mas_max = -1

# path_to_script = "/home/ilya/github/bk_transfer/scripts/final_clean_nw"
path_to_script = "/home/ilya/github/bk_transfer/scripts/script_clean_rms"
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
    # common_beam = (1.6, 1.6, 0)

elif data_origin == "mojave":
    # MOJAVE data
    need_downscale_uv = False
    template_uvfits_dict = {15.4: "/home/ilya/data/alpha/MOJAVE/1228+126.u.2020_07_02.uvf",
                            8.1: "/home/ilya/data/alpha/MOJAVE/1228+126.x.2006_06_15.uvf"}
    template_ccimage = {8.1: "/home/ilya/data/alpha/MOJAVE/template_cc_i_8.1.fits",
                        15.4: "/home/ilya/data/alpha/MOJAVE/template_cc_i_15.4.fits"}
    template_ccimage = create_clean_image_from_fits_file(template_ccimage[8.1])
    common_beam = template_ccimage.beam

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

##############################################
# No need to change anything below this line #
##############################################
# Plot only jet emission and do not plot counter-jet?
jet_only = False

freq_high = max(freqs_obs_ghz)
freq_low = min(freqs_obs_ghz)

if not only_make_pics or (only_make_pics and re_clean):
    for freq in freqs_obs_ghz:
        if not only_make_pics:
            uvdata = UVData(template_uvfits_dict[freq])
            noise = uvdata.noise(average_freq=False, use_V=False)
            uvdata.zero_data()
            # If one needs to decrease the noise this is the way to do it
            for baseline, baseline_noise_std in noise.items():
                noise.update({baseline: noise_scale_factor*baseline_noise_std})

            jm = JetImage(z=z, n_along=n_along, n_across=n_across,
                          lg_pixel_size_mas_min=lg_pixel_size_mas_min, lg_pixel_size_mas_max=lg_pixel_size_mas_max,
                          jet_side=True, rot=np.deg2rad(rot_angle_deg))
            cjm = JetImage(z=z, n_along=n_along, n_across=n_across,
                           lg_pixel_size_mas_min=lg_pixel_size_mas_min, lg_pixel_size_mas_max=lg_pixel_size_mas_max,
                           jet_side=False, rot=np.deg2rad(rot_angle_deg))
            if artificial_alpha:
                jm.load_image_stokes(stokes.upper(), "{}/jet_image_{}_{}_{}.txt".format(jetpol_files_directory, "i", freq_high, jet_model),
                                     scale=scale*(freq_high/freq)**(-alpha_true))
                cjm.load_image_stokes(stokes.upper(), "{}/cjet_image_{}_{}_{}.txt".format(jetpol_files_directory, "i", freq_high, jet_model),
                                      scale=scale*(freq_high/freq)**(-alpha_true))
            else:
                # jm.load_image_stokes(stokes.upper(), "{}/jet_image_{}_{}_{}.txt".format(jetpol_files_directory, "i", freq, jet_model), scale=scale)
                # cjm.load_image_stokes(stokes.upper(), "{}/cjet_image_{}_{}_{}.txt".format(jetpol_files_directory, "i", freq, jet_model), scale=scale)
                # FIXME: Debug FT of imager_dev
                jm.load_image_stokes(stokes.upper(), "/home/ilya/github/imager_dev/jet_image_i_15.4.txt", scale=scale)
                cjm.load_image_stokes(stokes.upper(), "/home/ilya/github/imager_dev/cjet_image_i_15.4.txt", scale=scale)
            js = TwinJetImage(jm, cjm)

            # Convert to difmap model format
            # TODO: Check
            js.save_image_to_difmap_format("{}/true_jet_model_i_{}.txt".format(save_dir, freq))

            # Rotate
            rotate_difmap_model("{}/true_jet_model_i_{}.txt".format(save_dir, freq),
                                "{}/true_jet_model_i_{}_rotated.txt".format(save_dir, freq),
                                PA_deg=-rot_angle_deg)
            if not only_clean:
                # Convolve with beam
                convert_difmap_model_file_to_CCFITS("{}/true_jet_model_i_{}_rotated.txt".format(save_dir, freq), "I", common_mapsize,
                                                    common_beam, template_uvfits_dict[freq],
                                                    "{}/convolved_true_jet_model_i_rotated_{}.fits".format(save_dir, freq))
            uvdata.substitute([js])
            uvdata.noise_add(noise)
            uvdata.save(os.path.join(save_dir, "template_{}.uvf".format(freq)), rewrite=True, downscale_by_freq=need_downscale_uv)

            if n_mc > 1:
                for i in range(n_mc):
                    uvdata = UVData(template_uvfits_dict[freq])
                    uvdata.substitute([js])
                    uvdata.noise_add(noise)
                    uvdata.save(os.path.join(save_dir, "template_{}_mc_{}.uvf".format(freq, i+1)), rewrite=True, downscale_by_freq=need_downscale_uv)


        # If doing from scratch or only making pics, but with re-CLEANing
        if not only_make_pics or (only_make_pics and re_clean):
            outfname = "model_cc_i_{}.fits".format(freq)

            if os.path.exists(outfname):
                os.unlink(outfname)

            if use_uvtaper:
                path_to_script = path_to_scripts[freq]

            if data_origin == "vlba":
                txt_box = boxes[freq]
            else:
                txt_box = None

            clean_difmap(fname="template_{}.uvf".format(freq), path=save_dir,
                         outfname=outfname, outpath=save_dir, stokes=stokes,
                         mapsize_clean=common_mapsize, path_to_script=path_to_script,
                         show_difmap_output=True, text_box=txt_box,
                         beam_restore=common_beam)
                         # dfm_model=os.path.join(save_dir, "model_cc_i_{}.mdl".format(freq)))
            if n_mc > 1:
                for i in range(n_mc):
                    outfname = "model_cc_i_{}_mc_{}.fits".format(freq, i+1)
                    clean_difmap(fname="template_{}_mc_{}.uvf".format(freq, i+1), path=save_dir,
                                 outfname=outfname, outpath=save_dir, stokes=stokes,
                                 mapsize_clean=common_mapsize, path_to_script=path_to_script,
                                 show_difmap_output=True, text_box=txt_box,
                                 beam_restore=common_beam)

if done_true_convolved:
    # Create image of alpha made from true jet models convolved with beam
    itrue_convolved_low = create_image_from_fits_file("{}/convolved_true_jet_model_i_rotated_{}.fits".format(save_dir, freq_low)).image
    itrue_convolved_high = create_image_from_fits_file("{}/convolved_true_jet_model_i_rotated_{}.fits".format(save_dir, freq_high)).image
    true_convolved_spix_array = np.log(itrue_convolved_low/itrue_convolved_high)/np.log(freq_low/freq_high)

# Observed images of CLEANed artificial UV-data
ccimages = {freq: create_clean_image_from_fits_file(os.path.join(save_dir, "model_cc_i_{}.fits".format(freq)))
            for freq in freqs_obs_ghz}
ipol_low = ccimages[freq_low].image
ipol_high = ccimages[freq_high].image
beam_low = ccimages[freq_low].beam
beam_high = ccimages[freq_high].beam
# Number of pixels in beam
npixels_beam_low = np.pi*beam_low[0]*beam_low[1]/(4*np.log(2)*common_mapsize[1]**2)
npixels_beam_high = np.pi*beam_high[0]*beam_high[1]/(4*np.log(2)*common_mapsize[1]**2)


std_low = find_image_std(ipol_low, beam_npixels=npixels_beam_low)
print("IPOL image std = {} mJy/beam".format(1000*std_low))
blc_low, trc_low = find_bbox(ipol_low, level=3*std_low, min_maxintensity_mjyperbeam=10*std_low,
                             min_area_pix=10*npixels_beam_low, delta=10)
blc_low, trc_low = convert_blc_trc(blc_low, trc_low, ipol_low)

std_high = find_image_std(ipol_high, beam_npixels=npixels_beam_high)
print("IPOL image std = {} mJy/beam".format(1000*std_high))
blc_high, trc_high = find_bbox(ipol_high, level=3*std_high, min_maxintensity_mjyperbeam=10*std_high,
                               min_area_pix=10*npixels_beam_high, delta=10)
blc_high, trc_high = convert_blc_trc(blc_high, trc_high, ipol_high)

if data_origin == "mojave":
    blc = (450, 450)
    trc = (950, 700)
    blc_true = (400, 400)
    trc_true = (1000, 800)
elif data_origin in ("bk145", "vsop"):
    blc = (400, 430)
    trc = (980, 710)
    blc_true = (400, 430)
    trc_true = (980, 710)
elif data_origin == "vlba":
    blc = (1900, 1900)
    trc = (2300, 2200)
    blc_true = (1900, 1900)
    trc_true = (2300, 2200)

# I high
fig = iplot(ipol_high, x=ccimages[freq_high].x, y=ccimages[freq_high].y,
            min_abs_level=3*std_high, blc=blc, trc=trc, beam=beam_low, close=True, show_beam=True, show=False,
            contour_color='gray', contour_linewidth=0.25, beam_place="lr")
fig.savefig(os.path.join(save_dir, "observed_ipol_{}GHz.png".format(freq_high)), dpi=600, bbox_inches="tight")

# sys.exit(0)

if done_true_convolved:
    # I high bias
    fig = iplot(ipol_high, (ipol_high - itrue_convolved_high)/itrue_convolved_high,
                x=ccimages[freq_high].x, y=ccimages[freq_high].y,
                colors_mask=ipol_high < 3*std_high,
                color_clim=[-0.25, 0.25],
                min_abs_level=3*std_high, blc=blc, trc=trc, beam=beam_high, close=True, show_beam=True, show=False,
                contour_color='black', contour_linewidth=0.25, colorbar_label="I frac. bias", plot_colorbar=True,
                cmap='bwr', beam_place="lr")
    fig.savefig(os.path.join(save_dir, "bias_ipol_{}GHz.png".format(freq_high)), dpi=600, bbox_inches="tight")
    # True I convolved
    fig = iplot(itrue_convolved_high,
                x=ccimages[freq_high].x, y=ccimages[freq_high].y,
                min_abs_level=0.0001*np.max(itrue_convolved_high), blc=blc_true, trc=trc_true, beam=common_beam, close=True, show_beam=True, show=False,
                contour_color='gray', contour_linewidth=0.25, beam_place="lr")
    fig.savefig(os.path.join(save_dir, "ipol_true_conv_{}GHz.png".format(freq_high)), dpi=600, bbox_inches="tight")

# I low
fig = iplot(ipol_low, x=ccimages[freq_low].x, y=ccimages[freq_low].y,
            min_abs_level=450*1e-06, blc=blc, trc=trc, beam=beam_low, close=True, show_beam=True, show=False,
            contour_color='gray', contour_linewidth=0.25, beam_place="lr")
fig.savefig(os.path.join(save_dir, "observed_ipol_{}GHz.png".format(freq_low)), dpi=600, bbox_inches="tight")

if done_true_convolved:
    # I low bias
    fig = iplot(ipol_low, (ipol_low - itrue_convolved_low)/itrue_convolved_low,
                x=ccimages[freq_low].x, y=ccimages[freq_low].y, colors_mask=ipol_low < 3*std_low, color_clim=[-0.25, 0.25],
                min_abs_level=3*std_low, blc=blc, trc=trc, beam=beam_low, close=True, show_beam=True, show=False,
                contour_color='black', contour_linewidth=0.25, colorbar_label="I frac. bias", plot_colorbar=True,
                cmap="bwr", beam_place="lr")
    fig.savefig(os.path.join(save_dir, "bias_ipol_{}GHz.png".format(freq_low)), dpi=600, bbox_inches="tight")
    # True I convolved
    fig = iplot(itrue_convolved_low,
                x=ccimages[freq_low].x, y=ccimages[freq_low].y,
                min_abs_level=0.0001*np.max(itrue_convolved_low), blc=blc_true, trc=trc_true, beam=common_beam, close=True, show_beam=True, show=False,
                contour_color='gray', contour_linewidth=0.25, beam_place="lr")
    fig.savefig(os.path.join(save_dir, "ipol_true_conv_{}GHz.png".format(freq_high)), dpi=600, bbox_inches="tight")


ipol_arrays = dict()
sigma_ipol_arrays = dict()
masks_dict = dict()
std_dict = dict()
for freq in freqs_obs_ghz:

    ipol = ccimages[freq].image
    ipol_arrays[freq] = ipol

    std = find_image_std(ipol, beam_npixels={freq_high: npixels_beam_high, freq_low: npixels_beam_low}[freq])
    std_dict[freq] = std
    masks_dict[freq] = ipol < 3*std
    sigma_ipol_arrays[freq] = np.ones(ipol.shape)*std

common_imask = np.logical_or.reduce([masks_dict[freq] for freq in freqs_obs_ghz])
spix_array = np.log(ipol_arrays[freq_low]/ipol_arrays[freq_high])/np.log(freq_low/freq_high)
# spix_array, sigma_spix_array, spix_chisq_array = spix_map(np.array(freqs_obs_ghz)*1e9, ipol_arrays, sigma_ipol_arrays,
#                                                           mask=common_imask, outfile=None, outdir=None,
#                                                           mask_on_chisq=False, ampcal_uncertainties=None)

# True spix
if artificial_alpha:
    color_clim = [alpha_true-0.5, alpha_true+0.5]
else:
    color_clim = [-1.5, 0.5]

if done_true_convolved:
    fig = iplot(ipol_arrays[freq_low], true_convolved_spix_array, x=ccimages[freq_low].x, y=ccimages[freq_low].y,
                min_abs_level=3*std_dict[freq_low], colors_mask=common_imask, color_clim=color_clim, blc=blc, trc=trc,
                beam=common_beam, close=True, colorbar_label=r"$\alpha$", show_beam=True, show=False,
                cmap='bwr', contour_color='black', plot_colorbar=True,
                contour_linewidth=0.25, beam_place="lr")
    fig.savefig(os.path.join(save_dir, "true_conv_spix_{}GHz_{}GHz_I_{}GHz.png".format(freq_high, freq_low, freq_low)),
                dpi=600, bbox_inches="tight")
    fig = iplot(itrue_convolved_low, true_convolved_spix_array, x=ccimages[freq_low].x, y=ccimages[freq_low].y,
                min_abs_level=0.0001*np.max(itrue_convolved_low), colors_mask=itrue_convolved_low < 0.0001*np.max(itrue_convolved_low),
                color_clim=color_clim, blc=blc_true, trc=trc_true,
                beam=common_beam, close=True, colorbar_label=r"$\alpha$", show_beam=True, show=False,
                cmap='bwr', contour_color='black', plot_colorbar=True,
                contour_linewidth=0.25, beam_place="lr")
    fig.savefig(os.path.join(save_dir, "true_conv_spix_{}GHz_{}GHz_true_I_{}GHz.png".format(freq_high, freq_low, freq_low)),
                dpi=600, bbox_inches="tight")

# Observed spix
if artificial_alpha:
    color_clim = [alpha_true-0.5, alpha_true+0.5]
else:
    color_clim = [-1.0, 0.0]
fig = iplot(ipol_arrays[freq_low], spix_array, x=ccimages[freq_low].x, y=ccimages[freq_low].y,
            min_abs_level=3*std_dict[freq_low], colors_mask=common_imask, color_clim=color_clim, blc=blc, trc=trc,
            beam=common_beam, close=True, colorbar_label=r"$\alpha$", show_beam=True, show=False,
            cmap='nipy_spectral', contour_color='black', plot_colorbar=True,
            contour_linewidth=0.25, beam_place="lr")
fig.savefig(os.path.join(save_dir, "spix_{}GHz_{}GHz_I_{}GHz.png".format(freq_high, freq_low, freq_low)), dpi=600, bbox_inches="tight")

if done_true_convolved:
    # Bias spix
    # FIXME: Try coolwarm or seismic cmap
    fig = iplot(ipol_arrays[freq_low], spix_array - true_convolved_spix_array, x=ccimages[freq_low].x, y=ccimages[freq_low].y,
                min_abs_level=3*std_dict[freq_low], colors_mask=common_imask, color_clim=[-0.3, 0.3], blc=blc, trc=trc,
                beam=common_beam, close=True, colorbar_label=r"$\alpha$ bias", show_beam=True, show=False,
                cmap='bwr', contour_color='black', plot_colorbar=True,
                contour_linewidth=0.25, beam_place="lr")
    fig.savefig(os.path.join(save_dir, "bias_spix_{}GHz_{}GHz_I_{}GHz.png".format(freq_high, freq_low, freq_low)), dpi=600, bbox_inches="tight")


if n_mc > 1:
    images = dict()
    for freq in freqs_obs_ghz:
        images[freq] = list()
    # Read all MC data
    for i in range(n_mc):
        for freq in freqs_obs_ghz:
            images[freq].append(create_clean_image_from_fits_file(os.path.join(save_dir, "model_cc_i_{}_mc_{}.fits".format(freq, i+1))).image)

    image_low = np.sum(images[freq_low], axis=0)/n_mc
    image_high = np.sum(images[freq_high], axis=0)/n_mc
    std = find_image_std(image_low, beam_npixels=npixels_beam_low)
    stack_mask = image_low < 5*std
    spix_array = np.log(image_low/image_high)/np.log(freq_low/freq_high)


    # Observed stacked spix
    if artificial_alpha:
        color_clim = [alpha_true-0.5, alpha_true+0.5]
    else:
        color_clim = [-1.0, 0.0]
    fig = iplot(image_low, spix_array, x=ccimages[freq_low].x, y=ccimages[freq_low].y,
                min_abs_level=4*std, colors_mask=stack_mask, color_clim=color_clim, blc=blc, trc=trc,
                beam=common_beam, close=True, colorbar_label=r"$\alpha$", show_beam=True, show=False,
                cmap='nipy_spectral', contour_color='black', plot_colorbar=True,
                contour_linewidth=0.25, beam_place="lr")
    fig.savefig(os.path.join(save_dir, "stack_spix_{}GHz_{}GHz_I_{}GHz.png".format(freq_high, freq_low, freq_low)), dpi=600, bbox_inches="tight")

    if not only_clean:
        # Bias of stackedspix
        fig = iplot(image_low, spix_array - true_convolved_spix_array, x=ccimages[freq_low].x, y=ccimages[freq_low].y,
                    min_abs_level=4*std, colors_mask=stack_mask, color_clim=[-0.5, 0.5], blc=blc, trc=trc,
                    beam=common_beam, close=True, colorbar_label=r"$\alpha$ bias", show_beam=True, show=False,
                    cmap='bwr', contour_color='black', plot_colorbar=True,
                    contour_linewidth=0.25, beam_place="lr")
        fig.savefig(os.path.join(save_dir, "stack_bias_spix_{}GHz_{}GHz_I_{}GHz.png".format(freq_high, freq_low, freq_low)), dpi=600, bbox_inches="tight")

        # Bias of stacked i_low
        fig = iplot(image_low, (image_low - itrue_convolved_low)/itrue_convolved_low, x=ccimages[freq_low].x, y=ccimages[freq_low].y,
                    min_abs_level=3*std, colors_mask=stack_mask, color_clim=[-0.5, 0.5], blc=blc, trc=trc,
                    beam=common_beam, close=True, colorbar_label=r"$I$ frac. bias", show_beam=True, show=False,
                    cmap='bwr', contour_color='black', plot_colorbar=True,
                    contour_linewidth=0.25, beam_place="lr")
        fig.savefig(os.path.join(save_dir, "stack_bias_I_{}GHz.png".format(freq_low)), dpi=600, bbox_inches="tight")
