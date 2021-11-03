import os
from pathlib import Path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import numpy as np
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
only_make_pics = True
# If ``only_make_pics = True``, do we need to re-CLEAN?
re_clean = False

# Make model low frequency image from high frequency one using `alpha_true`
artificial_alpha = False
# S ~ nu^{+alpha}
alpha_true = -0.5

# Use final CLEAN iterations with UVTAPER?
use_uvtaper = False

jet_model = "bk"
# jet_model = "2ridges"
# jet_model = "3ridges"
# jet_model = "kh"
if jet_model not in ("bk", "2ridges", "3ridges", "kh"):
    raise Exception
# data_origin = "mojave"
data_origin = "bk145"
# data_origin = "vsop"
if data_origin not in ("mojave", "bk145", "vsop"):
    raise Exception

# Saving intermediate files
if data_origin == "mojave":
    save_dir = os.path.join("/home/ilya/data/alpha/results/MOJAVE", jet_model)
elif data_origin == "bk145":
    save_dir = os.path.join("/home/ilya/data/alpha/results/BK145", jet_model)
elif data_origin == "vsop":
    save_dir = os.path.join("/home/ilya/data/alpha/results/VSOP", jet_model)
else:
    raise Exception("data_origin must be vsop, mojave of bk145!")
Path(save_dir).mkdir(parents=True, exist_ok=True)


# Observed frequencies of simulations
if data_origin == "mojave" or data_origin == "bk145":
    freqs_obs_ghz = [8.1, 15.4]
elif data_origin == "vsop":
    freqs_obs_ghz = [1.6, 4.8]


# -107 for M87
rot_angle_deg = -107.0

# Scale model image to obtain ~ 3 Jy
scale = 1.0

# Multiplicative factor for noise added to model visibilities.
noise_scale_factor = 1.0

# Common size of the map and pixel size (mas)
common_mapsize = (2048, 0.1)

# Common beam size (mas, mas, deg)
# Circular
common_beam = (1.56, 1.56, 0)


# C++ code run parameters
jetpol_files_directory = "/home/ilya/github/bk_transfer/Release"
z = 0.00436
n_along = 1024
n_across = 150
lg_pixel_size_mas_min = -2
lg_pixel_size_mas_max = -1.0


# path_to_script = "/home/ilya/github/bk_transfer/scripts/final_clean_nw"
path_to_script = "/home/ilya/github/bk_transfer/scripts/script_clean_rms"
path_to_scripts = {15.4: "/home/ilya/github/bk_transfer/scripts/script_clean_rms_u",
                   8.1: "/home/ilya/github/bk_transfer/scripts/script_clean_rms_x"}

if data_origin == "bk145":
    # Lesha's data
    need_downscale_uv = True
    template_uvfits_dict = {15.4: "/home/ilya/data/alpha/BK145/1228+126.U.2009_05_23C_ta60.uvf_cal",
                            8.1: "/home/ilya/data/alpha/BK145/1228+126.X.2009_05_23_ta60.uvf_cal"}
    # Low freq
    # FIXME: Create U-template beam
    template_x_ccimage = create_clean_image_from_fits_file("/home/ilya/data/alpha/BK145/X_template_beam.fits")
    common_beam = template_x_ccimage.beam

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
    common_beam = template_ccimages[1.6]
    path_to_script = "/home/ilya/data/alpha/VSOP/final_clean_vsop"

##############################################
# No need to change anything below this line #
##############################################
# Plot only jet emission and do not plot counter-jet?
jet_only = False

freq_high = max(freqs_obs_ghz)
freq_low = min(freqs_obs_ghz)

if not only_make_pics or (not only_make_pics and re_clean):
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
                jm.load_image_stokes("I", "{}/jet_image_{}_{}.txt".format(jetpol_files_directory, "i", freq_high),
                                     scale=scale*(freq_high/freq)**(-alpha_true))
                cjm.load_image_stokes("I", "{}/cjet_image_{}_{}.txt".format(jetpol_files_directory, "i", freq_high),
                                      scale=scale*(freq_high/freq)**(-alpha_true))
            else:
                jm.load_image_stokes("I", "{}/jet_image_{}_{}.txt".format(jetpol_files_directory, "i", freq), scale=scale)
                cjm.load_image_stokes("I", "{}/cjet_image_{}_{}.txt".format(jetpol_files_directory, "i", freq), scale=scale)
            js = TwinJetImage(jm, cjm)

            # Convert to difmap model format
            # TODO: Check
            js.save_image_to_difmap_format("{}/true_jet_model_i_{}.txt".format(save_dir, freq))

            # Rotate
            rotate_difmap_model("{}/true_jet_model_i_{}.txt".format(save_dir, freq),
                                "{}/true_jet_model_i_{}_rotated.txt".format(save_dir, freq),
                                PA_deg=-rot_angle_deg)
            # Convolve with beam
            convert_difmap_model_file_to_CCFITS("{}/true_jet_model_i_{}_rotated.txt".format(save_dir, freq), "I", common_mapsize,
                                                common_beam, template_uvfits_dict[freq],
                                                "{}/convolved_true_jet_model_i_rotated_{}.fits".format(save_dir, freq))
            uvdata.substitute([js])
            uvdata.noise_add(noise)
            uvdata.save(os.path.join(save_dir, "template_{}.uvf".format(freq)), rewrite=True, downscale_by_freq=need_downscale_uv)

        # If doing from scratch or only making pics, but with re-CLEANing
        if not only_make_pics or (only_make_pics and re_clean):
            outfname = "model_cc_i_{}.fits".format(freq)
            if os.path.exists(outfname):
                os.unlink(outfname)
            if use_uvtaper:
                path_to_script = path_to_scripts[freq]
            clean_difmap(fname="template_{}.uvf".format(freq), path=save_dir,
                         outfname=outfname, outpath=save_dir, stokes="i",
                         mapsize_clean=common_mapsize, path_to_script=path_to_script,
                         show_difmap_output=True,
                         beam_restore=common_beam)
                         # dfm_model=os.path.join(save_dir, "model_cc_i_{}.mdl".format(freq)))

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

# By hand
# blc = (459, 474)
# trc = (836, 654)

# bk
blc = (900, 930)
trc = (1500, 1240)

blc_low = blc
trc_low = trc
blc_high = blc
trc_high = trc

# blc = blc_low
# trc = trc_low

# I high
fig = iplot(ipol_high, x=ccimages[freq_high].x, y=ccimages[freq_high].y,
            min_abs_level=3*std_high, blc=blc_low, trc=trc_low, beam=beam_low, close=True, show_beam=True, show=False,
            contour_color='gray', contour_linewidth=0.25)
fig.savefig(os.path.join(save_dir, "observed_ipol_{}GHz.png".format(freq_high)), dpi=600, bbox_inches="tight")

# sys.exit(0)

# I high bias
fig = iplot(ipol_high, (ipol_high - itrue_convolved_high)/itrue_convolved_high,
            x=ccimages[freq_high].x, y=ccimages[freq_high].y,
            colors_mask=ipol_high < 3*std_high,
            color_clim=[-0.25, 0.25],
            min_abs_level=3*std_high, blc=blc_high, trc=trc_high, beam=beam_high, close=True, show_beam=True, show=False,
            contour_color='black', contour_linewidth=0.25, colorbar_label="I frac. bias", plot_colorbar=True, cmap='bwr')
fig.savefig(os.path.join(save_dir, "bias_ipol_{}GHz.png".format(freq_high)), dpi=600, bbox_inches="tight")

print("DEBUG")


# I low
fig = iplot(ipol_low, x=ccimages[freq_low].x, y=ccimages[freq_low].y,
            min_abs_level=3*std_low, blc=blc_low, trc=trc_low, beam=beam_low, close=True, show_beam=True, show=False,
            contour_color='gray', contour_linewidth=0.25)
fig.savefig(os.path.join(save_dir, "observed_ipol_{}GHz.png".format(freq_low)), dpi=600, bbox_inches="tight")
# I low bias
fig = iplot(ipol_low, (ipol_low - itrue_convolved_low)/itrue_convolved_low,
            x=ccimages[freq_low].x, y=ccimages[freq_low].y, colors_mask=ipol_low < 3*std_low, color_clim=[-0.25, 0.25],
            min_abs_level=3*std_low, blc=blc_low, trc=trc_low, beam=beam_low, close=True, show_beam=True, show=False,
            contour_color='black', contour_linewidth=0.25, colorbar_label="I frac. bias", plot_colorbar=True, cmap="bwr")
fig.savefig(os.path.join(save_dir, "bias_ipol_{}GHz.png".format(freq_low)), dpi=600, bbox_inches="tight")

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
fig = iplot(ipol_arrays[freq_low], true_convolved_spix_array, x=ccimages[freq_low].x, y=ccimages[freq_low].y,
            min_abs_level=3*std_dict[freq_low], colors_mask=common_imask, color_clim=color_clim, blc=blc, trc=trc,
            beam=common_beam, close=True, colorbar_label=r"$\alpha$", show_beam=True, show=False,
            cmap='bwr', contour_color='black', plot_colorbar=True,
            contour_linewidth=0.25)
fig.savefig(os.path.join(save_dir, "true_conv_spix_{}GHz_{}GHz_I_{}GHz.png".format(freq_high, freq_low, freq_low)),
            dpi=600, bbox_inches="tight")

# Observed spix
if artificial_alpha:
    color_clim = [alpha_true-0.5, alpha_true+0.5]
else:
    color_clim = [-1.5, 0.5]
fig = iplot(ipol_arrays[freq_low], spix_array, x=ccimages[freq_low].x, y=ccimages[freq_low].y,
            min_abs_level=3*std_dict[freq_low], colors_mask=common_imask, color_clim=color_clim, blc=blc, trc=trc,
            beam=common_beam, close=True, colorbar_label=r"$\alpha$", show_beam=True, show=False,
            cmap='bwr', contour_color='black', plot_colorbar=True,
            contour_linewidth=0.25)
fig.savefig(os.path.join(save_dir, "spix_{}GHz_{}GHz_I_{}GHz.png".format(freq_high, freq_low, freq_low)), dpi=600, bbox_inches="tight")

# Bias spix
fig = iplot(ipol_arrays[freq_low], spix_array - true_convolved_spix_array, x=ccimages[freq_low].x, y=ccimages[freq_low].y,
            min_abs_level=3*std_dict[freq_low], colors_mask=common_imask, color_clim=[-0.5, 0.5], blc=blc, trc=trc,
            beam=common_beam, close=True, colorbar_label=r"$\alpha$", show_beam=True, show=False,
            cmap='bwr', contour_color='black', plot_colorbar=True,
            contour_linewidth=0.25)
fig.savefig(os.path.join(save_dir, "bias_spix_{}GHz_{}GHz_I_{}GHz.png".format(freq_high, freq_low, freq_low)), dpi=600, bbox_inches="tight")