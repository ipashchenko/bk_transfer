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


jet_model = "bk"
# jet_model = "2ridges"
# jet_model = "3ridges"
# jet_model = "kh"
data_origin = "mojave"
# data_origin = "bk145"
# data_origin = "vsop"
# data_origin = "vlba"

n_mc = 30

# Saving intermediate files
if data_origin == "mojave":
    save_dir = os.path.join("/home/ilya/data/alpha/results/MOJAVE", jet_model)
elif data_origin == "bk145":
    save_dir = os.path.join("/home/ilya/data/alpha/results/BK145", jet_model)
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
rot_angle_deg = -107.0

# Scale model image to obtain ~ 3 Jy
scale = 1.0

# Multiplicative factor for noise added to model visibilities.
noise_scale_factor = 1.0

# Common size of the map and pixel size (mas)
common_mapsize = (1024, 0.1)

if data_origin == "bk145":
    # Low freq
    # FIXME: Create U-template beam
    # template_x_ccimage = create_clean_image_from_fits_file("/home/ilya/data/alpha/BK145/X_template_beam.fits")
    # common_beam = template_x_ccimage.beam
    common_beam = (1.6, 1.6, 0)

elif data_origin == "mojave":
    template_ccimage = {8.1: "/home/ilya/data/alpha/MOJAVE/template_cc_i_8.1.fits",
                        15.4: "/home/ilya/data/alpha/MOJAVE/template_cc_i_15.4.fits"}
    template_ccimage = create_clean_image_from_fits_file(template_ccimage[8.1])
    common_beam = template_ccimage.beam

elif data_origin == "vsop":
    template_ccimages = {1.6: create_clean_image_from_fits_file("/home/ilya/data/alpha/VSOP/template_l_beam.fits"),
                         4.8: create_clean_image_from_fits_file("/home/ilya/data/alpha/VSOP/template_c_beam.fits")}
    # common_beam = template_ccimages[1.6].beam
    common_beam = (1.0, 1.0, 0)

elif data_origin == "vlba":
    common_mapsize = (4096, 0.025)
    common_beam = (0.5, 0.5, 0)



freq_high = max(freqs_obs_ghz)
freq_low = min(freqs_obs_ghz)


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
color_clim = [-1.5, 0.5]

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
color_clim = [-1.0, 0.0]
fig = iplot(ipol_arrays[freq_low], spix_array, x=ccimages[freq_low].x, y=ccimages[freq_low].y,
            min_abs_level=3*std_dict[freq_low], colors_mask=common_imask, color_clim=color_clim, blc=blc, trc=trc,
            beam=common_beam, close=True, colorbar_label=r"$\alpha$", show_beam=True, show=False,
            cmap='nipy_spectral', contour_color='black', plot_colorbar=True,
            contour_linewidth=0.25, beam_place="lr")
fig.savefig(os.path.join(save_dir, "spix_{}GHz_{}GHz_I_{}GHz.png".format(freq_high, freq_low, freq_low)), dpi=600, bbox_inches="tight")

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
    color_clim = [-1.0, 0.0]
    fig = iplot(image_low, spix_array, x=ccimages[freq_low].x, y=ccimages[freq_low].y,
                min_abs_level=4*std, colors_mask=stack_mask, color_clim=color_clim, blc=blc, trc=trc,
                beam=common_beam, close=True, colorbar_label=r"$\alpha$", show_beam=True, show=False,
                cmap='nipy_spectral', contour_color='black', plot_colorbar=True,
                contour_linewidth=0.25, beam_place="lr")
    fig.savefig(os.path.join(save_dir, "stack_spix_{}GHz_{}GHz_I_{}GHz.png".format(freq_high, freq_low, freq_low)), dpi=600, bbox_inches="tight")

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