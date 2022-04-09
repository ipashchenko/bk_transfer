import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/ilya/github/ve/vlbi_errors')
sys.path.insert(0, '/home/ilya/github/bk_transfer')
from vlbi_utils import (find_image_std, find_bbox, convert_difmap_model_file_to_CCFITS, rotate_difmap_model,
                        convert_blc_trc)
from from_fits import create_clean_image_from_fits_file, create_image_from_fits_file
from image import plot as iplot


deep = True

if deep:
    # data_dir = "/home/ilya/data/alpha/blind_clean/MOJAVE/deep/kh"
    data_dir = "/home/ilya/data/alpha/blind_clean/MOJAVE/deep/kh/nw/deep"
else:
    # data_dir = "/home/ilya/data/alpha/blind_clean/MOJAVE/deep/kh/nodeep"
    data_dir = "/home/ilya/data/alpha/blind_clean/MOJAVE/deep/kh/nw/nodeep"

true_model_dir = "/home/ilya/data/alpha/blind_clean/MOJAVE/deep/kh"
freqs_obs_ghz = [8.1, 15.4]
freq_high = max(freqs_obs_ghz)
freq_low = min(freqs_obs_ghz)
common_mapsize = (1024, 0.1)


itrue_convolved_low = create_image_from_fits_file("{}/convolved_true_jet_model_i_rotated_{}.fits".format(true_model_dir, freq_low)).image
itrue_convolved_high = create_image_from_fits_file("{}/convolved_true_jet_model_i_rotated_{}.fits".format(true_model_dir, freq_high)).image
true_convolved_spix_array = np.log(itrue_convolved_low/itrue_convolved_high)/np.log(freq_low/freq_high)


# Observed images of CLEANed artificial UV-data
# ccimages = {freq: create_clean_image_from_fits_file(os.path.join(data_dir, "model_cc_i_{}.fits".format(freq)))
#             for freq in freqs_obs_ghz}
if deep:
    ccimages = {freq: create_clean_image_from_fits_file(os.path.join(data_dir, "deep_cc_{}.fits".format(freq)))
                for freq in freqs_obs_ghz}
else:
    ccimages = {freq: create_clean_image_from_fits_file(os.path.join(data_dir, "nodeep_cc_{}.fits".format(freq)))
                for freq in freqs_obs_ghz}
ipol_low = ccimages[freq_low].image
ipol_high = ccimages[freq_high].image
beam_low = ccimages[freq_low].beam
beam_high = ccimages[freq_high].beam
common_beam = beam_low
# Number of pixels in beam
npixels_beam_low = np.pi*beam_low[0]*beam_low[1]/(4*np.log(2)*common_mapsize[1]**2)
npixels_beam_high = np.pi*beam_high[0]*beam_high[1]/(4*np.log(2)*common_mapsize[1]**2)
std_low = find_image_std(ipol_low, beam_npixels=npixels_beam_low)
std_high = find_image_std(ipol_high, beam_npixels=npixels_beam_high)
blc = (450, 450)
trc = (800, 630)

ipol_arrays = dict()
masks_dict = dict()
std_dict = dict()
for freq in freqs_obs_ghz:

    ipol = ccimages[freq].image
    ipol_arrays[freq] = ipol

    std = find_image_std(ipol, beam_npixels={freq_high: npixels_beam_high, freq_low: npixels_beam_low}[freq])
    std_dict[freq] = std
    masks_dict[freq] = ipol < 3*std

common_imask = np.logical_or.reduce([masks_dict[freq] for freq in freqs_obs_ghz])
spix_array = np.log(ipol_arrays[freq_low]/ipol_arrays[freq_high])/np.log(freq_low/freq_high)

fig = iplot(ipol_arrays[freq_low], spix_array-true_convolved_spix_array, x=ccimages[freq_low].x, y=ccimages[freq_low].y,
            min_abs_level=3*std_dict[freq_low], colors_mask=common_imask, color_clim=[-0.55, 0.55], blc=blc, trc=trc,
            beam=common_beam, close=True, colorbar_label=r"$\alpha$ bias", show_beam=True, show=False,
            cmap='coolwarm', contour_color='black', plot_colorbar=True,
            contour_linewidth=0.25, beam_place="lr")
fig.savefig(os.path.join(data_dir, "bias_spix_{}GHz_{}GHz_I_{}GHz.png".format(freq_high, freq_low, freq_low)), dpi=600,
            bbox_inches="tight")
plt.show()

fig = iplot(ipol_arrays[freq_low], spix_array, x=ccimages[freq_low].x, y=ccimages[freq_low].y,
            min_abs_level=3*std_dict[freq_low], colors_mask=common_imask, color_clim=[-1.5, 1], blc=blc, trc=trc,
            beam=common_beam, close=True, colorbar_label=r"$\alpha$", show_beam=True, show=False,
            cmap='nipy_spectral', contour_color='black', plot_colorbar=True,
            contour_linewidth=0.25, beam_place="lr")
fig.savefig(os.path.join(data_dir, "spix_{}GHz_{}GHz_I_{}GHz.png".format(freq_high, freq_low, freq_low)), dpi=600,
            bbox_inches="tight")
plt.show()


# Stacked ==============================================================================================================
n_mc = 30
images = dict()
for freq in (freq_low, freq_high):
    images[freq] = list()
# Read all MC data
for i in range(n_mc):
    print("i_mc = ", i)
    for freq in (freq_low, freq_high):
        images[freq].append(create_clean_image_from_fits_file(os.path.join(data_dir, "model_cc_i_{}_mc_{}.fits".format(freq, i+1))).image)

npixels_beam = np.pi*common_beam[0]*common_beam[1]/(4*np.log(2)*0.1**2)
image_low = np.sum(images[freq_low], axis=0)/n_mc
image_high = np.sum(images[freq_high], axis=0)/n_mc
spix_arrays = [np.log(images[freq_low][i]/images[freq_high][i])/np.log(freq_low/freq_high) for i in range(n_mc)]
std_spix = np.std(spix_arrays, axis=0)
mean_spix = np.mean(spix_arrays, axis=0)
std_low = find_image_std(image_low, beam_npixels=npixels_beam)
std_high = find_image_std(image_high, beam_npixels=npixels_beam)
stack_mask = image_low < 5*std_low
spix_array = np.log(image_low/image_high)/np.log(freq_low/freq_high)
common_imask = np.logical_or.reduce([image_low < 3*std_low, image_high < 3*std_high])

if not deep:
    np.savetxt(os.path.join(true_model_dir, "nw_stack_alpha_mask.txt"), common_imask)
    np.savetxt(os.path.join(true_model_dir, "nw_stack_alpha_stdlow.txt"), np.array([std_low]))
else:
    common_imask = np.loadtxt(os.path.join(true_model_dir, "nw_stack_alpha_mask.txt"))
    common_imask = np.array(common_imask, dtype=int)
    std_low = np.loadtxt(os.path.join(true_model_dir, "nw_stack_alpha_stdlow.txt"))
    std_low = float(std_low)


fig = iplot(image_low, spix_array, x=ccimages[freq_low].x, y=ccimages[freq_low].y,
            min_abs_level=3*std_low, colors_mask=common_imask, color_clim=[-1.5, 1], blc=blc, trc=trc,
            beam=common_beam, close=True, colorbar_label=r"$\alpha$", show_beam=True, show=False,
            cmap='nipy_spectral', contour_color='black', plot_colorbar=True,
            contour_linewidth=0.25, beam_place="lr")
fig.savefig(os.path.join(data_dir, "stack_spix_{}GHz_{}GHz_I_{}GHz.png".format(freq_high, freq_low, freq_low)),
            dpi=600, bbox_inches="tight")
plt.show()

fig = iplot(image_low, spix_array-true_convolved_spix_array, x=ccimages[freq_low].x, y=ccimages[freq_low].y,
            min_abs_level=3*std_low, colors_mask=common_imask, color_clim=[-0.55, 0.55], blc=blc, trc=trc,
            beam=common_beam, close=True, colorbar_label=r"$\alpha$ bias", show_beam=True, show=False,
            cmap='coolwarm', contour_color='black', plot_colorbar=True,
            contour_linewidth=0.25, beam_place="lr")
fig.savefig(os.path.join(data_dir, "stack_bias_spix_{}GHz_{}GHz_I_{}GHz.png".format(freq_high, freq_low, freq_low)),
            dpi=600, bbox_inches="tight")
plt.show()