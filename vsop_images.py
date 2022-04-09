import numpy as np
import os
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from vlbi_utils import find_image_std
import sys
sys.path.insert(0, '/home/ilya/github/ve/vlbi_errors')
from image import plot
from from_fits import create_image_from_fits_file


# jet_models = ("bk", "2ridges", "3ridges", "kh")
jet_models = ("2ridges",)
# images_dir = "/home/ilya/data/alpha/blind_clean/VSOP"
images_dir = "/home/ilya/Downloads/VSOP_last_SC"
true_dir = "/home/ilya/fs/sshfs/calculon/data/alpha/results/final_run/VSOP"
models_dict = {"bk": "n1", "2ridges": "n2", "3ridges": "n3", "kh": "n4"}
labels_dict = {"bk": "BK", "2ridges": "2 ridges", "3ridges": "3 ridges", "kh": "KH"}

npixels_beam = np.pi*1*1/(4*np.log(2)*0.1**2)

i_images = dict()
alpha_images = dict()
std_dict = dict()
masks_dict = dict()
color_masks_dict = dict()
true_alpha_dict = dict()

beam = (1.0, 1.0, 0)

for jet_model in jet_models:
    jet_true_dir = os.path.join(true_dir, jet_model)
    itrue_convolved_low = create_image_from_fits_file(os.path.join(jet_true_dir, "convolved_true_jet_model_i_rotated_1.6.fits")).image
    itrue_convolved_high = create_image_from_fits_file(os.path.join(jet_true_dir, "convolved_true_jet_model_i_rotated_4.8.fits")).image
    true_convolved_spix_array = np.log(itrue_convolved_low/itrue_convolved_high)/np.log(1.6/4.8)

    ccimage_low = create_image_from_fits_file(os.path.join(images_dir, "1.6_{}_unif_circbeam_cc.fits".format(models_dict[jet_model])))
    image_low = create_image_from_fits_file(os.path.join(images_dir, "1.6_{}_unif_circbeam_cc.fits".format(models_dict[jet_model]))).image
    image_high = create_image_from_fits_file(os.path.join(images_dir, "4.8_{}_unif_circbeam_cc.fits".format(models_dict[jet_model]))).image
    alpha = np.log(image_low/image_high)/np.log(1.6/4.8)
    std_low = find_image_std(image_low, beam_npixels=npixels_beam)
    std_high = find_image_std(image_high, beam_npixels=npixels_beam)
    mask_low = image_low < 3*std_low
    mask_high = image_high < 3*std_high
    common_imask = np.logical_or(mask_low, mask_high)

    color_masks_dict[jet_model] = common_imask
    i_images[jet_model] = image_low
    alpha_images[jet_model] = alpha
    std_dict[jet_model] = std_low
    masks_dict[jet_model] = mask_low
    true_alpha_dict[jet_model] = true_convolved_spix_array


blc = (470, 470)
trc = (830, 650)
fig, axes = plt.subplots(len(jet_models), 1, figsize=(int(4*len(jet_models)), 10), sharey=True, sharex=True)
plt.subplots_adjust(hspace=0, wspace=0)


# plot(i_images["bk"], alpha_images["bk"]-true_alpha_dict["bk"], x=ccimage_low.x, y=ccimage_low.y,
#      min_abs_level=2*std_dict["bk"], colors_mask=masks_dict["bk"],
#      color_clim=[-0.5, 0.5], blc=blc, trc=trc,
#      beam=beam, colorbar_label=r"$\alpha$ bias", show_beam=True,
#      cmap='coolwarm', contour_color='black', plot_colorbar=True,
#      contour_linewidth=0.25, beam_place="lr", close=False, show=False,
#      axes=axes[0], show_xlabel_on_current_axes=False, show_ylabel_on_current_axes=True)
# plot(i_images["bk"], alpha_images["bk"], x=ccimage_low.x, y=ccimage_low.y,
#      min_abs_level=5*std_dict["bk"], colors_mask=masks_dict["bk"],
#      color_clim=[-1, 0], blc=blc, trc=trc,
#      beam=beam_low, colorbar_label=r"$\alpha$", show_beam=True,
#      cmap='nipy_spectral', contour_color='black', plot_colorbar=True,
#      contour_linewidth=0.25, beam_place="lr", close=False, show=False,
#      axes=axes[0], show_xlabel_on_current_axes=False, show_ylabel_on_current_axes=True)

plot(i_images["2ridges"], alpha_images["2ridges"],#-true_alpha_dict["2ridges"],
     x=ccimage_low.x, y=ccimage_low.y,
     min_abs_level=3*std_dict["2ridges"], colors_mask=color_masks_dict["2ridges"],
     color_clim=[-1., 1], blc=blc, trc=trc,
     beam=beam, colorbar_label=r"$\alpha$", show_beam=True,
     cmap='bwr', contour_color='black', plot_colorbar=True,
     contour_linewidth=0.25, beam_place="lr", close=False, show=False,
     axes=axes, show_xlabel_on_current_axes=True, show_ylabel_on_current_axes=True)
# plot(i_images["2ridges"], alpha_images["2ridges"], x=ccimage_low.x, y=ccimage_low.y,
#      min_abs_level=5*std_dict["bk"], colors_mask=masks_dict["2ridges"],
#      color_clim=[-1, 0], blc=blc, trc=trc,
#      beam=beam_low, colorbar_label=r"$\alpha$", show_beam=True,
#      cmap='nipy_spectral', contour_color='black', plot_colorbar=True,
#      contour_linewidth=0.25, beam_place="lr", close=False, show=False,
#      axes=axes[1], show_xlabel_on_current_axes=False, show_ylabel_on_current_axes=True)

# plot(i_images["3ridges"], alpha_images["3ridges"]-true_alpha_dict["3ridges"], x=ccimage_low.x, y=ccimage_low.y,
#      min_abs_level=2*std_dict["bk"], colors_mask=masks_dict["3ridges"],
#      color_clim=[-0.5, 0.5], blc=blc, trc=trc,
#      beam=beam, colorbar_label=r"$\alpha$ bias", show_beam=True,
#      cmap='coolwarm', contour_color='black', plot_colorbar=True,
#      contour_linewidth=0.25, beam_place="lr", close=False, show=False,
#      axes=axes[2], show_xlabel_on_current_axes=True, show_ylabel_on_current_axes=True)
# plot(i_images["3ridges"], alpha_images["3ridges"], x=ccimage_low.x, y=ccimage_low.y,
#      min_abs_level=5*std_dict["bk"], colors_mask=masks_dict["3ridges"],
#      color_clim=[-1, 0], blc=blc, trc=trc,
#      beam=beam_low, colorbar_label=r"$\alpha$ bias", show_beam=True,
#      cmap='nipy_spectral', contour_color='black', plot_colorbar=True,
#      contour_linewidth=0.25, beam_place="lr", close=False, show=False,
#      axes=axes[2], show_xlabel_on_current_axes=True, show_ylabel_on_current_axes=True)

if len(jet_models) == 4:
    plot(i_images["kh"], alpha_images["kh"]-true_alpha_dict["kh"], x=ccimage_low.x, y=ccimage_low.y,
         min_abs_level=2*std_dict["kh"], colors_mask=masks_dict["kh"],
         color_clim=[-0.5, 0.5], blc=blc, trc=trc,
         beam=beam, colorbar_label=r"$\alpha$ bias", show_beam=True,
         cmap='coolwarm', contour_color='black', plot_colorbar=True,
         contour_linewidth=0.25, beam_place="lr", close=False, show=False,
         axes=axes[3], show_xlabel_on_current_axes=True, show_ylabel_on_current_axes=True)
    # plot(i_images["kh"], alpha_images["kh"], x=ccimage_low.x, y=ccimage_low.y,
    #      min_abs_level=5*std_dict["kh"], colors_mask=masks_dict["kh"],
    #      color_clim=[-1, 0], blc=blc, trc=trc,
    #      beam=beam_low, colorbar_label=r"$\alpha$ bias", show_beam=True,
    #      cmap='nipy_spectral', contour_color='black', plot_colorbar=True,
    #      contour_linewidth=0.25, beam_place="lr", close=False, show=False,
    #      axes=axes[3], show_xlabel_on_current_axes=True, show_ylabel_on_current_axes=True)

# for i in range(len(jet_models)):
#     axes[i].text(5, 15, labels_dict[jet_models[i]], fontsize="x-large")
# fig.savefig("/home/ilya/fs/sshfs/calculon/data/alpha/results/final_run/VSOP/alpha_biases_orig_beam.png", bbox_inches="tight", dpi=600)
fig.savefig("/home/ilya/Downloads/VSOP_last_SC/alpha_orig_beam.png", bbox_inches="tight", dpi=600)
plt.show()