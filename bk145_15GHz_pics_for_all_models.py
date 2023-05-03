import os
import matplotlib.pyplot as plt
import numpy as np
from vlbi_utils import find_image_std
import sys
sys.path.insert(0, '/home/ilya/github/ve/vlbi_errors')
from image import plot as iplot
from from_fits import create_clean_image_from_fits_file


# Full width
blc = (430, 460)
trc = (970, 710)
jet_models = ("bk", "2ridges", "3ridges", "kh")
labels_dict = {"bk": "BK", "2ridges": "2 ridges", "3ridges": "3 ridges", "kh": "KH"}
figsize = (16, 10)
fig, axes = plt.subplots(4, 1, figsize=figsize, sharey=True, sharex=True)
plt.subplots_adjust(hspace=0, wspace=0)


for i, jet_model in enumerate(jet_models):

    save_dir = os.path.join("/home/ilya/fs/sshfs/calculon/data/alpha/results/final_run/BK145/orig_beam/uv_clipping/fix_beam_bpa/revision2", jet_model)
    ccimage = create_clean_image_from_fits_file(os.path.join(save_dir, "model_cc_i_8.1.fits"))
    beam = ccimage.beam
    beam = (beam[0], beam[1], np.rad2deg(beam[2]))

    mapsize = (1024, 0.1)
    npixels_beam = np.pi*beam[0]*beam[1]/(4*np.log(2)*mapsize[1]**2)
    std = find_image_std(ccimage.image, beam_npixels=npixels_beam)

    if i == 3:
        show_xlabel_on_current_axes = True
    else:
        show_xlabel_on_current_axes = False

    iplot(ccimage.image, x=ccimage.x, y=ccimage.y, min_abs_level=3*std, blc=blc, trc=trc,
          beam=beam, show_beam=True, contour_color='black', plot_colorbar=False,
          contour_linewidth=0.25, beam_place="lr", close=False, show=False,
          axes=axes[i], show_xlabel_on_current_axes=show_xlabel_on_current_axes, show_ylabel_on_current_axes=True)
    axes[i].text(5, 15, labels_dict[jet_models[i]], fontsize="x-large")

axes[i].invert_xaxis()


fig.savefig("/home/ilya/Dropbox/papers/alpha_bias/BK145_all_models_8GHz_CLEAN_contours.pdf", dpi=600, bbox_inches="tight")
plt.close(fig)