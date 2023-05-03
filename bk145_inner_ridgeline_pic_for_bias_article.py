import matplotlib.pyplot as plt
import numpy as np
from vlbi_utils import find_image_std
import sys
sys.path.insert(0, '/home/ilya/github/ve/vlbi_errors')
from image import plot as iplot
from from_fits import create_clean_image_from_fits_file


# ccfits = "/home/ilya/Downloads/model_cc_i_15.4.fits"
ccfits = "/home/ilya/Documents/EVN2022/data/model_cc_i_15.4_0.12pc.fits"
ccimage = create_clean_image_from_fits_file(ccfits)
beam = ccimage.beam
beam = (beam[0], beam[1], np.rad2deg(beam[2]))

# figsize = (12, 10)
# fig, axes = plt.subplots(2, 1, figsize=figsize, sharey=True, sharex=True)
# plt.subplots_adjust(hspace=0, wspace=0)
blc = (480, 480)
trc = (750, 630)
fig = iplot(colors=1000*ccimage.image, x=ccimage.x, y=ccimage.y,
            log_color=True, interp="gaussian", dynamic_range=1e+03,
            blc=blc, trc=trc,
            beam=beam, colorbar_label=r"$\log_{10} I$, mJy/beam", show_beam=True,
            cmap='hot', plot_colorbar=True,
            beam_place="lr", close=False, show=False)
            # ,axes=axes[0],
            # show_xlabel_on_current_axes=False, show_ylabel_on_current_axes=True)
plt.savefig("/home/ilya/Dropbox/papers/alpha_bias/BK145_2ridge_model_15GHz_CLEAN_colors.pdf", bbox_inches="tight", dpi=600)
plt.close(fig)

# sys.exit(0)
blc = (460, 460)
trc = (950, 690)
mapsize = (1024, 0.1)
npixels_beam = np.pi*beam[0]*beam[1]/(4*np.log(2)*mapsize[1]**2)
std = find_image_std(ccimage.image, beam_npixels=npixels_beam)
fig = iplot(ccimage.image, x=ccimage.x, y=ccimage.y,
            min_abs_level=3*std, blc=blc, trc=trc, beam=beam, close=True, show_beam=True, show=False,
            contour_color='black', beam_place="lr", contour_linewidth=0.25,
            plot_colorbar=False)
            # ,axes=axes[1],
            # show_xlabel_on_current_axes=True, show_ylabel_on_current_axes=True, plot_colorbar=False)
fig.savefig("/home/ilya/Dropbox/papers/alpha_bias/BK145_2ridge_model_15GHz_CLEAN_contours.pdf", dpi=600, bbox_inches="tight")
plt.close(fig)
