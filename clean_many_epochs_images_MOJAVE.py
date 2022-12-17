# This script was used to generate maps for Vlad kinematic tests
import os
import shutil
import sys
import astropy.io.fits as pf
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
import astropy.units as u
from vlbi_utils import find_image_std, find_bbox, downscale_uvdata_by_freq
sys.path.insert(0, '/home/ilya/github/ve/vlbi_errors')
from uv_data import UVData
from spydiff import clean_difmap, modelfit_difmap, import_difmap_model, modelfit_core_wo_extending
from image import plot as iplot
from from_fits import create_clean_image_from_fits_file
from jet_image import JetImage
import matplotlib
matplotlib.use("Qt5Agg")
from astropy.io.fits.verify import VerifyWarning
import warnings
warnings.simplefilter('ignore', category=VerifyWarning)


def change_date_ccfits(ccfits, n_days):
    t0 = Time('2000-01-01', format='iso', out_subfmt='date')
    t = t0 + int(n_days)*u.day
    # pf.setval(ccfits, "DATE-OBS", value=t.iso)
    hdulist = pf.open(ccfits)
    hdulist[0].header.set("DATE-OBS", value=t.iso)
    hdulist.writeto(ccfits, output_verify='ignore', overwrite=True)


# convert -delay 10 -loop 0 `ls -tr raw_nu*.png` animation_raw_nu.gif
# convert -delay 10 -loop 0 `ls -tr observed_i_u_*.png` animation_clean.gif
# data_dir = "/home/ilya/data/WISE/data"
# source_template = "test"
z = 1.0
n_along = 400
n_across = 80
match_resolution = False
# if match_resolution:
#     lg_pixsize_min = {2.3: -2.5, 8.6: -2.5-np.log10(8.6/2.3)}
#     lg_pixsize_max = {2.3: -0.5, 8.6: -0.5-np.log10(8.6/2.3)}
# else:
#     lg_pixsize_min = {2.3: -2.5, 8.6: -2.5}
#     lg_pixsize_max = {2.3: -0.5, 8.6: -0.5}


lg_pixsize_min = {15.4: -2.5}
lg_pixsize_max = {15.4: -0.5}

epochs = np.linspace(0.0, 20*360, 241)

rot_angle_deg = -90.0
freqs_ghz = [15.4]
freq_names = {15.4: "u"}
# Directory to save files
save_dir = "/home/ilya/data/WISE/new_results/0.33_0_300_0.2_smallnoise"
if not os.path.exists(save_dir):
    print("Creating directory ", save_dir)
    os.mkdir(save_dir)
# Some template UVFITS with full polarization. Its uv-coverage and noise will be used while creating fake data
template_uvfits = {15.4: "/home/ilya/data/WISE/data/1458+718.u.2006_09_06.uvf"}
bmaj_mas = 0.8072557389469439
bmin_mas = 0.6743727088663318
beam_mas = np.sqrt(bmin_mas*bmaj_mas)
# Multiplicative factor for noise added to model visibilities.
noise_scale_factor = 0.01
# CLEAN image parameters
mapsizes = {15.4: (1024, 0.1,)}
# Directory with C++ generated txt-files with model images
jetpol_run_directory = "/home/ilya/github/bk_transfer/Release"
path_to_script = "/home/ilya/github/bk_transfer/scripts/script_clean_rms"

only_plot_raw = False
reclean = True


std = None
blc = None
trc = None
for freq_ghz in freqs_ghz:
    # if freq_ghz != 8.6:
    #     continue
    for epoch in epochs:
        uvdata = UVData(template_uvfits[freq_ghz])
        downscale_uvdata_by_freq_flag = downscale_uvdata_by_freq(uvdata)
        noise = uvdata.noise(average_freq=False, use_V=False)
        # If one needs to decrease the noise this is the way to do it
        for baseline, baseline_noise_std in noise.items():
            noise.update({baseline: noise_scale_factor*baseline_noise_std})

        jm = JetImage(z=z, n_along=n_along, n_across=n_across,
                      lg_pixel_size_mas_min=lg_pixsize_min[freq_ghz], lg_pixel_size_mas_max=lg_pixsize_max[freq_ghz],
                      jet_side=True, rot=np.deg2rad(rot_angle_deg))

        image_file = "{}/jet_image_{}_{}_{:.1f}.txt".format(jetpol_run_directory, "i", freq_names[freq_ghz], epoch)
        image = np.loadtxt(image_file)
        print(f"Flux = {np.sum(image)}")


        # Plot raw image
        fig, axes = plt.subplots(1, 1)
        axes.matshow(image, cmap="inferno", aspect="auto")
        axes.set_xlabel("Along, nu pixels")
        axes.set_ylabel("Across, nu pixels")
        axes.annotate("{:05.1f} months".format(epoch/30), xy=(0.03, 0.9), xycoords="axes fraction", color="pink",
                      weight='bold', ha='left', va='center', size=20)
        fig.savefig(os.path.join(save_dir, "raw_nupixel_{}_{:.1f}.png".format(freq_names[freq_ghz], epoch)))
        plt.close(fig)

        jm.load_image_stokes("I", "{}/jet_image_{}_{}_{:.1f}.txt".format(jetpol_run_directory, "i", freq_names[freq_ghz], epoch), scale=1.0)
        # jm.plot(outfile=os.path.join(save_dir, "raw_{}_{:.1f}.png".format(freq_names[freq_ghz], epoch)),
        #         zoom_fr=0.2, cmap="inferno")
        if only_plot_raw:
            continue

        uvdata.zero_data()
        uvdata.substitute([jm])
        uvdata.noise_add(noise)
        uvdata.save(os.path.join(save_dir, "template.uvf"), rewrite=True,  downscale_by_freq=False)

        if reclean:

            outfname = "model_cc_i_{}_{:.1f}.fits".format(freq_names[freq_ghz], epoch)
            if os.path.exists(os.path.join(save_dir, outfname)):
                os.unlink(os.path.join(save_dir, outfname))
            clean_difmap(fname="template.uvf", path=save_dir,
                         outfname=outfname, outpath=save_dir, stokes="i",
                         mapsize_clean=mapsizes[freq_ghz], path_to_script=path_to_script,
                         show_difmap_output=True,
                         beam_restore=(beam_mas, beam_mas, 0))
            change_date_ccfits(os.path.join(save_dir, outfname), int(epoch))

        ccimage = create_clean_image_from_fits_file(os.path.join(save_dir, "model_cc_i_{}_{:.1f}.fits".format(freq_names[freq_ghz], epoch)))
        ipol = ccimage.image
        beam = ccimage.beam
        # Number of pixels in beam
        npixels_beam = np.pi*beam[0]*beam[1]/(4*np.log(2)*mapsizes[freq_ghz][1]**2)

        if std is None:
            std = find_image_std(ipol, beam_npixels=npixels_beam)
            print("IPOL image std = {} mJy/beam".format(1000*std))
            blc, trc = find_bbox(ipol, level=4*std, min_maxintensity_mjyperbeam=20*std,
                                 min_area_pix=10*npixels_beam, delta=5)
            print("blc = ", blc, ", trc = ", trc)
            if blc[0] == 0: blc = (blc[0]+1, blc[1])
            if blc[1] == 0: blc = (blc[0], blc[1]+1)
            if trc[0] == ipol.shape: trc = (trc[0]-1, trc[1])
            if trc[1] == ipol.shape: trc = (trc[0], trc[1]-1)

        # IPOL contours
        fig = iplot(ipol, x=ccimage.x, y=ccimage.y,
                    min_abs_level=4*std, blc=blc, trc=trc, beam=beam, close=True, show_beam=True, show=False,
                    contour_color='gray', contour_linewidth=0.25, plot_colorbar=False)
        axes = fig.get_axes()[0]
        axes.annotate("{:05.1f} months".format(epoch/30), xy=(0.03, 0.9), xycoords="axes fraction", color="gray",
                      weight='bold', ha='left', va='center', size=10)

        fig.savefig(os.path.join(save_dir, "observed_i_{}_{:.1f}.png".format(freq_names[freq_ghz], epoch)),
                    bbox_inches="tight", dpi=200)
