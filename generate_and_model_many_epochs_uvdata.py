import os
import sys
import numpy as np
from vlbi_utils import find_image_std, find_bbox, downscale_uvdata_by_freq
sys.path.insert(0, '../ve/vlbi_errors')
from uv_data import UVData
from spydiff import clean_difmap, modelfit_difmap, import_difmap_model, modelfit_core_wo_extending
from image import plot as iplot
from from_fits import create_clean_image_from_fits_file
from jet_image import JetImage


data_dir = "/home/ilya/data/rfc"
source_template = "J0102+5824"
epochs = np.loadtxt(os.path.join(data_dir, "{}_times.txt".format(source_template)))
z = 0.5
n_along = 1024
n_across = 256
lg_pixsize_min = -2.0
lg_pixsize_max = -1.0

rot_angle_deg = 0.0
freqs_ghz = [2.4, 8.1]
# Directory to save files
save_dir = "/home/ilya/data/rfc/results"
# Some template UVFITS with full polarization. Its uv-coverage and noise will be used while creating fake data
template_uvfits = {2.4: "/home/ilya/data/rfc/J0102+5824/J0102+5824_S_2017_10_21_pus_vis.fits",
                   8.1: "/home/ilya/data/rfc/J0102+5824/J0102+5824_X_2017_10_21_pus_vis.fits"}
# Multiplicative factor for noise added to model visibilities.
noise_scale_factor = 1.0
# CLEAN image parameters
# FIXME: Plavin+ used (2048, 0.05)
mapsizes = {2.4: (2048, 0.05,), 8.1: (2048, 0.05,)}
# Directory with C++ generated txt-files with model images
jetpol_run_directory = "/home/ilya/github/bk_transfer/Release"
path_to_script = "/home/ilya/github/bk_transfer/scripts/script_clean_rms"


core_positions = dict()
core_positions_err = dict()
core_fluxes = dict()
core_fluxes_err = dict()

for freq_ghz in freqs_ghz:
    core_positions[freq_ghz] = list()
    core_positions_err[freq_ghz] = list()
    core_fluxes[freq_ghz] = list()
    core_fluxes_err[freq_ghz] = list()
    for epoch in epochs:
        uvdata = UVData(template_uvfits[freq_ghz])
        downscale_uvdata_by_freq_flag = downscale_uvdata_by_freq(uvdata)
        noise = uvdata.noise(average_freq=False, use_V=False)
        # If one needs to decrease the noise this is the way to do it
        for baseline, baseline_noise_std in noise.items():
            noise.update({baseline: noise_scale_factor*baseline_noise_std})

        jm = JetImage(z=z, n_along=n_along, n_across=n_across,
                      lg_pixel_size_mas_min=lg_pixsize_min, lg_pixel_size_mas_max=lg_pixsize_max,
                      jet_side=True, rot=np.deg2rad(rot_angle_deg))

        jm.load_image_stokes("I", "{}/jet_image_{}_{}_{}.txt".format(jetpol_run_directory, "i", freq_ghz, epoch), scale=1.0)
        uvdata.zero_data()
        uvdata.substitute([jm])
        uvdata.noise_add(noise)
        uvdata.save(os.path.join(save_dir, "template.uvf"), rewrite=True,  downscale_by_freq=False)

        beam_fractions = np.linspace(0.5, 1.5, 11)
        results = modelfit_core_wo_extending("template.uvf",
                                             beam_fractions,
                                             path=save_dir,
                                             mapsize_clean=mapsizes[freq_ghz],
                                             path_to_script=path_to_script,
                                             niter=100,
                                             out_path=save_dir,
                                             use_brightest_pixel_as_initial_guess=True,
                                             estimate_rms=True,
                                             stokes="i",
                                             use_ell=False)

        # Flux of the core
        flux = np.median([results[frac]['flux'] for frac in beam_fractions])
        rms_flux = np.std([results[frac]['flux'] for frac in beam_fractions])

        # Position of the core
        r = np.median([np.hypot(results[frac]['ra'], results[frac]['dec']) for frac in beam_fractions])
        rms_r = np.std([np.hypot(results[frac]['ra'], results[frac]['dec']) for frac in beam_fractions])

        core_fluxes[freq_ghz].append(flux)
        core_fluxes_err[freq_ghz].append(rms_flux)
        core_positions[freq_ghz].append(r)
        core_positions_err[freq_ghz].append(rms_r)

        # Post-fit rms in the core region
        postfit_rms = np.median([results[frac]['rms'] for frac in beam_fractions])

        print("Core flux : {:.2f}+/-{:.2f} Jy".format(flux, rms_flux))
        print("Core position : {:.2f}+/-{:.2f} Jy".format(r, rms_r))

        outfname = "model_cc_i.fits"
        if os.path.exists(os.path.join(save_dir, outfname)):
            os.unlink(os.path.join(save_dir, outfname))
        clean_difmap(fname="template.uvf", path=save_dir,
                     outfname=outfname, outpath=save_dir, stokes="i",
                     mapsize_clean=mapsizes[freq_ghz], path_to_script=path_to_script,
                     show_difmap_output=True,
                     dfm_model=os.path.join(save_dir, "model_cc_i.mdl"))

        ccimage = create_clean_image_from_fits_file(os.path.join(save_dir, "model_cc_i.fits"))
        ipol = ccimage.image
        beam = ccimage.beam
        # Number of pixels in beam
        npixels_beam = np.pi*beam[0]*beam[1]/(4*np.log(2)*mapsize[1]**2)

        std = find_image_std(ipol, beam_npixels=npixels_beam)
        print("IPOL image std = {} mJy/beam".format(1000*std))
        blc, trc = find_bbox(ipol, level=4*std, min_maxintensity_mjyperbeam=10*std,
                             min_area_pix=4*npixels_beam, delta=10)
        print("blc = ", blc, ", trc = ", trc)
        if blc[0] == 0: blc = (blc[0]+1, blc[1])
        if blc[1] == 0: blc = (blc[0], blc[1]+1)
        if trc[0] == ipol.shape: trc = (trc[0]-1, trc[1])
        if trc[1] == ipol.shape: trc = (trc[0], trc[1]-1)

        # IPOL contours
        fig = iplot(ipol, x=ccimage.x, y=ccimage.y,
                    min_abs_level=4*std, blc=blc, trc=trc, beam=beam, close=True, show_beam=True, show=False,
                    contour_color='gray', contour_linewidth=0.25)
        fig.savefig(os.path.join(save_dir, "observed_i_{}_{}.png".format(freq_ghz, epoch)), dpi=600, bbox_inches="tight")
