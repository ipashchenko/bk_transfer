import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
from vlbi_utils import find_image_std, find_bbox, downscale_uvdata_by_freq
sys.path.insert(0, '/home/ilya/github/ve/vlbi_errors')
from uv_data import UVData
from spydiff import clean_difmap, modelfit_difmap, import_difmap_model, modelfit_core_wo_extending
from image import plot as iplot
from from_fits import create_clean_image_from_fits_file
from jet_image import JetImage
import matplotlib
matplotlib.use("Qt5Agg")


data_dir = "/home/ilya/data/rfc"
source_template = "J0102+5824"
epochs = np.loadtxt(os.path.join(data_dir, "{}_times.txt".format(source_template)))
z = 1.0
n_along = 400
n_across = 80
match_resolution = False
if match_resolution:
    lg_pixsize_min = {2.3: -2.5, 8.6: -2.5-np.log10(8.6/2.3)}
    lg_pixsize_max = {2.3: -0.5, 8.6: -0.5-np.log10(8.6/2.3)}
else:
    lg_pixsize_min = {2.3: -2.5, 8.6: -2.5}
    lg_pixsize_max = {2.3: -0.5, 8.6: -0.5}


epochs = [300.0]
epochs = np.linspace(0, 10*360, 30)

rot_angle_deg = -90.0
freqs_ghz = [2.3, 8.6]
freq_names = {2.3: "S", 8.6: "X"}
# Directory to save files
save_dir = "/home/ilya/data/rfc/results"
# Some template UVFITS with full polarization. Its uv-coverage and noise will be used while creating fake data
template_uvfits = {2.3: "/home/ilya/data/rfc/J0102+5824/J0102+5824_S_2017_10_21_pus_vis.fits",
                   8.6: "/home/ilya/data/rfc/J0102+5824/J0102+5824_X_2017_10_21_pus_vis.fits"}
# Multiplicative factor for noise added to model visibilities.
noise_scale_factor = 1.0
# CLEAN image parameters
# FIXME: Plavin+ used (2048, 0.05)
mapsizes = {2.3: (1024, 0.2,), 8.6: (1024, 0.2,)}
# Directory with C++ generated txt-files with model images
jetpol_run_directory = "/home/ilya/github/bk_transfer/Release"
path_to_script = "/home/ilya/github/bk_transfer/scripts/script_clean_rms"

do_plots = True


core_positions = dict()
core_positions_err = dict()
core_fluxes = dict()
core_fluxes_err = dict()

for freq_ghz in freqs_ghz:
    # if freq_ghz != 8.6:
    #     continue
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
                      lg_pixel_size_mas_min=lg_pixsize_min[freq_ghz], lg_pixel_size_mas_max=lg_pixsize_max[freq_ghz],
                      jet_side=True, rot=np.deg2rad(rot_angle_deg))

        jm.load_image_stokes("I", "{}/jet_image_{}_{}_{:.1f}.txt".format(jetpol_run_directory, "i", freq_names[freq_ghz], epoch), scale=1.0)
        uvdata.zero_data()
        uvdata.substitute([jm])
        uvdata.noise_add(noise)
        uvdata.save(os.path.join(save_dir, "template.uvf"), rewrite=True,  downscale_by_freq=True)

        modelfit_difmap("template.uvf", "in4.mdl", "out4.mdl", niter=200, stokes='i',
                        path=save_dir, mdl_path=save_dir, out_path=save_dir,
                        show_difmap_output=True,
                        save_dirty_residuals_map=False,
                        dmap_name=None, dmap_size=(1024, 0.1))

        shutil.copy(os.path.join(save_dir, "out4.mdl"),
                    os.path.join(save_dir, "out4_{:.1f}.mdl".format(epoch)))
        components = import_difmap_model("out4.mdl", save_dir)
        # Find closest to phase center component
        comps = sorted(components, key=lambda x: np.hypot(x.p[1], x.p[2]))
        core = comps[0]
        # core = components[0]

        # Flux of the core
        flux = core.p[0]

        # Position of the core
        r = np.hypot(core.p[1], core.p[2])

        core_fluxes[freq_ghz].append(flux)
        core_positions[freq_ghz].append(r)

        if do_plots:

            outfname = "model_cc_i.fits"
            if os.path.exists(os.path.join(save_dir, outfname)):
                os.unlink(os.path.join(save_dir, outfname))

            clean_difmap(fname="template.uvf", path=save_dir,
                         outfname=outfname, outpath=save_dir, stokes="i",
                         mapsize_clean=mapsizes[freq_ghz], path_to_script=path_to_script,
                         show_difmap_output=False,
                         dfm_model=os.path.join(save_dir, "out.mdl"))

            ccimage = create_clean_image_from_fits_file(os.path.join(save_dir, "model_cc_i.fits"))
            ipol = ccimage.image
            beam = ccimage.beam
            # Number of pixels in beam
            npixels_beam = np.pi*beam[0]*beam[1]/(4*np.log(2)*mapsizes[freq_ghz][1]**2)

            std = find_image_std(ipol, beam_npixels=npixels_beam)
            print("IPOL image std = {} mJy/beam".format(1000*std))
            blc, trc = find_bbox(ipol, level=4*std, min_maxintensity_mjyperbeam=20*std,
                                 min_area_pix=10*npixels_beam, delta=10)
            print("blc = ", blc, ", trc = ", trc)
            if blc[0] == 0: blc = (blc[0]+1, blc[1])
            if blc[1] == 0: blc = (blc[0], blc[1]+1)
            if trc[0] == ipol.shape: trc = (trc[0]-1, trc[1])
            if trc[1] == ipol.shape: trc = (trc[0], trc[1]-1)

            # # S
            # if freq_ghz == 2.3:
            #     blc = (805, 813)
            #     rc = (1698, 1218)
            # # X
            # else:
            #     blc = (929, 958)
            #     trc = (1424, 1087)

            # IPOL contours
            fig = iplot(ipol, x=ccimage.x, y=ccimage.y,
                        min_abs_level=4*std, blc=blc, trc=trc, beam=beam, close=True, show_beam=True, show=False,
                        contour_color='gray', contour_linewidth=0.25, components=components)
            fig.savefig(os.path.join(save_dir, "observed_i_{}_{:.1f}.png".format(freq_names[freq_ghz], epoch)), dpi=600, bbox_inches="tight")


# for freq_ghz in freqs_ghz:
#     flux = core_fluxes[freq_ghz][0]
#     r = core_positions[freq_ghz][0]
#     print("Core flux : {:.2f} Jy".format(flux))
#     print("Core position : {:.2f} mas".format(r))


fig, axes = plt.subplots(1, 1)
axes.plot(epochs/30, core_fluxes[8.6], label="Flux", color="C0")
axes.plot(epochs/30, core_positions[8.6], label="Position", color="C1")
axes.scatter(epochs/30, core_fluxes[8.6], color="C0")
axes.scatter(epochs/30, core_positions[8.6], color="C1")
axes.set_xlabel("Time, months")
plt.legend()
plt.show()