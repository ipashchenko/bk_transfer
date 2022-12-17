import os
import sys
import shutil
import numpy as np
from vlbi_utils import find_image_std, find_bbox, downscale_uvdata_by_freq
sys.path.insert(0, '/home/ilya/github/ve/vlbi_errors')
from uv_data import UVData
from spydiff import clean_difmap, modelfit_difmap, import_difmap_model, modelfit_core_wo_extending
from image import plot as iplot
from from_fits import create_clean_image_from_fits_file
from jet_image import JetImage
import matplotlib.pyplot as plt


data_dir = "/home/ilya/data/rfc"
source_template = "J0102+5824"
# epochs = np.loadtxt(os.path.join(data_dir, "{}_times.txt".format(source_template)))
z = 1.0
# n_along = 400
n_along = 2000
# n_across = 80
n_across = 1400
match_resolution = False
if match_resolution:
    lg_pixsize_min = {2.3: -2.5, 8.6: -2.5-np.log10(8.6/2.3)}
    lg_pixsize_max = {2.3: -0.5, 8.6: -0.5-np.log10(8.6/2.3)}
else:
    lg_pixsize_min = {2.3: -2.5, 8.6: -2.5}
    lg_pixsize_max = {2.3: -0.5, 8.6: -0.5}

lg_pixsize_min = {2.3: -1., 8.6: -1.}
lg_pixsize_max = {2.3: -1., 8.6: -1.}


# w/o LTTD
# epochs = np.linspace(0, 40*360, 60)
# w LTTD
# ts_obs_days = np.linspace(0.0, 8*360, 20)
ts_obs_days = [0.0]
epochs = ts_obs_days
# basename = "flare_shape_20_width_0.1_ampN_5.0_t0_0"
basename = "test"
plot_clean = True
only_plot_raw = False
extract_extended = True

rot_angle_deg = -90.0
freqs_ghz = [2.3, 8.6]
freq_names = {2.3: "S", 8.6: "X"}
# Directory to save files
save_dir = "/home/ilya/github/bk_transfer/pics/flares"
# Some template UVFITS with full polarization. Its uv-coverage and noise will be used while creating fake data
template_uvfits = {2.3: "/home/ilya/data/rfc/J0102+5824/J0102+5824_S_2017_10_21_pus_vis.fits",
                   8.6: "/home/ilya/data/rfc/J0102+5824/J0102+5824_X_2017_10_21_pus_vis.fits"}
# Multiplicative factor for noise added to model visibilities.
noise_scale_factor = 1.0
# CLEAN image parameters
# FIXME: Plavin+ used (2048, 0.05)
# mapsizes = {2.3: (2048, 0.05,), 8.6: (2048, 0.05,)}
mapsizes = {2.3: (1024, 0.1,), 8.6: (1024, 0.1,)}
# Directory with C++ generated txt-files with model images
jetpol_run_directory = "/home/ilya/github/bk_transfer/Release"
path_to_script = "/home/ilya/github/bk_transfer/scripts/script_clean_rms"

core_positions = dict()
core_positions_err = dict()
core_fluxes = dict()
core_fluxes_err = dict()

std = None
blc = None
trc = None

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
        fig.savefig(os.path.join(save_dir, "{}_raw_nupixel_{}_{:.1f}.png".format(basename, freq_names[freq_ghz], epoch)))
        plt.close(fig)
        if only_plot_raw:
            continue

        jm.load_image_stokes("I", "{}/jet_image_{}_{}_{:.1f}.txt".format(jetpol_run_directory, "i", freq_names[freq_ghz], epoch), scale=1.0)
        uvdata.zero_data()
        uvdata.substitute([jm])
        uvdata.noise_add(noise)
        uvdata.save(os.path.join(save_dir, "template_{}_{:.1f}.uvf".format(freq_names[freq_ghz], epoch)), rewrite=True, downscale_by_freq=True)

        if extract_extended:
            # beam_fractions = np.linspace(0.5, 1.5, 11)
            beam_fractions = [1.0]
            results = modelfit_core_wo_extending("template_{}_{:.1f}.uvf".format(freq_names[freq_ghz], epoch),
                                                 beam_fractions,
                                                 path=save_dir,
                                                 mapsize_clean=mapsizes[freq_ghz],
                                                 path_to_script=path_to_script,
                                                 niter=500,
                                                 out_path=save_dir,
                                                 use_brightest_pixel_as_initial_guess=True,
                                                 estimate_rms=True,
                                                 stokes="i",
                                                 use_ell=False)

            # Flux of the core
            flux = np.mean([results[frac]['flux'] for frac in beam_fractions])
            rms_flux = np.std([results[frac]['flux'] for frac in beam_fractions])

            # Position of the core
            r = np.mean([np.hypot(results[frac]['ra'], results[frac]['dec']) for frac in beam_fractions])
            rms_r = np.std([np.hypot(results[frac]['ra'], results[frac]['dec']) for frac in beam_fractions])

            core_fluxes[freq_ghz].append(flux)
            core_fluxes_err[freq_ghz].append(rms_flux)
            core_positions[freq_ghz].append(r)
            core_positions_err[freq_ghz].append(rms_r)

            # Post-fit rms in the core region
            postfit_rms = np.median([results[frac]['rms'] for frac in beam_fractions])

            print("Core flux : {:.2f}+/-{:.2f} Jy".format(flux, rms_flux))
            print("Core position : {:.2f}+/-{:.2f} mas".format(r, rms_r))

        else:
            modelfit_difmap("template_{}_{:.1f}.uvf".format(freq_names[freq_ghz], epoch), "in4.mdl", "out4.mdl", niter=200, stokes='i',
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
            # Flux of the core
            flux = core.p[0]
            # Position of the core
            r = np.hypot(core.p[1], core.p[2])
            core_fluxes[freq_ghz].append(flux)
            core_positions[freq_ghz].append(r)

        if plot_clean:

            outfname = "model_cc_i_{}_{:.1f}.fits".format(freq_names[freq_ghz], epoch)
            if os.path.exists(os.path.join(save_dir, outfname)):
                os.unlink(os.path.join(save_dir, outfname))

            clean_difmap(fname="template_{}_{:.1f}.uvf".format(freq_names[freq_ghz], epoch), path=save_dir,
                         outfname=outfname, outpath=save_dir, stokes="i",
                         mapsize_clean=(512, 0.2), path_to_script=path_to_script,
                         show_difmap_output=False,
                         dfm_model=os.path.join(save_dir, "model_cc_i.mdl"))

            ccimage = create_clean_image_from_fits_file(os.path.join(save_dir, outfname))
            ipol = ccimage.image
            # Here beam in rad
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

            if freq_ghz == 8.6:
                blc = (240, 230)
                trc = (400, 285)
            else:
                blc = (220, 210)
                trc = (420, 305)


            if len(beam_fractions) == 1:
                components = import_difmap_model("it2.mdl", save_dir)
            else:
                components = None

            # IPOL contours
            # Beam must be in deg
            beam_deg = (beam[0], beam[1], np.rad2deg(beam[2]))
            fig = iplot(ipol, x=ccimage.x, y=ccimage.y,
                        min_abs_level=4*std, blc=blc, trc=trc, beam=beam_deg, close=True, show_beam=True, show=False,
                        contour_color='gray', contour_linewidth=0.25, components=components)
            axes = fig.get_axes()[0]
            axes.annotate("{:05.1f} months".format(epoch/30), xy=(0.03, 0.9), xycoords="axes fraction", color="gray",
                          weight='bold', ha='left', va='center', size=10)
            fig.savefig(os.path.join(save_dir, "{}_observed_i_{}_{:.1f}.png".format(basename, freq_names[freq_ghz], epoch)), dpi=600, bbox_inches="tight")


if only_plot_raw:
    sys.exit(0)

for freq_ghz in freqs_ghz:
    flux = core_fluxes[freq_ghz][0]
    rms_flux = core_fluxes_err[freq_ghz][0]
    r = core_positions[freq_ghz][0]
    rms_r = core_positions_err[freq_ghz][0]
    print("Core flux : {:.2f}+/-{:.2f} Jy".format(flux, rms_flux))
    print("Core position : {:.2f}+/-{:.2f} mas".format(r, rms_r))


CS = np.array(core_positions[2.3])-np.array(core_positions[8.6])

fig, axes = plt.subplots(1, 1, figsize=(15, 15))
axes.set_xlabel("Time, months")
axes2 = axes.twinx()
axes2.set_ylabel("Flux density, Jy")
axes.set_ylabel("Core position, mas")
axes.tick_params("y")
axes2.tick_params("y")

axes.plot([], [], color="C0", label=r"$S_{\rm core,8 GHz}$")
axes.plot([], [], color="C1", label=r"$S_{\rm core,2 GHz}$")

axes.plot(epochs/30, CS, "--", label="CS", color="black")
axes.scatter(epochs/30, CS, color="black")

axes.plot(epochs/30, core_positions[8.6], "--", label=r"$r_{\rm 8 GHz}$", color="C0")
axes.scatter(epochs/30, core_positions[8.6], color="C0")
axes.plot(epochs/30, core_positions[2.3], "--", label=r"$r_{\rm 2 GHz}$", color="C1")
axes.scatter(epochs/30, core_positions[2.3], color="C1")
axes2.plot(epochs/30, core_fluxes[8.6], color="C0")
axes2.scatter(epochs/30, core_fluxes[8.6], color="C0")
axes2.plot(epochs/30, core_fluxes[2.3], color="C1")
axes2.scatter(epochs/30, core_fluxes[2.3], color="C1")

axes.legend()
fig.savefig(os.path.join(save_dir, "{}_CS_rc_Sc_vs_epoch.png".format(basename)), bbox_inches="tight")
plt.show()