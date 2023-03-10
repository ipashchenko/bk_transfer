import os
import sys
import shutil
import json
import glob
import numpy as np
from vlbi_utils import find_image_std, find_bbox, downscale_uvdata_by_freq
sys.path.insert(0, '/home/ilya/github/ve/vlbi_errors')
from uv_data import UVData
from components import CGComponent
from spydiff import clean_difmap, modelfit_difmap, import_difmap_model, modelfit_core_wo_extending, find_nw_beam
from image import plot as iplot
from from_fits import create_clean_image_from_fits_file
from jet_image import JetImage
import matplotlib.pyplot as plt


def make_and_model_visibilities(basename = "test", only_band=None, z = 1.0,
                                lg_pixsize_min_mas=-2.5, lg_pixsize_max_mas=-0.5, n_along = 400, n_across = 80, match_resolution = False,
                                ts_obs_days = np.linspace(-400.0, 8*360, 20),
                                noise_scale_factor = 1.0, mapsizes_dict = {2.3: (1024, 0.1,), 8.6: (1024, 0.1,)},
                                plot_clean = True, only_plot_raw = False,
                                extract_extended = True, use_scipy = False, beam_fractions = (1.0,), two_stage=True,
                                n_components=4,
                                save_dir = "/home/ilya/github/bk_transfer/pics/flares",
                                jetpol_run_directory = "/home/ilya/github/bk_transfer/Release",
                                path_to_script = "/home/ilya/github/bk_transfer/scripts/script_clean_rms",
                                calculon=True):
    """
    :param basename:
    :param only_band:
    :param z:
    :param lg_pixsize_min_mas:
    :param lg_pixsize_max_mas:
    :param n_along:
    :param n_across:
    :param match_resolution:
    :param ts_obs_days:
    :param noise_scale_factor:
        Multiplicative factor for noise added to model visibilities.
    :param mapsizes_dict:
        Dictionary with keys - frequencies in GHz (e.g. 2.3) and values - CLEAN map parameters for imaging to extract
        the extended emission.
    :param plot_clean: (optional)
        Plot CLEAN image with model? (default: ``True``)
    :param only_plot_raw: (optional)
        Plot only raw images with NU pixel? (default: ``False``)
    :param extract_extended: (optional)
        Use extraction of the extended emission to fit the core? (default: ``True``).
    :param use_scipy: (optional)
        Use scipy to fit core to the extracted from the extended emission data?
    :param beam_fractions: (optional)
        Iterable of the beam fractions to use for extended emission extraction. (default: ``(1.0, )``)
    :param two_stage: (opitional)
        Use two stage core parameters estimation procedure? (default: ``True``)
    :param n_components: (optional)
        Number of components to fir if ``extract_extended = False``. (default: ``4``)
    :param save_dir:
    :param jetpol_run_directory: (optional)
        Directory with C++ generated txt-files with model images.
    :param path_to_script:
    """
    if match_resolution:
        lg_pixsize_min = {2.3: lg_pixsize_min_mas, 8.6: lg_pixsize_min_mas-np.log10(8.6/2.3)}
        lg_pixsize_max = {2.3: lg_pixsize_max_mas, 8.6: lg_pixsize_max_mas-np.log10(8.6/2.3)}
    else:
        lg_pixsize_min = {2.3: lg_pixsize_min_mas, 8.6: lg_pixsize_min_mas}
        lg_pixsize_max = {2.3: lg_pixsize_max_mas, 8.6: lg_pixsize_max_mas}

    epochs = ts_obs_days
    rot_angle_deg = -90.0
    freqs_ghz = [2.3, 8.6]
    freq_names = {2.3: "S", 8.6: "X"}
    # Some template UVFITS with full polarization. Its uv-coverage and noise will be used while creating fake data
    template_uvfits = {2.3: "/home/ilya/data/rfc/J0102+5824/J0102+5824_S_2017_10_21_pus_vis.fits",
                       8.6: "/home/ilya/data/rfc/J0102+5824/J0102+5824_X_2017_10_21_pus_vis.fits"}
    # template_uvfits = {2.3: "/home/ilya/data/rfc/J0102+5824/J0102+5824_S_2017_10_21_pus_vis_doscat.fits",
    #                    8.6: "/home/ilya/data/rfc/J0102+5824/J0102+5824_X_2017_10_21_pus_vis_doscat.fits"}
    core_positions = dict()
    core_positions_err = dict()
    core_fluxes = dict()
    core_fluxes_err = dict()
    core_sizes = dict()
    core_sizes_err = dict()

    std = None
    blc = None
    trc = None

    # Remove old json files
    json_files = glob.glob(os.path.join(save_dir, "*.json"))
    for jfn in json_files:
        try:
            os.unlink(jfn)
        except:
            pass

    for freq_ghz in freqs_ghz:
        if only_band is not None and only_band != freq_names[freq_ghz]:
            continue
        core_positions[freq_ghz] = list()
        core_positions_err[freq_ghz] = list()
        core_fluxes[freq_ghz] = list()
        core_fluxes_err[freq_ghz] = list()
        core_sizes[freq_ghz] = list()
        core_sizes_err[freq_ghz] = list()
        nw_beam_size = None
        for epoch in epochs:
            if nw_beam_size is None:
                nw_beam = find_nw_beam(template_uvfits[freq_ghz], "i", mapsize=mapsizes_dict[freq_ghz])
                nw_beam_size = np.sqrt(nw_beam[0]*nw_beam[1])
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

            # # Plot raw image
            # fig, axes = plt.subplots(1, 1)
            # axes.matshow(image, cmap="inferno", aspect="auto")
            # axes.set_xlabel("Along, nu pixels")
            # axes.set_ylabel("Across, nu pixels")
            # axes.annotate("{:05.1f} months".format(epoch/30), xy=(0.03, 0.9), xycoords="axes fraction", color="pink",
            #               weight='bold', ha='left', va='center', size=20)
            # fig.savefig(os.path.join(save_dir, "{}_raw_nupixel_{}_{:.1f}.png".format(basename, freq_names[freq_ghz], epoch)))
            # plt.close(fig)
            # if only_plot_raw:
            #     continue

            jm.load_image_stokes("I", "{}/jet_image_{}_{}_{:.1f}.txt".format(jetpol_run_directory, "i", freq_names[freq_ghz], epoch), scale=1.0)
            uvdata.zero_data()
            uvdata.substitute([jm])
            uvdata.noise_add(noise)
            uvdata.save(os.path.join(save_dir, "template_{}_{:.1f}.uvf".format(freq_names[freq_ghz], epoch)), rewrite=True, downscale_by_freq=True)


        if extract_extended:
            beam_fracs = " ".join([str(bf) for bf in beam_fractions])
            fnames = " ".join(["template_{}_{:.1f}.uvf".format(freq_names[freq_ghz], epoch) for epoch in epochs])
            script_dir = os.path.split(jetpol_run_directory)[0]


            if calculon:
                n_jobs = 44
            else:
                n_jobs = 4
            os.system(f"parallel -k --jobs {n_jobs} python {script_dir}/modelfit_single_epoch.py --beam_fractions \"{beam_fracs}\" --mapsize_clean \"{mapsizes_dict[freq_ghz][0]} {mapsizes_dict[freq_ghz][1]}\" --save_dir \"{save_dir}\" --path_to_script \"{path_to_script}\"  --nw_beam_size \"{nw_beam_size}\" --fname ::: {fnames}")




            # sys.exit(0)




            # Gather results
            for epoch in epochs:
                fname = "template_{}_{:.1f}.uvf".format(freq_names[freq_ghz], epoch)
                base = fname.split(".")[:-1]
                base = ".".join(base)
                with open(os.path.join(save_dir, f"{base}_core_modelfit_result.json"), "r") as fo:
                    results = json.load(fo)
                    # Flux of the core
                    flux = np.mean([results[str(frac)]['flux'] for frac in beam_fractions])
                    rms_flux = np.std([results[str(frac)]['flux'] for frac in beam_fractions])

                    # Position of the core
                    r = np.mean([np.hypot(results[str(frac)]['ra'], results[str(frac)]['dec']) for frac in beam_fractions])
                    rms_r = np.std([np.hypot(results[str(frac)]['ra'], results[str(frac)]['dec']) for frac in beam_fractions])

                    # Size of the core
                    size = np.mean([results[str(frac)]['size'] for frac in beam_fractions])
                    rms_size = np.std([results[str(frac)]['size'] for frac in beam_fractions])

                    core_fluxes[freq_ghz].append(flux)
                    core_fluxes_err[freq_ghz].append(rms_flux)
                    core_positions[freq_ghz].append(r)
                    core_positions_err[freq_ghz].append(rms_r)
                    core_sizes[freq_ghz].append(size)
                    core_sizes_err[freq_ghz].append(rms_size)

                    # Post-fit rms in the core region
                    postfit_rms = np.median([results[str(frac)]['rms'] for frac in beam_fractions])

                    print("Core flux : {:.2f}+/-{:.2f} Jy".format(flux, rms_flux))
                    print("Core position : {:.2f}+/-{:.2f} mas".format(r, rms_r))

            # Re-calculate it for next frequency
            nw_beam_size = None

        else:
            for i, epoch in enumerate(epochs):
                if i == 0:
                    mdl_fname = f"in_{n_components}_{freq_names[freq_ghz]}_last.mdl"
                else:
                    mdl_fname = "out{}_{:.1f}.mdl".format(n_components, epochs[i-1])
                modelfit_difmap("template_{}_{:.1f}.uvf".format(freq_names[freq_ghz], epoch),
                                mdl_fname=mdl_fname, out_fname="out{}_{:.1f}.mdl".format(n_components, epoch), niter=200, stokes='i',
                                path=save_dir, mdl_path=save_dir, out_path=save_dir,
                                show_difmap_output=True,
                                save_dirty_residuals_map=False,
                                dmap_name=None, dmap_size=(1024, 0.1))

                components = import_difmap_model("out{}_{:.1f}.mdl".format(n_components, epoch), save_dir)
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
            for i, epoch in enumerate(epochs):

                outfname = "model_cc_i_{}_{:.1f}.fits".format(freq_names[freq_ghz], epoch)
                if os.path.exists(os.path.join(save_dir, outfname)):
                    os.unlink(os.path.join(save_dir, outfname))

                clean_difmap(fname="template_{}_{:.1f}.uvf".format(freq_names[freq_ghz], epoch), path=save_dir,
                             outfname=outfname, outpath=save_dir, stokes="i",
                             mapsize_clean=(512, 0.2), path_to_script=path_to_script,
                             show_difmap_output=False)
                             # dfm_model=os.path.join(save_dir, "model_cc_i.mdl"))

                ccimage = create_clean_image_from_fits_file(os.path.join(save_dir, outfname))
                ipol = ccimage.image
                # Here beam in rad
                beam = ccimage.beam
                # Number of pixels in beam
                npixels_beam = np.pi*beam[0]*beam[1]/(4*np.log(2)*mapsizes_dict[freq_ghz][1]**2)

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

                if extract_extended:
                    if len(beam_fractions) == 1:
                        components = [CGComponent(core_fluxes[freq_ghz][i], core_positions[freq_ghz][i], 0, core_sizes[freq_ghz][i])]
                        # components = import_difmap_model("it2.mdl", save_dir)
                    else:
                        components = None
                else:
                    components = import_difmap_model(os.path.join(save_dir, "out{}_{:.1f}.mdl".format(n_components, epoch)))

                # IPOL contours
                # Beam must be in deg
                beam_deg = (beam[0], beam[1], np.rad2deg(beam[2]))
                fig = iplot(ipol, x=ccimage.x, y=ccimage.y,
                            min_abs_level=4*std, blc=blc, trc=trc, beam=beam_deg, close=True, show_beam=True, show=False,
                            contour_color='gray', contour_linewidth=0.25, components=components)
                axes = fig.get_axes()[0]
                axes.annotate("{:05.1f} months".format((1+z)*epoch/30), xy=(0.03, 0.9), xycoords="axes fraction", color="gray",
                              weight='bold', ha='left', va='center', size=10)
                fig.savefig(os.path.join(save_dir, "{}_observed_i_{}_{:.1f}.png".format(basename, freq_names[freq_ghz], epoch)), dpi=600, bbox_inches="tight")


    if only_plot_raw:
        sys.exit(0)

    # for freq_ghz in freqs_ghz:
    #     if only_band is not None and only_band != freq_names[freq_ghz]:
    #         continue
    #     flux = core_fluxes[freq_ghz][0]
    #     if extract_extended:
    #         rms_flux = core_fluxes_err[freq_ghz][0]
    #     else:
    #         rms_flux = 0.0
    #     r = core_positions[freq_ghz][0]
    #     if extract_extended:
    #         rms_r = core_positions_err[freq_ghz][0]
    #     else:
    #         rms_r = 0.0
    #     print("Core flux : {:.2f}+/-{:.2f} Jy".format(flux, rms_flux))
    #     print("Core position : {:.2f}+/-{:.2f} mas".format(r, rms_r))

    if only_band is None:
        CS = np.array(core_positions[2.3])-np.array(core_positions[8.6])
        np.savetxt(os.path.join(save_dir, f"source_{basename}_CS.txt"), CS)

    np.savetxt(os.path.join(save_dir, f"source_{basename}_epochs.txt"), epochs*(1+z))
    np.savetxt(os.path.join(save_dir, f"source_{basename}_S_2.txt"), core_fluxes[2.3])
    np.savetxt(os.path.join(save_dir, f"source_{basename}_S_8.txt"), core_fluxes[8.6])

    fig, axes = plt.subplots(1, 1, figsize=(15, 15))
    axes.set_xlabel("Obs. Time, years")
    axes2 = axes.twinx()
    axes2.set_ylabel("Flux density, Jy")
    axes.set_ylabel("Core position, mas")
    axes.tick_params("y")
    axes2.tick_params("y")

    axes.plot([], [], color="C0", label=r"$S_{\rm core,8 GHz}$")
    axes.plot([], [], color="C1", label=r"$S_{\rm core,2 GHz}$")

    if only_band is None:
        axes.plot(epochs*(1+z)/30/12, CS, "--", label="CS", color="black")
        axes.scatter(epochs*(1+z)/30/12, CS, color="black")

    if only_band is None:
        axes.plot(epochs*(1+z)/30/12, core_positions[8.6], "--", label=r"$r_{\rm 8 GHz}$", color="C0")
        axes.scatter(epochs*(1+z)/30/12, core_positions[8.6], color="C0")
        axes2.plot(epochs*(1+z)/30/12, core_fluxes[8.6], color="C0")
        axes2.scatter(epochs*(1+z)/30/12, core_fluxes[8.6], color="C0")
        axes.plot(epochs*(1+z)/30/12, core_positions[2.3], "--", label=r"$r_{\rm 2 GHz}$", color="C1")
        axes.scatter(epochs*(1+z)/30/12, core_positions[2.3], color="C1")
        axes2.plot(epochs*(1+z)/30/12, core_fluxes[2.3], color="C1")
        axes2.scatter(epochs*(1+z)/30/12, core_fluxes[2.3], color="C1")

    if only_band is not None and only_band == freq_names[8.6]:
        axes.plot(epochs*(1+z)/30/12, core_positions[8.6], "--", label=r"$r_{\rm 8 GHz}$", color="C0")
        axes.scatter(epochs*(1+z)/30/12, core_positions[8.6], color="C0")
        axes2.plot(epochs*(1+z)/30/12, core_fluxes[8.6], color="C0")
        axes2.scatter(epochs*(1+z)/30/12, core_fluxes[8.6], color="C0")

    if only_band is not None and only_band == freq_names[2.3]:
        axes.plot(epochs*(1+z)/30/12, core_positions[2.3], "--", label=r"$r_{\rm 2 GHz}$", color="C1")
        axes.scatter(epochs*(1+z)/30/12, core_positions[2.3], color="C1")
        axes2.plot(epochs*(1+z)/30/12, core_fluxes[2.3], color="C1")
        axes2.scatter(epochs*(1+z)/30/12, core_fluxes[2.3], color="C1")

    axes.legend()
    axes2.set_ylim([0, None])
    fig.savefig(os.path.join(save_dir, "{}_CS_rc_Sc_vs_obs_epoch.png".format(basename)), bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    basename = "test"
    only_band = None
    z = 1.0,
    lg_pixsize_min_mas=-2.5
    lg_pixsize_max_mas=-0.5
    n_along = 400
    n_across = 80
    match_resolution = False
    ts_obs_days = np.linspace(-400.0, 8*360, 20)
    noise_scale_factor = 1.0
    mapsizes_dict = {2.3: (1024, 0.1,), 8.6: (1024, 0.1,)}
    plot_clean = True
    only_plot_raw = False
    extract_extended = True
    use_scipy_for_extract_extended = False
    beam_fractions = (1.0,)
    two_stage = True
    n_components = 4
    save_dir = "/home/ilya/github/bk_transfer/pics/flares"
    jetpol_run_directory = "/home/ilya/github/bk_transfer/Release"
    path_to_script = "/home/ilya/github/bk_transfer/scripts/script_clean_rms"

    make_and_model_visibilities(basename, only_band, z, lg_pixsize_min_mas, lg_pixsize_max_mas, n_along, n_across, match_resolution,
                                ts_obs_days,
                                noise_scale_factor, mapsizes_dict,
                                plot_clean, only_plot_raw,
                                extract_extended, use_scipy_for_extract_extended, beam_fractions, two_stage,
                                n_components,
                                save_dir, jetpol_run_directory, path_to_script)