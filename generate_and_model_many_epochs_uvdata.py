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
from astropy.stats import mad_std


def find_iqu_image_std(i_image_array, q_image_array, u_image_array, beam_npixels):
    # Robustly estimate image pixels std
    std = mad_std(i_image_array)

    # Find preliminary bounding box
    blc, trc = find_bbox(i_image_array, level=4*std,
                         min_maxintensity_mjyperbeam=4*std,
                         min_area_pix=2*beam_npixels,
                         delta=0)

    # Now mask out source emission using found bounding box and estimate std
    # more accurately
    mask = np.zeros(i_image_array.shape)
    mask[blc[1]: trc[1], blc[0]: trc[0]] = 1
    outside_icn = np.ma.array(i_image_array, mask=mask)
    outside_qcn = np.ma.array(q_image_array, mask=mask)
    outside_ucn = np.ma.array(u_image_array, mask=mask)
    return {"I": mad_std(outside_icn), "Q": mad_std(outside_qcn), "U": mad_std(outside_ucn)}


def pol_mask(stokes_image_dict, beam_pixels, n_sigma=2., return_quantile=False):
    """
    Find mask using stokes 'I' map and 'PPOL' map using specified number of
    sigma.

    :param stokes_image_dict:
        Dictionary with keys - stokes, values - arrays with images.
    :param beam_pixels:
        Number of pixels in beam.
    :param n_sigma: (optional)
        Number of sigma to consider for stokes 'I' and 'PPOL'. 1, 2 or 3.
        (default: ``2``)
    :return:
        Dictionary with Boolean array of masks and P quantile (optionally).
    """
    quantile_dict = {1: 0.6827, 2: 0.9545, 3: 0.9973, 4: 0.99994}
    rms_dict = find_iqu_image_std(*[stokes_image_dict[stokes] for stokes in ('I', 'Q', 'U')],  beam_pixels)

    qu_rms = np.mean([rms_dict[stoke] for stoke in ('Q', 'U')])
    ppol_quantile = qu_rms * np.sqrt(-np.log((1. - quantile_dict[n_sigma]) ** 2.))
    i_cs_mask = stokes_image_dict['I'] < n_sigma * rms_dict['I']
    ppol_cs_image = np.hypot(stokes_image_dict['Q'], stokes_image_dict['U'])
    ppol_cs_mask = ppol_cs_image < ppol_quantile
    mask_dict = {"I": i_cs_mask, "P": np.logical_or(i_cs_mask, ppol_cs_mask)}
    if not return_quantile:
        return mask_dict
    else:
        return mask_dict, ppol_quantile



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
        lg_pixsize_min = {2.3: lg_pixsize_min_mas, 8.6: lg_pixsize_min_mas, 15.4: lg_pixsize_min_mas}
        lg_pixsize_max = {2.3: lg_pixsize_max_mas, 8.6: lg_pixsize_max_mas, 15.4: lg_pixsize_max_mas}

    epochs = ts_obs_days
    rot_angle_deg = 0.0
    freqs_ghz = [15.4]
    freq_names = {15.4: "u"}
    # Some template UVFITS with full polarization. Its uv-coverage and noise will be used while creating fake data
    template_uvfits = {2.3: "/home/ilya/data/rfc/J0102+5824/J0102+5824_S_2017_10_21_pus_vis.fits",
                       8.6: "/home/ilya/data/rfc/J0102+5824/J0102+5824_X_2017_10_21_pus_vis.fits",
                       15.4: "/home/ilya/github/bk_transfer/uvfits/1458+718.u.2006_09_06.uvf"}
    # template_uvfits = {2.3: "/home/ilya/data/rfc/J0102+5824/J0102+5824_S_2017_10_21_pus_vis_doscat.fits",
    #                    8.6: "/home/ilya/data/rfc/J0102+5824/J0102+5824_X_2017_10_21_pus_vis_doscat.fits"}

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
        nw_beam_size = None
        for epoch in epochs:
            if nw_beam_size is None:
                nw_beam = find_nw_beam(template_uvfits[freq_ghz], "i", mapsize=mapsizes_dict[freq_ghz])
                nw_beam_size = np.sqrt(nw_beam[0]*nw_beam[1])
                print("NW beam size = ", nw_beam_size)
            uvdata = UVData(template_uvfits[freq_ghz])
            downscale_uvdata_by_freq_flag = downscale_uvdata_by_freq(uvdata)
            noise = uvdata.noise(average_freq=False, use_V=False)
            # If one needs to decrease the noise this is the way to do it
            for baseline, baseline_noise_std in noise.items():
                noise.update({baseline: noise_scale_factor*baseline_noise_std})

            jm_i = JetImage(z=z, n_along=n_along, n_across=n_across,
                            lg_pixel_size_mas_min=lg_pixsize_min[freq_ghz], lg_pixel_size_mas_max=lg_pixsize_max[freq_ghz],
                            jet_side=True, rot=np.deg2rad(rot_angle_deg))
            jm_q = JetImage(z=z, n_along=n_along, n_across=n_across,
                            lg_pixel_size_mas_min=lg_pixsize_min[freq_ghz], lg_pixel_size_mas_max=lg_pixsize_max[freq_ghz],
                            jet_side=True, rot=np.deg2rad(rot_angle_deg))
            jm_u = JetImage(z=z, n_along=n_along, n_across=n_across,
                            lg_pixel_size_mas_min=lg_pixsize_min[freq_ghz], lg_pixel_size_mas_max=lg_pixsize_max[freq_ghz],
                            jet_side=True, rot=np.deg2rad(rot_angle_deg))

            image_file = "{}/jet_image_{}_{}_{:.1f}.txt".format(jetpol_run_directory, "i", freq_names[freq_ghz], epoch)
            image = np.loadtxt(image_file)
            print(f"Flux = {np.sum(image)}")


            jm_i.load_image_stokes("I", "{}/jet_image_{}_{}_{:.1f}.txt".format(jetpol_run_directory, "i", freq_names[freq_ghz], epoch), scale=1.0)
            jm_q.load_image_stokes("Q", "{}/jet_image_{}_{}_{:.1f}.txt".format(jetpol_run_directory, "q", freq_names[freq_ghz], epoch), scale=1.0)
            jm_u.load_image_stokes("U", "{}/jet_image_{}_{}_{:.1f}.txt".format(jetpol_run_directory, "u", freq_names[freq_ghz], epoch), scale=1.0)
            uvdata.zero_data()
            uvdata.substitute([jm_i, jm_q, jm_u])
            uvdata.noise_add(noise)
            uvdata.save(os.path.join(save_dir, "template_{}_{:.1f}.uvf".format(freq_names[freq_ghz], epoch)), rewrite=True, downscale_by_freq=False)

        for i, epoch in enumerate(epochs):

            outfname_i = "model_cc_i_{}_{:.1f}.fits".format(freq_names[freq_ghz], epoch)
            outfname_q = "model_cc_q_{}_{:.1f}.fits".format(freq_names[freq_ghz], epoch)
            outfname_u = "model_cc_u_{}_{:.1f}.fits".format(freq_names[freq_ghz], epoch)
            if os.path.exists(os.path.join(save_dir, outfname_i)):
                os.unlink(os.path.join(save_dir, outfname_i))

            clean_difmap(fname="template_{}_{:.1f}.uvf".format(freq_names[freq_ghz], epoch), path=save_dir,
                         outfname=outfname_i, outpath=save_dir, stokes="i",
                         mapsize_clean=(1024, 0.1), path_to_script=path_to_script,
                         show_difmap_output=True)
            clean_difmap(fname="template_{}_{:.1f}.uvf".format(freq_names[freq_ghz], epoch), path=save_dir,
                         outfname=outfname_q, outpath=save_dir, stokes="q",
                         mapsize_clean=(1024, 0.1), path_to_script=path_to_script,
                         show_difmap_output=True)
            clean_difmap(fname="template_{}_{:.1f}.uvf".format(freq_names[freq_ghz], epoch), path=save_dir,
                         outfname=outfname_u, outpath=save_dir, stokes="u",
                         mapsize_clean=(1024, 0.1), path_to_script=path_to_script,
                         show_difmap_output=True)
                         # dfm_model=os.path.join(save_dir, "model_cc_i.mdl"))

            ccimage_i = create_clean_image_from_fits_file(os.path.join(save_dir, outfname_i))
            ccimage_q = create_clean_image_from_fits_file(os.path.join(save_dir, outfname_q))
            ccimage_u = create_clean_image_from_fits_file(os.path.join(save_dir, outfname_u))
            ipol = ccimage_i.image
            qpol = ccimage_q.image
            upol = ccimage_u.image
            ppol = np.hypot(qpol, upol)
            fpol = ppol/ipol
            pang = 0.5*np.arctan2(upol, qpol)
            # Here beam in rad
            beam = ccimage_i.beam
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

            # blc = (300, 300)
            # trc = (800, 700)

            masks_dict, ppol_quantile = pol_mask({"I": ipol, "Q": qpol, "U": upol}, npixels_beam, n_sigma=4,
                                                 return_quantile=True)

            # IPOL contours
            # Beam must be in deg
            beam_deg = (beam[0], beam[1], np.rad2deg(beam[2]))
            print("Plotting beam (deg) = ", beam_deg)
            # PPOL contours
            fig = iplot(contours=ppol, x=ccimage_i.x, y=ccimage_i.y, min_abs_level=ppol_quantile,
                        blc=blc, trc=trc, beam=beam_deg, close=False,
                        contour_color='gray', contour_linewidth=0.25)
            # Add single IPOL contour and vectors of the PANG
            fig = iplot(contours=ipol, vectors=pang,
                        x=ccimage_i.x, y=ccimage_i.y, vinc=4, contour_linewidth=1.0,
                        vectors_mask=masks_dict["P"], abs_levels=[2*std], blc=blc, trc=trc,
                        beam=beam, close=True, show_beam=True, show=False,
                        contour_color='gray', fig=fig, vector_color="black", plot_colorbar=False)
            axes = fig.get_axes()[0]
            axes.annotate("{:05.1f} months".format((1+z)*epoch/30), xy=(0.03, 0.9), xycoords="axes fraction", color="gray",
                          weight='bold', ha='left', va='center', size=10)
            fig.savefig(os.path.join(save_dir, "{}_observed_pol_{}_{:.1f}.png".format(basename, freq_names[freq_ghz], epoch)), dpi=600, bbox_inches="tight")
