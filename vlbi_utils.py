import os
import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import generate_binary_structure
from skimage.measure import regionprops
from astropy.stats import mad_std
from astropy.convolution import convolve, Gaussian2DKernel
from scipy.stats import percentileofscore, scoreatpercentile
import astropy.io.fits as pf
from astropy.wcs import WCS
from astropy import units as u
import sys
sys.path.insert(0, '/home/ilya/github/ve/vlbi_errors')
from from_fits import create_clean_image_from_fits_file


def pol_mask(stokes_image_dict, beam_pixels, n_sigma=2., return_quantile=False, blc=None, trc=None):
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
    rms_dict = find_iqu_image_std(*[stokes_image_dict[stokes] for stokes in ('I', 'Q', 'U')],  beam_pixels, blc=blc, trc=trc)

    print("RMS_dict :", rms_dict)

    qu_rms = np.mean([rms_dict[stoke] for stoke in ('Q', 'U')])
    print("QU_rms = ", qu_rms)
    ppol_quantile = qu_rms * np.sqrt(-np.log((1. - quantile_dict[n_sigma]) ** 2.))
    i_cs_mask = stokes_image_dict['I'] < n_sigma * rms_dict['I']
    ppol_cs_image = np.hypot(stokes_image_dict['Q'], stokes_image_dict['U'])
    ppol_cs_mask = ppol_cs_image < ppol_quantile
    mask_dict = {"I": i_cs_mask, "P": np.logical_or(i_cs_mask, ppol_cs_mask)}
    if not return_quantile:
        return mask_dict
    else:
        return mask_dict, ppol_quantile


def correct_ppol_bias(ipol_array, ppol_array, q_array, u_array, beam_npixels):
    std_dict = find_iqu_image_std(ipol_array, q_array, u_array, beam_npixels)
    rms = 0.5*(std_dict["Q"] + std_dict["U"])
    snr = ppol_array / rms
    factor = 1-1/snr**2
    factor[factor < 0] = 0
    return ppol_array*np.sqrt(factor)


def check_bbox(blc, trc, image_size):
    """
    :note:
        This can make quadratic image rectangular.
    """
    # If some bottom corner coordinate become negative
    blc = list(blc)
    trc = list(trc)
    if blc[0] < 0:
        blc[0] = 0
    if blc[1] < 0:
        blc[1] = 0
    # If some top corner coordinate become large than image size
    if trc[0] > image_size:
        delta = abs(trc[0]-image_size)
        blc[0] -= delta
        # Check if shift have not made it negative
        if blc[0] < 0 and trc[0] > image_size:
            blc[0] = 0
        trc[0] -= delta
    if trc[1] > image_size:
        delta = abs(trc[1]-image_size)
        blc[1] -= delta
        # Check if shift have not made it negative
        if blc[1] < 0 and trc[1] > image_size:
            blc[1] = 0
        trc[1] -= delta
    return tuple(blc), tuple(trc)


def find_bbox(array, level, min_maxintensity_mjyperbeam, min_area_pix,
              delta=0.):
    """
    Find bounding box for part of image containing source.

    :param array:
        Numpy 2D array with image.
    :param level:
        Level at which threshold image in image units.
    :param min_maxintensity_mjyperbeam:
        Minimum of the maximum intensity in the region to include.
    :param min_area_pix:
        Minimum area for region to include.
    :param delta: (optional)
        Extra space to add symmetrically [pixels]. (default: ``0``)
    :return:
        Tuples of BLC & TRC.

    :note:
        This is BLC, TRC for numpy array (i.e. transposed source map as it
        conventionally seen on VLBI maps).
    """
    signal = array > level
    s = generate_binary_structure(2, 2)
    labeled_array, num_features = label(signal, structure=s)
    props = regionprops(labeled_array, intensity_image=array)

    signal_props = list()
    for prop in props:
        if prop.max_intensity > min_maxintensity_mjyperbeam/1000 and prop.area > min_area_pix:
            signal_props.append(prop)

    # Sometimes no regions are found. In that case return full image
    if not signal_props:
        return (0, 0,), (array.shape[1], array.shape[1],)

    blcs = list()
    trcs = list()

    for prop in signal_props:
        bbox = prop.bbox
        blc = (int(bbox[1]), int(bbox[0]))
        trc = (int(bbox[3]), int(bbox[2]))
        blcs.append(blc)
        trcs.append(trc)

    min_blc_0 = min([blc[0] for blc in blcs])
    min_blc_1 = min([blc[1] for blc in blcs])
    max_trc_0 = max([trc[0] for trc in trcs])
    max_trc_1 = max([trc[1] for trc in trcs])
    blc_rec = (min_blc_0-delta, min_blc_1-delta,)
    trc_rec = (max_trc_0+delta, max_trc_1+delta,)

    blc_rec_ = blc_rec
    trc_rec_ = trc_rec
    blc_rec_, trc_rec_ = check_bbox(blc_rec_, trc_rec_, array.shape[0])

    # Enlarge 10% each side
    delta_ra = abs(trc_rec[0]-blc_rec[0])
    delta_dec = abs(trc_rec[1]-blc_rec[1])
    blc_rec = (blc_rec[0] - int(0.1*delta_ra), blc_rec[1] - int(0.1*delta_dec))
    trc_rec = (trc_rec[0] + int(0.1*delta_ra), trc_rec[1] + int(0.1*delta_dec))

    blc_rec, trc_rec = check_bbox(blc_rec, trc_rec, array.shape[0])

    return blc_rec, trc_rec


def find_image_std(image_array, beam_npixels, min_num_pixels_used_to_estimate_std=100):
    # Robustly estimate image pixels std
    std = mad_std(image_array)

    # Find preliminary bounding box
    blc, trc = find_bbox(image_array, level=4*std,
                         min_maxintensity_mjyperbeam=4*std,
                         min_area_pix=2*beam_npixels,
                         delta=0)
    print("Found bounding box : ", blc, trc)

    # Now mask out source emission using found bounding box and estimate std
    # more accurately
    mask = np.zeros(image_array.shape)
    mask[blc[1]: trc[1], blc[0]: trc[0]] = 1
    if mask.shape[0]*mask.shape[1] - np.count_nonzero(mask) < min_num_pixels_used_to_estimate_std:
        return mad_std(image_array)
        # raise Exception("Too small area outside found box with source emission to estimate std - try decrease beam_npixels!")
    outside_icn = np.ma.array(image_array, mask=mask)
    return mad_std(outside_icn)


def find_iqu_image_std(i_image_array, q_image_array, u_image_array, beam_npixels, blc=None, trc=None):
    # Robustly estimate image pixels std
    std = mad_std(i_image_array)

    # Find preliminary bounding box
    if blc is None or trc is None:
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


def correct_ppol_bias(ipol_array, ppol_array, q_array, u_array, beam_npixels):
    std_dict = find_iqu_image_std(ipol_array, q_array, u_array, beam_npixels)
    rms = 0.5*(std_dict["Q"] + std_dict["U"])
    snr = ppol_array / rms
    factor = 1-1/snr**2
    factor[factor < 0] = 0
    return ppol_array*np.sqrt(factor)


# TODO: Add restriction on spatial closeness of the outliers to include them in the range
def choose_range_from_positive_tailed_distribution(data, min_fraction=95):
    """
    Suitable for PANG and FPOL maps.

    :param data:
        Array of values in masked region. Only small fraction (in positive side
        tail) is supposed to be noise.
    :param min_fraction: (optional)
        If no gaps in data distribution than choose this fraction range (in
        percents). (default: ``95``)
    :return:
    """
    mstd = mad_std(data)
    min_fraction_range = scoreatpercentile(data, min_fraction)
    hp_indexes = np.argsort(data)[::-1][np.argsort(np.diff(np.sort(data)[::-1]))]
    for ind in hp_indexes:
        hp = data[ind]
        hp_low = np.sort(data)[hp - np.sort(data) > 0][-1]
        diff = hp - hp_low
        frac = percentileofscore(data, hp_low)
        if diff < mstd/2 and frac < 95:
            break
    if diff > mstd/2:
        return min_fraction_range, 95
    else:
        return hp_low, frac


def filter_CC(ccfits, mask, outname=None, plotsave_fn=None):
    """
    :param ccfits:
    :param mask:
        Mask with region of source flux being True.
    :param outname:
    :return:
    """
    hdus = pf.open(ccfits)
    hdus.verify("silentfix")
    data = hdus[1].data
    data_ = data.copy()
    deg2mas = u.deg.to(u.mas)

    header = pf.getheader(ccfits)
    imsize = header["NAXIS1"]
    wcs = WCS(header)
    # Ignore FREQ, STOKES - only RA, DEC matters here
    wcs = wcs.celestial

    # Make offset coordinates
    wcs.wcs.crval = 0, 0
    wcs.wcs.ctype = 'XOFFSET', 'YOFFSET'
    wcs.wcs.cunit = 'deg', 'deg'

    xs = list()
    ys = list()
    xs_del = list()
    ys_del = list()
    fs_del = list()
    for flux, x_orig, y_orig in zip(data['FLUX'], data['DELTAX'], data['DELTAY']):
        x, y = wcs.world_to_array_index(x_orig*u.deg, y_orig*u.deg)

        if x >= imsize:
            x = imsize - 1
        if y >= imsize:
            y = imsize - 1
        if mask[x, y]:
            # Keep this component
            xs.append(x)
            ys.append(y)
        else:
            # Remove row from rec_array
            xs_del.append(x_orig)
            ys_del.append(y_orig)
            fs_del.append(flux)

    for (x, y, f) in zip(xs_del, ys_del, fs_del):
        local_mask = ~np.logical_and(np.logical_and(data_["DELTAX"] == x, data_["DELTAY"] == y),
                                     data_["FLUX"] == f)
        data_ = data_.compress(local_mask, axis=0)
    print("Deleted {} components".format(len(xs_del)))

    if plotsave_fn is not None:
        a = data_['DELTAX']*deg2mas
        b = data_['DELTAY']*deg2mas
        a_all = data['DELTAX']*deg2mas
        b_all = data['DELTAY']*deg2mas

        fig, axes = plt.subplots(1, 1)
        # im = axes.scatter(a, b, c=np.log10(1000*data_["FLUX"]), vmin=0, s=1, cmap="jet")
        axes.scatter(a_all, b_all, color="gray", alpha=0.25, s=2)
        # axes.scatter(a, b, color="red", alpha=0.5, s=1)
        # from mpl_toolkits.axes_grid1 import make_axes_locatable
        # divider = make_axes_locatable(axes)
        # cax = divider.append_axes("right", size="5%", pad=0.00)
        # cb = fig.colorbar(im, cax=cax)
        # cb.set_label("CC Flux, Jy")
        axes.invert_xaxis()
        axes.set_aspect("equal")
        axes.set_xlabel("RA, mas")
        axes.set_ylabel("DEC, mas")
        plt.show()
        plt.savefig(plotsave_fn, bbox_inches="tight", dpi=300)
        plt.close()

    hdus[1].data = data_
    hdus[1].header["NAXIS2"] = len(data_)
    if outname is None:
        outname = ccfits
    hdus.writeto(outname, overwrite=True)


def filter_CC_dimap_by_rmax(dfm_model_in, r_max, dfm_model_out=None):
    comps = np.loadtxt(dfm_model_in, comments="!")
    len_in = len(comps)
    comps = comps[comps[:, 1] < r_max]
    len_out = len(comps)
    print("{} CCs left from initial {}\n".format(len_out, len_in))
    if dfm_model_out is None:
        dfm_model_out = dfm_model_in
    np.savetxt(dfm_model_out, comps)


def CCFITS_to_difmap(ccfits, difmap_mdl_file, shift=None):
    hdus = pf.open(ccfits)
    hdus.verify("silentfix")
    data = hdus[1].data
    deg2mas = u.deg.to(u.mas)
    with open(difmap_mdl_file, "w") as fo:
        for flux, ra, dec in zip(data['FLUX'], data['DELTAX'], data['DELTAY']):
            ra *= deg2mas
            dec *= deg2mas
            if shift is not None:
                ra -= shift[0]
                dec -= shift[1]
            theta = np.rad2deg(np.arctan2(ra, dec))
            r = np.hypot(ra, dec)
            fo.write("{} {} {}\n".format(flux, r, theta))


def get_mask_for_ccstack(iccfiles, cc_conv_cutoff_mjy=0.0075, kern_width=10):
    """
    Get mask containing CC of stack.
    :param iccfiles:
        Iterable of CC-files.
    :param cc_conv_cutoff_mjy:
    :param kern_width:
    :return:
        Masked numpy 2D array.
    """
    # iccfiles = glob.glob(os.path.join(cc_dir, "cc_I_*.fits"))
    iccimages = [create_clean_image_from_fits_file(ccfile) for ccfile in iccfiles]
    icconly = np.mean([ccimage.cc for ccimage in iccimages], axis=0)
    cconly_conv = convolve(1000*icconly, Gaussian2DKernel(x_stddev=kern_width))
    signal = cconly_conv > cc_conv_cutoff_mjy
    s = generate_binary_structure(2, 2)
    labeled_array, num_features = label(signal, structure=s)
    props = regionprops(labeled_array, intensity_image=cconly_conv)
    max_prop = sorted(props, key=lambda x: x.area, reverse=True)[0]
    mask = np.zeros((512, 512), dtype=bool)
    bbox = max_prop.bbox
    mask[bbox[0]:bbox[2], bbox[1]:bbox[3]] = max_prop.filled_image
    return mask


def rotate_difmap_model(difmap_infile, difmap_outfile, PA_deg):
    comps = np.loadtxt(difmap_infile, comments="!")
    # https://stackoverflow.com/a/49805976
    axes = plt.subplot(111, projection='polar')
    # theta=0 at the top
    axes.set_theta_zero_location("N")
    # Theta increases in the counterclockwise direction
    axes.set_theta_direction(1)
    axes.plot(np.deg2rad(comps[:, 2]), comps[:, 1], '.', color="C0", label="before")
    comps[:, 2] = comps[:, 2] + PA_deg
    axes.plot(np.deg2rad(comps[:, 2]), comps[:, 1], '.', color="C1", label="after")
    plt.legend()
    plt.show()
    if difmap_outfile is None:
        difmap_outfile = difmap_infile
    np.savetxt(difmap_outfile, comps)


def convert_difmap_model_file_to_CCFITS(difmap_model_file, stokes, mapsize, restore_beam, uvfits_template, out_ccfits,
                                        shift=None, show_difmap_output=True):
    """
    Using difmap-formated model file (e.g. flux, r, theta) obtain convolution of your model with the specified beam.

    :param difmap_model_file:
        Difmap-formated model file. Use ``JetImage.save_image_to_difmap_format`` to obtain it.
    :param stokes:
        Stokes parameter.
    :param mapsize:
        Iterable of image size and pixel size (mas).
    :param restore_beam:
        Beam to restore: bmaj(mas), bmin(mas), bpa(deg).
    :param uvfits_template:
        Template uvfits observation to use. Difmap can't read model without having observation at hand.
    :param out_ccfits:
        File name to save resulting convolved map.
    :param shift: (optional)
        Shift to apply. Need this because wmodel doesn't apply shift. If
        ``None`` then do not apply shift. (default: ``None``)
    :param show_difmap_output: (optional)
        Boolean. Show Difmap output? (default: ``True``)
    """
    from subprocess import Popen, PIPE

    cmd = "observe " + uvfits_template + "\n"
    cmd += "select " + stokes + "\n"
    cmd += "rmodel " + difmap_model_file + "\n"
    cmd += "mapsize " + str(mapsize[0] * 2) + "," + str(mapsize[1]) + "\n"
    if shift is not None:
        # Here we need shift, because in CLEANing shifts are not applied to
        # saving model files!
        cmd += "shift " + str(shift[0]) + ', ' + str(shift[1]) + "\n"
    print("Restoring difmap model with BEAM : bmin = " + str(restore_beam[1]) + ", bmaj = " + str(restore_beam[0]) + ", " + str(restore_beam[2]) + " deg")
    # default dimfap: false,true (parameters: omit_residuals, do_smooth)
    cmd += "restore " + str(restore_beam[1]) + "," + str(restore_beam[0]) + "," + str(restore_beam[2]) + "," + "true,false" + "\n"
    cmd += "wmap " + out_ccfits + "\n"
    cmd += "exit\n"

    with Popen('difmap', stdin=PIPE, stdout=PIPE, stderr=PIPE, universal_newlines=True) as difmap:
        outs, errs = difmap.communicate(input=cmd)
    if show_difmap_output:
        print(outs)
        print(errs)


def get_transverse_profile(ccfits, PA, nslices=200, plot_zobs_min=0, plot_zobs_max=None, beam=None, pixsize_mas=None,
                           treat_as_numpy_array=False, save_dir=None, save_prefix=None, save_figs=True, fig=None,
                           alpha=1.0, n_good_min=10, fig_res=None):
    from scipy.ndimage import rotate
    from astropy.stats import mad_std
    from astropy.modeling import fitting
    from astropy.modeling.models import custom_model, Gaussian1D

    if save_dir is None:
        save_dir = os.getcwd()
    if save_prefix is None:
        save_prefix = "transverse_profiles"

    if not treat_as_numpy_array:
        ccimage = create_clean_image_from_fits_file(ccfits)
        pixsize_mas = abs(ccimage.pixsize[0])*u.rad.to(u.mas)
        beam = ccimage.beam
        print("Beam (mas) : ", beam)
        image = ccimage.image
    else:
        image = ccfits

    size = image.shape[0]
    delta = round(size/2/nslices)
    print("Pixsize = {:.2f} mas".format(pixsize_mas))
    # Make jet directing down when plotting with origin=lower in matshow
    std = mad_std(image)
    print("std = {:.2f} mJy/beam".format(1000*std))
    image = rotate(image, PA, reshape=False)
    widths_mas = list()
    pos_mas = list()
    for i in range(nslices):
        imslice = image[int(size/2) - delta*i, :]
        g_init = Gaussian1D(amplitude=np.max(imslice), mean=size/2, stddev=beam[0]/pixsize_mas, fixed={'mean': True})
        fit_g = fitting.LevMarLSQFitter()
        x = np.arange(size)
        y = imslice
        mask = imslice > 5*std
        n_good = np.count_nonzero(mask)
        print("Number of unmasked elements for z = {:.2f} is N = {}".format(delta*i*pixsize_mas, n_good))
        if n_good < n_good_min:
            continue
        g = fit_g(g_init, x[mask], y[mask], weights=1/std)
        print("Convolved FWHM = {:.2f} mas".format(g.fwhm*pixsize_mas))
        width_mas_deconvolved = np.sqrt((g.fwhm*pixsize_mas)**2 - beam[0]**2)
        print("Deconvolved FWHM = {:.2f} mas".format(width_mas_deconvolved))
        if np.isnan(width_mas_deconvolved):
            continue
        widths_mas.append(width_mas_deconvolved)
        pos_mas.append(delta*i*pixsize_mas)

    pos_mas = np.array(pos_mas)
    widths_mas = np.array(widths_mas)
    if fig is None:
        fig, axes = plt.subplots(1, 1)
    else:
        axes = fig.get_axes()[0]
    if plot_zobs_max is not None:
        assert plot_zobs_max > plot_zobs_min
        axes.set_xlim([plot_zobs_min, plot_zobs_max])
        mask = np.logical_and(pos_mas < plot_zobs_max, pos_mas > plot_zobs_min)
        pos_to_plot = pos_mas[mask]
        widths_to_plot = widths_mas[mask]
    else:
        widths_to_plot = widths_mas
        pos_to_plot = pos_mas

    axes.plot(pos_to_plot, widths_to_plot, color="C0", alpha=alpha)
    axes.set_xlabel(r"$z_{\rm obs}$, mas")
    axes.set_ylabel("FWHM, mas")
    plt.xscale("log")
    plt.yscale("log")
    if save_figs:
        fig.savefig(os.path.join(save_dir, "{}.png".format(save_prefix)), bbox_inches="tight", dpi=300)
    plt.show()

    # Now fit profile
    @custom_model
    def power_law(r, amp=1.0, r0=0.0, k=0.5):
        return amp*(r + r0)**k
    pl_init = power_law(fixed={"r0": True})
    fit_pl = fitting.LevMarLSQFitter()
    pl = fit_pl(pl_init, pos_to_plot, widths_to_plot, maxiter=10000)
    print(fit_pl.fit_info)
    print("k = ", pl.k)
    print("r0 = ", pl.r0)
    print("amp = ", pl.amp)

    # Plot fit
    xx = np.linspace(np.min(pos_to_plot), np.max(pos_to_plot), 1000)
    yy = pl(xx)
    fig_, axes = plt.subplots(1, 1)
    axes.plot(xx, yy, color="C1", label="k = {:.2f}".format(pl.k.value))
    axes.scatter(pos_to_plot, widths_to_plot, color="C0", label="data", s=2)
    axes.set_xlabel(r"$z_{\rm obs}$, mas")
    axes.set_ylabel("FWHM, mas")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    if save_figs:
        fig_.savefig(os.path.join(save_dir, "{}_fit.png".format(save_prefix)), bbox_inches="tight", dpi=300)
    plt.show()
    plt.close(fig_)

    # Make residuals and plot them
    res = widths_to_plot - pl(pos_to_plot)
    max_res = 1.2*np.max(np.abs(res))

    if fig_res is None:
        fig_res, axes = plt.subplots(1, 1)
    else:
        axes = fig_res.get_axes()[0]

    axes.plot(pos_to_plot, res, color="C0", alpha=1.0)
    # axes.set_ylim([-max_res, max_res])
    axes.set_ylim([-2, 2])
    axes.set_xlabel(r"$z_{\rm obs}$, mas")
    axes.set_ylabel("residual FWHM, mas")
    if save_figs:
        fig_res.savefig(os.path.join(save_dir, "{}_residual_width.png".format(save_prefix)), bbox_inches="tight", dpi=300)
    plt.show()

    return fig, fig_res


def time_average(uvfits, outfname, time_sec=120, show_difmap_output=True,
                 reweight=True):
    stamp = datetime.datetime.now()
    command_file = "difmap_commands_{}".format(stamp.isoformat())

    difmapout = open(command_file, "w")
    if reweight:
        difmapout.write("observe " + uvfits + ", {}, true\n".format(time_sec))
    else:
        difmapout.write("observe " + uvfits + ", {}, false\n".format(time_sec))
    difmapout.write("wobs {}\n".format(outfname))
    difmapout.write("exit\n")
    difmapout.close()
    # TODO: Use subprocess for silent cleaning?
    shell_command = "difmap < " + command_file + " 2>&1"
    if not show_difmap_output:
        shell_command += " >/dev/null"
    os.system(shell_command)

    # Remove command file
    os.unlink(command_file)