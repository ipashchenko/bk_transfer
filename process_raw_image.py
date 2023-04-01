import sys
import os
import math
import numpy as np
from astropy import cosmology
import astropy.units as u
sys.path.insert(0, '/home/ilya/github/ve/vlbi_errors')
from image import plot as iplot
import matplotlib
label_size = 20
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams['ytick.labelsize'] = label_size
matplotlib.rcParams['axes.titlesize'] = label_size
matplotlib.rcParams['axes.labelsize'] = label_size
matplotlib.rcParams['font.size'] = label_size
matplotlib.rcParams['legend.fontsize'] = label_size
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

cosmo = cosmology.WMAP9
# cm/s
c = 29979245800.0
# cm
pc = 3.0856775814671913e+18
#
day_to_sec = 86400.0
# g
m_e = 9.10938356e-28


def plot_function(contours=None, colors=None, vectors=None, vectors_values=None,
         cmap='gist_rainbow', abs_levels=None, rel_levels=None, min_abs_level=None,
         min_rel_level=None, k=2, vinc=2, contours_mask=None, colors_mask=None,
         vectors_mask=None, color_clim=None, outfile=None, outdir=None, close=False,
         colorbar_label=None, show=True, contour_color='k', vector_color="k", plot_colorbar=True,
         max_vector_value_length=5., mas_in_pixel=None, vector_enlarge_factor=1.0,
         label_size=14, figsize=(20, 5), fig=None):
    """
    :param contours: (optional)
        Numpy 2D array (possibly masked) that should be plotted using contours.
    :param colors: (optional)
        Numpy 2D array (possibly masked) that should be plotted using colors.
    :param vectors: (optional)
        Numpy 2D array (possibly masked) that should be plotted using vectors.
    :param vectors_values: (optional)
        Numpy 2D array (possibly masked) that should be used as vector's lengths
        when plotting ``vectors`` array.
    :param cmap: (optional)
        Colormap to use for plotting colors.
        (default: ``gist_rainbow``)
    :param abs_levels: (optional)
        Iterable of absolute levels. If ``None`` then construct levels in other
        way. (default: ``None``)
    :param min_abs_level: (optional)
        Values of minimal absolute level. Used with conjunction of ``k``
        argument for building sequence of absolute levels. If ``None`` then
        construct levels in other way. (default: ``None``)
    :param rel_levels: (optional)
        Iterable of relative levels. If ``None`` then construct levels in other
        way. (default: ``None``)
    :param min_rel_level: (optional)
        Values of minimal relative level. Used with conjunction of ``k``
        argument for building sequence of relative levels. If ``None`` then
        construct levels in other way. (default: ``None``)
    :param k: (optional)
        Factor of incrementation for levels. (default: ``2.0``)
    :param colorbar_label: (optional)
        String to label colorbar. If ``None`` then don't label. (default:
        ``None``)
    :param plot_colorbar: (optional)
        If colors is set then should we plot colorbar? (default: ``True``).
    :param max_vector_value_length: (optional)
        Determines what part of the image is the length of the vector with
        maximum magnitude. E.g. if ``5`` then maximum value of vector quantity
        corresponds to arrow with length equal to 1/5 of the image length.
        (default: ``5``)
    :param mas_in_pixel: (optonal)
        Number of milliarcseconds in one pixel. If ``None`` then plot in pixels.
        (default: ``None``)
    :param vector_enlarge_factor: (optional)
        Additional factor to increase length of vectors representing direction and values of linear polarization.
    """
    matplotlib.rcParams['xtick.labelsize'] = label_size
    matplotlib.rcParams['ytick.labelsize'] = label_size
    matplotlib.rcParams['axes.titlesize'] = label_size
    matplotlib.rcParams['axes.labelsize'] = label_size
    matplotlib.rcParams['font.size'] = label_size
    matplotlib.rcParams['legend.fontsize'] = label_size
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    image = None
    if contours is not None:
        image = contours
    elif colors is not None and image is None:
        image = colors
    elif vectors is not None and image is None:
        image = vectors
    else:
        raise Exception("No image to plot")

    x = np.arange(image.shape[0]) - image.shape[0]/2
    y = np.arange(image.shape[1]) - image.shape[1]/2
    if mas_in_pixel is not None:
        x *= mas_in_pixel
        y *= mas_in_pixel

    # Optionally mask arrays
    if contours is not None and contours_mask is not None:
        contours = np.ma.array(contours, mask=contours_mask)
    if colors is not None and colors_mask is not None:
        colors = np.ma.array(colors, mask=colors_mask)
    if vectors is not None and vectors_mask is not None:
        vectors = np.ma.array(vectors, mask=vectors_mask)

    # Actually plotting
    if fig is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    else:
        ax = fig.get_axes()[0]

    # Plot contours
    if contours is not None:
        if abs_levels is None:
            max_level = np.nanmax(contours)
            if rel_levels is not None:
                abs_levels = [-max_level] + [max_level * i for i in rel_levels]
            else:
                if min_abs_level is not None:
                    n_max = int(math.ceil(math.log(max_level / min_abs_level, k)))
                elif min_rel_level is not None:
                    min_abs_level = min_rel_level * max_level / 100.
                    n_max = int(math.ceil(math.log(max_level / min_abs_level, k)))
                else:
                    raise Exception("Not enough information for levels")
                abs_levels = [-min_abs_level] + [min_abs_level * k ** i for i in
                                                 range(n_max)]
        co = ax.contour(y, x, contours, abs_levels, colors=contour_color)
    if colors is not None:
        im = ax.imshow(colors, interpolation='none',
                       origin='lower', extent=[y[0], y[-1], x[0], x[-1]],
                       cmap=plt.get_cmap(cmap), clim=color_clim)
    if vectors is not None:
        if vectors_values is not None:
            u = vectors_values * np.cos(vectors)
            v = vectors_values * np.sin(vectors)
            max_vector_value = np.max(np.abs(vectors_values))
            scale = max_vector_value_length*max_vector_value/vector_enlarge_factor
        else:
            u = np.cos(vectors)
            v = np.sin(vectors)
            scale = None

        if vectors_mask is not None:
            u = np.ma.array(u, mask=vectors_mask)
            v = np.ma.array(v, mask=vectors_mask)

        vec = ax.quiver(y[::vinc], x[::vinc], u[::vinc, ::vinc],
                        v[::vinc, ::vinc], angles='uv',
                        units='width', headwidth=0., headlength=0., scale=scale,
                        width=0.0025, headaxislength=0., pivot='middle',
                        scale_units='width', color=vector_color)

    # Set equal aspect
    ax.set_aspect('equal')

    if colors is not None:
        if plot_colorbar:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.00)
            cb = fig.colorbar(im, cax=cax)
            if colorbar_label is not None:
                cb.set_label(colorbar_label)

    # Saving output
    if outfile:
        if outdir is None:
            outdir = '.'
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        path = os.path.join(outdir, outfile)
        plt.savefig("{}.png".format(path), bbox_inches='tight', dpi=300)

    if show:
        plt.ioff()
        plt.show()
    if close:
        plt.close()

    return fig



def generalized1_gaussian1d(x, loc, scale, shape):
    return np.exp(-(np.abs(x-loc)/scale)**shape)


def equipartition_bsq_coefficient(s, gamma_min, gamma_max):

    if s != 2.0:
        return (s - 2)/(s - 1) / (8*np.pi*m_e*c*c) * (gamma_min**(1.-s) - gamma_max**(1.-s)) / (gamma_min**(2.-s) - gamma_max**(2.-s))
    else:
        return 1.0/(8*np.pi*m_e*c*c*gamma_min*np.log(gamma_max/gamma_min))


def N(r_proj_pc, t_obs_days, t_start_days, amp, l_pc, Gamma, theta_deg, N_1, n, z, shape=2):
    theta = np.deg2rad(theta_deg)
    r_pc = r_proj_pc/np.sin(theta)
    beta = np.sqrt(Gamma**2 - 1)/Gamma
    # beta_obs = beta/(1 - beta*np.cos(theta))/(1+z)
    # This is velocity at given z. We correct for this just expanding time intervals!
    beta_obs = beta/(1 - beta*np.cos(theta))
    return N_1*r_pc**(-n)*(1 + amp * generalized1_gaussian1d(r_pc, beta_obs*c*(t_obs_days - t_start_days)*day_to_sec/pc, l_pc, shape))


def B(r_proj_pc, t_obs_days, t_start_days, amp, l_pc, Gamma, theta_deg, B_1, b, z, shape=2):
    theta = np.deg2rad(theta_deg)
    r_pc = r_proj_pc/np.sin(theta)
    beta = np.sqrt(Gamma**2 - 1)/Gamma
    # beta_obs = beta/(1 - beta*np.cos(theta))/(1+z)
    # This is velocity at given z. We correct for this just expanding time intervals!
    beta_obs = beta/(1 - beta*np.cos(theta))
    return B_1*r_pc**(-b)*(1 + amp * generalized1_gaussian1d(r_pc, beta_obs*c*(t_obs_days - t_start_days)*day_to_sec/pc, l_pc, shape))


def ang_to_dist(z):
    return cosmo.kpc_proper_per_arcmin(z)


def mas_to_pc(mas_coordinates, z):
    return (mas_coordinates * u.mas * ang_to_dist(z)).to(u.pc).value


def get_proj_core_position(image_txt, tau_txt, z, lg_pixel_size_mas_min, lg_pixel_size_mas_max, n_along, n_across):
    print("Using logspace coordinates!")
    resolutions = np.logspace(lg_pixel_size_mas_min, lg_pixel_size_mas_max, n_along)
    pixsize_array = np.tile(resolutions, n_across).reshape(n_across, n_along).T
    intensity_factor = (pixsize_array/np.min(pixsize_array))**2
    pixel_coordinates = np.cumsum(resolutions) - resolutions/2

    tau = np.loadtxt(tau_txt)
    I_orig = np.loadtxt(image_txt)
    I = I_orig/intensity_factor.T
    midle = int(n_across/2)
    tau_stripe = 0.5*(tau[midle-1, :] + tau[midle, :])
    I_stripe = 0.5*(I[midle-1, :] + I[midle, :])
    I_mean = np.sum(I, axis=0)

    idx_max_I_stripe = np.argmax(I_stripe)
    idx_max_I_mean = np.argmax(I_mean)

    # FIXME: If several tau = 1 region are present - get the first close to 1
    # First one is jet apex
    idx_tau_1 = np.where(tau_stripe < 1)[0][0]
    if idx_tau_1 == 0:
        idx_tau_1 = np.where(tau_stripe < 1)[0][1]
    if idx_tau_1 == 1:
        idx_tau_1 = np.where(tau_stripe < 1)[0][2]
    if idx_tau_1 == 2:
        idx_tau_1 = np.where(tau_stripe < 1)[0][3]
    if idx_tau_1 == 3:
        idx_tau_1 = np.where(tau_stripe < 1)[0][4]

    # Find if there is another tau = 1 region due to flare down the core
    tau_down_core = tau_stripe[idx_tau_1:]
    try:
        idx_tau_1_2 = np.where(tau_down_core > 1)[0][-1]
        # Index in full size
        idx_tau_1_2 = np.arange(len(tau_stripe))[idx_tau_1:][idx_tau_1_2]
    # No second tau = 1 region
    except IndexError:
        print("No second tau = 1 region")
        idx_tau_1_2 = None

    tau_at_max_I_stripe = tau_stripe[idx_max_I_stripe]
    tau_at_max_I_mean = tau_stripe[idx_max_I_mean]

    print("Tau at max I stripe = ", tau_at_max_I_stripe)
    print("Tau at max I mean = ", tau_at_max_I_mean)

    pos_tau_1_1_mas = pixel_coordinates[idx_tau_1]
    pos_tau_1_1_pc = mas_to_pc(pos_tau_1_1_mas, z)
    pos_tau_1_2_mas = None
    pos_tau_1_2_pc = None
    if idx_tau_1_2 is not None:
        pos_tau_1_2_mas = pixel_coordinates[idx_tau_1_2]
        pos_tau_1_2_pc = mas_to_pc(pos_tau_1_2_mas, z)
    pos_max_I_stripe_mas = pixel_coordinates[idx_max_I_stripe]
    pos_max_I_stripe_pc = mas_to_pc(pos_max_I_stripe_mas, z)
    pos_max_I_mean_mas = pixel_coordinates[idx_max_I_mean]
    pos_max_I_mean_pc = mas_to_pc(pos_max_I_mean_mas, z)

    print("Shape = ", I_orig.shape)
    print("idx_tau_1 = ", idx_tau_1)
    core_flux = np.sum(I_orig[:, :idx_tau_1])
    jet_flux = np.sum(I_orig[:, idx_tau_1:])

    return {"tau_1_1": pos_tau_1_1_pc, "tau_1_2": pos_tau_1_2_pc,
            "max_I_stripe": pos_max_I_stripe_pc, "max_I_mean": pos_max_I_mean_pc,
            "core_flux": core_flux, "tau_1_1_mas": pos_tau_1_1_mas, "tau_1_2_mas": pos_tau_1_2_mas,
            "max_I_stripe_mas": pos_max_I_stripe_mas, "max_I_mean_mas": pos_max_I_mean_mas,
            "core_flux": core_flux, "jet_flux": jet_flux}


def process_raw_images(basename, txt_dir, save_dir, z, plot, match_resolution,
                       n_along, n_across, lg_pixsize_min_mas, lg_pixsize_max_mas,
                       ts_obs_days, flare_params, flare_shape,
                       Gamma, LOS_coeff, b, B_1, n, gamma_min, gamma_max, s):
    if match_resolution:
        lg_pixel_size_mas_min = {2.3: lg_pixsize_min_mas, 8.6: lg_pixsize_min_mas-np.log10(8.6/2.3)}
        lg_pixel_size_mas_max = {2.3: lg_pixsize_max_mas, 8.6: lg_pixsize_max_mas-np.log10(8.6/2.3)}
    else:
        lg_pixel_size_mas_min = {2.3: lg_pixsize_min_mas, 8.6: lg_pixsize_min_mas}
        lg_pixel_size_mas_max = {2.3: lg_pixsize_max_mas, 8.6: lg_pixsize_max_mas}

    for t_obs_days in ts_obs_days:
        print("Processing raw images for T[days] = {:.1f}".format(t_obs_days))
        imagei_txt = os.path.join(txt_dir, "jet_image_i_u_{:.1f}.txt".format(t_obs_days))
        imageq_txt = os.path.join(txt_dir, "jet_image_q_u_{:.1f}.txt".format(t_obs_days))
        imageu_txt = os.path.join(txt_dir, "jet_image_u_u_{:.1f}.txt".format(t_obs_days))
        imagetau_txt = os.path.join(txt_dir, "jet_image_tau_u_{:.1f}.txt".format(t_obs_days))

        # convert -delay 10 -loop 0 `ls -tr tobs*.png` animation.gif
        if plot:
            # fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 12))
            # fig, axes = plt.subplots(1, 1)
            imagei = np.loadtxt(imagei_txt)
            imageq = np.loadtxt(imageq_txt)
            imageu = np.loadtxt(imageu_txt)
            imagetau = np.loadtxt(imagetau_txt)

            min_abs_lev = 0.001*np.max(imagei)
            colors_mask = imagei > min_abs_lev

            mask = imagei == 0
            imagei[mask] = np.nan
            imageq[mask] = np.nan
            imageu[mask] = np.nan
            imagetau[mask] = np.nan

            # plt.matshow(imagei)
            # plt.show()
            # plt.matshow(imageq)
            # plt.show()
            # plt.matshow(imageu)
            # plt.show()
            # plt.matshow(imagetau)
            # plt.show()

            imagep = np.hypot(imageq, imageu)
            imagef = imagep/imagei
            imagepang = 0.5*np.arctan2(imageu, imageq)

            tau_mask = np.logical_and(np.log10(imagetau) > 1, np.log10(imagetau) < -1)
            imagetau[tau_mask] = np.nan

            fig = plot_function(contours=imagep, colors=np.log10(imagetau), vectors=imagepang,
                       vectors_values=None, min_rel_level=0.05,
                       vinc=4, contour_color="gray", vector_color="k", cmap="gist_rainbow",
                       vector_enlarge_factor=8, colorbar_label=r"$\lg{\tau}$")
            fig = plot_function(contours=imagei, abs_levels=[0.01*np.max(imagei)], fig=fig)
            axes = fig.get_axes()[0]
            axes.annotate("{:05.1f} months".format((1+z)*t_obs_days/30), xy=(0.03, 0.9), xycoords="axes fraction", color="gray",
                          weight='bold', ha='left', va='center', size=10)
            fig.savefig(os.path.join(save_dir, "{}_true_poltau_{}_{:.1f}.png".format(basename, "u", t_obs_days)), dpi=600, bbox_inches="tight")
            plt.close()

            fig = plot_function(contours=imagep, colors=imagef, vectors=imagepang,
                                vectors_values=None, min_rel_level=0.05,
                                vinc=4, contour_color="gray", vector_color="k", cmap="gist_rainbow",
                                vector_enlarge_factor=8, colorbar_label="FPOL")
            fig = plot_function(contours=imagei, abs_levels=[0.01*np.max(imagei)], fig=fig)
            axes = fig.get_axes()[0]
            axes.annotate("{:05.1f} months".format((1+z)*t_obs_days/30), xy=(0.03, 0.9), xycoords="axes fraction", color="gray",
                          weight='bold', ha='left', va='center', size=10)
            fig.savefig(os.path.join(save_dir, "{}_true_polfrac_{}_{:.1f}.png".format(basename, "u", t_obs_days)), dpi=600, bbox_inches="tight")
            plt.close()

            # # PPOL contours
            # fig = iplot(contours=imagep, min_abs_level=min_abs_lev,
            #             close=False, contour_color='gray', contour_linewidth=0.25)
            # # Add single IPOL contour and vectors of the PANG
            # fig = iplot(contours=imagei, vectors=imagepang,
            #             vinc=4, contour_linewidth=1.0,
            #             vectors_mask=colors_mask, abs_levels=[2*min_abs_lev],
            #             close=True, show=False,
            #             contour_color='gray', fig=fig, vector_color="black", plot_colorbar=False,
            #             vector_scale=4)
            # axes = fig.get_axes()[0]
            # axes.annotate("{:05.1f} months".format((1+z)*t_obs_days/30), xy=(0.03, 0.9), xycoords="axes fraction", color="gray",
            #               weight='bold', ha='left', va='center', size=10)
            # fig.savefig(os.path.join(save_dir, "{}_observed_pol_{}_{:.1f}.png".format(basename, "u", t_obs_days)), dpi=600, bbox_inches="tight")



if __name__ == "__main__":
    txt_dir = "/home/ilya/data/flares/pol"
    save_dir = "/home/ilya/data/flares/pol"
    z = 1.0
    plot = True
    match_resolution = False
    lg_pixsize_min_mas = -2.5
    lg_pixsize_max_mas = -0.5
    n_along = 400
    n_across = 80

    basename = "pol_0"
    # ts_obs_days = np.linspace(-400.0, 8*360, 20)
    ts_obs_days = [-400.0]

    flare_params = [(2.0, 0.0, 0.0, 0.2)]
    flare_shape = 20.0
    Gamma = 10.0
    LOS_coeff = 0.5
    b = 1.0
    B_1 = 2.0
    n = 2.0
    gamma_min = 10.
    gamma_max = 1E+04
    s = 2.

    process_raw_images(basename, txt_dir, save_dir, z, plot, match_resolution,
                       n_along, n_across, lg_pixsize_min_mas, lg_pixsize_max_mas,
                       ts_obs_days, flare_params, flare_shape,
                       Gamma, LOS_coeff, b, B_1, n, gamma_min, gamma_max, s)
